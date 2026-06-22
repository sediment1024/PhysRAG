"""
Select Physical Feature using VideoCLIP-XL + FAISS RAG

This script provides functions to retrieve the best matching physical feature
for a text prompt using VideoCLIP-XL text encoding and FAISS vector search.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import faiss

# Add VideoCLIP-XL to path
# File is in PhysicalDB/utils/, VideoCLIP-XL source code is in PhysicalDB/VideoCLIP-XL/
# __file__ -> utils/ -> PhysicalDB/ -> PhysicalDB/VideoCLIP-XL/
videoclip_xl_dir = Path(__file__).resolve().parent.parent / "VideoCLIP-XL"
videoclip_xl_dir_str = str(videoclip_xl_dir.resolve())
if videoclip_xl_dir_str not in sys.path:
    sys.path.insert(0, videoclip_xl_dir_str)

# Verify path exists
if not videoclip_xl_dir.exists():
    raise ImportError(
        f"VideoCLIP-XL directory not found at: {videoclip_xl_dir}\n"
        f"Please ensure VideoCLIP-XL is in PhysicalDB/VideoCLIP-XL/ directory."
    )

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder


class VideoCLIPXLRetriever:
    """
    VideoCLIP-XL + FAISS retriever for physical features.
    
    This class loads the FAISS index and VideoCLIP-XL model once,
    then can be reused for multiple queries.
    """
    
    def __init__(
        self,
        index_dir: str,
        model_path: str,
        device: torch.device = None,
        temperature: float = 100.0,
        verbose: bool = False,
    ):
        """
        Initialize the retriever.
        
        Args:
            index_dir: Directory containing FAISS index and metadata
            model_path: Path to VideoCLIP-XL.bin
            device: Device to use (default: cuda if available)
            temperature: Temperature parameter for similarity scaling (default: 100.0)
            verbose: Whether to print loading messages (default: False)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.temperature = temperature
        self.verbose = verbose
        
        # Load FAISS index
        index_dir_path = Path(index_dir)
        index_file = index_dir_path / "video_features.index"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        if self.verbose:
            print(f"✓ Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load metadata
        metadata_file = index_dir_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        for item in self.metadata:
            for key in ("video_path", "feature_path"):
                value = item.get(key)
                if value and not Path(value).is_absolute():
                    item[key] = str((index_dir_path / value).resolve())
            entry = item.get("entry")
            if isinstance(entry, dict):
                value = entry.get("video_path")
                if value and not Path(value).is_absolute():
                    entry["video_path"] = str((index_dir_path / value).resolve())
        
        if self.verbose:
            print(f"✓ Loaded metadata: {len(self.metadata)} entries")
        
        # Load config
        config_file = index_dir_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Load VideoCLIP-XL model
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_obj}")
        
        if self.verbose:
            print(f"Loading VideoCLIP-XL model from: {model_path_obj}")
        self.model = VideoCLIP_XL()
        state_dict = torch.load(str(model_path_obj), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        if self.verbose:
            print(f"✓ Model loaded successfully")
    
    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt using VideoCLIP-XL.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Normalized text feature vector [768]
        """
        with torch.no_grad():
            text_inputs = text_encoder.tokenize([prompt], truncate=True).to(self.device)
            text_features = self.model.text_model.encode_text(text_inputs).float()
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Shape: [1, 768]
            return text_features.squeeze(0)  # [768]
    
    def search(self, prompt: str, k: int = 1) -> Tuple[Dict[str, Any], float]:
        """
        Search for best matching physical feature.
        
        Args:
            prompt: Text prompt to search for
            k: Number of top results to return (default: 1)
            
        Returns:
            Tuple of (best_metadata, best_score):
            - best_metadata: Metadata dict for the best match
            - best_score: Similarity score (after temperature scaling)
        """
        results = self.search_topk(prompt, k=k)
        best = results[0]
        return best["metadata"], best["score"]

    def search_topk(self, prompt: str, k: int = 1) -> List[Dict[str, Any]]:
        """
        Search for the top-k matching physical features.

        Returns one dict per retrieved video with:
        - metadata: metadata entry
        - raw_score: raw FAISS inner product / cosine similarity
        - score: raw_score multiplied by self.temperature, matching search()
        - rank: 1-based rank
        """
        text_feat = self.encode_text(prompt)  # [768]
        text_feat_np = text_feat.cpu().numpy().astype('float32').reshape(1, -1)

        k = min(max(int(k), 1), self.index.ntotal)
        scores, indices = self.index.search(text_feat_np, k)

        results = []
        for rank, (idx, raw_score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            raw = float(raw_score)
            results.append(
                {
                    "metadata": self.metadata[int(idx)],
                    "raw_score": raw,
                    "score": raw * self.temperature,
                    "rank": rank,
                }
            )
        if not results:
            raise ValueError(f"No valid FAISS results for prompt: {prompt}")
        return results


# Global retriever instance (lazy loading)
# Note: This is now only used in the main process (not in DataLoader workers).
# Physical feature loading has been moved from collate_fn (workers) to compute_loss (main process)
# to avoid loading models in worker processes.
_global_retriever = None


def get_retriever(
    index_dir: str = None,
    model_path: str = None,
    device: torch.device = None,
    temperature: float = 100.0,
    verbose: bool = False,
) -> VideoCLIPXLRetriever:
    """
    Get or create global retriever instance.
    
    Note: This function is now only called from the main process (in compute_loss).
    Physical feature loading has been moved from collate_fn to compute_loss to avoid
    loading VideoCLIP-XL models in DataLoader worker processes.
    
    Args:
        index_dir: Directory containing FAISS index (default: PhysicalDB/faiss_index)
        model_path: Path to VideoCLIP-XL.bin (default: PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin)
        device: Device to use (default: cuda if available, else cpu)
        temperature: Temperature parameter
        
    Returns:
        VideoCLIPXLRetriever instance
    """
    global _global_retriever
    
    # Set device (default to CUDA if available, since we're in main process)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lazy initialization: create retriever on first call
    if _global_retriever is None:
        # File is in PhysicalDB/utils/, VideoCLIP-XL is in PhysicalDB/VideoCLIP-XL/
        physicaldb_dir = Path(__file__).resolve().parent.parent
        
        if index_dir is None:
            # FAISS index is in PhysicalDB/faiss_index/
            index_dir = str(physicaldb_dir / "faiss_index")
        
        if model_path is None:
            # Model weight file is in PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin
            model_path = str(physicaldb_dir / "VideoCLIP-XL" / "VideoCLIP-XL.bin")
        
        _global_retriever = VideoCLIPXLRetriever(
            index_dir=index_dir,
            model_path=model_path,
            device=device,
            temperature=temperature,
            verbose=verbose,
        )
    
    return _global_retriever


def _resolve_feature_path(metadata: Dict[str, Any]) -> Path:
    feature_path = Path(metadata["feature_path"])
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            f"Please run extract_features_new_physical_db.py to extract features for this video.\n"
            f"Video: {metadata.get('video_path', 'unknown')}"
        )
    return feature_path


def select_best_feature_for_prompt(
    prompt: str,
    db_json: str = None,  # Not used, kept for compatibility
    feature_root: str = None,  # Not used, kept for compatibility
    clip_model_name: str = None,  # Not used, kept for compatibility
    num_frames: int = None,  # Not used, kept for compatibility
    index_dir: str = None,
    model_path: str = None,
    device: torch.device = None,
    temperature: float = 100.0,
    verbose: bool = False,
) -> Tuple[Optional[Path], Optional[Dict], Optional[float]]:
    """
    Select best physical feature for a prompt using VideoCLIP-XL + FAISS.
    
    This function maintains the same interface as select_physical_feature_xclip.py
    for compatibility, but uses VideoCLIP-XL + FAISS instead.
    
    Args:
        prompt: Text prompt to match against
        db_json: (Deprecated, kept for compatibility) Path to physical database JSON
        feature_root: (Deprecated, kept for compatibility) Root directory for features
        clip_model_name: (Deprecated, kept for compatibility) CLIP model name
        num_frames: (Deprecated, kept for compatibility) Number of frames
        index_dir: Directory containing FAISS index (default: PhysicalDB/faiss_index)
        model_path: Path to VideoCLIP-XL.bin (default: PhysicalDB/VideoCLIP-XL/VideoCLIP-XL.bin)
        device: Device to use
        temperature: Temperature parameter for similarity scaling (default: 100.0)
        
    Returns:
        Tuple of (feature_path, entry_dict, score):
        - feature_path: Path to the best matching feature file
        - entry_dict: Database entry dict from PhyDB
        - score: VideoCLIP-XL similarity score (after temperature scaling)
        
    Raises:
        FileNotFoundError: If index, metadata, or feature file is not found
        ValueError: If no valid match is found
    """
    # Get retriever
    retriever = get_retriever(
        index_dir=index_dir,
        model_path=model_path,
        device=device,
        temperature=temperature,
        verbose=verbose,
    )
    
    # Search for best match
    best_metadata, best_score = retriever.search(prompt, k=1)
    
    feature_path = _resolve_feature_path(best_metadata)
    
    # Get entry dict from metadata
    entry_dict = best_metadata.get("entry", {})
    
    return feature_path, entry_dict, best_score


def select_topk_features_for_prompt(
    prompt: str,
    k: int = 1,
    index_dir: str = None,
    model_path: str = None,
    device: torch.device = None,
    temperature: float = 100.0,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Select top-k physical features for a prompt using VideoCLIP-XL + FAISS.

    Each returned item contains:
    - feature_path: resolved feature file path
    - entry: metadata entry dict
    - raw_score: raw cosine / inner-product similarity
    - score: temperature-scaled score, matching select_best_feature_for_prompt
    - rank: retrieval rank
    """
    retriever = get_retriever(
        index_dir=index_dir,
        model_path=model_path,
        device=device,
        temperature=temperature,
        verbose=verbose,
    )
    results = []
    for item in retriever.search_topk(prompt, k=k):
        metadata = item["metadata"]
        results.append(
            {
                "feature_path": _resolve_feature_path(metadata),
                "entry": metadata.get("entry", {}),
                "raw_score": float(item["raw_score"]),
                "score": float(item["score"]),
                "rank": int(item["rank"]),
            }
        )
    return results


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VideoCLIP-XL + FAISS retrieval")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A ball bouncing on the ground",
        help="Text prompt to search for"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="faiss_index",
        help="Directory containing FAISS index (relative to PhysicalDB/ directory, or absolute path)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="VideoCLIP-XL/VideoCLIP-XL.bin",
        help="Path to VideoCLIP-XL.bin (relative to PhysicalDB/ directory, or absolute path)"
    )
    
    args = parser.parse_args()
    
    # File is in PhysicalDB/utils/, PhysicalDB dir is parent.parent
    physicaldb_dir = Path(__file__).resolve().parent.parent
    project_root = physicaldb_dir.parent
    
    # Resolve paths (defaults are relative to PhysicalDB)
    if Path(args.index_dir).is_absolute():
        index_dir = Path(args.index_dir).resolve()
    else:
        index_dir = (physicaldb_dir / args.index_dir).resolve()
    
    if Path(args.model_path).is_absolute():
        model_path = Path(args.model_path).resolve()
    else:
        model_path = (physicaldb_dir / args.model_path).resolve()
    
    print("=" * 60)
    print("Testing VideoCLIP-XL + FAISS Retrieval")
    print("=" * 60)
    print(f"\nPrompt: {args.prompt}")
    
    feature_path, entry, score = select_best_feature_for_prompt(
        prompt=args.prompt,
        index_dir=str(index_dir),
        model_path=str(model_path),
        verbose=True,  # Enable verbose for testing
    )
    
    print(f"\n✓ Best match found:")
    print(f"  Score: {score:.4f}")
    print(f"  Feature path: {feature_path}")
    print(f"  Category: {entry.get('category', 'unknown')}")
    print(f"  Video: {entry.get('video_path', 'unknown')}")
    print(f"  Prompt: {entry.get('prompt', 'unknown')}")
