"""
Build FAISS Index for VideoCLIP-XL RAG

This script:
1. Loads VideoCLIP-XL model
2. Encodes all videos in PhyDB using VideoCLIP-XL
3. Builds a FAISS IndexFlatIP index
4. Saves index and metadata to PhysicalDB/faiss_index/
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import cv2
import faiss

# Add VideoCLIP-XL source code to path
# File location: PhysicalDB/utils/build_faiss_index.py
# VideoCLIP-XL location: PhysicalDB/VideoCLIP-XL/
videoclip_xl_dir = Path(__file__).resolve().parent.parent / "VideoCLIP-XL"
if not videoclip_xl_dir.exists():
    raise ImportError(f"VideoCLIP-XL directory not found at: {videoclip_xl_dir}")

sys.path.insert(0, str(videoclip_xl_dir))
from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder


def _frame_from_video(video):
    """Generator to read frames from video."""
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


# ImageNet normalization
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    """Normalize image data using ImageNet statistics."""
    return (data / 255.0 - v_mean) / v_std


def video_preprocessing(video_path, fnum=8):
    """Preprocess video for VideoCLIP-XL."""
    video = cv2.VideoCapture(str(video_path))
    frames = list(_frame_from_video(video))
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    # Uniform sampling
    step = max(1, len(frames) // fnum)
    frames = frames[::step][:fnum]
    
    # Repeat last frame if not enough
    while len(frames) < fnum:
        frames.append(frames[-1])
    
    vid_tube = []
    for fr in frames:
        fr = fr[:, :, ::-1]  # BGR -> RGB
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), axis=(0, 1))
        vid_tube.append(fr)
    
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    return torch.from_numpy(vid_tube)


def build_faiss_index(db_json_path: str, model_path: str, feature_root: str, output_dir: str, device=None):
    """Build FAISS index for all videos in PhyDB."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Building FAISS Index for VideoCLIP-XL RAG")
    print("=" * 60)
    
    # Load PhyDB
    db_path = Path(db_json_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    with open(db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
    
    print(f"\n1. Loaded PhyDB: {len(db_data)} entries")
    
    # Load VideoCLIP-XL model
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")
    
    print(f"\n2. Loading VideoCLIP-XL model from: {model_path_obj}")
    model = VideoCLIP_XL()
    model.load_state_dict(torch.load(str(model_path_obj), map_location="cpu"))
    model.to(device).eval()
    print(f"   ✓ Model loaded successfully")
    
    # Encode all videos
    print(f"\n3. Encoding {len(db_data)} videos...")
    video_features_list = []
    metadata_list = []
    failed_count = 0
    
    with torch.no_grad():
        for idx, entry in enumerate(db_data):
            video_path = Path(entry.get("video_path", ""))
            
            if not video_path.exists():
                print(f"  ⚠ [{idx+1}/{len(db_data)}] Video not found: {video_path}")
                failed_count += 1
                continue
            
            try:
                # Preprocess and encode video
                video_input = video_preprocessing(video_path, fnum=8).float().to(device)
                video_feat = model.vision_model.get_vid_features(video_input).float()
                video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)  # L2 normalize
                video_feat_np = video_feat.cpu().numpy().astype('float32').squeeze(0)
                
                video_features_list.append(video_feat_np)
                
                # Prepare metadata
                category = entry.get("category", "unknown")
                video_name = entry.get("video_name", "")
                video_stem = Path(video_name).stem
                feature_path = Path(feature_root) / category / f"{video_stem}.pt"
                
                metadata_list.append({
                    "index_id": len(metadata_list),
                    "video_path": str(video_path),
                    "video_name": video_name,
                    "category": category,
                    "feature_path": str(feature_path),
                    "entry": entry
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"  ✓ Processed {idx+1}/{len(db_data)} videos")
                    
            except Exception as e:
                print(f"  ✗ [{idx+1}/{len(db_data)}] Failed: {video_path.name}: {e}")
                failed_count += 1
    
    if len(video_features_list) == 0:
        raise ValueError("No videos were successfully encoded!")
    
    print(f"\n✓ Successfully encoded {len(video_features_list)} videos")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} videos")
    
    # Build FAISS index
    print(f"\n4. Building FAISS index...")
    video_features = np.vstack(video_features_list).astype('float32')  # [N, 768]
    index = faiss.IndexFlatIP(video_features.shape[1])
    index.add(video_features)
    print(f"   ✓ Index built with {index.ntotal} vectors (dim={video_features.shape[1]})")
    
    # Save index and metadata
    print(f"\n5. Saving index and metadata...")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_file = output_dir_path / "video_features.index"
    faiss.write_index(index, str(index_file))
    print(f"   ✓ Saved index: {index_file}")
    
    # Save metadata
    metadata_file = output_dir_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved metadata: {metadata_file}")
    
    # Save config
    config_file = output_dir_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_path": str(model_path_obj),
            "feature_dim": int(video_features.shape[1]),
            "num_videos": len(metadata_list),
            "created_at": datetime.now().isoformat(),
            "db_json_path": str(db_path),
            "feature_root": str(feature_root),
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved config: {config_file}")
    
    print("\n" + "=" * 60)
    print("✓ FAISS Index Build Complete!")
    print("=" * 60)
    print(f"Summary: {len(metadata_list)}/{len(db_data)} videos encoded, {failed_count} failed")
    print(f"Output: {output_dir_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for VideoCLIP-XL RAG")
    parser.add_argument("--db_json", type=str, default="PhysicalDB/new_physical_db.json",
                       help="Path to new_physical_db.json (relative to project root)")
    parser.add_argument("--model_path", type=str, default="VideoCLIP-XL/VideoCLIP-XL.bin",
                       help="Path to VideoCLIP-XL.bin (relative to PhysicalDB/ or absolute)")
    parser.add_argument("--feature_root", type=str, default="dataset/physical_features",
                       help="Root directory for physical features (relative to project root)")
    parser.add_argument("--output_dir", type=str, default="PhysicalDB/faiss_index",
                       help="Output directory (relative to project root)")
    
    args = parser.parse_args()
    
    # Get base directories
    physicaldb_dir = Path(__file__).resolve().parent.parent
    project_root = physicaldb_dir.parent
    
    # Resolve paths
    db_json_path = (project_root / args.db_json).resolve()
    feature_root = (project_root / args.feature_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    
    # Model path: try PhysicalDB first, then project root
    model_path_arg = Path(args.model_path)
    if model_path_arg.is_absolute():
        model_path = model_path_arg.resolve()
    else:
        model_path = (physicaldb_dir / args.model_path).resolve()
        if not model_path.exists():
            model_path = (project_root / args.model_path).resolve()
    
    build_faiss_index(
        db_json_path=str(db_json_path),
        model_path=str(model_path),
        feature_root=str(feature_root),
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
