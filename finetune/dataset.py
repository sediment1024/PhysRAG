import hashlib
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required for video loading") from exc


def _load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def _sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    pad = np.full(num_frames - total_frames, total_frames - 1, dtype=np.int64)
    return np.concatenate([np.arange(total_frames, dtype=np.int64), pad], axis=0)


def _read_video_cv2(path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _sample_frame_indices(total_frames, num_frames)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")

    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    frames_np = np.stack(frames, axis=0)  # [F, H, W, C]
    frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()
    frames_tensor = frames_tensor / 127.5 - 1.0
    return frames_tensor  # [C, F, H, W]


def _atomic_save(obj: torch.Tensor, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _pad_prompt_embedding(embedding: torch.Tensor, text_len: Optional[int]) -> torch.Tensor:
    if text_len is None:
        return embedding
    if embedding.size(0) >= text_len:
        return embedding[:text_len]
    pad = torch.zeros(
        text_len - embedding.size(0),
        embedding.size(1),
        dtype=embedding.dtype,
        device=embedding.device,
    )
    return torch.cat([embedding, pad], dim=0)


class VideoTextDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        num_frames: int,
        height: int,
        width: int,
        encode_text: Optional[Callable[[str], torch.Tensor]] = None,
        encode_video: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cache_root: Optional[str] = None,
        text_len: Optional[int] = None,
        resolution: Optional[str] = None,
    ) -> None:
        super().__init__()
        root = Path(data_root)
        caption_path = root / caption_column if caption_column else Path(caption_column)
        video_path = root / video_column if video_column else Path(video_column)

        self.prompts = _load_lines(caption_path)
        self.videos = []
        for value in _load_lines(video_path):
            path = Path(value)
            self.videos.append(path if path.is_absolute() else root / path)

        if len(self.prompts) != len(self.videos):
            raise ValueError(
                f"prompts/videos length mismatch: {len(self.prompts)} vs {len(self.videos)}"
            )

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.encode_text = encode_text
        self.encode_video = encode_video
        self.text_len = text_len
        self.resolution = resolution or f"{num_frames}x{height}x{width}"

        self.cache_root = Path(cache_root) if cache_root else None
        self.prompt_cache_dir = None
        self.video_cache_dir = None
        if self.cache_root:
            self.prompt_cache_dir = self.cache_root / "prompt_embeddings"
            self.video_cache_dir = self.cache_root / "video_latent" / self.resolution
            self.prompt_cache_dir.mkdir(parents=True, exist_ok=True)
            self.video_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.videos)

    def _prompt_cache_path(self, prompt: str) -> Optional[Path]:
        if self.prompt_cache_dir is None:
            return None
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.prompt_cache_dir / f"{prompt_hash}.pt"

    def _video_cache_path(self, video_path: Path) -> Optional[Path]:
        if self.video_cache_dir is None:
            return None
        video_hash = hashlib.sha256(str(video_path).encode("utf-8")).hexdigest()
        return self.video_cache_dir / f"{video_hash}.pt"

    def _get_prompt_embedding(self, prompt: str) -> Optional[torch.Tensor]:
        cache_path = self._prompt_cache_path(prompt)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        if self.encode_text is None:
            return None
        embedding = self.encode_text(prompt)
        embedding = _pad_prompt_embedding(embedding, self.text_len)
        embedding = embedding.detach().cpu()
        if cache_path is not None:
            _atomic_save(embedding, cache_path)
        return embedding

    def _get_encoded_video(self, video_path: Path, video: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        cache_path = self._video_cache_path(video_path)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        if self.encode_video is None:
            return None
        if video is None:
            video = _read_video_cv2(video_path, self.num_frames, self.height, self.width)
        latent = self.encode_video(video)
        latent = latent.detach().cpu()
        if cache_path is not None:
            _atomic_save(latent, cache_path)
        return latent

    def __getitem__(self, idx: int):
        prompt = self.prompts[idx]
        video_path = self.videos[idx]

        prompt_embedding = self._get_prompt_embedding(prompt)

        encoded_video = self._get_encoded_video(video_path, None)
        video = None
        if encoded_video is None:
            video = _read_video_cv2(video_path, self.num_frames, self.height, self.width)
            encoded_video = self._get_encoded_video(video_path, video)

        return {
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video": video,
            "video_path": str(video_path),
        }


def parse_resolution(resolution: str) -> Tuple[int, int, int]:
    parts = resolution.lower().split("x")
    if len(parts) != 3:
        raise ValueError("train_resolution must be in format 'framesxheightxwidth'")
    frames, height, width = (int(p) for p in parts)
    return frames, height, width
