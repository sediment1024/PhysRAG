import argparse
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import VideoTextDataset, parse_resolution


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def load_ti2v_config():
    import importlib.util

    if "wan.configs.shared_config" not in sys.modules:
        if "wan.configs" not in sys.modules:
            configs_module = types.ModuleType("wan.configs")
            sys.modules["wan.configs"] = configs_module
        shared_config_file = REPO_ROOT / "wan" / "configs" / "shared_config.py"
        shared_spec = importlib.util.spec_from_file_location(
            "wan.configs.shared_config", shared_config_file
        )
        shared_module = importlib.util.module_from_spec(shared_spec)
        shared_module.__package__ = "wan.configs"
        sys.modules["wan.configs.shared_config"] = shared_module
        shared_spec.loader.exec_module(shared_module)

    ti2v_5b_file = REPO_ROOT / "wan" / "configs" / "wan_ti2v_5B.py"
    spec = importlib.util.spec_from_file_location("wan.configs.wan_ti2v_5B", ti2v_5b_file)
    ti2v_5b_module = importlib.util.module_from_spec(spec)
    ti2v_5b_module.__package__ = "wan.configs"
    if "wan.configs" not in sys.modules:
        configs_module = types.ModuleType("wan.configs")
        sys.modules["wan.configs"] = configs_module
    sys.modules["wan.configs.wan_ti2v_5B"] = ti2v_5b_module
    spec.loader.exec_module(ti2v_5b_module)
    return ti2v_5b_module.ti2v_5B


def build_text_encoder(ckpt_dir: Path, config, device: torch.device, t5_cpu: bool):
    if "wan.modules" not in sys.modules:
        modules_pkg = types.ModuleType("wan.modules")
        modules_pkg.__path__ = [str(REPO_ROOT / "wan" / "modules")]
        sys.modules["wan.modules"] = modules_pkg

    import importlib.util

    tokenizers_file = REPO_ROOT / "wan" / "modules" / "tokenizers.py"
    tokenizers_spec = importlib.util.spec_from_file_location("wan.modules.tokenizers", tokenizers_file)
    tokenizers_module = importlib.util.module_from_spec(tokenizers_spec)
    tokenizers_module.__package__ = "wan.modules"
    sys.modules["wan.modules.tokenizers"] = tokenizers_module
    tokenizers_spec.loader.exec_module(tokenizers_module)

    from wan.modules.t5 import T5EncoderModel

    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=str(ckpt_dir / config.t5_checkpoint),
        tokenizer_path=str(ckpt_dir / config.t5_tokenizer),
        shard_fn=None,
    )
    if not t5_cpu:
        text_encoder.model.to(device)
    text_encoder.model.eval()
    text_encoder.model.requires_grad_(False)
    return text_encoder


def build_vae(ckpt_dir: Path, device: torch.device):
    import importlib.util

    if "wan.modules" not in sys.modules:
        modules_pkg = types.ModuleType("wan.modules")
        modules_pkg.__path__ = [str(REPO_ROOT / "wan" / "modules")]
        sys.modules["wan.modules"] = modules_pkg

    vae_file = REPO_ROOT / "wan" / "modules" / "vae2_2.py"
    vae_spec = importlib.util.spec_from_file_location("wan.modules.vae2_2", vae_file)
    vae_module = importlib.util.module_from_spec(vae_spec)
    vae_module.__package__ = "wan.modules"
    sys.modules["wan.modules.vae2_2"] = vae_module
    vae_spec.loader.exec_module(vae_module)
    Wan2_2_VAE = vae_module.Wan2_2_VAE

    config = load_ti2v_config()
    vae = Wan2_2_VAE(vae_pth=str(ckpt_dir / config.vae_checkpoint), device=device)
    vae.model.eval()
    vae.model.requires_grad_(False)
    return vae


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Wan2.2 cache (video latents + prompt embeddings).")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--caption_column", type=str, default="prompts_new.txt")
    parser.add_argument("--video_column", type=str, default="videos_new.txt")
    parser.add_argument("--train_resolution", type=str, default="25x480x720")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--skip_text", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    startup = time.time()
    cache_dir = args.cache_dir or str(Path(args.data_root) / "cache_wan")
    log_rank0(
        rank,
        "Initializing cache build: "
        f"device={device}, ckpt_dir={args.ckpt_dir}, data_root={args.data_root}, "
        f"cache_dir={cache_dir}, resolution={args.train_resolution}",
    )

    stage_start = time.time()
    config = load_ti2v_config()
    log_rank0(rank, f"Loaded TI2V config in {time.time() - stage_start:.1f}s")
    ckpt_dir = Path(args.ckpt_dir)
    frames, height, width = parse_resolution(args.train_resolution)

    text_encoder = None
    if not args.skip_text:
        log_rank0(rank, f"Loading T5 text encoder on {'cpu' if args.t5_cpu else device}...")
        stage_start = time.time()
        text_encoder = build_text_encoder(ckpt_dir, config, device, args.t5_cpu)
        log_rank0(rank, f"T5 text encoder ready in {time.time() - stage_start:.1f}s")
    else:
        log_rank0(rank, "Skipping text encoder; only video latents will be cached.")

    log_rank0(rank, f"Loading Wan VAE on {device}...")
    stage_start = time.time()
    vae = build_vae(ckpt_dir, device)
    log_rank0(rank, f"Wan VAE ready in {time.time() - stage_start:.1f}s")

    def encode_text_fn(prompt: str) -> torch.Tensor:
        if text_encoder is None:
            return None
        with torch.no_grad():
            if args.t5_cpu:
                return text_encoder([prompt], torch.device("cpu"))[0]
            return text_encoder([prompt], device)[0]

    def encode_video_fn(video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.encode([video.to(device=device)])[0]

    log_rank0(rank, "Building dataset index...")
    stage_start = time.time()
    dataset = VideoTextDataset(
        data_root=args.data_root,
        caption_column=args.caption_column,
        video_column=args.video_column,
        num_frames=frames,
        height=height,
        width=width,
        encode_text=None if args.skip_text else encode_text_fn,
        encode_video=encode_video_fn,
        cache_root=cache_dir,
        text_len=config.text_len,
        resolution=args.train_resolution,
    )
    log_rank0(
        rank,
        f"Dataset ready in {time.time() - stage_start:.1f}s with {len(dataset)} items "
        f"(startup total {time.time() - startup:.1f}s)",
    )

    indices = list(range(rank, len(dataset), world_size))
    total = len(indices)
    start = time.time()

    log_rank0(
        rank,
        f"Cache build start: dataset={len(dataset)}, world_size={world_size}, "
        f"per_rank={total}, resolution={args.train_resolution}",
    )

    for idx, dataset_idx in enumerate(indices, 1):
        if idx == 1:
            log_rank0(rank, f"[Rank {rank}] Processing first sample (dataset index {dataset_idx})...")
        _ = dataset[dataset_idx]
        if idx == 1 and args.log_every != 1:
            log_rank0(rank, f"[Rank {rank}] First sample cached.")
        if idx % args.log_every == 0 or idx == total:
            elapsed = time.time() - start
            rate = elapsed / idx if idx else 0.0
            remaining = (total - idx) * rate
            print(
                f"[Rank {rank}] {idx}/{total} cached | "
                f"elapsed: {elapsed:.1f}s | remaining: {remaining:.1f}s",
                flush=True,
            )

    if dist.is_initialized():
        # Allow skipping barrier to avoid long waits if one rank finishes earlier.
        if os.environ.get("DISABLE_DIST_BARRIER") != "1":
            dist.barrier()
        dist.destroy_process_group()

    log_rank0(rank, "Cache build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
