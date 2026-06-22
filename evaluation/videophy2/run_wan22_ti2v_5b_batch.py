import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import save_video
import wan


def _slug(prompt: str, max_len: int = 120) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt.strip())
    safe = "_".join(filter(None, safe.split("_")))
    if len(safe) > max_len:
        h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        safe = safe[: max_len - 9] + "_" + h
    return safe or "prompt"


def _load_physical_checkpoint(path: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def _filter_state_dict(state: dict, model_state: dict) -> tuple[dict, int]:
    filtered = {}
    skipped = 0
    for key, value in state.items():
        target = model_state.get(key)
        if target is None:
            skipped += 1
            continue
        if value is None or not hasattr(value, "shape"):
            skipped += 1
            continue
        if value.numel() == 0 or value.shape != target.shape:
            skipped += 1
            continue
        filtered[key] = value
    return filtered, skipped


def _load_physical_features(path: Path) -> torch.Tensor:
    ref_features = torch.load(path, map_location="cpu")
    if isinstance(ref_features, dict):
        ref_features = ref_features.get("features", list(ref_features.values())[0])
    if not isinstance(ref_features, torch.Tensor):
        ref_features = torch.tensor(ref_features)
    if ref_features.dim() == 2:
        ref_features = ref_features.unsqueeze(0)
    return ref_features


def _compute_target_shape_and_seq_len(frame_num, size, vae_stride, patch_size, sp_size, z_dim):
    f_lat = (frame_num - 1) // vae_stride[0] + 1
    h_lat = size[1] // vae_stride[1]
    w_lat = size[0] // vae_stride[2]
    h_lat = h_lat - (h_lat % patch_size[1])
    w_lat = w_lat - (w_lat % patch_size[2])
    h_tokens = h_lat // patch_size[1]
    w_tokens = w_lat // patch_size[2]
    seq_len = int((f_lat * h_tokens * w_tokens + sp_size - 1) // sp_size * sp_size)
    target_shape = torch.Size([1, z_dim, f_lat, h_lat, w_lat])
    return target_shape, int(seq_len)


def setup_physical_injection(pipe, args, size):
    from wan.modules.physical_adapter import PhysicalAdapter
    from wan.modules.physical_injection import PhysicalInjectionManager

    target_shape, seq_len = _compute_target_shape_and_seq_len(
        frame_num=args.frame_num,
        size=size,
        vae_stride=pipe.vae_stride,
        patch_size=pipe.patch_size,
        sp_size=pipe.sp_size,
        z_dim=pipe.vae.model.z_dim,
    )

    adapter = None
    if args.physical_token_mode == "adapter":
        adapter = PhysicalAdapter(
            input_dim=768,
            hidden_dim=768,
            output_channels=args.physical_adapter_dim,
            num_queries=args.physical_num_queries,
        ).to(dtype=pipe.config.param_dtype)

    injection_layers = [int(x) for x in args.physical_injection_layers.split(",") if x.strip()]
    manager = PhysicalInjectionManager(
        model=pipe.model,
        physical_adapter=adapter,
        injection_layers=injection_layers,
        hidden_size=pipe.model.dim,
        adapter_dim=args.physical_adapter_dim,
        seq_len=seq_len,
        patch_size=pipe.patch_size,
        dtype=pipe.config.param_dtype,
        use_direct_tokens=(args.physical_token_mode == "direct"),
        injection_position=args.physical_injection_position,
    )

    if args.physical_ckpt:
        state = _load_physical_checkpoint(args.physical_ckpt)
        model_state = pipe.model.state_dict()
        filtered_state, skipped = _filter_state_dict(state, model_state)
        missing, unexpected = pipe.model.load_state_dict(filtered_state, strict=False)
        logging.info(
            "Physical checkpoint loaded: used=%d skipped=%d missing=%d unexpected=%d",
            len(filtered_state),
            skipped,
            len(missing),
            len(unexpected),
        )

    pipe.physical_injection_manager = manager
    logging.info(
        "Physical injection enabled: layers=%s seq_len=%d target_shape=%s",
        injection_layers,
        seq_len,
        target_shape,
    )
    return manager, target_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with upsampled_caption column")
    ap.add_argument("--ckpt_dir", required=True, help="Base Wan2.2-TI2V-5B directory")
    ap.add_argument("--physical_ckpt", default=None, help="Path to merged_model.pt")
    ap.add_argument("--out_dir", required=True, help="Output directory for videos")
    ap.add_argument("--size", default="1280*704", choices=SIZE_CONFIGS.keys())
    ap.add_argument("--frame_num", type=int, default=49)
    ap.add_argument("--sample_steps", type=int, default=None)
    ap.add_argument("--sample_guide_scale", type=float, default=None)
    ap.add_argument("--sample_shift", type=float, default=None)
    ap.add_argument("--sample_solver", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t5_cpu", action="store_true", default=False)
    ap.add_argument("--convert_model_dtype", action="store_true")
    ap.add_argument("--offload_model", action="store_true")
    ap.add_argument("--disable_rag", action="store_true", help="Disable RAG physical injection.")
    ap.add_argument("--cogvideo_root", type=str, default=str(REPO_ROOT))
    ap.add_argument("--faiss_index_dir", type=str, default=str(REPO_ROOT / "data" / "rag" / "faiss_index"))
    ap.add_argument(
        "--videoclip_xl_model_path",
        type=str,
        default=str(REPO_ROOT / "PhysicalDB" / "VideoCLIP-XL" / "VideoCLIP-XL.bin"),
    )
    ap.add_argument("--physical_injection_layers", type=str, default="0,1,2")
    ap.add_argument("--physical_adapter_dim", type=int, default=16)
    ap.add_argument("--physical_num_queries", type=int, default=128)
    ap.add_argument(
        "--physical_token_mode",
        type=str,
        default="adapter",
        choices=["adapter", "direct"],
        help="Use adapter+pooling tokens or directly inject MAE tokens.",
    )
    ap.add_argument(
        "--physical_injection_position",
        type=str,
        default="layers",
        choices=["layers", "post"],
        help="Inject tokens at specific layers or once after all blocks (before head).",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    ap.add_argument("--start", type=int, default=0, help="Start index for sharding")
    ap.add_argument("--end", type=int, default=None, help="End index (exclusive) for sharding")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output exists")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "upsampled_caption" not in df.columns:
        raise ValueError("CSV missing 'upsampled_caption' column")

    prompts = df["upsampled_caption"].tolist()
    if args.end is None:
        args.end = len(prompts)
    prompts = prompts[args.start:args.end]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    cfg = WAN_CONFIGS["ti2v-5B"]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.sample_solver is None:
        args.sample_solver = getattr(cfg, "sample_solver", "unipc")

    size = SIZE_CONFIGS[args.size]
    logging.info("Loading Wan TI2V pipeline...")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    rag_enabled = not args.disable_rag
    manager = None
    target_shape = None
    select_best_feature_for_prompt = None
    if rag_enabled:
        cogvideo_root = Path(args.cogvideo_root)
        sys.path.append(str(cogvideo_root))
        from PhysicalDB.utils.select_physical_feature_videoclip_xl import (
            select_best_feature_for_prompt,
        )
        manager, target_shape = setup_physical_injection(wan_ti2v, args, size=size)
    else:
        if args.physical_ckpt:
            logging.info("Loading physical checkpoint (no RAG)...")
            state = _load_physical_checkpoint(args.physical_ckpt)
            model_state = wan_ti2v.model.state_dict()
            filtered_state, skipped = _filter_state_dict(state, model_state)
            missing, unexpected = wan_ti2v.model.load_state_dict(filtered_state, strict=False)
            logging.info(
                "Physical checkpoint loaded: used=%d skipped=%d missing=%d unexpected=%d",
                len(filtered_state),
                skipped,
                len(missing),
                len(unexpected),
            )
    total = len(prompts)
    logging.info("Starting inference: %d prompts", total)
    for offset, prompt in enumerate(prompts):
        idx = args.start + offset
        fname = f"{idx:04d}_{_slug(prompt)}.mp4"
        out_path = out_dir / fname
        if args.skip_existing and out_path.exists():
            logging.info("[%d/%d] Skip existing: %s", offset + 1, total, out_path)
            continue

        logging.info("[%d/%d] Prompt: %s", offset + 1, total, prompt[:100])
        try:
            if rag_enabled:
                feature_path, _, _ = select_best_feature_for_prompt(
                    prompt=prompt,
                    index_dir=args.faiss_index_dir,
                    model_path=args.videoclip_xl_model_path,
                    device=torch.device("cuda"),
                )
                ref_features = _load_physical_features(Path(feature_path))
                manager.set_ref_features(ref_features, target_shape)

            video = wan_ti2v.generate(
                prompt,
                size=size,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.seed + idx,
                offload_model=args.offload_model,
            )

            save_video(
                tensor=video[None],
                save_file=str(out_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info("Saved: %s", out_path)
        except Exception as exc:
            logging.exception("Failed on %d: %s", idx, exc)
            if os.environ.get("WAN_FAIL_FAST") == "1":
                raise

    logging.info("Finished.")


if __name__ == "__main__":
    main()
