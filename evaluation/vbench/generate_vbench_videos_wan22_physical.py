#!/usr/bin/env python3
"""
VBench inference script (not training).

Generate videos for VBench 1st-gen evaluation using Wan2.2 Physical TI2V.
Same loading pattern as run_wan22_ti2v_5b_batch.py: WanTI2V from base ckpt_dir,
optional physical weights (merged_model.pt) via _filter_state_dict + load_state_dict,
optional setup_physical_injection (PhysicalAdapter, RAG) when RAG is enabled.
Reads prompts from VBench txt files; generates 5 videos per prompt per category.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Wan2.2 repo root (script lives under Benchmark/VBench/DiT-Mem/evaluation/vbench/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_WAN_REPO_ROOT = _SCRIPT_DIR.parents[5]
if str(_WAN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAN_REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

# Set CUDA device before importing torch (for single-process inference)
for i, a in enumerate(sys.argv):
    if a == "--cuda_device" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

import torch

# -----------------------------------------------------------------------------
# Helpers for loading physical checkpoint at inference (same as run_wan22_ti2v_5b_batch.py)
# -----------------------------------------------------------------------------


def _load_physical_checkpoint(path: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _load_physical_features(path: Path) -> torch.Tensor:
    """Load ref features from disk (same as run_wan22_ti2v_5b_batch.py)."""
    ref_features = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ref_features, dict):
        ref_features = ref_features.get("features", list(ref_features.values())[0])
    if not isinstance(ref_features, torch.Tensor):
        ref_features = torch.tensor(ref_features)
    if ref_features.dim() == 2:
        ref_features = ref_features.unsqueeze(0)
    return ref_features


def _filter_state_dict(state: dict, model_state: dict):
    """Return (filtered_dict, skipped_count)."""
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


def setup_physical_injection(pipe, opts, size):
    """Setup PhysicalInjectionManager for inference; optionally load physical_ckpt into pipe.model."""
    from wan.modules.physical_adapter import PhysicalAdapter
    from wan.modules.physical_injection import PhysicalInjectionManager

    target_shape, seq_len = _compute_target_shape_and_seq_len(
        frame_num=opts.frame_num,
        size=size,
        vae_stride=pipe.vae_stride,
        patch_size=pipe.patch_size,
        sp_size=pipe.sp_size,
        z_dim=pipe.vae.model.z_dim,
    )

    adapter = None
    if getattr(opts, "physical_token_mode", "adapter") == "adapter":
        adapter = PhysicalAdapter(
            input_dim=768,
            hidden_dim=768,
            output_channels=opts.physical_adapter_dim,
            num_queries=opts.physical_num_queries,
        ).to(dtype=pipe.config.param_dtype)

    injection_layers = [int(x) for x in getattr(opts, "physical_injection_layers", "0,1,2").split(",") if x.strip()]
    manager = PhysicalInjectionManager(
        model=pipe.model,
        physical_adapter=adapter,
        injection_layers=injection_layers,
        hidden_size=pipe.model.dim,
        adapter_dim=opts.physical_adapter_dim,
        seq_len=seq_len,
        patch_size=pipe.patch_size,
        dtype=pipe.config.param_dtype,
        use_direct_tokens=(getattr(opts, "physical_token_mode", "adapter") == "direct"),
        injection_position=getattr(opts, "physical_injection_position", "layers"),
    )

    if getattr(opts, "physical_ckpt", None):
        state = _load_physical_checkpoint(opts.physical_ckpt)
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
        "Physical injection enabled: layers=%s seq_len=%d",
        injection_layers,
        seq_len,
    )
    return manager, target_shape


def load_prompts_from_txt(txt_path: Path) -> list:
    prompts = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_wan22_physical_pipeline(
    ckpt_dir: str,
    physical_ckpt=None,
    device_id: int = 0,
    disable_rag: bool = False,  # PhysRAG: RAG on by default
    t5_cpu: bool = False,
    size_key: str = "1280*704",
    frame_num: int = 49,
    physical_injection_layers: str = "0,1,2",
    physical_adapter_dim: int = 16,
    physical_num_queries: int = 128,
    physical_token_mode: str = "adapter",
    physical_injection_position: str = "layers",
):
    """
    Load Wan2.2 TI2V 5B for inference; optionally load physical fine-tuned weights (same pattern as run_wan22_ti2v_5b_batch.py).

    - ckpt_dir: base Wan2.2-TI2V-5B directory.
    - physical_ckpt: path to merged_model.pt; if None, use base model only.
    - disable_rag: if True, only load physical weights into model (no RAG/ref_features per prompt).
    - If disable_rag=False, CogVideo/PhysicalDB must be available and ref_features are set per prompt.
    """
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    import wan

    cfg = WAN_CONFIGS["ti2v-5B"]
    size = SIZE_CONFIGS.get(size_key, (1280, 704))

    logging.info("Loading Wan TI2V pipeline for inference from %s", ckpt_dir)
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=t5_cpu,
        convert_model_dtype=False,
    )

    class Opts:
        pass

    opts = Opts()
    opts.physical_ckpt = physical_ckpt
    opts.frame_num = frame_num
    opts.physical_injection_layers = physical_injection_layers
    opts.physical_adapter_dim = physical_adapter_dim
    opts.physical_num_queries = physical_num_queries
    opts.physical_token_mode = physical_token_mode
    opts.physical_injection_position = physical_injection_position

    if not disable_rag and physical_ckpt:
        manager, target_shape = setup_physical_injection(pipe, opts, size=size)
        pipe._vbench_rag = True
        pipe._vbench_target_shape = target_shape
        pipe._vbench_manager = manager
    elif physical_ckpt:
        # Inference with physical weights only (no RAG)
        state = _load_physical_checkpoint(physical_ckpt)
        model_state = pipe.model.state_dict()
        filtered_state, skipped = _filter_state_dict(state, model_state)
        missing, unexpected = pipe.model.load_state_dict(filtered_state, strict=False)
        logging.info(
            "Physical checkpoint (no RAG) loaded: used=%d skipped=%d missing=%d unexpected=%d",
            len(filtered_state),
            skipped,
            len(missing),
            len(unexpected),
        )
        pipe._vbench_rag = False
    else:
        pipe._vbench_rag = False

    return pipe, cfg, size


def save_video_tensor(video_tensor, out_path: str, fps: int = 16, value_range=(-1, 1)):
    """Save (C, N, H, W) inference output to mp4 via wan.utils.save_video."""
    from wan.utils.utils import save_video as wan_save_video

    # WanTI2V.generate returns (C, N, H, W)
    if video_tensor.dim() == 4:
        wan_save_video(
            tensor=video_tensor.unsqueeze(0),
            save_file=out_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=value_range,
        )
    else:
        wan_save_video(tensor=video_tensor, save_file=out_path, fps=fps)


def generate_videos_for_category(
    pipe,
    config,
    size,
    max_area: int,
    category_name: str,
    prompts: list,
    output_dir: Path,
    num_inference_steps: int = 40,
    fps: int = 16,
    frame_num: int = 49,
    skip_existing: bool = True,
    offload_model: bool = True,
    faiss_index_dir=None,
    videoclip_xl_model_path=None,
):
    """Run inference: 5 videos per prompt for one VBench category."""
    n_prompt = getattr(
        config,
        "sample_neg_prompt",
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    category_dir = output_dir / category_name
    category_dir.mkdir(parents=True, exist_ok=True)

    success, skipped, failed = 0, 0, 0
    sample_solver = getattr(config, "sample_solver", "unipc")
    guide_scale = getattr(config, "sample_guide_scale", 5.0)
    shift = getattr(config, "sample_shift", 5.0)
    sample_fps = getattr(config, "sample_fps", 16)

    for prompt_idx, prompt in enumerate(prompts, 1):
        for video_idx in range(5):
            safe_prompt = prompt.replace("/", "_").replace("\\", "_")
            filename = f"{safe_prompt}-{video_idx}.mp4"
            output_path = category_dir / filename

            if skip_existing and output_path.exists():
                skipped += 1
                continue

            seed = 42 + prompt_idx * 10 + video_idx
            try:
                if getattr(pipe, "_vbench_rag", False) and faiss_index_dir and videoclip_xl_model_path:
                    try:
                        from PhysicalDB.utils.select_physical_feature_videoclip_xl import (
                            select_best_feature_for_prompt,
                        )
                    except ModuleNotFoundError as e:
                        raise ModuleNotFoundError(
                            "RAG is enabled but PhysicalDB not found. Add CogVideo-main to PYTHONPATH "
                            "(e.g. export PYTHONPATH=/path/to/CogVideo-main:$PYTHONPATH) or run with --disable_rag."
                        ) from e
                    feature_path, _, _ = select_best_feature_for_prompt(
                        prompt=prompt,
                        index_dir=faiss_index_dir,
                        model_path=videoclip_xl_model_path,
                        device=torch.device("cuda"),
                    )
                    ref_features = _load_physical_features(Path(feature_path))
                    pipe._vbench_manager.set_ref_features(ref_features, pipe._vbench_target_shape)

                video = pipe.generate(
                    prompt,
                    size=size,
                    max_area=max_area,
                    frame_num=frame_num,
                    shift=shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=guide_scale,
                    n_prompt=n_prompt,
                    seed=seed,
                    offload_model=offload_model,
                )
                if video is not None:
                    save_video_tensor(video, str(output_path), fps=fps or sample_fps)
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logging.exception("Failed %s: %s", filename, e)
                failed += 1

        # Progress: one prompt done (5 videos)
        print(
            f"  progress: prompt {prompt_idx}/{len(prompts)} (success={success}, skipped={skipped}, failed={failed})",
            flush=True,
        )

    print(f"Category {category_name} done: success={success}, skipped={skipped}, failed={failed}", flush=True)
    return success, skipped, failed


def main():
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="VBench inference: generate videos with Wan2.2 Physical TI2V (run_wan22 style)")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Base Wan2.2-TI2V-5B directory")
    parser.add_argument(
        "--physical_ckpt",
        type=str,
        default=None,
        help="Path to merged_model.pt (default: <checkpoint_dir>/merged_model.pt when checkpoint_dir is a checkpoint dir)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="If set, directory containing merged_model.pt; then physical_ckpt defaults to checkpoint_dir/merged_model.pt and ckpt_dir must be base model dir (use with --ckpt_dir as base)",
    )
    parser.add_argument("--base_model_dir", type=str, default=None, help="Alias for ckpt_dir (base model)")
    parser.add_argument("--vbench_prompt_dir", type=Path, default=_SCRIPT_DIR / "prompts")
    parser.add_argument("--output_dir", type=str, default="./vbench_outputs_wan22")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frame_num", type=int, default=49, help="Same default as run_wan22_ti2v_5b_batch.py")
    parser.add_argument("--size", type=str, default="1280*704", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    parser.add_argument("--force_overwrite", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument(
        "--disable_rag",
        action="store_true",
        help="Disable RAG physical injection; only load physical weights into model (no CogVideo/PhysicalDB per-prompt retrieval)",
    )
    parser.add_argument(
        "--faiss_index_dir",
        type=str,
        default=str(Path(_WAN_REPO_ROOT) / "data" / "rag" / "faiss_index"),
        help="Same default as run_wan22_ti2v_5b_batch.py",
    )
    parser.add_argument(
        "--videoclip_xl_model_path",
        type=str,
        default=str(Path(_WAN_REPO_ROOT) / "PhysicalDB" / "VideoCLIP-XL" / "VideoCLIP-XL.bin"),
        help="Same default as run_wan22_ti2v_5b_batch.py",
    )
    parser.add_argument("--physical_injection_layers", type=str, default="0,1,2")
    parser.add_argument("--physical_adapter_dim", type=int, default=16)
    parser.add_argument("--physical_num_queries", type=int, default=128)
    parser.add_argument("--physical_token_mode", type=str, default="adapter", choices=["adapter", "direct"])
    parser.add_argument("--physical_injection_position", type=str, default="layers", choices=["layers", "post"])
    args = parser.parse_args()

    repo_root = Path(_WAN_REPO_ROOT)

    def resolve(p: str):
        path = Path(p)
        return path if path.is_absolute() else (repo_root / path)

    ckpt_dir = resolve(args.ckpt_dir or args.base_model_dir or "")
    if not ckpt_dir.is_dir():
        print(f"Error: ckpt_dir not found: {ckpt_dir}")
        sys.exit(1)

    if args.checkpoint_dir:
        checkpoint_dir = resolve(args.checkpoint_dir)
        physical_ckpt = args.physical_ckpt or str(checkpoint_dir / "merged_model.pt")
    else:
        physical_ckpt = args.physical_ckpt
        if physical_ckpt:
            physical_ckpt = str(resolve(physical_ckpt)) if not Path(physical_ckpt).is_absolute() else physical_ckpt

    if physical_ckpt and not Path(physical_ckpt).is_absolute():
        physical_ckpt = str(resolve(physical_ckpt))
    if physical_ckpt and not Path(physical_ckpt).exists():
        print(f"Error: physical_ckpt not found: {physical_ckpt}")
        sys.exit(1)

    output_dir = resolve(args.output_dir)
    vbench_prompt_dir = args.vbench_prompt_dir if args.vbench_prompt_dir.is_absolute() else (repo_root / args.vbench_prompt_dir)
    if not vbench_prompt_dir.exists():
        print(f"Error: prompt dir not found: {vbench_prompt_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    size = SIZE_CONFIGS[args.size]
    max_area = MAX_AREA_CONFIGS[args.size]
    # Always use device_id=0: CUDA_VISIBLE_DEVICES (set by shell) restricts to one GPU, visible as cuda:0.
    # Passing physical GPU id here causes "invalid device ordinal" when e.g. CUDA_VISIBLE_DEVICES=7.
    device_id = 0

    # When RAG is enabled, add CogVideo-main to path once so PhysicalDB can be imported later.
    if not args.disable_rag and args.faiss_index_dir:
        faiss_dir = Path(args.faiss_index_dir).resolve()
        cogvideo_root = faiss_dir.parents[2] if len(faiss_dir.parts) >= 3 else None  # .../CogVideo-main/PhysicalDB/...
        if cogvideo_root and cogvideo_root.exists() and str(cogvideo_root) not in sys.path:
            sys.path.insert(0, str(cogvideo_root))
        elif not (cogvideo_root and cogvideo_root.exists()):
            logging.warning(
                "RAG enabled but CogVideo path not found (faiss_index_dir=%s). Run with --disable_rag or set correct --faiss_index_dir.",
                args.faiss_index_dir,
            )

    print("Loading Wan2.2 Physical TI2V for inference (run_wan22 style)...")
    pipe, config, size_tuple = load_wan22_physical_pipeline(
        ckpt_dir=str(ckpt_dir),
        physical_ckpt=physical_ckpt,
        device_id=device_id,
        disable_rag=args.disable_rag,
        t5_cpu=args.t5_cpu,
        size_key=args.size,
        frame_num=args.frame_num,
        physical_injection_layers=args.physical_injection_layers,
        physical_adapter_dim=args.physical_adapter_dim,
        physical_num_queries=args.physical_num_queries,
        physical_token_mode=args.physical_token_mode,
        physical_injection_position=args.physical_injection_position,
    )

    txt_files = sorted(vbench_prompt_dir.glob("*.txt"))
    if args.categories:
        txt_files = [f for f in txt_files if f.stem in args.categories]
    if not txt_files:
        print("No prompt files found")
        sys.exit(1)

    total_categories = len(txt_files)
    total_success, total_skipped, total_failed = 0, 0, 0
    print(f"Starting inference: {total_categories} categories, 5 videos per prompt", flush=True)
    try:
        for cat_idx, txt_file in enumerate(txt_files, 1):
            category_name = txt_file.stem
            prompts = load_prompts_from_txt(txt_file)
            print(f"Category: {category_name} ({cat_idx}/{total_categories}), prompts: {len(prompts)} (5 videos each)", flush=True)
            s, sk, f = generate_videos_for_category(
                pipe,
                config,
                size_tuple,
                max_area,
                category_name,
                prompts,
                output_dir,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                frame_num=args.frame_num,
                skip_existing=not args.force_overwrite,
                offload_model=args.offload_model,
                faiss_index_dir=args.faiss_index_dir or None,
                videoclip_xl_model_path=args.videoclip_xl_model_path or None,
            )
            total_success += s
            total_skipped += sk
            total_failed += f
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done. Success: {total_success}, Skipped: {total_skipped}, Failed: {total_failed}", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
