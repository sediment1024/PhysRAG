import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import best_output_size, save_video
import wan


def load_prompts(prompts_json: Path) -> list[str]:
    if not prompts_json.exists():
        raise FileNotFoundError(f"prompts.json not found: {prompts_json}")
    with prompts_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("prompts.json must be a list of dicts")
    prompts = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "caption" not in item:
            raise ValueError(f"prompts.json item {idx} missing 'caption'")
        caption = str(item["caption"]).strip()
        if caption:
            prompts.append(caption)
    if not prompts:
        raise ValueError("No valid prompts found in prompts.json")
    return prompts


def load_manual_ref_map(path: Path) -> Dict[int, str]:
    """
    Load manual reference mapping for prompt index -> feature path.

    Supported formats:
    1) Dict JSON: {"79": "/abs/path/xxx.pt", "80": "..."}
    2) List JSON: [{"idx": 79, "feature_path": "/abs/path/xxx.pt"}, ...]
    Prompt index is 1-based, aligned with output_video_{idx}.mp4 naming.
    """
    if not path.exists():
        raise FileNotFoundError(f"manual_ref_json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[int, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            mapping[int(k)] = str(v)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "idx" not in item or "feature_path" not in item:
                continue
            mapping[int(item["idx"])] = str(item["feature_path"])
    else:
        raise ValueError("manual_ref_json must be a dict or list")
    return mapping


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


def _load_physical_feature_tokens(path: Path) -> torch.Tensor:
    ref_features = torch.load(path, map_location="cpu")
    if isinstance(ref_features, dict):
        ref_features = ref_features.get("features", list(ref_features.values())[0])
    if not isinstance(ref_features, torch.Tensor):
        ref_features = torch.tensor(ref_features)
    if ref_features.dim() == 3 and ref_features.shape[0] == 1:
        ref_features = ref_features.squeeze(0)
    if ref_features.dim() == 1:
        ref_features = ref_features.unsqueeze(0)
    if ref_features.dim() != 2:
        raise RuntimeError(f"Unsupported physical feature shape from {path}: {tuple(ref_features.shape)}")
    return ref_features


def _fuse_topk_physical_features(retrieved, aggregation: str, weight_temperature: float) -> tuple[torch.Tensor, list[float]]:
    features = [_load_physical_feature_tokens(Path(item["feature_path"])) for item in retrieved]
    if aggregation == "concat":
        return torch.cat(features, dim=0).unsqueeze(0), [1.0 for _ in features]
    if aggregation != "weighted_sum":
        raise ValueError(f"Unsupported rag_aggregation: {aggregation}")

    stacked = torch.stack(features, dim=0)
    scores = torch.tensor([float(item["raw_score"]) for item in retrieved], dtype=torch.float32)
    weights = torch.softmax(scores / max(float(weight_temperature), 1e-6), dim=0).to(dtype=stacked.dtype)
    fused = (weights[:, None, None] * stacked).sum(dim=0)
    return fused.unsqueeze(0), [float(x) for x in weights.cpu().tolist()]


def _compute_target_shape_and_seq_len(frame_num, size, vae_stride, patch_size, sp_size, z_dim):
    f_lat = (frame_num - 1) // vae_stride[0] + 1
    h_lat = size[1] // vae_stride[1]
    w_lat = size[0] // vae_stride[2]
    h_lat = h_lat - (h_lat % patch_size[1])
    w_lat = w_lat - (w_lat % patch_size[2])
    h_tokens = h_lat // patch_size[1]
    w_tokens = w_lat // patch_size[2]
    seq_len = int(math.ceil((f_lat * h_tokens * w_tokens) / sp_size) * sp_size)
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
        adapter_dim = args.physical_adapter_dim
        adapter = PhysicalAdapter(
            input_dim=768,
            hidden_dim=768,
            output_channels=adapter_dim,
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
            "Loaded physical checkpoint: used=%d skipped=%d missing=%d unexpected=%d",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch infer PhyGenBench prompts with Wan TI2V + physical injection.")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--physical_ckpt", type=str, required=True)
    parser.add_argument(
        "--phygenbench_root",
        type=str,
        default=None,
        help="PhyGenBench root containing PhyGenBench/prompts.json.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Generate one video from a prompt.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for --prompt mode (default: outputs/<modelname>).",
    )
    parser.add_argument("--modelname", type=str, default="phyrag-wan22-ti2v-5b")
    parser.add_argument("--cogvideo_root", type=str, default=str(REPO_ROOT))
    parser.add_argument("--size", type=str, default="704*480", choices=SIZE_CONFIGS.keys())
    parser.add_argument("--frame_num", type=int, default=49)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_solver", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--physical_injection_layers", type=str, default="0,1,2")
    parser.add_argument("--physical_adapter_dim", type=int, default=16)
    parser.add_argument("--physical_num_queries", type=int, default=128)
    parser.add_argument(
        "--physical_token_mode",
        type=str,
        default="adapter",
        choices=["adapter", "direct"],
        help="Use adapter+pooling tokens or directly inject MAE tokens.",
    )
    parser.add_argument(
        "--physical_injection_position",
        type=str,
        default="layers",
        choices=["layers", "post"],
        help="Inject tokens at specific layers or once after all blocks (before head).",
    )
    parser.add_argument(
        "--disable_rag",
        action="store_true",
        help="Disable RAG retrieval and physical injection during inference.",
    )
    parser.add_argument(
        "--faiss_index_dir",
        type=str,
        default=str(REPO_ROOT / "data" / "rag" / "faiss_index"),
    )
    parser.add_argument(
        "--videoclip_xl_model_path",
        type=str,
        default=str(REPO_ROOT / "PhysicalDB" / "VideoCLIP-XL" / "VideoCLIP-XL.bin"),
    )
    parser.add_argument("--rag_top_k", type=int, default=1, help="Number of retrieved physical reference videos per prompt.")
    parser.add_argument(
        "--rag_aggregation",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "concat"],
        help="Fuse top-k RAG features by score-weighted sum or keep them as separate tokens.",
    )
    parser.add_argument(
        "--rag_weight_temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for score-weighted top-k RAG fusion, using raw FAISS cosine scores.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="Start index for prompt sharding")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) for prompt sharding")
    parser.add_argument(
        "--manual_ref_json",
        type=str,
        default=None,
        help="Optional JSON mapping prompt idx (1-based) -> feature .pt path for manual override.",
    )
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()
    if args.rag_top_k < 1:
        raise ValueError("--rag_top_k must be >= 1")
    if args.rag_weight_temperature <= 0:
        raise ValueError("--rag_weight_temperature must be > 0")

    cfg = WAN_CONFIGS["ti2v-5B"]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.sample_solver is None:
        args.sample_solver = getattr(cfg, "sample_solver", "unipc")

    single_prompt = args.prompt is not None
    if single_prompt:
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("--prompt must not be empty")
        prompts = [prompt]
        args.start = 0
        args.end = 1
        output_dir = Path(args.output_dir or (Path("outputs") / args.modelname))
    else:
        if args.phygenbench_root is None:
            raise ValueError("Provide either --prompt or --phygenbench_root")
        phy_root = Path(args.phygenbench_root)
        prompts_json = phy_root / "PhyGenBench" / "prompts.json"
        prompts = load_prompts(prompts_json)
        if args.end is None:
            args.end = len(prompts)
        prompts = prompts[args.start:args.end]
        if args.limit is not None:
            prompts = prompts[: args.limit]
        output_dir = phy_root / "PhyVideos" / args.modelname
    output_dir.mkdir(parents=True, exist_ok=True)

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

    manager = None
    target_shape = None
    manual_ref_map: Dict[int, str] = {}
    if not args.disable_rag:
        # RAG retriever (VideoCLIP-XL + FAISS)
        cogvideo_root = Path(args.cogvideo_root)
        sys.path.append(str(cogvideo_root))
        from PhysicalDB.utils.select_physical_feature_videoclip_xl import (
            select_topk_features_for_prompt,
        )

        manager, target_shape = setup_physical_injection(wan_ti2v, args, size=size)
        if args.manual_ref_json:
            manual_ref_map = load_manual_ref_map(Path(args.manual_ref_json))
            logging.info("Loaded manual ref override map: %d entries", len(manual_ref_map))

    logging.info("Starting PhyGenBench inference: %d prompts (start=%d, end=%d)", len(prompts), args.start, args.end)
    for offset, prompt in enumerate(prompts, start=1):
        idx = args.start + offset
        output_path = output_dir / ("output.mp4" if single_prompt else f"output_video_{idx}.mp4")
        if args.skip_existing and output_path.exists():
            logging.info("[%d/%d] Skip existing: %s", offset, len(prompts), output_path)
            continue

        logging.info("[%d/%d] Prompt: %s", offset, len(prompts), prompt[:100])
        try:
            if not args.disable_rag:
                manual_path = manual_ref_map.get(idx)
                if manual_path:
                    feature_path = Path(manual_path)
                    if not feature_path.exists():
                        raise FileNotFoundError(f"Manual feature path not found for idx={idx}: {feature_path}")
                    logging.info("[%d/%d] Using manual ref feature: %s", offset, len(prompts), feature_path)
                else:
                    retrieved = select_topk_features_for_prompt(
                        prompt=prompt,
                        k=args.rag_top_k,
                        index_dir=args.faiss_index_dir,
                        model_path=args.videoclip_xl_model_path,
                        device=torch.device("cuda"),
                    )
                    ref_features, weights = _fuse_topk_physical_features(
                        retrieved,
                        aggregation=args.rag_aggregation,
                        weight_temperature=args.rag_weight_temperature,
                    )
                    logging.info(
                        "[%d/%d] RAG top_k=%d aggregation=%s weights=%s",
                        offset,
                        len(prompts),
                        args.rag_top_k,
                        args.rag_aggregation,
                        ",".join(f"{w:.3f}" for w in weights),
                    )
                if manual_path:
                    ref_features = _load_physical_features(feature_path)
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
                save_file=str(output_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info("Saved: %s", output_path)
        except Exception as exc:
            logging.exception("Failed on %d: %s", idx, exc)
            if os.environ.get("WAN_FAIL_FAST") == "1":
                raise

    logging.info("Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
