import argparse
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import dataset first (no CUDA dependencies)
from finetune.dataset import VideoTextDataset, parse_resolution


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # Only set CUDA seed if CUDA is available and initialized
    # This avoids CUDA initialization errors during module import
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def _get_device_count():
        try:
            return torch.cuda.device_count()
        except Exception:
            return "unknown"

    try:
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA not available after setting device. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}, "
                f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<not set>')}, "
                f"torch.version.cuda={torch.version.cuda}, "
                f"torch.cuda.device_count()={_get_device_count()}, local_rank={local_rank}"
            )
    except RuntimeError as e:
        error_msg = str(e)
        if any(keyword in error_msg.upper() for keyword in ["CUDA", "DEVICE", "UNKNOWN ERROR"]):
            raise RuntimeError(
                f"Failed to initialize CUDA for local_rank={local_rank}. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}, "
                f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<not set>')}, "
                f"torch.version.cuda={torch.version.cuda}, "
                f"torch.cuda.device_count()={_get_device_count()}. Original error: {error_msg}"
            ) from e
        raise
    
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def get_grad_norm(model, use_deepspeed: bool, scaler=None) -> float:
    """Get gradient norm for logging."""
    if use_deepspeed:
        try:
            if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'get_global_grad_norm'):
                return model.optimizer.get_global_grad_norm()
        except Exception:
            pass
        return 0.0
    else:
        total_norm = sum(p.grad.data.norm(2).item() ** 2 
                         for p in model.parameters() if p.grad is not None)
        return total_norm ** 0.5 if total_norm > 0 else 0.0


def get_learning_rate(optimizer, use_deepspeed: bool, model=None) -> float:
    """Get current learning rate."""
    opt = model.optimizer if (use_deepspeed and hasattr(model, 'optimizer')) else optimizer
    if opt is None:
        return 0.0
    try:
        if hasattr(opt, 'param_groups'):
            return opt.param_groups[0].get('lr', 0.0)
        elif hasattr(opt, 'get_lr'):
            return opt.get_lr()[0]
    except Exception:
        pass
    return 0.0


def log_training_progress(
    global_step: int,
    total_steps: int,
    loss: float,
    start_time: float,
    step_start_time: float,
    optimizer,
    model,
    use_deepspeed: bool,
    scaler=None,
):
    """Log detailed training progress similar to transformers Trainer."""
    elapsed_time = time.time() - start_time
    step_time = time.time() - step_start_time
    remaining_time = (elapsed_time / global_step * (total_steps - global_step)) if global_step > 0 else 0
    
    print(
        f"Training steps: {global_step * 100 // total_steps}% | {global_step}/{total_steps} | "
        f"loss: {loss:.4f} | grad_norm: {get_grad_norm(model, use_deepspeed, scaler):.2f} | "
        f"lr: {get_learning_rate(optimizer, use_deepspeed, model):.2e} | "
        f"elapsed: {format_time(elapsed_time)} | remaining: {format_time(remaining_time)} | "
        f"s/it: {step_time:.2f}"
    )


def load_deepspeed_config(path: str, args, world_size: int):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["train_micro_batch_size_per_gpu"] = args.batch_size
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["train_batch_size"] = args.batch_size * args.gradient_accumulation_steps * world_size

    # Set learning rate if optimizer config uses "auto"
    if "optimizer" in config and "params" in config["optimizer"]:
        if config["optimizer"]["params"].get("lr") == "auto":
            config["optimizer"]["params"]["lr"] = args.learning_rate
        if config["optimizer"]["params"].get("weight_decay") == "auto":
            config["optimizer"]["params"]["weight_decay"] = args.weight_decay

    if args.mixed_precision == "bf16":
        config["bf16"] = {"enabled": True}
        config.pop("fp16", None)
    elif args.mixed_precision == "fp16":
        config["fp16"] = {"enabled": True}
        config.pop("bf16", None)
    else:
        config.pop("bf16", None)
        config.pop("fp16", None)

    # Set gradient_clipping if config uses "auto"
    if config.get("gradient_clipping") == "auto":
        config["gradient_clipping"] = args.max_grad_norm if args.max_grad_norm is not None else 1.0

    return config


def compute_seq_len(frames: int, height: int, width: int, vae_stride, patch_size, sp_size: int) -> int:
    f_lat = (frames - 1) // vae_stride[0] + 1
    h_lat = height // vae_stride[1]
    w_lat = width // vae_stride[2]
    seq_len = f_lat * (h_lat // patch_size[1]) * (w_lat // patch_size[2])
    return int(math.ceil(seq_len / sp_size)) * sp_size


def build_text_encoder(ckpt_dir: Path, config, device, t5_cpu: bool):
    # Import here to avoid CUDA initialization during module import
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
    return text_encoder


def build_physical_loader(cogvideo_root: Path, device=None):
    """
    Build physical feature loader using VideoCLIP-XL + FAISS RAG.
    
    Args:
        cogvideo_root: Path to CogVideo root directory
        device: Device to use for VideoCLIP-XL model (default: cuda if available, else cpu)
    
    Returns:
        Function to load physical features for a batch of prompts
    """
    if not cogvideo_root.exists():
        raise FileNotFoundError(f"cogvideo_root not found: {cogvideo_root}")
    sys.path.append(str(cogvideo_root))
    try:
        from PhysicalDB.utils.select_physical_feature_videoclip_xl import (
            select_best_feature_for_prompt,
            select_topk_features_for_prompt,
        )
    except Exception as exc:
        raise ImportError(
            "Failed to import VideoCLIP-XL retriever from CogVideo-main"
        ) from exc

    # Determine device for VideoCLIP-XL model
    # Use GPU if available, but allow CPU fallback if needed
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    def _load_feature_tensor(feature_path: Path, dtype: torch.dtype) -> torch.Tensor:
        ref_features = torch.load(str(feature_path), map_location="cpu")
        if isinstance(ref_features, dict):
            ref_features = ref_features.get("features", list(ref_features.values())[0])
        if not isinstance(ref_features, torch.Tensor):
            ref_features = torch.tensor(ref_features)
        if ref_features.dim() == 3 and ref_features.shape[0] == 1:
            ref_features = ref_features.squeeze(0)
        if ref_features.dim() == 1:
            ref_features = ref_features.unsqueeze(0)
        if ref_features.dim() != 2:
            raise RuntimeError(f"Unsupported physical feature shape from {feature_path}: {tuple(ref_features.shape)}")
        return ref_features.to(dtype=dtype)

    def _fuse_topk_features(features, scores, aggregation: str, weight_temperature: float) -> tuple[torch.Tensor, list]:
        if aggregation == "concat":
            return torch.cat(features, dim=0), [1.0 for _ in features]
        if aggregation != "weighted_sum":
            raise ValueError(f"Unsupported rag_aggregation: {aggregation}")

        stacked = torch.stack(features, dim=0)  # [K, S, D]
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        temperature = max(float(weight_temperature), 1e-6)
        weights = torch.softmax(score_tensor / temperature, dim=0).to(dtype=stacked.dtype)
        fused = (weights[:, None, None] * stacked).sum(dim=0)
        return fused, [float(x) for x in weights.cpu().tolist()]

    def _load_physical_feature_batch(
        prompts,
        dtype,
        index_dir,
        model_path,
        rag_top_k=1,
        rag_aggregation="weighted_sum",
        rag_weight_temperature=0.07,
    ):
        ref_features_list = []
        entries_list = []

        for prompt in prompts:
            top_k = max(int(rag_top_k), 1)
            if top_k == 1:
                feature_path, entry, score = select_best_feature_for_prompt(
                    prompt=prompt,
                    index_dir=index_dir,
                    model_path=model_path,
                    device=device,  # Use GPU if available
                )
                ref_features = _load_feature_tensor(Path(feature_path), dtype=dtype)
                entries = [{"entry": entry, "score": float(score), "raw_score": float(score), "rank": 1, "weight": 1.0}]
            else:
                retrieved = select_topk_features_for_prompt(
                    prompt=prompt,
                    k=top_k,
                    index_dir=index_dir,
                    model_path=model_path,
                    device=device,
                )
                features = [_load_feature_tensor(Path(item["feature_path"]), dtype=dtype) for item in retrieved]
                scores = [float(item["raw_score"]) for item in retrieved]
                ref_features, weights = _fuse_topk_features(
                    features,
                    scores,
                    aggregation=rag_aggregation,
                    weight_temperature=rag_weight_temperature,
                )
                entries = []
                for item, weight in zip(retrieved, weights):
                    entries.append(
                        {
                            "entry": item["entry"],
                            "score": float(item["score"]),
                            "raw_score": float(item["raw_score"]),
                            "rank": int(item["rank"]),
                            "weight": float(weight),
                        }
                    )
            ref_features = ref_features.unsqueeze(0)
            ref_features_list.append(ref_features)
            entries_list.append(entries)

        ref_features_batch = torch.cat(ref_features_list, dim=0)
        return ref_features_batch, entries_list

    return _load_physical_feature_batch


def save_checkpoint(model, optimizer, scaler, step: int, output_dir: Path, max_keep: int, rank: int):
    if not is_main_process(rank):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(state, ckpt_dir / "merged_model.pt")

    trainer_state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(trainer_state, ckpt_dir / "trainer_state.pt")

    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if max_keep is not None and len(checkpoints) > max_keep:
        for stale in checkpoints[:-max_keep]:
            for item in stale.glob("*"):
                item.unlink()
            stale.rmdir()


def _resolve_resume_paths(resume_from: Path) -> tuple[Path, Path | None]:
    if resume_from.is_dir():
        model_path = resume_from / "merged_model.pt"
        trainer_path = resume_from / "trainer_state.pt"
    else:
        if resume_from.name == "merged_model.pt":
            model_path = resume_from
            trainer_path = resume_from.with_name("trainer_state.pt")
        elif resume_from.name == "trainer_state.pt":
            trainer_path = resume_from
            model_path = resume_from.with_name("merged_model.pt")
        else:
            raise ValueError(
                f"resume_from must be a checkpoint dir or merged_model.pt/trainer_state.pt, got {resume_from}"
            )

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model not found: {model_path}")
    if not trainer_path.exists():
        trainer_path = None
    return model_path, trainer_path


def load_resume_state(resume_from: str, model, optimizer, scaler, rank: int) -> int:
    resume_path = Path(resume_from)
    model_path, trainer_path = _resolve_resume_paths(resume_path)

    state = torch.load(model_path, map_location="cpu")
    target_model = model.module if hasattr(model, "module") else model
    missing, unexpected = target_model.load_state_dict(state, strict=False)
    if is_main_process(rank):
        print(
            f"Loaded checkpoint weights from {model_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    step = 0
    if trainer_path is not None:
        trainer_state = torch.load(trainer_path, map_location="cpu")
        step = int(trainer_state.get("step", 0))
        opt_state = trainer_state.get("optimizer")
        if optimizer is not None and opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as exc:
                if is_main_process(rank):
                    print(f"Warning: failed to load optimizer state, continuing with weights only: {exc}")
        scaler_state = trainer_state.get("scaler")
        if scaler is not None and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                if is_main_process(rank):
                    print(f"Warning: failed to load scaler state, continuing with weights only: {exc}")
        if is_main_process(rank):
            print(f"Loaded trainer state from {trainer_path} (step={step})")
    elif is_main_process(rank):
        print("Warning: trainer_state.pt not found; resuming weights only.")

    return step


def freeze_backbone_enable_plugin(model):
    """
    Freeze WanModel backbone and enable only physical adapter + injection modules.
    Use when train_mode=plugin (RAG injection + freeze backbone).
    """
    # 1) Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze: injection modules (SequenceConcatInjection inside block/head wrappers)
    for m in model.modules():
        if m.__class__.__name__ == "SequenceConcatInjection":
            for p in m.parameters():
                p.requires_grad = True

    # 3) Unfreeze: physical adapter (only in adapter mode)
    if hasattr(model, "physical_adapter") and model.physical_adapter is not None:
        for p in model.physical_adapter.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--caption_column", type=str, required=True)
    parser.add_argument("--video_column", type=str, required=True)
    parser.add_argument("--train_resolution", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if loss doesn't improve for this many logging steps (0 to disable).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum loss improvement to reset early stop counter.",
    )
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--noise_shift", type=float, default=1.0)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--physical_injection_layers", type=str, default="0,1,2")
    parser.add_argument("--physical_adapter_dim", type=int, default=16)
    parser.add_argument("--physical_num_queries", type=int, default=128)
    parser.add_argument(
        "--physical_query_mode",
        type=str,
        default="learned",
        choices=["learned", "dit"],
        help="Query source for adapter cross-attention: learned queries or DiT hidden states.",
    )
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
    parser.add_argument("--cogvideo_root", type=str, default="../CogVideo-main")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="full",
        choices=["baseline", "plugin", "full"],
        help=(
            "baseline: no RAG injection (same as --disable_rag). "
            "plugin: RAG injection + freeze backbone (only train adapter + injection). "
            "full: RAG injection + train backbone (current default)."
        ),
    )
    parser.add_argument(
        "--disable_rag",
        action="store_true",
        help="Disable RAG physical feature retrieval and injection (overridden by --train_mode baseline).",
    )

    parser.add_argument("--faiss_index_dir", type=str, default=None)
    parser.add_argument("--videoclip_xl_model_path", type=str, default=None)
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
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")

    args = parser.parse_args()

    if args.physical_query_mode == "dit" and args.physical_token_mode == "direct":
        raise ValueError("physical_query_mode=dit requires physical_token_mode=adapter")
    if args.rag_top_k < 1:
        raise ValueError("--rag_top_k must be >= 1")
    if args.rag_weight_temperature <= 0:
        raise ValueError("--rag_weight_temperature must be > 0")

    # Initialize distributed training FIRST, before importing any modules that might access CUDA
    # This must be done before any CUDA operations
    rank, world_size, local_rank = init_distributed()
    
    # After distributed init, CUDA should be available
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available after distributed initialization. "
            f"This may indicate a CUDA initialization issue. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}, "
            f"local_rank={local_rank}"
        )
    
    device = torch.device(f"cuda:{local_rank}")

    set_seed(args.seed + rank)
    # Only enable cuDNN benchmark if CUDA is available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # NOW import wan modules after distributed initialization
    # This ensures CUDA is available when modules are loaded
    import importlib.util
    import types
    
    # Load shared_config first (doesn't trigger CUDA)
    if 'wan.configs.shared_config' not in sys.modules:
        if 'wan.configs' not in sys.modules:
            configs_module = types.ModuleType('wan.configs')
            sys.modules['wan.configs'] = configs_module
        shared_config_file = REPO_ROOT / "wan" / "configs" / "shared_config.py"
        shared_spec = importlib.util.spec_from_file_location("wan.configs.shared_config", shared_config_file)
        shared_module = importlib.util.module_from_spec(shared_spec)
        shared_module.__package__ = "wan.configs"
        sys.modules['wan.configs.shared_config'] = shared_module
        shared_spec.loader.exec_module(shared_module)
    
    # Load wan_ti2v_5B config
    ti2v_5B_file = REPO_ROOT / "wan" / "configs" / "wan_ti2v_5B.py"
    spec = importlib.util.spec_from_file_location("wan.configs.wan_ti2v_5B", ti2v_5B_file)
    ti2v_5B_module = importlib.util.module_from_spec(spec)
    ti2v_5B_module.__package__ = "wan.configs"
    if 'wan.configs' not in sys.modules:
        configs_module = types.ModuleType('wan.configs')
        sys.modules['wan.configs'] = configs_module
    sys.modules['wan.configs.wan_ti2v_5B'] = ti2v_5B_module
    spec.loader.exec_module(ti2v_5B_module)
    ti2v_5B = ti2v_5B_module.ti2v_5B
    
    # Import modules directly to avoid triggering package __init__.py
    # First, create the wan.modules package with proper __path__ attribute
    if 'wan.modules' not in sys.modules:
        modules_pkg = types.ModuleType('wan.modules')
        modules_pkg.__path__ = [str(REPO_ROOT / "wan" / "modules")]
        sys.modules['wan.modules'] = modules_pkg
    
    # Import attention module first (needed by model.py)
    attention_file = REPO_ROOT / "wan" / "modules" / "attention.py"
    attention_spec = importlib.util.spec_from_file_location("wan.modules.attention", attention_file)
    attention_module = importlib.util.module_from_spec(attention_spec)
    attention_module.__package__ = "wan.modules"
    sys.modules['wan.modules.attention'] = attention_module
    attention_spec.loader.exec_module(attention_module)
    
    # Import tokenizers module (needed by t5.py)
    tokenizers_file = REPO_ROOT / "wan" / "modules" / "tokenizers.py"
    tokenizers_spec = importlib.util.spec_from_file_location("wan.modules.tokenizers", tokenizers_file)
    tokenizers_module = importlib.util.module_from_spec(tokenizers_spec)
    tokenizers_module.__package__ = "wan.modules"
    sys.modules['wan.modules.tokenizers'] = tokenizers_module
    tokenizers_spec.loader.exec_module(tokenizers_module)
    
    # Import WanModel (depends on attention module)
    model_file = REPO_ROOT / "wan" / "modules" / "model.py"
    model_spec = importlib.util.spec_from_file_location("wan.modules.model", model_file)
    model_module = importlib.util.module_from_spec(model_spec)
    model_module.__package__ = "wan.modules"
    sys.modules['wan.modules.model'] = model_module
    model_spec.loader.exec_module(model_module)
    WanModel = model_module.WanModel

    # train_mode: baseline => no RAG; plugin/full => RAG (unless --disable_rag)
    use_rag = (args.train_mode != "baseline") and not args.disable_rag
    PhysicalAdapter = None
    PhysicalInjectionManager = None
    if use_rag:
        # Import PhysicalAdapter
        adapter_file = REPO_ROOT / "wan" / "modules" / "physical_adapter.py"
        adapter_spec = importlib.util.spec_from_file_location("wan.modules.physical_adapter", adapter_file)
        adapter_module = importlib.util.module_from_spec(adapter_spec)
        adapter_module.__package__ = "wan.modules"
        sys.modules['wan.modules.physical_adapter'] = adapter_module
        adapter_spec.loader.exec_module(adapter_module)
        PhysicalAdapter = adapter_module.PhysicalAdapter

        # Import PhysicalInjectionManager
        injection_file = REPO_ROOT / "wan" / "modules" / "physical_injection.py"
        injection_spec = importlib.util.spec_from_file_location("wan.modules.physical_injection", injection_file)
        injection_module = importlib.util.module_from_spec(injection_spec)
        injection_module.__package__ = "wan.modules"
        sys.modules['wan.modules.physical_injection'] = injection_module
        injection_spec.loader.exec_module(injection_module)
        PhysicalInjectionManager = injection_module.PhysicalInjectionManager
    
    # Import Wan2_2_VAE
    vae_file = REPO_ROOT / "wan" / "modules" / "vae2_2.py"
    vae_spec = importlib.util.spec_from_file_location("wan.modules.vae2_2", vae_file)
    vae_module = importlib.util.module_from_spec(vae_spec)
    vae_module.__package__ = "wan.modules"
    sys.modules['wan.modules.vae2_2'] = vae_module
    vae_spec.loader.exec_module(vae_module)
    Wan2_2_VAE = vae_module.Wan2_2_VAE
    
    # Import FlowDPMSolverMultistepScheduler
    if 'wan.utils' not in sys.modules:
        utils_pkg = types.ModuleType('wan.utils')
        sys.modules['wan.utils'] = utils_pkg
    solvers_file = REPO_ROOT / "wan" / "utils" / "fm_solvers.py"
    solvers_spec = importlib.util.spec_from_file_location("wan.utils.fm_solvers", solvers_file)
    solvers_module = importlib.util.module_from_spec(solvers_spec)
    solvers_module.__package__ = "wan.utils"
    sys.modules['wan.utils.fm_solvers'] = solvers_module
    solvers_spec.loader.exec_module(solvers_module)
    FlowDPMSolverMultistepScheduler = solvers_module.FlowDPMSolverMultistepScheduler

    config = ti2v_5B
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    frames, height, width = parse_resolution(args.train_resolution)
    seq_len = compute_seq_len(frames, height, width, config.vae_stride, config.patch_size, args.sp_size)

    text_encoder = build_text_encoder(ckpt_dir, config, device, args.t5_cpu)
    text_encoder.model.eval()
    text_encoder.model.requires_grad_(False)

    vae = Wan2_2_VAE(vae_pth=str(ckpt_dir / config.vae_checkpoint), device=device)
    vae.model.eval()
    vae.model.requires_grad_(False)

    def encode_text_fn(prompt: str) -> torch.Tensor:
        if args.t5_cpu:
            return text_encoder([prompt], torch.device("cpu"))[0]
        return text_encoder([prompt], device)[0]

    def encode_video_fn(video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.encode([video.to(device=device)])[0]

    cache_dir = args.cache_dir or str(Path(args.data_root) / "cache_wan")
    dataset = VideoTextDataset(
        data_root=args.data_root,
        caption_column=args.caption_column,
        video_column=args.video_column,
        num_frames=frames,
        height=height,
        width=width,
        encode_text=encode_text_fn,
        encode_video=encode_video_fn,
        cache_root=cache_dir,
        text_len=config.text_len,
        resolution=args.train_resolution,
    )

    def collate_fn(samples):
        prompts = [sample["prompt"] for sample in samples]
        prompt_embeddings = [sample["prompt_embedding"] for sample in samples]
        encoded_videos = [sample["encoded_video"] for sample in samples]
        batch = {"prompt": prompts}
        if all(emb is not None for emb in prompt_embeddings):
            batch["prompt_embedding"] = prompt_embeddings
        if all(latent is not None for latent in encoded_videos):
            batch["encoded_video"] = torch.stack(encoded_videos, dim=0)
        return batch

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = WanModel.from_pretrained(str(ckpt_dir))
    model.to(device=device, dtype=config.param_dtype)
    model.train()
    
    # Enable gradient checkpointing if requested (important for memory optimization)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main_process(rank):
            print("✓ Gradient checkpointing enabled")

    injection_manager = None
    if use_rag:
        use_direct_tokens = args.physical_token_mode == "direct"
        injection_layers = [int(x) for x in args.physical_injection_layers.split(",") if x]
        invalid_layers = [idx for idx in injection_layers if idx < 0 or idx >= len(model.blocks)]
        if invalid_layers:
            raise ValueError(
                f"Invalid physical_injection_layers={invalid_layers}; "
                f"model has {len(model.blocks)} blocks, valid indices are 0-{len(model.blocks) - 1}."
            )
        adapter = None
        if not use_direct_tokens:
            adapter = PhysicalAdapter(
                input_dim=768,
                hidden_dim=768,
                query_dim=model.dim,
                output_channels=args.physical_adapter_dim,
                num_queries=args.physical_num_queries,
            ).to(device=device, dtype=config.param_dtype)

        injection_manager = PhysicalInjectionManager(
            model=model,
            physical_adapter=adapter,
            injection_layers=injection_layers,
            hidden_size=model.dim,
            adapter_dim=args.physical_adapter_dim,
            seq_len=seq_len,
            patch_size=config.patch_size,
            dtype=config.param_dtype,
            use_direct_tokens=use_direct_tokens,
            injection_position=args.physical_injection_position,
            query_mode=args.physical_query_mode,
        )

    # Trainable parameters: plugin => only adapter + injection; else => full model
    if args.train_mode == "plugin":
        if not use_rag:
            raise ValueError("train_mode=plugin requires RAG enabled (do not use --train_mode baseline or --disable_rag).")
        trainable_params = freeze_backbone_enable_plugin(model)
        if is_main_process(rank):
            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"train_mode=plugin: {n_trainable} trainable params ({100.0 * n_trainable / n_total:.2f}% of model)")
    else:
        trainable_params = list(model.parameters())

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=args.noise_shift,
        use_dynamic_shifting=False,
    )

    use_deepspeed = args.deepspeed
    if use_deepspeed:
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError("deepspeed is required when --deepspeed is set") from exc

        ds_config = load_deepspeed_config(
            args.deepspeed_config or str(REPO_ROOT / "finetune" / "deepspeed_zero3.json"),
            args, world_size
        )
        model, optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=trainable_params, config=ds_config
        )
        if not hasattr(model.optimizer, "overlapping_partition_gradients_reduce_epilogue"):
            if is_main_process(rank):
                print("warning: deepspeed optimizer missing overlapping_partition_gradients_reduce_epilogue; using no-op")
            model.optimizer.overlapping_partition_gradients_reduce_epilogue = types.MethodType(lambda self: None, model.optimizer)
        scaler = None
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    resume_step = 0
    if args.resume_from:
        resume_step = load_resume_state(args.resume_from, model, optimizer, scaler, rank)

    load_physical_feature_batch = None
    faiss_index_dir = None
    videoclip_xl_model_path = None
    if use_rag:
        if is_main_process(rank):
            print("Loading physical feature loader (VideoCLIP-XL + FAISS)...")
        # Use GPU for VideoCLIP-XL if available (faster retrieval)
        # VideoCLIP-XL will use the same GPU as the training process
        videoclip_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        load_physical_feature_batch = build_physical_loader(Path(args.cogvideo_root), device=videoclip_device)
        cogvideo_path = Path(args.cogvideo_root)
        faiss_index_dir = args.faiss_index_dir or str(cogvideo_path / "PhysicalDB" / "faiss_index")
        videoclip_xl_model_path = args.videoclip_xl_model_path or str(cogvideo_path / "PhysicalDB" / "VideoCLIP-XL" / "VideoCLIP-XL.bin")

        # Pre-warm VideoCLIP-XL retriever by loading it once before training starts
        # This avoids the first-call delay during training
        # IMPORTANT: Each process (rank) needs to pre-warm its own retriever
        print(f"[Rank {rank}] Pre-warming VideoCLIP-XL retriever...")
        warmup_start = time.time()
        try:
            # Trigger initialization by calling with a dummy prompt
            _ = load_physical_feature_batch(
                prompts=["dummy prompt for warmup"],
                dtype=torch.float32,
                index_dir=faiss_index_dir,
                model_path=videoclip_xl_model_path,
                rag_top_k=args.rag_top_k,
                rag_aggregation=args.rag_aggregation,
                rag_weight_temperature=args.rag_weight_temperature,
            )
            warmup_time = time.time() - warmup_start
            print(f"[Rank {rank}] ✓ VideoCLIP-XL retriever pre-warmed in {warmup_time:.2f}s")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Failed to pre-warm VideoCLIP-XL retriever: {e}")

        if is_main_process(rank):
            print(
                "✓ Physical feature loader ready "
                f"(index: {faiss_index_dir}, model: {videoclip_xl_model_path}, device: {videoclip_device}, "
                f"top_k: {args.rag_top_k}, aggregation: {args.rag_aggregation}, "
                f"weight_temperature: {args.rag_weight_temperature})"
            )

    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    global_step = resume_step
    total_steps = args.max_train_steps
    if total_steps is None:
        total_steps = math.ceil(len(data_loader) * args.train_epochs / args.gradient_accumulation_steps)
    shape_warned = False

    steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    resume_epoch = min(global_step // steps_per_epoch, args.train_epochs)
    resume_batch_idx = (global_step % steps_per_epoch) * args.gradient_accumulation_steps
    
    # Track training time
    training_start_time = time.time()
    step_start_time = training_start_time

    if is_main_process(rank):
        print(f"train_mode: {args.train_mode} (RAG={'on' if use_rag else 'off'})")
        print(f"Starting training: {args.train_epochs} epochs, {total_steps} total steps")
        print(f"Dataset size: {len(dataset)}, DataLoader batches: {len(data_loader)}")
        print(f"Logging every {args.log_steps} steps, saving every {args.save_steps} steps")
        if global_step > 0:
            print(f"Resuming at step {global_step} (epoch {resume_epoch + 1}, batch {resume_batch_idx})")
        print("=" * 60)

    best_loss = float("inf")
    no_improve = 0
    stop_training = False

    for epoch in range(resume_epoch, args.train_epochs):
        if is_main_process(rank):
            print(f"\nEpoch {epoch + 1}/{args.train_epochs}")
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(data_loader):
            if global_step >= total_steps:
                break
            if epoch == resume_epoch and batch_idx < resume_batch_idx:
                continue

            prompts: List[str] = batch["prompt"]
            if "encoded_video" in batch:
                x0 = batch["encoded_video"].to(device=device, non_blocking=True)
            else:
                raise RuntimeError("encoded_video not found in batch; cache may be misconfigured")

            noise = torch.randn_like(x0)
            schedule_timesteps = scheduler.timesteps.to(device=device)
            step_indices = torch.randint(
                0,
                schedule_timesteps.numel(),
                (x0.size(0),),
                device=device,
            )
            timesteps = schedule_timesteps[step_indices].to(torch.float32)

            sigmas = scheduler.sigmas.to(device=device, dtype=x0.dtype)
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < len(x0.shape):
                sigma = sigma.unsqueeze(-1)
            alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
            x_t = alpha_t * x0 + sigma_t * noise
            x_t_list = [x_t[i] for i in range(x_t.size(0))]

            if "prompt_embedding" in batch:
                context = [emb.to(device=device) for emb in batch["prompt_embedding"]]
            else:
                if args.t5_cpu:
                    context = text_encoder(prompts, torch.device("cpu"))
                    context = [c.to(device=device) for c in context]
                else:
                    context = text_encoder(prompts, device)

            autocast_dtype = None
            if args.mixed_precision == "bf16":
                autocast_dtype = torch.bfloat16
            elif args.mixed_precision == "fp16":
                autocast_dtype = torch.float16

            if use_deepspeed and autocast_dtype is not None:
                x_t_list = [u.to(dtype=autocast_dtype) for u in x_t_list]
                context = [c.to(dtype=autocast_dtype) for c in context]

            if use_rag:
                # Load physical features using VideoCLIP-XL + FAISS RAG
                # This is done in main process (not in DataLoader workers) to avoid loading models in each worker
                if is_main_process(rank) and global_step <= 10:
                    rag_start = time.time()
                ref_features, _ = load_physical_feature_batch(
                    prompts=prompts,
                    dtype=torch.float32,
                    index_dir=faiss_index_dir,
                    model_path=videoclip_xl_model_path,
                    rag_top_k=args.rag_top_k,
                    rag_aggregation=args.rag_aggregation,
                    rag_weight_temperature=args.rag_weight_temperature,
                )
                if is_main_process(rank) and global_step <= 10:
                    rag_time = time.time() - rag_start
                    print(f"[TIMING] Step {global_step}: RAG loading took {rag_time:.2f}s")

                # Move ref_features to device only when needed (after loading)
                # Keep in float32 initially, will be converted in injection_manager if needed
                injection_manager.set_ref_features(ref_features, x0.shape)

            use_autocast = autocast_dtype is not None and not use_deepspeed and torch.cuda.is_available()
            # Use appropriate device type based on CUDA availability
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
                preds = model(x_t_list, t=timesteps, context=context, seq_len=seq_len)
                pred = preds if isinstance(preds, torch.Tensor) else torch.stack(preds, dim=0)
                target = noise - x0
                if pred.shape != target.shape:
                    if pred.dim() != target.dim():
                        raise RuntimeError(f"pred/target rank mismatch: {pred.shape} vs {target.shape}")
                    if is_main_process(rank) and not shape_warned:
                        print(f"warning: pred/target shape mismatch, cropping to min shape: {pred.shape} vs {target.shape}")
                        shape_warned = True
                    min_shape = [min(p, t) for p, t in zip(pred.shape, target.shape)]
                    slices = tuple(slice(0, m) for m in min_shape)
                    pred = pred[slices]
                    target = target[slices]
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                if not use_deepspeed:
                    loss = loss / args.gradient_accumulation_steps
            
            # Note: We don't clear ref_features here to ensure gradient checkpointing compatibility
            # The ref_features will be overwritten in the next forward pass

            # Backward and optimize
            if use_deepspeed:
                model.backward(loss)
                model.step()
                should_update = model.is_gradient_accumulation_boundary()
                actual_loss = loss.item()
                current_optimizer = model.optimizer if hasattr(model, 'optimizer') else None
            else:
                (scaler.scale(loss) if scaler else loss).backward()
                should_update = (batch_idx + 1) % args.gradient_accumulation_steps == 0
                if should_update:
                    if args.max_grad_norm is not None:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    (scaler.step(optimizer) if scaler else optimizer.step())
                    if scaler:
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                actual_loss = loss.item() * args.gradient_accumulation_steps
                current_optimizer = optimizer

            # Update step and log/save
            if should_update:
                global_step += 1
                is_log_step = is_main_process(rank) and global_step % args.log_steps == 0
                is_save_step = global_step % args.save_steps == 0

                if is_log_step:
                    current_time = time.time()
                    log_training_progress(
                        global_step, total_steps, actual_loss, training_start_time,
                        step_start_time, current_optimizer, model, use_deepspeed, scaler
                    )
                    step_start_time = current_time

                    if args.early_stop_patience > 0:
                        if actual_loss < best_loss - args.early_stop_min_delta:
                            best_loss = actual_loss
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= args.early_stop_patience:
                            if is_main_process(rank):
                                print(
                                    f"Early stopping triggered: loss did not improve for "
                                    f"{no_improve} logging steps."
                                )
                            stop_training = True

                if world_size > 1:
                    stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    stop_training = bool(stop_tensor.item())

                if stop_training:
                    break

                if is_save_step:
                    if is_main_process(rank):
                        print(f"Saving checkpoint at step {global_step}...")
                    save_checkpoint(model, optimizer, scaler, global_step, output_dir, args.save_total_limit, rank)
                    if is_main_process(rank):
                        print(f"✓ Checkpoint saved")

        if stop_training:
            if is_main_process(rank):
                print("Early stopping triggered. Exiting training loop.")
            break
        if global_step >= total_steps:
            break

    save_checkpoint(model, optimizer, scaler, global_step, output_dir, args.save_total_limit, rank)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
