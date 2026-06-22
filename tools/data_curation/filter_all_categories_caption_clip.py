import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


LABEL_PROMPTS = {
    "collision": [
        "a photo of a physical collision between objects",
        "violent impact",
        "objects crashing",
    ],
    "explosion": [
        "a photo of an explosion",
        "fire and smoke exploding",
        "blast",
    ],
    "deformation": [
        "a photo of an object deforming",
        "bending and twisting",
        "soft body deformation",
    ],
    "melting": [
        "a photo of something melting",
        "solid turning into liquid",
        "ice melting",
    ],
    "combustion": [
        "a photo of fire and burning",
        "flames",
        "object catching fire",
    ],
    "liquid motion": [
        "a photo of liquid moving",
        "splashing water",
        "fluid dynamics",
    ],
    "gas motion": [
        "a photo of smoke or gas moving",
        "steam rising",
        "fog flowing",
    ],
    "rigid body motion": [
        "a photo of a solid object moving",
        "tumbling rocks",
        "falling object",
    ],
    "elastic motion": [
        "a photo of an elastic object bouncing",
        "jelly shaking",
        "bouncing ball",
    ],
    "reflection": [
        "a photo of a reflection in a mirror or water",
        "shiny surface reflection",
    ],
    "refraction": [
        "a photo of light bending through glass or water",
        "distortion through transparency",
    ],
    "scattering": [
        "a photo of light scattering",
        "foggy light",
        "diffused light",
    ],
    "interference and diffraction": [
        "optical interference pattern",
        "diffraction of light",
        "iridescent colors",
    ],
    "solidification": [
        "liquid freezing into solid",
        "water turning to ice",
        "freezing",
    ],
    "liquefaction": [
        "solid turning into liquid",
        "melting metal",
        "dissolving",
    ],
    "vaporization": [
        "liquid turning into gas",
        "boiling water",
        "steam evaporation",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter WISA videos by caption with CLIP text encoder.")
    model_default = os.environ.get("CLIP_MODEL_PATH")
    metadata_default = os.environ.get("WISA_METADATA_PATH")
    video_root_default = os.environ.get("WISA_VIDEO_ROOT")
    parser.add_argument(
        "--model_path",
        default=model_default,
        required=model_default is None,
        help="Local path to CLIP model.",
    )
    parser.add_argument(
        "--json_path",
        default=metadata_default,
        required=metadata_default is None,
        help="Path to wisa-80k.json.",
    )
    parser.add_argument(
        "--video_root",
        default=video_root_default,
        required=video_root_default is None,
        help="Root directory containing video folders (0,1,2,...).",
    )
    parser.add_argument(
        "--video_subdirs",
        default=None,
        help="Comma-separated subdirs under video_root to scan (e.g. 0,1,2).",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root for filtered videos (default: video_root).",
    )
    parser.add_argument(
        "--top_k_percent",
        type=float,
        default=0.1,
        help="Keep top k percent videos per category.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Only process one label (optional).",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit videos per label for quick test.",
    )
    parser.add_argument(
        "--caption_field",
        default="captions",
        help="Caption field name in the JSON.",
    )
    parser.add_argument(
        "--max_caption_tokens",
        type=int,
        default=77,
        help="Max token length for CLIP text encoder (default 77).",
    )
    parser.add_argument(
        "--min_caption_chars",
        type=int,
        default=10,
        help="Skip captions shorter than this length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for caption encoding.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing scores file if present.",
    )
    return parser.parse_args()


def build_video_index(video_root: str, subdirs: List[str] | None) -> Dict[str, str]:
    video_map: Dict[str, str] = {}
    if not os.path.isdir(video_root):
        return video_map

    for fname in os.listdir(video_root):
        fpath = os.path.join(video_root, fname)
        if os.path.isfile(fpath) and fname.endswith(".mp4"):
            video_map[fname] = fpath

    if subdirs is None:
        potential_dirs = [
            d for d in os.listdir(video_root)
            if os.path.isdir(os.path.join(video_root, d)) and d.isdigit()
        ]
    else:
        potential_dirs = subdirs

    for d in potential_dirs:
        dir_path = os.path.join(video_root, d)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".mp4"):
                video_map[fname] = os.path.join(dir_path, fname)

    return video_map


def load_existing_scores(scores_path: str) -> Dict[str, float]:
    if not os.path.exists(scores_path):
        return {}
    scores = {}
    with open(scores_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, val = line.split(":", 1)
            try:
                scores[name.strip()] = float(val.strip())
            except ValueError:
                continue
    return scores


def batch_iter(items: List[Tuple[str, str]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.video_root

    with open(args.json_path, "r") as f:
        data = json.load(f)

    subdirs = None
    if args.video_subdirs:
        subdirs = [s.strip() for s in args.video_subdirs.split(",") if s.strip()]
    video_map = build_video_index(args.video_root, subdirs)
    print(f"Indexed {len(video_map)} videos from {args.video_root}")

    device = torch.device(args.device)
    print(f"Loading CLIP from {args.model_path} to {device}...")
    model = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    labels = [args.label] if args.label else list(LABEL_PROMPTS.keys())
    for label in labels:
        prompts = LABEL_PROMPTS.get(label)
        if not prompts:
            print(f"Unknown label: {label}, skipping.")
            continue

        output_dir = os.path.join(output_root, f"{label.replace(' ', '_')}_caption_clip_top10")
        os.makedirs(output_dir, exist_ok=True)
        scores_path = os.path.join(output_root, f"{label.replace(' ', '_')}_caption_clip_scores.txt")

        existing_scores = load_existing_scores(scores_path) if args.resume else {}

        items: List[Tuple[str, str]] = []
        for entry in data:
            if entry.get("label") != label:
                continue
            caption = entry.get(args.caption_field, "")
            if not isinstance(caption, str) or len(caption.strip()) < args.min_caption_chars:
                continue
            video_name = entry.get("video_name")
            if not video_name or video_name not in video_map:
                continue
            if video_name in existing_scores:
                continue
            items.append((video_name, caption))

        if args.max_videos:
            items = items[: args.max_videos]

        print(f"[{label}] {len(items)} captions to score")

        with torch.no_grad():
            prompt_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_caption_tokens,
            )
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            prompt_feats = model.get_text_features(**prompt_inputs)
            prompt_feats = F.normalize(prompt_feats, dim=-1)

        scores: List[Tuple[str, float]] = []
        with open(scores_path, "a") as score_file:
            for batch in batch_iter(items, args.batch_size):
                names = [n for n, _ in batch]
                captions = [c for _, c in batch]
                inputs = processor(
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_caption_tokens,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    caption_feats = model.get_text_features(**inputs)
                    caption_feats = F.normalize(caption_feats, dim=-1)
                    sims = caption_feats @ prompt_feats.T
                    batch_scores = sims.max(dim=1).values.detach().cpu().tolist()

                for name, score in zip(names, batch_scores):
                    scores.append((video_map[name], float(score)))
                    score_file.write(f"{name}: {score}\n")
                score_file.flush()

        scores.sort(key=lambda x: x[1], reverse=True)
        num_keep = max(1, int(len(scores) * args.top_k_percent)) if scores else 0
        print(f"[{label}] Keeping top {num_keep} videos.")
        for src_path, score in scores[:num_keep]:
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print("Done.")


if __name__ == "__main__":
    main()
