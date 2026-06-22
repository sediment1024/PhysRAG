import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


LABEL_PROMPTS = {
    "collision": [
        "a photo of a physical collision between objects",
        "violent impact",
        "objects crashing",
        "real-world collision, not toy blocks",
        "not LEGO blocks",
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
    "unnatural light source": [
        "a photo of an unnatural or artificial light source",
        "unnatural lighting, abnormal illumination",
        "non-physical light source in the scene",
    ],
}

SYSTEM_PROMPT = (
    "You are a strict grader for video-text alignment. "
    "Only reply with a single number between 0 and 100. "
    "不要输出除数字以外的内容。"
)
COMMON_EXCLUSION_PROMPTS = [
    "no people",
    "no portraits",
    "no human faces",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter WISA videos with Qwen3-VL-4B.")
    model_default = os.environ.get("QWEN3_VL_MODEL_PATH")
    metadata_default = os.environ.get("WISA_METADATA_PATH")
    video_root_default = os.environ.get("WISA_VIDEO_ROOT")
    parser.add_argument(
        "--model_path",
        default=model_default,
        required=model_default is None,
        help="Local path to Qwen3-VL-4B-Instruct.",
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
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for mp4 files.",
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
        "--top_k_per_label",
        type=int,
        default=None,
        help="Keep exactly this many videos per category, capped by available scored videos.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=6,
        help="Number of frames sampled per video.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (e.g. cuda, cuda:1, cpu).",
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
        "--resume",
        action="store_true",
        help="Resume from existing scores file if present.",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=0,
        help="Print raw model outputs for the first N samples.",
    )
    parser.add_argument(
        "--use_caption",
        action="store_true",
        help="Score captions only (no video frames).",
    )
    parser.add_argument(
        "--caption_with_first_frame",
        action="store_true",
        help="When using --use_caption, also include the first video frame.",
    )
    parser.add_argument(
        "--caption_field",
        default="captions",
        help="Caption field in wisa-80k.json.",
    )
    parser.add_argument(
        "--min_caption_chars",
        type=int,
        default=20,
        help="Minimum caption length to score.",
    )
    parser.add_argument(
        "--max_caption_chars",
        type=int,
        default=2000,
        help="Maximum caption length (characters) to keep.",
    )
    parser.add_argument(
        "--max_caption_tokens",
        type=int,
        default=512,
        help="Max tokens for caption prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for caption scoring (higher is faster on large GPUs).",
    )

    parser.add_argument(
        "--first_frame_size",
        type=int,
        default=256,
        help="Resize first frame to NxN before scoring (0 to disable).",
    )
    return parser.parse_args()


def build_video_index(video_root: str, subdirs: List[str] | None, recursive: bool) -> Dict[str, str]:
    video_map: Dict[str, str] = {}
    if not os.path.isdir(video_root):
        return video_map

    # If video_root directly contains mp4 files, index them.
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
        if recursive:
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if fname.endswith(".mp4"):
                        video_map[fname] = os.path.join(root, fname)
        else:
            for fname in os.listdir(dir_path):
                if fname.endswith(".mp4"):
                    video_map[fname] = os.path.join(dir_path, fname)

    return video_map


def sample_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []
    if num_frames <= 1:
        indices = [frame_count // 2]
    else:
        indices = [
            int(i * (frame_count - 1) / (num_frames - 1))
            for i in range(num_frames)
        ]
    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def sample_first_frame(video_path: str, size: int) -> Image.Image | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if size and size > 0:
        image = image.resize((size, size), Image.BILINEAR)
    return image


def parse_score(text: str) -> float:
    matches = re.findall(r"\d+\.?\d*", text)
    if not matches:
        return 0.0
    score = float(matches[-1])
    if score <= 10.0:
        score *= 10.0
    return max(0.0, min(100.0, score))


def build_prompt(prompt: str) -> str:
    return (
        "Please rate how well the video matches the description. "
        "Respond with a single number between 0 and 100.\n"
        f"Description: {prompt}"
    )


def build_caption_prompt(prompt: str, caption: str) -> str:
    return (
        "Please rate how well the caption matches the description. "
        "Respond with a single number between 0 and 100.\n"
        f"Description: {prompt}\n"
        f"Caption: {caption}"
    )


def build_caption_frame_prompt(prompt: str, caption: str) -> str:
    return (
        "Please rate how well the caption and image match the description. "
        "Respond with a single number between 0 and 100.\n"
        f"Description: {prompt}\n"
        f"Caption: {caption}"
    )


def score_video(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frames: List[Image.Image],
    prompt: str,
    device: torch.device,
    debug: bool = False,
) -> float:
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": build_prompt(prompt)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    if "input_ids" in inputs:
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    else:
        gen_ids = output_ids[0]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    if debug:
        print(f"Model raw output: {response}")
    return parse_score(response)


def score_caption(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    caption: str,
    prompt: str,
    device: torch.device,
    max_tokens: int,
    debug: bool = False,
) -> float:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_caption_prompt(prompt, caption)},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    if "input_ids" in inputs:
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    else:
        gen_ids = output_ids[0]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    if debug:
        print(f"Model raw output: {response}")
    return parse_score(response)


def score_caption_with_frame(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frame: Image.Image,
    caption: str,
    prompt: str,
    device: torch.device,
    max_tokens: int,
    debug: bool = False,
) -> float:
    content = [
        {"type": "image", "image": frame},
        {"type": "text", "text": build_caption_frame_prompt(prompt, caption)},
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[frame],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    if "input_ids" in inputs:
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    else:
        gen_ids = output_ids[0]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    if debug:
        print(f"Model raw output: {response}")
    return parse_score(response)



def score_caption_batch(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    captions: List[str],
    prompt: str,
    device: torch.device,
    max_tokens: int,
    debug: bool = False,
) -> List[float]:
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_caption_prompt(prompt, caption)},
        ]
        for caption in captions
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = processor(
        text=texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    if "attention_mask" in inputs:
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        prompt_lens = [inputs["input_ids"].shape[1]] * len(texts)
    responses = []
    for i, prompt_len in enumerate(prompt_lens):
        gen_ids = output_ids[i][prompt_len:]
        responses.append(processor.decode(gen_ids, skip_special_tokens=True))
    if debug and responses:
        print(f"Model raw output (batch 0): {responses[0]}")
    return [parse_score(r) for r in responses]


def score_caption_with_frame_batch(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    frames: List[Image.Image],
    captions: List[str],
    prompt: str,
    device: torch.device,
    max_tokens: int,
    debug: bool = False,
) -> List[float]:
    contents = [
        [
            {"type": "image", "image": frame},
            {"type": "text", "text": build_caption_frame_prompt(prompt, caption)},
        ]
        for frame, caption in zip(frames, captions)
    ]
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        for content in contents
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = processor(
        text=texts,
        images=frames,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    if "attention_mask" in inputs:
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        prompt_lens = [inputs["input_ids"].shape[1]] * len(texts)
    responses = []
    for i, prompt_len in enumerate(prompt_lens):
        gen_ids = output_ids[i][prompt_len:]
        responses.append(processor.decode(gen_ids, skip_special_tokens=True))
    if debug and responses:
        print(f"Model raw output (batch 0): {responses[0]}")
    return [parse_score(r) for r in responses]

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


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.video_root

    with open(args.json_path, "r") as f:
        data = json.load(f)

    labels = [args.label] if args.label else list(LABEL_PROMPTS.keys())
    subdirs = None
    if args.video_subdirs:
        subdirs = [s.strip() for s in args.video_subdirs.split(",") if s.strip()]
    video_map = build_video_index(args.video_root, subdirs, args.recursive)
    print(f"Indexed {len(video_map)} videos from {args.video_root}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    print(f"Loading Qwen3-VL from {args.model_path} to {device}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    score_suffix = "qwen3vl"
    if args.use_caption:
        score_suffix = "qwen3vl_caption"
        if args.caption_with_first_frame:
            score_suffix = "qwen3vl_caption_frame"

    for label in labels:
        prompts = LABEL_PROMPTS.get(label)
        if not prompts:
            print(f"Unknown label: {label}, skipping.")
            continue
        merged_prompt = ", ".join(prompts + COMMON_EXCLUSION_PROMPTS)
        if args.top_k_per_label is not None:
            keep_suffix = f"top{args.top_k_per_label}"
        else:
            keep_suffix = f"top{int(args.top_k_percent * 100)}"
        output_dir = os.path.join(output_root, f"{label.replace(' ', '_')}_{score_suffix}_{keep_suffix}")
        os.makedirs(output_dir, exist_ok=True)
        scores_path = os.path.join(output_root, f"{label.replace(' ', '_')}_{score_suffix}_scores.txt")

        existing_scores = load_existing_scores(scores_path) if args.resume else {}
        scores: List[Tuple[str, float]] = []

        if args.use_caption:
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
                items.append((video_map[video_name], caption))

            if args.max_videos:
                items = items[: args.max_videos]

            print(f"[{label}] {len(items)} captions to score")

            with open(scores_path, "a") as score_file:
                        batch_size = max(1, args.batch_size)
                        processed = 0
                        for start_idx in range(0, len(items), batch_size):
                            batch = items[start_idx : start_idx + batch_size]
                            pending = []
                            for video_path, caption in batch:
                                video_name = os.path.basename(video_path)
                                if video_name in existing_scores:
                                    scores.append((video_path, existing_scores[video_name]))
                                    continue
                                pending.append((video_path, caption, video_name))

                            if not pending:
                                processed += len(batch)
                                if processed % 50 == 0:
                                    print(f"[{label}] Processed {processed}/{len(items)}")
                                continue

                            try:
                                if args.caption_with_first_frame:
                                    frames = []
                                    captions = []
                                    names = []
                                    paths = []
                                    for video_path, caption, video_name in pending:
                                        if args.max_caption_chars and len(caption) > args.max_caption_chars:
                                            caption = caption[: args.max_caption_chars]
                                        frame = sample_first_frame(video_path, args.first_frame_size)
                                        if frame is None:
                                            continue
                                        frames.append(frame)
                                        captions.append(caption)
                                        names.append(video_name)
                                        paths.append(video_path)
                                    if frames:
                                        batch_scores = score_caption_with_frame_batch(
                                            model,
                                            processor,
                                            frames,
                                            captions,
                                            merged_prompt,
                                            device,
                                            max_tokens=args.max_caption_tokens,
                                            debug=(args.debug_samples > 0 and start_idx < args.debug_samples),
                                        )
                                        for video_path, video_name, score in zip(paths, names, batch_scores):
                                            scores.append((video_path, score))
                                            score_file.write(f"{video_name}: {score}\n")
                                            score_file.flush()
                                else:
                                    captions = []
                                    names = []
                                    paths = []
                                    for video_path, caption, video_name in pending:
                                        captions.append(caption)
                                        names.append(video_name)
                                        paths.append(video_path)
                                    if captions:
                                        batch_scores = score_caption_batch(
                                            model,
                                            processor,
                                            captions,
                                            merged_prompt,
                                            device,
                                            max_tokens=args.max_caption_tokens,
                                            debug=(args.debug_samples > 0 and start_idx < args.debug_samples),
                                        )
                                        for video_path, video_name, score in zip(paths, names, batch_scores):
                                            scores.append((video_path, score))
                                            score_file.write(f"{video_name}: {score}\n")
                                            score_file.flush()
                            except Exception as exc:
                                print(f"[{label}] Batch error at {start_idx}: {exc}. Falling back to single scoring.")
                                for video_path, caption, video_name in pending:
                                    try:
                                        if args.caption_with_first_frame:
                                            if args.max_caption_chars and len(caption) > args.max_caption_chars:
                                                caption = caption[: args.max_caption_chars]
                                            frame = sample_first_frame(video_path, args.first_frame_size)
                                            if frame is None:
                                                continue
                                            score = score_caption_with_frame(
                                                model,
                                                processor,
                                                frame,
                                                caption,
                                                merged_prompt,
                                                device,
                                                max_tokens=args.max_caption_tokens,
                                                debug=(args.debug_samples > 0 and start_idx < args.debug_samples),
                                            )
                                        else:
                                            score = score_caption(
                                                model,
                                                processor,
                                                caption,
                                                merged_prompt,
                                                device,
                                                max_tokens=args.max_caption_tokens,
                                                debug=(args.debug_samples > 0 and start_idx < args.debug_samples),
                                            )
                                    except Exception as inner_exc:
                                        print(f"[{label}] Error on {video_name}: {inner_exc}")
                                        continue
                                    scores.append((video_path, score))
                                    score_file.write(f"{video_name}: {score}\n")
                                    score_file.flush()

                            processed += len(batch)
                            if processed % 50 == 0:
                                print(f"[{label}] Processed {processed}/{len(items)}")

        else:
            target_videos = [item["video_name"] for item in data if item.get("label") == label]
            available = [video_map[v] for v in target_videos if v in video_map]
            if args.max_videos:
                available = available[: args.max_videos]
            print(f"[{label}] {len(target_videos)} in metadata, {len(available)} on disk")

            with open(scores_path, "a") as score_file:
                for idx, video_path in enumerate(available):
                    video_name = os.path.basename(video_path)
                    if video_name in existing_scores:
                        scores.append((video_path, existing_scores[video_name]))
                        continue
                    frames = sample_frames(video_path, args.num_frames)
                    if not frames:
                        continue
                    try:
                        score = score_video(
                            model,
                            processor,
                            frames,
                            merged_prompt,
                            device,
                            debug=(args.debug_samples > 0 and idx < args.debug_samples),
                        )
                    except Exception as exc:
                        print(f"[{label}] Error on {video_name}: {exc}")
                        continue
                    scores.append((video_path, score))
                    score_file.write(f"{video_name}: {score}\n")
                    score_file.flush()
                    if (idx + 1) % 25 == 0:
                        print(f"[{label}] Processed {idx + 1}/{len(available)}")

        scores.sort(key=lambda x: x[1], reverse=True)
        if args.top_k_per_label is not None:
            num_keep = min(args.top_k_per_label, len(scores))
        else:
            num_keep = max(1, int(len(scores) * args.top_k_percent)) if scores else 0
        print(f"[{label}] Keeping top {num_keep} videos.")
        for src_path, score in scores[:num_keep]:
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print("Done.")


if __name__ == "__main__":
    main()
