#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from statistics import mean

import numpy as np
from PIL import Image

COMMON_EXCLUSION_PROMPTS = ["no people", "no portraits", "no human faces"]

# Light prompt mapping for WISA labels; timing task does not depend on exact wording.
LABEL_PROMPTS = {
    "collision": "a physical collision between objects",
    "combustion": "fire and burning",
    "deformation": "object deforming, bending, twisting",
    "elastic motion": "elastic object bouncing",
    "explosion": "an explosion with blast, fire or smoke",
    "gas motion": "gas or smoke moving",
    "interference and diffraction": "optical interference or diffraction",
    "liquefaction": "solid turning into liquid",
    "liquid motion": "liquid moving and splashing",
    "melting": "solid melting into liquid",
    "reflection": "reflection on mirror or water",
    "refraction": "light refraction through glass or water",
    "rigid body motion": "rigid object moving",
    "scattering": "light scattering",
    "solidification": "liquid freezing into solid",
    "unnatural light source": "unnatural or artificial light source",
    "vaporization": "liquid turning into gas",
}

SYSTEM_PROMPT = (
    "You are a strict grader for video-text alignment. "
    "Only reply with a single number between 0 and 100. "
    "不要输出除数字以外的内容。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-VL latency on one video: caption vs caption+first-frame vs video."
    )
    parser.add_argument("--video_path", required=True, help="Path to one mp4 file.")
    model_default = os.environ.get("QWEN3_VL_MODEL_PATH")
    metadata_default = os.environ.get("WISA_METADATA_PATH")
    parser.add_argument("--model_path", default=model_default, required=model_default is None)
    parser.add_argument(
        "--json_path",
        default=metadata_default,
        required=metadata_default is None,
        help="wisa-80k.json for auto loading label/caption by video_name.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype. fp16 is usually most stable for this benchmark.",
    )
    parser.add_argument(
        "--attn_impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. eager is slower but often avoids kernel crashes.",
    )
    parser.add_argument("--label", default=None, help="Optional label override.")
    parser.add_argument("--caption", default=None, help="Optional caption override.")
    parser.add_argument("--caption_field", default="captions")
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--first_frame_size", type=int, default=256)
    parser.add_argument("--max_caption_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1, help="Runs per mode.")
    parser.add_argument(
        "--only_mode",
        default="all",
        choices=["all", "caption", "caption_first_frame", "video"],
        help="Run only one mode for debugging stability.",
    )
    parser.add_argument("--include_model_load", action="store_true")
    return parser.parse_args()


def load_meta(json_path: str, video_name: str, caption_field: str):
    if not os.path.isfile(json_path):
        return None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item.get("video_name") == video_name:
            return item.get("label"), item.get(caption_field)
    return None, None


def merged_prompt(label: str) -> str:
    base = LABEL_PROMPTS.get(label, f"a video about {label}")
    return ", ".join([base] + COMMON_EXCLUSION_PROMPTS)


def parse_score(text: str) -> float:
    matches = re.findall(r"\d+\.?\d*", text)
    if not matches:
        return 0.0
    score = float(matches[-1])
    if score <= 10.0:
        score *= 10.0
    return max(0.0, min(100.0, score))


def _sample_frames_cv2(video_path: str, num_frames: int):
    import cv2

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
        indices = [int(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def _sample_first_frame_cv2(video_path: str, size: int):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if size > 0:
        image = image.resize((size, size), Image.BILINEAR)
    return image


def _sample_frames_imageio(video_path: str, num_frames: int):
    import imageio.v3 as iio

    frames_np = list(iio.imiter(video_path))
    if not frames_np:
        return []
    n = len(frames_np)
    if num_frames <= 1:
        indices = [n // 2]
    else:
        indices = [int(i * (n - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        arr = np.asarray(frames_np[idx])
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        frames.append(Image.fromarray(arr[:, :, :3].astype(np.uint8)))
    return frames


def _sample_first_frame_imageio(video_path: str, size: int):
    import imageio.v3 as iio

    try:
        arr = iio.imread(video_path, index=0)
    except Exception:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    image = Image.fromarray(arr[:, :, :3].astype(np.uint8))
    if size > 0:
        image = image.resize((size, size), Image.BILINEAR)
    return image


def sample_frames(video_path: str, num_frames: int):
    try:
        return _sample_frames_cv2(video_path, num_frames)
    except Exception:
        return _sample_frames_imageio(video_path, num_frames)


def sample_first_frame(video_path: str, size: int):
    try:
        return _sample_first_frame_cv2(video_path, size)
    except Exception:
        return _sample_first_frame_imageio(video_path, size)


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


def build_video_prompt(prompt: str) -> str:
    return (
        "Please rate how well the video matches the description. "
        "Respond with a single number between 0 and 100.\n"
        f"Description: {prompt}"
    )


def score_caption(model, processor, caption: str, prompt: str, device, max_tokens: int) -> float:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_caption_prompt(prompt, caption)},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    return parse_score(response)


def score_caption_with_frame(model, processor, frame: Image.Image, caption: str, prompt: str, device, max_tokens: int) -> float:
    content = [
        {"type": "image", "image": frame},
        {"type": "text", "text": build_caption_frame_prompt(prompt, caption)},
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[frame], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    return parse_score(response)


def score_video(model, processor, frames, prompt: str, device) -> float:
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": build_video_prompt(prompt)})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.batch_decode([gen_ids], skip_special_tokens=True)[0]
    return parse_score(response)


RUNTIME = {}


def load_runtime_dependencies() -> None:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Qwen3-VL runtime dependencies. "
            "Use an environment with transformers>=5 and qwen3vl support."
        ) from exc

    RUNTIME.update(
        {
            "torch": torch,
            "AutoProcessor": AutoProcessor,
            "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
            "no_grad": torch.no_grad,
        }
    )


def run_caption(model, processor, device, prompt: str, caption: str, max_caption_tokens: int):
    t0 = time.perf_counter()
    score = score_caption(model, processor, caption=caption, prompt=prompt, device=device, max_tokens=max_caption_tokens)
    return score, time.perf_counter() - t0


def run_caption_frame(model, processor, device, video_path: str, prompt: str, caption: str, first_frame_size: int, max_caption_tokens: int):
    t0 = time.perf_counter()
    frame = sample_first_frame(video_path, first_frame_size)
    if frame is None:
        raise RuntimeError("Failed to read first frame.")
    score = score_caption_with_frame(
        model,
        processor,
        frame=frame,
        caption=caption,
        prompt=prompt,
        device=device,
        max_tokens=max_caption_tokens,
    )
    return score, time.perf_counter() - t0


def run_video(model, processor, device, video_path: str, prompt: str, num_frames: int):
    t0 = time.perf_counter()
    frames = sample_frames(video_path, num_frames)
    if not frames:
        raise RuntimeError("Failed to sample frames from video.")
    score = score_video(model, processor, frames=frames, prompt=prompt, device=device)
    return score, time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()

    global no_grad
    no_grad = RUNTIME["no_grad"]

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(args.video_path)

    video_name = os.path.basename(args.video_path)
    meta_label, meta_caption = load_meta(args.json_path, video_name, args.caption_field)

    label = args.label or meta_label
    caption = args.caption or meta_caption

    if not label:
        raise ValueError("Cannot resolve label. Pass --label or provide json with this video.")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Cannot resolve caption. Pass --caption or provide json with this video.")

    prompt = merged_prompt(label)

    torch = RUNTIME["torch"]
    AutoProcessor = RUNTIME["AutoProcessor"]
    Qwen3VLForConditionalGeneration = RUNTIME["Qwen3VLForConditionalGeneration"]

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    device = torch.device(args.device)

    t_load0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map=None,
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    load_sec = time.perf_counter() - t_load0

    if args.include_model_load:
        print(f"model_load_sec={load_sec:.3f}")

    results = {"caption": [], "caption_first_frame": [], "video": []}
    scores = {"caption": [], "caption_first_frame": [], "video": []}

    for _ in range(max(1, args.repeat)):
        if args.only_mode in ("all", "caption"):
            print("[run] mode=caption")
            s, dt = run_caption(model, processor, device, prompt, caption, args.max_caption_tokens)
            scores["caption"].append(s)
            results["caption"].append(dt)

        if args.only_mode in ("all", "caption_first_frame"):
            print("[run] mode=caption_first_frame")
            s, dt = run_caption_frame(
                model,
                processor,
                device,
                args.video_path,
                prompt,
                caption,
                args.first_frame_size,
                args.max_caption_tokens,
            )
            scores["caption_first_frame"].append(s)
            results["caption_first_frame"].append(dt)

        if args.only_mode in ("all", "video"):
            print("[run] mode=video")
            s, dt = run_video(model, processor, device, args.video_path, prompt, args.num_frames)
            scores["video"].append(s)
            results["video"].append(dt)

    print(f"video={args.video_path}")
    print(f"video_name={video_name}")
    print(f"label={label}")
    print(f"repeat={max(1, args.repeat)}")
    print("---")
    if results["caption"]:
        print(f"caption_sec={mean(results['caption']):.3f} runs={results['caption']} score={mean(scores['caption']):.2f}")
    if results["caption_first_frame"]:
        print(
            f"caption_first_frame_sec={mean(results['caption_first_frame']):.3f} "
            f"runs={results['caption_first_frame']} score={mean(scores['caption_first_frame']):.2f}"
        )
    if results["video"]:
        print(f"video_sec={mean(results['video']):.3f} runs={results['video']} score={mean(scores['video']):.2f}")


if __name__ == "__main__":
    main()
