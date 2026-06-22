#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Iterable, List, Dict, Any, Set


def list_video_ids(video_dir: str) -> Set[str]:
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    ids: Set[str] = set()
    for name in os.listdir(video_dir):
        path = os.path.join(video_dir, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            ids.add(stem)
    return ids


def load_master_json(master_json_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(master_json_path):
        raise FileNotFoundError(f"Master JSON not found: {master_json_path}")
    with open(master_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept common layouts: list[...] or {"data": list[...]} or {"items": list[...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "samples", "annotations", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unsupported JSON structure: expected list or dict with 'data'/'items'.")


def match_entry_id(entry: Dict[str, Any]) -> Iterable[str]:
    # Try common id fields
    for key in ("id", "video_id", "videoId", "uid", "name", "filename", "file_name", "video_name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            stem, _ = os.path.splitext(value)
            yield stem
        elif isinstance(value, int):
            yield str(value)
    # Special cases: nested structures
    video = entry.get("video")
    if isinstance(video, dict):
        for key in ("id", "video_id", "name", "filename", "file_name"):
            value = video.get(key)
            if isinstance(value, str) and value:
                stem, _ = os.path.splitext(value)
                yield stem
            elif isinstance(value, int):
                yield str(value)


def filter_entries_by_ids(entries: List[Dict[str, Any]], wanted_ids: Set[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for entry in entries:
        for candidate in match_entry_id(entry):
            if candidate in wanted_ids:
                selected.append(entry)
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract entries from a master JSON whose IDs match video filenames.")
    parser.add_argument("--video-dir", required=True, help="Directory containing selected videos; filenames (without extension) are treated as IDs.")
    parser.add_argument("--master-json", required=True, help="Path to the master JSON file (e.g., wisa-80k.json).")
    parser.add_argument("--output", required=True, help="Path to write the filtered JSON.")
    args = parser.parse_args()

    try:
        wanted_ids = list_video_ids(args.video_dir)
        if not wanted_ids:
            print(f"[WARN] No video files found in {args.video_dir}", file=sys.stderr)
        entries = load_master_json(args.master_json)
        selected = filter_entries_by_ids(entries, wanted_ids)
        # Preserve as a list; do not wrap unless original was wrapped (we don't know), so keep it simple.
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        # Report basic stats
        print(f"Total master entries: {len(entries)}", file=sys.stderr)
        print(f"Video IDs found: {len(wanted_ids)}", file=sys.stderr)
        print(f"Matched entries: {len(selected)}", file=sys.stderr)
        # Also report missing IDs (those without a match), capped to a reasonable count
        matched_ids: Set[str] = set()
        for entry in selected:
            for candidate in match_entry_id(entry):
                if candidate in wanted_ids:
                    matched_ids.add(candidate)
        missing_ids = sorted(wanted_ids - matched_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:20])
            more = "" if len(missing_ids) <= 20 else f" (+{len(missing_ids) - 20} more)"
            print(f"[WARN] IDs without matches: {preview}{more}", file=sys.stderr)
        print(f"Wrote filtered JSON to: {args.output}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


