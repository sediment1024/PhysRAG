#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove duplicate .mp4 files (same basename) inside *_qwen3vl_caption_top10 dirs."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing *_qwen3vl_caption_top10 folders.",
    )
    parser.add_argument(
        "--suffix",
        default="_qwen3vl_caption_top10",
        help="Folder name suffix to match.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicate files.",
    )
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help="Remove empty directories after deletion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    total_removed = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(args.suffix)):
        files = sorted(folder.rglob("*.mp4"))
        seen = {}
        dupes = []
        for f in files:
            if f.name not in seen:
                seen[f.name] = f
            else:
                dupes.append(f)
        print(f"[{folder.name}] total={len(files)} unique={len(seen)} dupes={len(dupes)}")
        if args.apply and dupes:
            for f in dupes:
                try:
                    f.unlink()
                    total_removed += 1
                except Exception as exc:
                    print(f"  failed to delete: {f} ({exc})")
            if args.remove_empty_dirs:
                # Remove empty dirs bottom-up.
                for d in sorted((p for p in folder.rglob("*") if p.is_dir()), reverse=True):
                    try:
                        d.rmdir()
                    except OSError:
                        pass

    if args.apply:
        print(f"Done. Removed {total_removed} duplicate files.")
    else:
        print("Dry run only. Re-run with --apply to delete duplicates.")


if __name__ == "__main__":
    main()
