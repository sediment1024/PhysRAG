#!/usr/bin/env python3
"""Verify and safely extract PhysRAG dataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if root.resolve() not in destination.parents:
        raise RuntimeError(f"Unsafe tar member path: {member_name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "shard_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        shard = args.dataset_dir / row["shard"]
        if not args.skip_checksum:
            actual = sha256_file(shard)
            if actual != row["sha256"]:
                raise RuntimeError(f"Checksum mismatch for {shard}: {actual}")
        with tarfile.open(shard, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                destination = safe_destination(args.output_dir, member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Failed to read {member.name} from {shard}")
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        print(f"[{index}/{len(rows)}] extracted {shard.name}")

    for relative in ("metadata.jsonl", "prompts_new.txt", "videos_new.txt", "dataset_info.json"):
        shutil.copy2(args.dataset_dir / relative, args.output_dir / relative)
    shutil.copytree(args.dataset_dir / "rag", args.output_dir / "rag", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
