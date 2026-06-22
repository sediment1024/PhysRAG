#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from statistics import mean


CATEGORY_MAP = {
    "Force": "Mechanics",
    "Light": "Optics",
    "Heat": "Thermal",
    "Physical Properties": "Material",
    "Chemical Properties": "Material",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PhyGenBench category metrics from a result JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PhyGenBench result JSON (list of samples).",
    )
    parser.add_argument(
        "--modelname",
        default=None,
        help="Model name to resolve score key as '<modelname>_open'.",
    )
    parser.add_argument(
        "--score-key",
        default=None,
        help="Score field name in JSON (overrides --modelname).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize by dividing scores by 3.",
    )
    return parser.parse_args()


def get_score_key(args: argparse.Namespace) -> str:
    if args.score_key:
        return args.score_key
    if args.modelname:
        return f"{args.modelname}_open"
    raise ValueError("Provide --score-key or --modelname.")


def main() -> None:
    args = parse_args()
    score_key = get_score_key(args)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets = defaultdict(list)
    for row in data:
        main_category = row.get("main_category")
        mapped = CATEGORY_MAP.get(main_category)
        if not mapped:
            continue
        score = row.get(score_key)
        if score is None:
            continue
        buckets[mapped].append(score)

    results = {}
    for name in ("Mechanics", "Optics", "Thermal", "Material"):
        values = buckets.get(name, [])
        results[name] = mean(values) if values else None

    avg_values = [v for v in results.values() if v is not None]
    results["Average"] = mean(avg_values) if avg_values else None

    if args.normalize:
        results = {k: (v / 3 if v is not None else None) for k, v in results.items()}

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
