#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified entry script for prompt-based data synthesis.

This script dispatches to the dataset-specific synthesis scripts
(amazon/yelp/yahoo/mnli) under data_synthetic/.

Each underlying script already exposes a CLI via argparse with options like
--use_api, --model_path, --api_key, --base_url, --model_name, etc.

Example usages:

- Yelp synthesis with local model:
  python run_synthesis.py --dataset yelp \
    --script_args "--model_path /path/to/llama3-8b --samples_per_label 500"

- Amazon synthesis via API:
  python run_synthesis.py --dataset amazon \
    --script_args "--use_api --api_key sk-xxx --base_url https://api.your-endpoint.com --model_name your-model"
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import Dict


_SYNTHESIS_SCRIPT_MAP: Dict[str, str] = {
    "amazon": "amazon_data_synthesis.py",
    "yelp": "yelp_data_synthesis.py",
    "yahoo": "yahoo_data_synthesis.py",
    "mnli": "MNLI_data_synthesis.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry for prompt-based data synthesis scripts"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["amazon", "yelp", "yahoo", "mnli"],
        required=True,
        help="Dataset to synthesize: amazon, yelp, yahoo, or mnli.",
    )
    parser.add_argument(
        "--script_args",
        type=str,
        default="",
        help=(
            "Extra arguments for the underlying synthesis script, e.g. "
            "--script_args \"--use_api --samples_per_label 1000\""
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def resolve_synthesis_script(dataset: str) -> str:
    if dataset not in _SYNTHESIS_SCRIPT_MAP:
        raise ValueError(f"No synthesis script mapping for dataset={dataset}")
    script_name = _SYNTHESIS_SCRIPT_MAP[dataset]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Resolved synthesis script not found: {script_path}")
    return script_path


def main() -> None:
    args = parse_args()

    script_path = resolve_synthesis_script(args.dataset)
    extra_args = shlex.split(args.script_args) if args.script_args else []

    cmd = [sys.executable, script_path] + extra_args

    print("=== Data Synthesis Unified Entry ===")
    print(f"Dataset    : {args.dataset}")
    print(f"Script     : {script_path}")
    if extra_args:
        print(f"Script args: {' '.join(extra_args)}")

    if args.dry_run:
        print("[DRY RUN] Command not executed.")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
