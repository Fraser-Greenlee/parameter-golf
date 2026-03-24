#!/usr/bin/env python3
"""Sweep runner for interleaved draft token configs.

Runs configs sequentially on 8xH100 via torchrun.
Results are logged to sweep_results_interleaved/ directory and W&B.

Usage:
  python3 sweep_interleaved.py                         # run all configs
  python3 sweep_interleaved.py --configs baseline_sota draft_512_a03  # run subset
  python3 sweep_interleaved.py --dry-run               # print commands only
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Each config is a dict of env var overrides on top of SHARED_DEFAULTS.
CONFIGS = {
    # === Baselines ===
    "baseline_sota": {
        # Exact SOTA settings, no changes — establishes comparison
    },
    "baseline_leakyrelu": {
        "LEAKY_RELU_SLOPE": "0.5",
    },

    # === Full Interleaved (512 real → 1024 interleaved) ===
    "draft_512_a03": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
    },
    "draft_512_a05": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.5",
    },
    "draft_512_a00": {
        # Perfect drafts (GT only) — upper bound on draft benefit
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.0",
    },

    # === Full Interleaved (1024 real → 2048 interleaved, same context as baseline) ===
    "draft_1024_a03": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "1024",
        "DRAFT_ALPHA_END": "0.3",
    },

    # === Exposure Bias ===
    "draft_512_a03_exp10": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
        "DRAFT_EXPOSURE_FRAC": "0.1",
    },

    # === Auxiliary Draft Loss ===
    "draft_512_a03_aux": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
        "DRAFT_AUX_LOSS": "0.1",
    },

    # === Combined: Draft + LeakyReLU ===
    "draft_512_a03_leaky": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
        "LEAKY_RELU_SLOPE": "0.5",
    },

    # === Eval-only two-pass (control — no training change) ===
    "evalonly_twopass": {
        "EVAL_TWO_PASS": "1",
    },
    "evalonly_twopass_leaky": {
        "EVAL_TWO_PASS": "1",
        "LEAKY_RELU_SLOPE": "0.5",
    },

    # === Temperature sweep ===
    "draft_512_temp05": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
        "DRAFT_TEMP": "0.5",
    },
    "draft_512_temp20": {
        "DRAFT_ENABLED": "1",
        "DRAFT_TRAIN_SEQ_LEN": "512",
        "DRAFT_ALPHA_END": "0.3",
        "DRAFT_TEMP": "2.0",
    },
}

# Shared defaults for all configs
SHARED_DEFAULTS = {
    "ITERATIONS": "20000",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_SEQ_LEN": "2048",
    "MAX_WALLCLOCK_SECONDS": "600",
    "VAL_LOSS_EVERY": "2000",
    "EVAL_STRIDE": "64",
}


def run_config(name: str, overrides: dict, nproc: int, dry_run: bool, out_dir: Path,
               wandb_project: str = ""):
    env = {**os.environ, **SHARED_DEFAULTS, **overrides, "RUN_ID": name}
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    log_file = out_dir / f"{name}.log"

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "train_gpt.py",
    ]

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"Overrides: {overrides}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        env_diff = {k: v for k, v in {**SHARED_DEFAULTS, **overrides}.items()}
        for k, v in sorted(env_diff.items()):
            print(f"    {k}={v}")
        return

    with open(log_file, "w") as f:
        t0 = time.time()
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
    else:
        print(f"  DONE in {elapsed:.0f}s")

    # Extract final metrics from log
    try:
        with open(log_file) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "final_int6_sliding_window_exact" in line or "final_twopass_exact" in line:
                print(f"  Result: {line.strip()}")
                break
            if "val_bpb" in line:
                print(f"  Result: {line.strip()}")
                break
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Sweep interleaved draft token configs")
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of configs to run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--nproc", type=int, default=8, help="GPUs per run")
    parser.add_argument("--out-dir", type=str, default="sweep_results_interleaved", help="Output directory")
    parser.add_argument("--wandb-project", type=str, default="parameter-golf-interleaved",
                        help="W&B project (empty string to disable)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    configs_to_run = args.configs or list(CONFIGS.keys())
    print(f"Running {len(configs_to_run)} configs: {configs_to_run}")

    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"WARNING: Unknown config '{name}', skipping")
            continue
        run_config(name, CONFIGS[name], args.nproc, args.dry_run, out_dir, args.wandb_project)

    print(f"\nAll done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
