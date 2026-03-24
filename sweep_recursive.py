#!/usr/bin/env python3
"""Sweep runner for recursive diffusion LM configs.

Runs configs sequentially on 8xH100 via torchrun.
Results are logged to sweep_results/ directory.

Usage:
  python3 sweep_recursive.py                    # run all configs
  python3 sweep_recursive.py --configs baseline recurse2 recurse4_xsa  # run subset
  python3 sweep_recursive.py --dry-run          # print commands only
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Each config is a dict of env var overrides on top of defaults.
# The baseline config matches current SOTA minus competition-specific tricks.
CONFIGS = {
    # === MDLM single pass (no recursion) ===
    "mdlm_1pass": {
        "RECURSE_TRAIN_MAX": "1", "RECURSE_EVAL": "1",
    },

    # === Recursion depth ===
    "mdlm_r2": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2",
    },
    "mdlm_r4": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "4",
        "RECURSE_EVAL": "4",
    },
    "mdlm_r2_fixed": {
        "RECURSE_TRAIN_MIN": "2", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2",
    },

    # === Eval-time scaling ===
    "mdlm_r2_eval4": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "4",
    },
    "mdlm_r2_eval8": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "8",
    },

    # === Temperature ===
    "mdlm_r2_temp05": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "RECURSE_TEMP": "0.5",
    },
    "mdlm_r2_temp2": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "RECURSE_TEMP": "2.0",
    },

    # === EMA blending ===
    "mdlm_r2_ema08": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "RECURSE_EMA": "0.8",
    },

    # === Step weighting ===
    "mdlm_r2_uniform": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "RECURSE_STEP_WEIGHT": "uniform",
    },
    "mdlm_r2_last1": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "RECURSE_STEP_WEIGHT": "last_1",
    },

    # === MDLM loss weight ===
    "mdlm_r2_uniform_loss": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "MDLM_LOSS_WEIGHT": "uniform",
    },

    # === Eval stride ===
    "mdlm_r2_stride128": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "2", "EVAL_STRIDE": "128",
    },

    # === Combined ===
    "mdlm_r2_temp05_ema08": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "2",
        "RECURSE_EVAL": "4", "RECURSE_TEMP": "0.5", "RECURSE_EMA": "0.8",
    },
    "mdlm_r4_ema08": {
        "RECURSE_TRAIN_MIN": "1", "RECURSE_TRAIN_MAX": "4",
        "RECURSE_EVAL": "4", "RECURSE_EMA": "0.8",
    },
}

# Shared defaults for all configs
SHARED_DEFAULTS = {
    "ITERATIONS": "20000",
    "TRAIN_BATCH_TOKENS": "524288",
    "TRAIN_SEQ_LEN": "1024",
    "VAL_LOSS_EVERY": "1000",
    "MAX_WALLCLOCK_SECONDS": "600",
}


def run_config(name: str, overrides: dict, nproc: int, dry_run: bool, out_dir: Path):
    env = {**os.environ, **SHARED_DEFAULTS, **overrides, "RUN_ID": name}
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
        return

    with open(log_file, "w") as f:
        t0 = time.time()
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
    else:
        print(f"  DONE in {elapsed:.0f}s")

    # Extract final val_bpb from log
    try:
        with open(log_file) as f:
            for line in reversed(f.readlines()):
                if "val_bpb" in line:
                    print(f"  Result: {line.strip()}")
                    break
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Sweep recursive diffusion configs")
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of configs to run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--nproc", type=int, default=8, help="GPUs per run")
    parser.add_argument("--out-dir", type=str, default="sweep_results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    configs_to_run = args.configs or list(CONFIGS.keys())
    print(f"Running {len(configs_to_run)} configs: {configs_to_run}")

    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"WARNING: Unknown config '{name}', skipping")
            continue
        run_config(name, CONFIGS[name], args.nproc, args.dry_run, out_dir)

    print(f"\nAll done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
