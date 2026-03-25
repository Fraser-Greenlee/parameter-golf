#!/bin/bash
set -euo pipefail

# Time sweep: baseline vs draft at different sequence lengths
# Baseline seq_len=2048, 500 steps each to let torch.compile settle

# Baseline: 2048 seq len
# ./train.sh baseline-2048 ITERATIONS=500 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=9999 WANDB_MODE=disabled

# Draft: 2048 real tokens → 4096 interleaved (2x baseline)
# ./train.sh draft-2048r-4096i ITERATIONS=500 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=9999 WANDB_MODE=disabled DRAFT_ENABLED=1 DRAFT_TRAIN_SEQ_LEN=2048 DRAFT_ALPHA_END=0.3

# Draft: 1024 real tokens → 2048 interleaved (0.5x baseline)
./train.sh draft-1024r-2048i ITERATIONS=500 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=9999 WANDB_MODE=disabled DRAFT_ENABLED=1 DRAFT_TRAIN_SEQ_LEN=1024 DRAFT_ALPHA_END=0.3
