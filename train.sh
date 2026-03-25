#!/bin/bash
set -euo pipefail

# Usage: ./train.sh <name> [ENV=VAL ...]
# Examples:
#   ./train.sh baseline
#   ./train.sh draft-2048 DRAFT_ENABLED=1 DRAFT_TRAIN_SEQ_LEN=2048 DRAFT_ALPHA_END=0.3
#   ./train.sh time-test ITERATIONS=500 VAL_LOSS_EVERY=0 DRAFT_ENABLED=1

if [ $# -lt 1 ]; then
    echo "Usage: ./train.sh <name> [ENV=VAL ...]"
    exit 1
fi

NAME="$1"; shift

LOGDIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$LOGDIR"

TIMESTAMP="$(date +%H%M%S)"
LOGFILE="$LOGDIR/${TIMESTAMP}_${NAME}.log"

echo "Logging to: $LOGFILE"
echo "Config: $*"

# Store config in the log header
{
    echo "# run: $NAME"
    echo "# date: $(date -Iseconds)"
    echo "# env: $*"
    echo "---"
} > "$LOGFILE"

env "$@" torchrun --standalone --nproc_per_node=8 train_gpt_lookahead.py 2>&1 | tee -a "$LOGFILE"
