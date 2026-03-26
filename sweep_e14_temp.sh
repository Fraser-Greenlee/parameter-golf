#!/bin/bash
set -euo pipefail

# E14: Temperature scaling sweep
# Uses train_gpt_lookahead.py with EVAL_TEMPERATURE env var
# All runs train identically — only eval differs

LOGDIR="logs/$(date +%Y-%m-%d)"
mkdir -p "$LOGDIR"

for T in 0.85 0.90 0.95 1.00 1.05 1.10; do
    NAME="E14-temp-${T}"
    LOGFILE="$LOGDIR/$(date +%H%M%S)_${NAME}.log"
    echo "=== Starting $NAME ==="
    {
        echo "# run: $NAME"
        echo "# date: $(date -Iseconds)"
        echo "# env: EVAL_TEMPERATURE=$T LOOKAHEAD_SMEAR=1 TWOPASS_TRAIN_FRAC=0.05 LEAKY_RELU_SLOPE=0.5 EVAL_PASSES=2"
        echo "---"
    } > "$LOGFILE"
    srun --partition=dev --nodes=1 --exclusive --gpus=8 --cpus-per-task=208 --mem=0 \
        --time=00:20:00 --job-name="$NAME" \
        bash -c "source /home/fraser_convergence_ai/parameter-golf/.venv/bin/activate && \
        cd /home/fraser_convergence_ai/parameter-golf && \
        EVAL_TEMPERATURE=$T LOOKAHEAD_SMEAR=1 TWOPASS_TRAIN_FRAC=0.05 LEAKY_RELU_SLOPE=0.5 EVAL_PASSES=2 \
        WANDB_MODE=disabled \
        torchrun --standalone --nproc_per_node=8 train_gpt_lookahead.py 2>&1" \
        | tee -a "$LOGFILE"
    echo "=== Finished $NAME ==="
done
