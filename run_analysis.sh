#!/bin/bash
# Train the SOTA model, then run comprehensive analysis.
# Usage: srun ... bash run_analysis.sh
set -euo pipefail

cd /home/fraser_convergence_ai/parameter-golf
source .venv/bin/activate

SOTA_DIR="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"

echo "=== Phase 1: Training SOTA model ==="
# Use the SOTA config from abaybektursun's submission
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 "${SOTA_DIR}/train_gpt.py"

echo ""
echo "=== Phase 2: Model analysis ==="
# Run analysis on the trained checkpoint (single GPU)
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 \
python analyze_model.py --checkpoint final_model.pt --output-dir analysis_results
