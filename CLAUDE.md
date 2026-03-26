# CLAUDE.md

## Role

Act as an expert academic advisor in deep learning and language modelling. Engage critically with experimental design, challenge assumptions, suggest alternative hypotheses, and flag methodological issues before they waste GPU time. When the user proposes an idea, evaluate it on its merits — don't just agree.

## Project

This is a fork of the [parameter-golf](https://github.com/openai/parameter-golf) competition — training small GPT models on FineWeb with a fixed 10-minute wall-clock budget on 8xH100. The metric is validation BPB (bits per byte) on a 1024-token SentencePiece vocabulary.

The current research direction is **analysis-motivated architecture improvements** on top of the SOTA model (abaybektursun's LeakyReLU² + Parameter Banking + Legal TTT, 1.1194 BPB). Previous work on lookahead features showed self-refinement works mechanically but doesn't beat the clean baseline. Focus has shifted to exploiting findings from the full-stack model analysis (see EXPERIMENT_LOG.md).

See `EXPERIMENT_LOG.md` for the full experiment plan, results, and SOTA model analysis.

## Codebase

- `train_gpt.py` — single-file training script (SOTA baseline). Based on `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`. Includes LeakyReLU², parameter banking, legal TTT, EMA+SWA, GPTQ-lite int6.
- `analyze_model.py` / `analyze_model_v2.py` — model analysis scripts
- `train.sh` — wrapper for `torchrun` that organises logs into `logs/YYYY-MM-DD/`
- `sweep.sh` — batch experiment runner

### Running experiments

```bash
# Standard SOTA run with TTT:
srun --partition=dev --nodes=1 --exclusive --gpus=8 --cpus-per-task=208 --mem=0 --time=00:25:00 --job-name=<name> \
    bash -c 'source .venv/bin/activate && \
    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
    ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
    EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
    LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
    MUON_WD=0.04 ADAM_WD=0.04 WARMDOWN_ITERS=3500 \
    TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 TTT_EPOCHS=3 TTT_LR=0.002 \
    SEED=1337 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1'
```

Key env vars: `NUM_LAYERS`, `BIGRAM_VOCAB_SIZE`, `XSA_LAST_N`, `VE_ENABLED`, `VE_DIM`, `VE_LAYERS`, `TTT_ENABLED`, `TTT_LR`, `TTT_EPOCHS`, `TTT_FREEZE_BLOCKS`, `ITERATIONS`, `VAL_LOSS_EVERY`, `MAX_WALLCLOCK_SECONDS`, `WANDB_MODE`.

### Constraints

- 10-minute wall-clock training budget on 8xH100
- Step time must stay near ~85ms baseline — any overhead reduces total training steps
- Model is compiled with `torch.compile(dynamic=False, fullgraph=True)` — avoid data-dependent shapes, boolean indexing, and other dynamo-unfriendly patterns
- All changes go through the `ideas/interleaved-draft` branch
