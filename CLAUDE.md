# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Parameter Golf** is an OpenAI challenge to train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 8xH100s**, scored by compression on FineWeb validation (bits per byte, lower is better). Inspired by NanoGPT Speedrunning but optimizing L(N) — lowest loss given fixed parameter count.

## Key Commands

### Data Download
```bash
# Download cached FineWeb with 1024-token vocabulary (full: 80 shards / ~8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024
# Smaller subset for local iteration
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training (CUDA — the real benchmark)
```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 train_gpt.py
# 8xH100 (leaderboard config)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
All hyperparameters are set via environment variables (see `Hyperparameters` class at top of `train_gpt.py`). Key env vars: `RUN_ID`, `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `VAL_LOSS_EVERY`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`.

### Training (MLX — local Mac Apple Silicon)
```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

## Architecture

The codebase is intentionally minimal — two self-contained training scripts and a data pipeline:

- **`train_gpt.py`** (~1200 lines) — PyTorch/CUDA training script. Contains everything: model definition (GPT with GQA, RoPE, tied embeddings), Muon optimizer, int8 quantization, zlib compression, training loop, and validation. This is the script that matters for leaderboard runs. Launched via `torchrun` for distributed training.
- **`train_gpt_mlx.py`** (~1300 lines) — MLX port for local Mac development. Same model architecture, adapted for Apple Silicon. Single-process only.
- **`data/`** — Dataset download helpers. `cached_challenge_fineweb.py` downloads pre-tokenized FineWeb shards from HuggingFace. `download_hf_docs_and_tokenize.py` rebuilds tokenizers from source docs.

Both training scripts are capped at 1500 lines to stay readable for newcomers.

### Default Baseline Model
- 9 transformer layers, 512 model dim, 8 attention heads, 4 KV heads (GQA), 2x MLP expansion
- Vocab size 1024 (SentencePiece BPE), sequence length 1024, tied input/output embeddings
- Muon optimizer for matrix params, Adam for scalars/embeddings
- 524,288 tokens/step, ~10 minute wallclock cap, int8+zlib serialization

## Submission Structure

Submissions go in `records/track_10min_16mb/` (or `track_non_record_16mb/` for unlimited compute). Each submission folder contains:
- `train_gpt.py` — the modified training script (must run standalone from within the records folder)
- `README.md` — explanation of changes
- `submission.json` — metadata (author, val_bpb, etc.)
- `train.log` — training output log

New SOTA must beat existing record by >=0.005 nats at p < 0.01 significance.

## Scoring
- **Artifact size** = code bytes (`train_gpt.py`) + compressed model bytes (int8 + zlib) <= 16,000,000 bytes (decimal, not MiB)
- **Score** = `val_bpb` (bits per byte on FineWeb validation set, tokenizer-agnostic)
- Training capped at 10 minutes; evaluation also capped at 10 minutes (separate)
- No network access or training data during evaluation
