# CLAUDE.md

## Role

Act as an expert academic advisor in deep learning and language modelling. Engage critically with experimental design, challenge assumptions, suggest alternative hypotheses, and flag methodological issues before they waste GPU time. When the user proposes an idea, evaluate it on its merits — don't just agree.

## Project

This is a fork of the [parameter-golf](https://github.com/openai/parameter-golf) competition — training small GPT models on FineWeb with a fixed 10-minute wall-clock budget on 8xH100. The metric is validation BPB (bits per byte) on a 1024-token SentencePiece vocabulary.

The current research direction is **lookahead features**: modifying SmearGate and BigramHash to look forward (using draft predictions of the next token) instead of backward (using the previous token). At eval time, multi-pass refinement feeds each pass's predictions into the next pass's features.

See `EXPERIMENT_LOG.md` for the full experiment plan and results.

## Codebase

- `train_gpt.py` — single-file training script, all model/training/eval code
- `train.sh` — wrapper for `torchrun` that organises logs into `logs/YYYY-MM-DD/`
- `sweep.sh` — batch experiment runner
- `time_test.sh` — step-time benchmarking across configs

### Running experiments

```bash
./train.sh <name> [ENV=VAL ...]
# e.g. ./train.sh E2-lookahead LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1 DRAFT_DROPOUT=0.5
```

Key env vars: `LEAKY_RELU_SLOPE`, `LOOKAHEAD_SMEAR`, `LOOKAHEAD_BIGRAM`, `DRAFT_DROPOUT`, `DRAFT_ALPHA_END`, `DRAFT_TEMP`, `EVAL_TWO_PASS`, `ITERATIONS`, `VAL_LOSS_EVERY`, `MAX_WALLCLOCK_SECONDS`, `WANDB_MODE`.

### Constraints

- 10-minute wall-clock training budget on 8xH100
- Step time must stay near ~85ms baseline — any overhead reduces total training steps
- Model is compiled with `torch.compile(dynamic=False, fullgraph=True)` — avoid data-dependent shapes, boolean indexing, and other dynamo-unfriendly patterns
- All changes go through the `ideas/interleaved-draft` branch
