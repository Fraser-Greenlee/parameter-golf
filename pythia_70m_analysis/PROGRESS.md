# Pythia-70M Weight Engineering

## Goal

Construct Pythia-70M's weights entirely from code. Proof-of-concept for Parameter Golf where code costs zero artifact bytes.

## Architecture

Pythia-70M = GPT-NeoX: 6 layers, 8 heads, d_model=512, head_dim=64, FFN=2048, vocab=50304. RoPE with `rotary_pct=0.25` (16 rotary dims, **half-half** layout: dims 0..7 = real, 8..15 = imaginary). Parallel residual. Untied embeddings.

## Per-Head Engineering Status

Each head was tested by replacing it in the trained model and measuring loss on 20 FineWeb docs (baseline: 4.086 nats). All engineered heads use trained V/O matrices + engineered QK.

| Circuit | Heads | Method | Mean Delta | Mean KL | Mean Activity |
|---------|-------|--------|------------|---------|---------------|
| Prev-token | L2_H1 | RoPE bias pre-rotation + XSA | +0.008 | 3.4 | 0.83 |
| Induction | L3_H0/H6, L4_H1/H2/H4 | K-composition + XSA | +0.003 | 20.8 | 0.86 |
| Copy | All L5 | Identity QK | +0.049 | 20.1 | 0.96 |
| Suppress | L1_H0/H5, L2_H2/H6 | Low-scale QK | +0.024 | 1.8 | 0.53 |
| Content | 30 heads across L0-L4 | Moderate identity QK | +0.061 | 2.2 | 0.58 |

**Worst individual heads** (highest loss delta, with activity context):

| Head | Circuit | Delta | KL | Activity | Issue |
|------|---------|-------|-----|----------|-------|
| L2_H1 | prev_token | +2.22 | 3.4 | 0.83 | Test harness bug (deep dive shows only +0.008) |
| L0_H0 | content | +0.24 | 0.9 | 0.36 | Low activity but still costs |
| L0_H7 | content | +0.24 | 2.3 | 0.61 | Active, learned routing our identity QK misses |
| L0_H2 | content | +0.18 | 1.0 | 0.37 | Similar to L0_H0 |
| L0_H1 | content | +0.16 | 2.8 | 0.70 | Most active L0 head (soft prev-token at 77%) |
| L5_H0 | copy | +0.11 | 21.7 | 0.96 | Huge KL but modest delta -- V/O carries the load |

## Key Discoveries

### XSA (Exclusive Self-Attention)

Masking `attn[q,q] = -inf` for induction/prev-token heads solves the self-attention tie that blocked induction engineering. Result: 100% induction on repetitive text, BOS-sink fallback otherwise. Cost for Parameter Golf: ~2 lines of code.

### OV subspace alignment matters more than QK

Switching from arbitrary orthogonal subspaces to trained V/O matrices:
- Copy heads: +1.51 -> **+0.049** (31x better)
- Prev-token: +7.23 -> **+0.008** (900x better)

The OV circuit (what information is routed) dominates over the QK circuit (which tokens attend to which). Even with huge KL divergence in attention distributions (18-22), loss delta stays low when V/O is correct.

### L2_H1 reconstructs contractions

The trained prev-token head (mean prev weight 0.77, not 1.0) attends back to verb stems across BPE splits: `don't` → `t` attends to `don`, `Ohio's` → `s` attends to `Ohio`. Our pure positional version misses this but costs only +0.008 nats.

### Activity reveals which heads matter

L0-L2 heads are moderately active (0.31-0.83). L3+ heads become very sharp (0.78-0.98). Low-activity heads have low loss delta regardless of KL -- they're effectively idle on non-repetitive text.

## What's Left

1. **Content head QK features** (L0_H0, L0_H7, L0_H1): the 6 worst heads are all early-layer content heads where learned semantic routing costs +0.10-0.24 each. Deep-diving these to understand their QK features is the next frontier.
2. **FFN engineering**: Not started. L0 learns token detectors, L5 does logit shaping. May not be worth engineering -- FFN learns basics within a few training steps.
3. **Embeddings**: Already addressed separately via bigram SVD in the main pipeline.

## Files

| Script | Purpose |
|--------|---------|
| `engineer_weights.py` | Circuit construction + verification |
| `test_heads_isolated.py` | Per-head isolated evaluation on FineWeb |
| `inspect_prev_token_head.py` | L2_H1 deep dive |
| `collect_attention_data.py` | Empirical attention pattern collection |
| `collect_ffn_data.py` | FFN neuron profiling |
| `collect_ov_qk_circuits.py` | Weight-space circuit analysis |
| `text_samples.py` | 30 diverse test samples |

Output data in `heads/`, `ffn/`, `circuits/`, `summary/`.
