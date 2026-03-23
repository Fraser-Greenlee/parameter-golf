# Pythia-70M Weight Engineering

## Goal

Build a library of scripts that can construct the weights of Pythia-70M (or a model with equivalent behavior) entirely from code -- no pre-trained weights loaded. Each attention head and FFN neuron should be engineered from first principles based on the circuits we've discovered through empirical analysis.

This serves as a proof-of-concept for the Parameter Golf competition: any structure we can express as code costs zero artifact bytes.

## Architecture Reference

Pythia-70M = GPT-NeoX: 6 layers, 8 heads, d_model=512, head_dim=64, FFN=2048, vocab=50304

- RoPE: `rotary_pct=0.25` -> 16 rotary dims, 48 content dims
- RoPE layout: **half-half** (dims 0..7 = real, 8..15 = imaginary), NOT interleaved
- Parallel residual: attention + FFN run simultaneously per layer
- Untied embeddings: separate embed_in and embed_out
- QKV fused: `[1536, 512]` interleaved `[Q0,K0,V0, Q1,K1,V1, ...]`
- Output proj: `[512, 512]`, head h = columns `[h*64:(h+1)*64]`

## What We've Done

### Phase 1: Empirical Analysis (complete)

Ran 30 diverse text samples through the trained Pythia-70M and collected:

| Output | Files | Description |
|--------|-------|-------------|
| `heads/L*_H*.txt` | 48 files | Per-head attention patterns with top-K attended tokens per position |
| `ffn/L*_neurons.txt` | 6 files | Per-neuron firing rates, trigger tokens, output effects |
| `circuits/L*_H*_{ov,qk}.txt` | 96 files | Weight-space QK/OV circuits projected to token space |
| `summary/*.md` | 3 files | Classification of all heads, FFN analysis, implications |

Key findings:
- **L2_H1**: Pure previous-token head (98% prev-token rate)
- **L3_H0, L3_H6, L4_H1, L4_H2, L4_H4**: Induction heads (appear as BOS-sinks on normal text, switch to textbook induction on repetitive sequences)
- **All L5 heads**: Copy heads (OV identity 0.70-0.96, entropy 0.17-0.36)
- **L1_H0, L1_H5, L2_H2, L2_H6**: Suppression heads (negative OV identity)
- Entropy decreases sharply through layers: 2.16 (L0) -> 0.24 (L5)
- FFN: zero dead neurons, activation magnitudes increase 7x from L0 to L5

### Phase 2: Weight Engineering (in progress)

`engineer_weights.py` constructs circuits from first principles and verifies them.

| Circuit | Status | Method | Result |
|---------|--------|--------|--------|
| Previous-token head | **DONE** | RoPE bias pre-rotation (half-half layout) | 100% prev-token (matches trained 98%) |
| Copy heads | **DONE** | Identity QK + identity OV with exponential SV decay | 97% self-attention, strong copying |
| Suppression heads | **DONE** | Low-scale QK (broad attention) + negative OV identity | Working, high entropy |
| Induction heads | **DONE** | K-composition + XSA (exclusive self-attention) | 100% induction match on repetitive text, BOS-sink fallback on non-repetitive |
| Content heads | **DONE** | Moderate identity QK, weak OV | Default filler for unassigned heads |
| FFN layers | **NOT STARTED** | -- | Using random init |
| Embeddings | **NOT STARTED** | -- | Using random init |

## Key Breakthrough: XSA (Exclusive Self-Attention)

The induction head engineering was stuck at a 50/50 self vs induction-match tie. Both Q and K project from the prev-token head's output subspace, but the original embedding also lives there, so self-attention score (Q=K exactly) always ties with the match score.

**XSA solves this completely.** By masking the attention diagonal (`attn[q,q] = -inf` before softmax) for induction heads, self-attention is eliminated and the induction match wins with 100%:

```
on@10  -> on@3  (1.00)  -- predecessor "sat" matches, attends to earlier occurrence
the@11 -> the@4 (1.00)  -- predecessor "on" matches
The@14 -> The@7 (1.00)  -- predecessor "." matches
sat@9  -> BOS   (fallback) -- "dog" never appeared as predecessor, falls back to BOS
```

This exactly reproduces trained Pythia-70M behavior (induction on matches, BOS-sink on non-matches).

XSA is natural for heads that never want self-attention:
- **Induction heads**: always want a *different* position where the predecessor matched
- **Previous-token heads**: always want position i-1, never self

Copy heads and content heads keep standard attention (they benefit from self-attention).

**Cost for Parameter Golf**: ~2 lines of code (boolean flag + diagonal mask), zero artifact bytes.

## What's Left

### FFN engineering

The FFN layers in Pythia-70M learn:
- L0: Basic token detectors (comma, period, "in", "is", numbers, plural nouns)
- L3: Code structure, sentence structure
- L5: Sentence boundary detection, final logit shaping (activations up to 29x)

Possible approaches:
- **Token detector neurons**: For the top-K most common tokens, construct a neuron whose input weight is the token's embedding direction and whose output weight boosts the appropriate next-token logits.
- **Bigram neurons**: For the top-K bigrams, construct neurons that detect token A and boost token B.
- **May not be worth engineering**: FFN learns basic detectors within a few training steps anyway. The attention circuits are higher leverage.

### Embedding engineering

Already addressed separately via bigram SVD embeddings in the main experiment pipeline (`train_gpt_mlx_sweep.py`). Could integrate here for completeness.

## File Inventory

```
pythia_70m_analysis/
  PROGRESS.md                    # This file
  text_samples.py                # 30 diverse text samples for testing
  collect_attention_data.py      # Empirical attention pattern collection
  collect_ffn_data.py            # FFN neuron activation profiling
  collect_ov_qk_circuits.py      # Weight-space QK/OV circuit analysis
  engineer_weights.py            # Weight engineering + verification
  heads/L{0-5}_H{0-7}.txt       # 48 attention pattern dumps
  ffn/L{0-5}_neurons.txt         # 6 FFN neuron profile dumps
  circuits/L{0-5}_H{0-7}_{ov,qk}.txt  # 96 circuit analysis dumps
  summary/head_classification.md # Per-head type classification
  summary/ffn_analysis.md        # FFN layer analysis
  summary/implications_for_parameter_golf.md  # Actionable init changes
```
