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

### Phase 2: Weight Engineering

`engineer_weights.py` constructs circuits from first principles and verifies them.

| Circuit | Status | Method | Result |
|---------|--------|--------|--------|
| Previous-token head | **DONE** | RoPE bias pre-rotation (half-half layout) | 100% prev-token, +0.008 nats isolated loss |
| Induction heads | **DONE** | K-composition + XSA (exclusive self-attention) | 100% induction on repetitive, BOS-sink fallback |
| Copy heads | **DONE** | Identity QK + trained V/O | +0.049 mean loss with trained V/O |
| Suppression heads | **DONE** | Low-scale QK + trained V/O | +0.019 mean loss |
| Content heads | **DONE** | Moderate identity QK + trained V/O | +0.048 mean loss |
| FFN layers | **NOT STARTED** | -- | Using random init |
| Embeddings | **NOT STARTED** | -- | Using random init |

### Phase 3: Per-Head Isolated Evaluation

`test_heads_isolated.py` replaces each head individually in the trained model (keeping all other heads trained) and measures loss impact. Uses 20 real FineWeb validation documents.

**Key finding: OV subspace alignment is critical.** With arbitrary orthogonal subspaces, copy heads cost +1.51 nats each. With trained V/O matrices, they cost only +0.049 nats -- a **31x improvement**.

| Circuit | Arbitrary V/O | Trained V/O | Improvement |
|---------|--------------|-------------|-------------|
| Copy (L5, mean) | +1.51 | **+0.049** | 31x |
| Prev-token (L2_H1) | +7.23 | **+0.008** | 900x |
| Induction (mean) | +0.028 | **+0.003** | 9x |
| Suppress (mean) | +0.024 | **+0.019** | modest |
| Content (mean) | +0.061 | **+0.048** | modest |

The remaining loss gap is almost entirely from content heads and copy heads having engineered QK patterns (identity-like) instead of the trained model's learned content-specific QK circuits.

### Phase 4: Deep Dive into Individual Heads

`inspect_prev_token_head.py` analyzes L2_H1 on 40 FineWeb docs (10,482 positions).

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

## L2_H1 Deep Dive: Contraction Reconstruction

The trained prev-token head is NOT purely positional. On FineWeb data:

- **Mean prev-token weight: 0.77** (not ~1.0). 23% of attention goes elsewhere.
- **Entropy: 0.92** vs 0.02 for our engineered version.
- **83 of 10,482 positions (0.8%)** attend to something other than prev-token.

The non-prev-token behavior has a clear semantic pattern: **contraction and possessive reconstruction**. When BPE splits `don't` into `don`, `'`, `t`:
- Token `t` attends back to `don` (0.35 weight), not to `'` (the literal prev token)
- Same for: `can't`→`can`, `won't`→`won`, `didn't`→`didn`
- Possessives: `Ohio's`→`s` attends to `Ohio` (0.58), not to `'`

**Tokens with weakest prev-token signal**: `t` (0.35), `,"` (0.38), `s` (0.43) -- contraction/possessive suffixes. **Strongest**: subword continuations like `N` (0.98), `00` (0.97) -- mid-word tokens that always follow their prefix.

Despite this rich content behavior, **our pure positional version costs only +0.008 nats** -- the contraction feature is linguistically interesting but not critical for loss.

## What's Left

### Engineering real content features

The per-head evaluation shows that with trained V/O, most heads have small loss deltas. The remaining gap is in the QK circuits -- which tokens attend to which. The content heads (L0-L4) use learned semantic features we haven't replicated:
- Which tokens are semantically similar (for identity QK heads)
- Which positions carry relevant context (for distant-content heads)
- How to modulate attention sharpness by content

Next steps for content head engineering:
- Deep-dive the highest-loss content heads (L0_H0, L0_H7, L1_H2) on FineWeb
- Identify the semantic features their QK circuits detect
- Attempt to reproduce with engineered Q/K weight constructions

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
  compare_engineered_attention.py # Full model comparison (all heads replaced)
  test_heads_isolated.py         # Per-head isolated evaluation
  inspect_prev_token_head.py     # Deep dive into L2_H1
  heads/L{0-5}_H{0-7}.txt       # 48 attention pattern dumps
  ffn/L{0-5}_neurons.txt         # 6 FFN neuron profile dumps
  circuits/L{0-5}_H{0-7}_{ov,qk}.txt  # 96 circuit analysis dumps
  summary/head_classification.md # Per-head type classification
  summary/ffn_analysis.md        # FFN layer analysis
  summary/implications_for_parameter_golf.md  # Actionable init changes
  summary/per_head_comparison.md # Isolated per-head loss deltas
  summary/prev_token_head_deep_dive.md  # L2_H1 contraction reconstruction
```
