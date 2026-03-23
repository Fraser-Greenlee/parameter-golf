# Pythia-70M Weight Engineering

## Goal

Recreate Pythia-70M's behaviour by keeping the **input embedding matrix** (embed_in) and deriving all other weights (attention, FFN, layer norms, embed_out) from code that reads the embedding structure.

The embeddings encode token features (semantic clusters, syntactic categories, frequency) that can be leveraged to construct content-aware attention heads, copy circuits, and FFN neurons. Code that inspects the embedding to classify tokens and build weight matrices costs zero artifact bytes in the Parameter Golf competition.

**Long-term**: apply the same approach to the competition model (1024 vocab, 512 dim) where the embedding is only 0.52 MB at int8 -- leaving 15.5 MB of the 16 MB budget for trained corrections on top of engineered weights.

## Architecture

Pythia-70M = GPT-NeoX: 6 layers, 8 heads, d_model=512, head_dim=64, FFN=2048, vocab=50304. RoPE `rotary_pct=0.25` (16 rotary dims, **half-half** layout). Parallel residual. Untied embeddings.

## What We Know

### Attention Head Types (from empirical analysis on FineWeb)

| Type | Heads | Behaviour | Engineering Status |
|------|-------|-----------|-------------------|
| Previous-token | L2_H1 | 98% attends to position i-1 | **DONE** -- RoPE bias pre-rotation + XSA |
| Induction | L3_H0/H6, L4_H1/H2/H4 | Pattern-match on repeated sequences, BOS-sink otherwise | **DONE** -- K-composition + XSA |
| Copy | All L5 | Sharp attention + identity OV (copies attended token to output) | Structural QK done, needs embedding-aware OV |
| Suppression | L1_H0/H5, L2_H2/H6 | Broad attention + negative OV | Structural done |
| Content | 30 heads in L0-L4 | Content-dependent routing (e.g. L0_H7 function-word detector) | **Needs embedding-derived QK** |

### Key Discoveries

**XSA (Exclusive Self-Attention)**: Masking `attn[q,q] = -inf` for induction/prev-token heads eliminates the self-attention tie. Gives 100% induction on repetitive text, BOS-sink fallback otherwise. ~2 lines of code.

**OV alignment matters more than QK**: Using the correct V/O subspace reduced copy head loss delta from +1.51 to +0.049 (31x). The OV circuit dominates.

**L0_H7 is a function-word detector**: Articles/conjunctions/punctuation (`the` 99%, `a` 96%, `,` 78%) → self-attend. Content words/subwords → attend to previous token. The QK circuit separates closed-class from open-class tokens using subtle embedding geometry (self-QK scores differ by only ±0.01).

**Embedding structure**: NMF with 30 factors explains 39% of embedding variance. Factors correspond to semantic clusters (food, physics, countries, emotions, etc.). Surface features (POS, character composition) explain only 10%. The remaining 90% is distributional structure from training.

**Embedding quantization**: 8-bit preserves model quality (+0.06 nats). 4-bit is catastrophic (+18 nats). The model is extremely sensitive to embedding precision.

### Per-Head Isolated Evaluation

Each head replaced individually in trained model, loss measured on 20 FineWeb docs (baseline: 4.086 nats). Using trained V/O + engineered QK:

| Circuit | Mean Delta | Mean KL | Mean Activity |
|---------|------------|---------|---------------|
| Prev-token | +0.008 | 3.4 | 0.83 |
| Induction | +0.003 | 20.8 | 0.86 |
| Copy | +0.049 | 20.1 | 0.96 |
| Suppress | +0.024 | 1.8 | 0.53 |
| Content | +0.061 | 2.2 | 0.58 |

Worst heads: L0_H0 (+0.24), L0_H7 (+0.24), L0_H2 (+0.18) -- all early-layer content heads where the trained QK uses embedding-derived semantic features our generic identity QK misses.

## Current Focus: Embedding-Derived Weight Construction

The input embedding matrix E [50304, 512] encodes:
- **Token similarity** -- cosine distance ≈ semantic relatedness
- **Syntactic categories** -- function words vs content words cluster differently
- **Semantic domains** -- NMF reveals food, science, legal, etc. clusters
- **Subword structure** -- fragments vs complete words occupy different subspaces

We can leverage E to construct:

1. **Content-aware QK circuits**: Instead of generic identity QK, project Q and K onto embedding directions that separate relevant token categories (e.g. function words for L0_H7). The embedding tells us which tokens ARE function words.

2. **Embedding-space OV circuits**: Set W_OV to operate in the embedding's natural basis. For copy heads: OV ≈ E_out @ E_in.T (copy in token space). For suppression: negative of that.

3. **FFN as embedding-based bigram predictor**: Each neuron detects an embedding direction (token category) and outputs the appropriate next-token distribution.

4. **embed_out from embed_in**: Derive the output embedding as a function of the input embedding (e.g. shifted by the "next-token" direction in embedding space).

## Files

| Script | Purpose |
|--------|---------|
| `engineer_weights.py` | Circuit construction + verification |
| `test_heads_isolated.py` | Per-head isolated evaluation on FineWeb |
| `inspect_prev_token_head.py` | L2_H1 deep dive |
| `inspect_L0_H7.py` | Function-word detector deep dive |
| `inspect_embeddings.py` | Embedding structure analysis |
| `factorize_embeddings.py` | SVD + NMF factorization |
| `nmf_sweep.py` | NMF factor count vs R² (GPU-ready) |
| `collect_attention_data.py` | Empirical attention patterns |
| `collect_ffn_data.py` | FFN neuron profiling |
| `collect_ov_qk_circuits.py` | Weight-space circuit analysis |

Output data in `heads/`, `ffn/`, `circuits/`, `summary/`.
