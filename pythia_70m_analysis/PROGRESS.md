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
| Induction heads | **PARTIAL** | K-composition via shared prev-token subspace | 50/50 self vs matching position |
| Content heads | **DONE** | Moderate identity QK, weak OV | Default filler for unassigned heads |
| FFN layers | **NOT STARTED** | -- | Using random init |
| Embeddings | **NOT STARTED** | -- | Using random init |

## What's Left

### Induction heads (hardest problem)

Current issue: both Q and K project from the prev-token head's output subspace in the residual stream, but the original embedding also lives there. This means self-attention score (where Q=K exactly) ties with the induction match score. The trained model resolves this through learned selectivity that pure weight construction can't easily replicate.

Possible approaches:
1. **Dedicated communication channel**: Reserve a subspace of the residual stream exclusively for inter-layer signaling. Zero out the embedding's contribution to this subspace. The prev-token head writes predecessor content there; the induction head's K reads from there.
2. **Asymmetric Q/K projections**: Make Q read from the embedding space and K read from the prev-token OV space, with a learned alignment matrix between them.
3. **Positional tie-breaking**: Add a mild RoPE bias to induction heads that prefers earlier positions, breaking the 50/50 tie in favor of the first occurrence.
4. **Accept 50/50 as init**: The structural matching is correct even at 50/50. Training can sharpen it within a few steps. This may be good enough for Parameter Golf where we only need an init advantage.

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
