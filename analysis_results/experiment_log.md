# Experiment Log

This file tracks all experiments — completed results and planned future work.
See `WEIGHT_ENGINEERING.md` and `CIRCUIT_INIT.md` for research background.

## Research Direction

We are exploring **weight engineering**: replacing random/default initial weights with
code-generated weights that encode known linguistic and structural priors. The core idea
is that anything expressible as code costs zero artifact bytes (only the compressed model
counts toward 16MB). We study pre-trained models to identify universal weight structures,
then express those structures as parametric formulas in the training script.

**Important constraint**: whatever we produce must still fit within the contest's 16MB
artifact limit (code + int8+zlib compressed model). Weight subcloning from larger models
like GPT-2 is valid *only if* the resulting weights compress within budget after training.

---

## Completed Experiments

### 2026-03-22: Attention Init Strategy Sweep (MLX, 200 steps, 1 shard)

**Goal**: Test circuit-inspired attention initialization strategies derived from cross-model analysis of SmolLM2-135M, Pythia-70M, Qwen2.5-0.5B, and GPT-2.

**Setup**:
- Script: `train_gpt_mlx_sweep.py` (modified `train_gpt_mlx.py` with init injection)
- 200 training steps, batch=8192 tokens, 1 train shard (~100M tokens)
- Validation on ~20 batches of 524k tokens (subset of full val set)
- Model: 9 layers, 512 dim, 8 heads, 4 KV heads, vocab 1024

**Strategies tested**:
1. **baseline** — Default Kaiming uniform Q/K/V, zero-init output projection
2. **mimetic** — From Trockman & Kolter (ICML 2023): QK ~ alpha*noise + beta*I, OV ~ alpha*noise + beta*I, factored via SVD. alpha=beta=0.7 for QK, 0.4 for OV
3. **spectral** — Random orthogonal bases with exponential SV decay matching trained model profiles (qk_decay=0.06, ov_decay=0.04)
4. **identity_qk** — QK circuit biased toward identity (attend-to-self), beta=0.5 identity strength. V and O stay default
5. **ortho_head** — Each head gets an orthogonal subspace of residual stream via shared QR basis
6. **copy_ov** — OV circuit near identity with exponential SV decay (ov_decay=0.04). Q and K stay default
7. **full_circuit** — Combines: orthogonal subspaces + identity QK (beta=0.3) + copy OV + spectral decay + depth scaling + symmetry-breaking noise

**Results** (ranked by val_bpb, lower is better):

| Rank | Strategy | val_bpb | Train@25 | Train@50 | Train@100 | Train@200 | Time |
|------|----------|---------|----------|----------|-----------|-----------|------|
| 1 | **full_circuit** | **2.4194** | 5.213 | 4.719 | 4.397 | 3.897 | 91s |
| 2 | **identity_qk** | **2.4213** | 5.307 | 4.644 | 4.383 | 3.898 | 88s |
| 3 | copy_ov | 2.4265 | 5.197 | 4.644 | 4.376 | 3.922 | 88s |
| 4 | baseline | 2.4295 | 5.348 | 4.645 | 4.362 | 3.915 | 70s |
| 5 | spectral | 2.4342 | 5.176 | 4.711 | 4.429 | 3.923 | 89s |
| 6 | mimetic | 2.4360 | 5.514 | 4.764 | 4.403 | 3.922 | 81s |
| 7 | ortho_head | 2.4377 | 5.212 | 4.714 | 4.430 | 3.931 | 87s |

**Key findings**:
- **full_circuit wins** (2.4194 vs baseline 2.4295): -0.010 bpb combining identity QK + copy OV + orthogonal subspaces + spectral decay
- **identity_qk is the single most impactful component** (2.4213): just biasing QK toward identity captures most of the gain
- **copy_ov also helps** (2.4265): making OV circuits identity-like with spectral decay
- **mimetic init hurts** vs baseline: the noise+identity formulation from the ViT paper doesn't transfer well (possibly because RoPE + QK-norm already handle what mimetic was designed for)
- **ortho_head alone hurts**: forcing orthogonal subspaces without identity/copy structure is counterproductive
- **spectral alone is neutral**: matching SV decay profiles without structural bias adds nothing

**Interpretation**: The trained-model analysis showed QK circuits have positive identity cosine and copy heads dominate (31-47% across models). Directly encoding these biases at init (identity_qk, copy_ov) helps. The combination (full_circuit) is best. The spectral profile matching alone isn't enough — the *direction* (identity/copy) matters more than the *magnitude distribution* (SV decay).

**Caveats**:
- Only 200 steps on 1 shard — margins would likely grow over full 20k-step training
- Validation on subset (~10M tokens) rather than full val set
- All strategies converge to similar train_loss by step 200 — the val_bpb gap suggests init affects generalization, not just optimization speed

---

## Planned Experiments

### P1: Bigram-informed embedding init
**Priority: HIGH** — Encode actual FineWeb corpus statistics into weights, not just structural priors.
- Compute 1024×1024 bigram co-occurrence matrix from FineWeb train shards
- SVD → top-512 singular vectors → initialize tied embedding matrix
- The zero-layer model (embed → unembed) should already approximate bigram LM
- Combine with full_circuit attention init from the sweep above
- Expected impact: significant — this front-loads what the model learns first

### P2: RoPE-aware induction circuit construction
**Priority: HIGH** — Our identity_qk was generic. This is the specific formula.
- Layer 1: previous-token head via RoPE offset — W_Q and W_K project onto positional frequency dims with rotational offset for position i-1
- Layer 2: induction head via K-composition — keys read "what preceded me" from layer 1's OV output, queries read "who am I" from content embedding
- Uses 2 of 72 total heads — minimal capacity cost
- Need to account for QK-norm (RMSNorm on Q,K) and q_gain when computing target weights
- See WEIGHT_ENGINEERING.md "Hand-coding attention circuits" section for construction details

### P3: Log-unigram output bias
**Priority: MEDIUM** — Simplest possible change, zero parameters.
- Set output bias to `log(freq_i / total)` from FineWeb token frequencies
- Current architecture has no output bias (tied embeddings) — would need to add one or encode in embedding norms
- Lets model skip learning the unigram distribution entirely

### P4: Longer runs to validate init gap
**Priority: MEDIUM** — Confirm the 0.010 bpb gap from the sweep holds at scale.
- Run full_circuit vs baseline for 1000-2000 steps on MLX
- If gap closes: init mainly helps compression, not convergence
- If gap grows: real training dynamics advantage worth pursuing on CUDA

### P5: Compression-aware weight structure
**Priority: MEDIUM** — Exploit the compression angle more deliberately.
- full_circuit already compressed 13% smaller (4.7MB vs 5.4MB)
- Test: encourage weight clustering/sparsity during training via regularization
- Test: ALBERT-style weight sharing (share attention across layers) — fewer unique params, better compression
- Goal: fit a wider/deeper model in the same 16MB budget

### P6: Weight subcloning from GPT-2
**Priority: LOW (needs feasibility check)** — Claims 4× faster convergence.
- Slice GPT-2 Small (768→512 dim, 12→9 layers) via neuron importance ranking
- Remap embeddings from GPT-2's 50257-vocab to our 1024-vocab via sub-word averaging
- Must verify the resulting model still compresses within 16MB after training
- Risk: vocabulary mismatch may negate benefits; subcloned weights may not compress well
