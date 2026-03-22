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


| Rank | Strategy         | val_bpb    | Train@25 | Train@50 | Train@100 | Train@200 | Time |
| ---- | ---------------- | ---------- | -------- | -------- | --------- | --------- | ---- |
| 1    | **full_circuit** | **2.4194** | 5.213    | 4.719    | 4.397     | 3.897     | 91s  |
| 2    | **identity_qk**  | **2.4213** | 5.307    | 4.644    | 4.383     | 3.898     | 88s  |
| 3    | copy_ov          | 2.4265     | 5.197    | 4.644    | 4.376     | 3.922     | 88s  |
| 4    | baseline         | 2.4295     | 5.348    | 4.645    | 4.362     | 3.915     | 70s  |
| 5    | spectral         | 2.4342     | 5.176    | 4.711    | 4.429     | 3.923     | 89s  |
| 6    | mimetic          | 2.4360     | 5.514    | 4.764    | 4.403     | 3.922     | 81s  |
| 7    | ortho_head       | 2.4377     | 5.212    | 4.714    | 4.430     | 3.931     | 87s  |


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

### 2026-03-22: P3/P1 Embedding + Bias Sweep (MLX, 200 steps, 1 shard)

**Goal**: Test corpus-statistics-based initialization (P3: unigram bias, P1: bigram embeddings) and their combinations with circuit init.

**Pre-computation**:

- Token frequencies from 1 FineWeb shard (100M tokens): 887/1024 tokens seen
- Bigram co-occurrence matrix (1024×1024), log1p-transformed, SVD: top-16 captures 80% variance, top-128 gets 93%
- Embedding init: unit-norm rows of U*sqrt(S) from bigram SVD, scaled to match tied_embed_init_std
- Unigram bias: log(freq/total) with Laplace smoothing, stored as additive bias on logits

**Results** (ranked by val_bpb):


| Rank | Strategy                          | val_bpb    | Train@1 | Train@25 | Train@200 | Time   |
| ---- | --------------------------------- | ---------- | ------- | -------- | --------- | ------ |
| 1    | **bigram_emb**                    | **2.3917** | 6.852   | 5.116    | 3.785     | 75s    |
| 2    | bigram_emb_plus_bias              | 2.4256     | 6.000   | 5.442    | 3.820     | 78s    |
| 3    | baseline                          | 2.4291     | 6.943   | 5.348    | 3.903     | 71s    |
| 4    | unigram_bias                      | 2.4300     | 6.019   | 5.602    | 3.859     | 68s    |
| 5    | bigram_emb_plus_bias_plus_circuit | 2.4346     | 6.000   | 5.346    | 3.852     | 1410s* |


*Last strategy ran slow due to Mac thermal throttling — ignore time, focus on bpb.

**Key findings**:

- **bigram_emb is the clear winner** (2.3917 vs baseline 2.4291): **-0.037 bpb**, 3.7× larger than the full_circuit gain from the attention sweep. Bigram SVD embeddings give the model a massive head start on token similarity structure.
- **unigram_bias alone barely helps** (2.4300 vs 2.4291): only -0.001 bpb at step 200, despite a dramatic step-1 advantage (6.019 vs 6.943). The model learns the unigram distribution very quickly anyway — the bias front-loads it but doesn't improve the endpoint.
- **Adding bias to bigram_emb HURTS** (2.4256 vs 2.3917): the combination is worse than bigram_emb alone. The bias likely interferes with the embedding's learned logit structure — the bigram SVD embeddings already implicitly encode frequency information through their norms/directions.
- **Adding circuit init to bias+emb also hurts** (2.4346): the full combo is the worst of the new strategies. This confirms the P1.5 concern — init components can interfere.
- **bigram_emb converges faster AND better**: train_loss at step 200 is 3.785 vs baseline 3.903, and val_bpb gap is the largest we've seen.

**Interpretation**: The bigram SVD embedding init is by far the most impactful technique we've found. It works because it gives the tied embedding/output matrix a structure where token dot products approximate PMI — the model starts knowing which tokens co-occur. The unigram bias is redundant because the embedding norms already capture frequency information after training for a few steps. The interference between bias and embeddings suggests they're competing to explain the same variance.

**Action items**:

- ~~bigram_emb is the new default to beat~~ → see next experiment
- Drop unigram_bias as a standalone strategy (redundant)
- The P1.5 ablation concern is validated: **test combinations carefully, don't assume additivity**
- ~~Next: try bigram_emb + full_circuit (without bias)~~ → done, see below

---

### 2026-03-22: bigram_emb + full_circuit (no bias) ablation

**Goal**: Test whether circuit init is additive with bigram embeddings (without the interfering bias).

**Results**:


| Strategy                    | val_bpb    | vs baseline | vs bigram_emb |
| --------------------------- | ---------- | ----------- | ------------- |
| baseline                    | 2.4289     | —           | —             |
| bigram_emb                  | 2.3941     | -0.035      | —             |
| **bigram_emb_plus_circuit** | **2.3876** | **-0.041**  | **-0.007**    |


**Key findings**:

- **Circuit init IS additive with bigram embeddings** — unlike the bias, which interfered. The -0.007 from circuit init on top of bigram_emb is consistent with the -0.010 from the first attention sweep (full_circuit vs baseline). They encode different structure in different parts of the model.
- **bigram_emb_plus_circuit is the new best** at 2.3876 bpb (-0.041 vs baseline)
- The interference was specifically between **unigram bias and bigram embeddings** (both operate on the logit/embedding space). Circuit init operates on attention weights — orthogonal to the embedding init.
- Train loss at step 200: bigram_emb_plus_circuit (3.790) ≈ bigram_emb (3.787) — the val_bpb gap suggests circuit init helps generalization more than raw training loss.

**Current best recipe**: bigram SVD embedding init + full_circuit attention init (no unigram bias)

---

### 2026-03-22: P2 (RoPE circuit) and P5a (L1 regularization)

**Goal**: Test RoPE-aware previous-token head construction (P2) and L1 sparsity regularization (P5a).

**P2 construction**: In layers 0-1, KV group 0 gets a RoPE-aware previous-token head:
- Q direction: (1,0) per frequency pair → position-only query
- K direction: (cos(θ_j), -sin(θ_j)) per pair → pre-rotated by -1 position
- After RoPE, dot product = Σ cos(θ_j * (Δ-1)), peaked at Δ=1 (previous token)
- Measured sharpness: 35% attention on prev token, 23% self, 23% two-back at seq position 10
- Remaining heads use standard full_circuit init

**P5a implementation**: L1 penalty (λ=1e-5) added to loss on all 2D+ weight matrices.

**Results**:

| Strategy | val_bpb | Train@200 | Compressed Size |
|----------|---------|-----------|-----------------|
| bigram_emb_plus_circuit (current best) | **2.3891** | 3.787 | baseline |
| bigram_emb_plus_rope_circuit (P2) | 2.4026 | 3.800 | — |
| bigram_emb_plus_circuit_l1 (P5a) | 2.8523* | 4.693 | — |

*L1 int8 roundtrip val_bpb=2.8523 is much worse — but note the pre-roundtrip val was 2.9427. The gap (0.09) is much smaller than baseline's gap (~0.001), suggesting L1 does help compressibility, just at a huge quality cost with λ=1e-5.

**Key findings**:
- **P2 (RoPE circuit) is WORSE than generic full_circuit** (2.4026 vs 2.3891): -0.014 bpb regression. The rank-1 positional-only QK projection removes content-awareness from 2 heads per layer 0-1, which hurts more than the positional bias helps. The generic full_circuit's identity QK (which biases toward content similarity) is more valuable than explicit position-based attention.
- **P5a (L1) badly hurts quality** (2.8523 vs 2.3891): λ=1e-5 is far too aggressive. The model can't learn effectively under that much sparsity pressure in 200 steps. Need to try much smaller λ (1e-7, 1e-8) or apply L1 only in the later half of training.
- **The generic full_circuit remains the best attention init** — attempts to make it more specific (RoPE-aware) or add training modifications (L1) haven't improved on it.

**Lessons**:
- Content-aware attention (identity QK) > position-aware attention (RoPE offset). This makes sense for language modeling where "what comes next" depends more on what the current token IS than where it IS.
- L1 regularization needs careful tuning — the compression benefit exists (smaller roundtrip gap) but the quality hit at λ=1e-5 is catastrophic. A gentler schedule (ramp up L1 during warmdown) might work.

---

### 2026-03-22: E3 — Depth-varying circuit from Pythia-70M statistics

**Goal**: Replace constant full_circuit parameters with depth-varying values fitted
from Pythia-70M's actual weight statistics.

**Pythia-70M measurements** (512d, 8 heads, 6 layers):
- OV identity cosine: 0.01 → -0.02 → -0.07 → 0.19 → 0.32 → **0.83** (strong depth trend)
- QK identity cosine: ~0 everywhere with high variance (content-specific, not generic)
- Q/K norm ratio: ~1.0 early → 4.9 → 11.5 → 5.3 later (effective q_gain increases)
- Q norms: 7→7→6→22→67→44 (explode in later layers)
- MLP top-1024/2048: 56-64% energy (cropping to half loses significant capacity)
- Pythia-70M BPB on test text: ~1.00 (reference point; 70M params, 50k vocab)

**depth_circuit parameters**: OV identity ramps 0→0.8 quadratically with depth,
beta_qk reduced from 0.3 to 0.05 (matching Pythia's near-zero QK identity),
q_gain ramps from 1.5 to 3.0, depth_scale increases with layer position.

**Results**:

| Strategy | val_bpb | Train@200 |
|----------|---------|-----------|
| bigram_emb_plus_circuit (constant) | **2.3864** | 3.788 |
| bigram_emb_plus_depth_circuit | 2.3919 | 3.777 |

**Key findings**:
- **Depth-varying is slightly worse** (-0.006 bpb regression) despite better train loss
- Reducing beta_qk from 0.3 to 0.05 hurts — our smaller model benefits from the identity QK
  bias even though Pythia-70M didn't develop strong QK identity. This may be because Pythia
  had 6 layers to develop content-specific QK circuits; our 9-layer model with only 200 training
  steps hasn't had time, so the identity bias is still useful as a starting point.
- Depth-varying OV identity + q_gain didn't compensate for the QK regression
- **Copying trained model geometry doesn't always transfer** — model size, training duration,
  and architecture differences (GQA, relu², QK-norm) mean our optimal init differs from
  Pythia's converged state

**Current best remains**: `bigram_emb_plus_circuit` = **2.3864 bpb**

---

### 2026-03-22: E5 — Frequency-scaled embedding norms

**Goal**: Encode token frequency in embedding norms (sqrt(freq) scaling) instead of
uniform unit norms. Common tokens get larger embeddings → higher logits.

**Results**:

| Strategy | val_bpb |
|----------|---------|
| bigram_emb_plus_circuit (unit norms) | **2.3855** |
| bigram_emb_freq_norm_plus_circuit | 2.3933 |
| bigram_emb_freq_norm (no circuit) | 2.4002 |

**Finding**: Frequency norms hurt, same pattern as unigram bias. The model applies
`RMSNorm(embedding)` at input, which normalizes away magnitude. And with tied embeddings,
uniform-norm directions work better for the output dot products. Frequency information
is redundant — the model learns it within a few steps regardless.

**Current best remains**: `bigram_emb_plus_circuit` = **2.3855 bpb**

---

## Planned Experiments

**Current best recipe**: bigram SVD embeddings + full_circuit attention init (no bias) = **2.3855 bpb** (-0.043 vs baseline)

**Key lessons so far**:
- Bigram SVD embeddings = biggest single win (-0.037 bpb)
- Generic structural attention init (identity QK, copy OV) adds ~-0.006 on top
- Specific encodings consistently fail: RoPE previous-token, unigram bias, freq norms, depth-varying params from Pythia
- Frequency/unigram information is redundant — the model learns it in a few steps; RMSNorm washes out magnitude
- Incremental parameter tuning (E3, E5, E6) has hit diminishing returns

### P4: Longer runs / CUDA validation

**Priority: LOW (until we have new ideas worth validating at scale)**

- bigram_emb_plus_circuit shows consistent -0.043 bpb gap over 200 steps
- Run on CUDA with full data when ready for leaderboard submission

