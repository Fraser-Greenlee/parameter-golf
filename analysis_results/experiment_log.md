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

Priority ordering revised based on review feedback. Key strategic insight: pursue
init engineering (P3/P1/P2) and compression engineering (P5a) in parallel — these
are complementary axes with different ceilings.

### P3: Log-unigram output bias
**Priority: HIGH (do first)** — Lowest-effort, highest-certainty win.
- Pre-compute token frequencies from FineWeb train shards
- Set output bias to `log(freq_i / total)` — 1024 floats = 4KB, negligible
- Current architecture has no output bias (tied embeddings) — add a bias vector
- Every step the model spends learning "the" > "xylophone" is a step wasted
- The unigram distribution is the single largest source of predictable cross-entropy

### P1: Bigram-informed embedding init
**Priority: HIGH** — Encode actual FineWeb corpus statistics into weights.
- Compute 1024×1024 bigram co-occurrence matrix from FineWeb train shards
- SVD → top-512 singular vectors → initialize tied embedding matrix
- **Important**: with tied embeddings, this sets both input and output projection.
  SVD of co-occurrence gives vectors where dot products approximate PMI — reasonable
  embedding space but may have weird norm properties interacting with RMSNorm.
  Normalize each row to unit norm after SVD, store frequency info separately via P3 bias.
  Embeddings encode *similarity structure*, bias encodes *frequency structure* — cleaner separation.
- Combine with full_circuit attention init from the sweep

### P2: RoPE-aware induction circuit construction
**Priority: HIGH** — Our identity_qk was generic. This is the specific formula.
- Layer 1: previous-token head via RoPE offset — pick the highest-frequency RoPE pair (θ_i),
  set W_Q and W_K to project onto those 2 dimensions. RoPE rotation naturally creates offset-1 preference.
- Layer 2: induction head via K-composition — keys read "what preceded me" from layer 1's OV output,
  queries read "who am I" from content embedding
- Uses 2 of 72 total heads — minimal capacity cost
- **QK-norm complication**: with RMSNorm on Q and K, effective attention logit is
  `(Q/||Q|| . K/||K||) * q_gain * sqrt(d_head)`. Only *direction* matters, not magnitude.
  For the prev-token head, projecting onto just 2 of 64 dims means q_gain needs to be large
  enough that this head's attention is sharp. Prototype on paper first.
- See WEIGHT_ENGINEERING.md "Hand-coding attention circuits" section

### P1.5: Combined init ablation
**Priority: HIGH (run after P3+P1+P2 are individually implemented)** — Verify components are additive.
- Stack P3 + P1 + P2 + full_circuit incrementally
- The mimetic result already showed init strategies can interfere unexpectedly
- Test each combination: baseline, +P3, +P3+P1, +P3+P1+P2, +P3+P1+P2+full_circuit
- If any combination is worse than its subset, investigate why

### P5a: Compression via L1 regularization / gradual magnitude pruning
**Priority: HIGH (run in parallel with init work)** — Potentially larger gains than init.
- full_circuit already compressed 13% smaller (4.7MB vs 5.4MB int8+zlib)
- Add L1 regularization or gradual magnitude pruning during training — easy to implement
- If we achieve 2-3× better compression, we can fit **25-30M params** in 16MB instead of 17M
- That model size increase could be worth far more than any init trick
- Test: measure compressed size vs val_bpb tradeoff at different sparsity levels

### P7: MLP initialization
**Priority: MEDIUM** — Sweep was all attention-focused, but MLPs encode n-gram statistics.
- Geva et al. showed MLP layers act as key-value memories
- relu² activation means MLP neurons are very sparse by default
- Initialize a few MLP neurons in layer 1 to detect high-frequency bigram patterns:
  input weights match bigram embedding directions (from P1's SVD)
- Lower priority because MLP structure is less well-characterized than attention circuits

### P4: Longer runs to validate init gap
**Priority: LOW** — Better to spend compute on P1/P2/P3 which have higher ceilings.
- If bigram embeddings + full_circuit show a clear gap at 200 steps, that's validation enough
- Don't need a separate 1000-step confirmation of full_circuit alone
- Revisit if P1+P2+P3 results are ambiguous

### P5b: ALBERT-style weight sharing
**Priority: LOW** — More invasive architecture change, test after P5a.
- Share attention weights across all 9 layers — store 1 copy instead of 9
- Group FFN into 3 groups of 3 layers — store 3 copies instead of 9
- Freed budget enables wider d_model (768?) or more layers
- Risk: may hurt final quality even if compression improves

### P6: Weight subcloning from GPT-2
**Priority: LOW** — Vocabulary mismatch is a deeper problem than just remapping.
- GPT-2's weights are optimized for 50k-token distribution; after slicing to 1024 tokens,
  internal representations are organized around distinctions that don't exist in our vocabulary
- 1024-token BPE has very different granularity (subword fragments, common short words)
  than GPT-2's token space
- The 4× convergence claim is for *matched vocabulary* subcloning
- Revisit only if P1-P5a don't pan out
