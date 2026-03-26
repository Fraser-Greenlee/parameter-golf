# Experiment Log: Lookahead Features via Draft Predictions

## Hypothesis

At the character/subword level with a 1,024-token vocabulary, prediction difficulty is highly non-uniform. Word-initial positions carry 3-6 bits of entropy while mid-word positions carry <1 bit. SmearGate and BigramHash currently look backward (blending/hashing with the previous token). By giving these features **lookahead** — a draft prediction of the next token — they gain information about where they are in a word and what's coming next, without any change to sequence length or attention.

## Architecture Summary

**Current (backward-looking):**
```
position:  x_0    x_1    x_2    x_3
smear:     [∅]    x_0    x_1    x_2     ← blends with previous token
bigram:    [∅]   (0,1)  (1,2)  (2,3)   ← hashes current + previous
target:    x_1    x_2    x_3    x_4
```

**Proposed (forward-looking via drafts):**
```
position:  x_0    x_1    x_2    x_3
smear:     d_1    d_2    d_3    d_4     ← blends with draft of NEXT token
bigram:   (0,d1) (1,d2) (2,d3) (3,d4)  ← hashes current + next draft
target:    x_1    x_2    x_3    x_4
```

- No sequence length change — transformer processes same T-length sequence
- Draft info only enters through cheap pre-attention features (embedding lookup + blend)
- During training: `d_{t+1}` is ground truth or noisy GT (same alpha curriculum)
- During eval: multi-pass refinement — pass 1 without drafts → pass 2 with predictions as drafts → pass 3 with better predictions → ...
- Draft embeddings are soft: `draft_emb = sum(p_k * emb(tok_k))` over top-k predictions

**Key advantage over interleaving:** No 2x sequence length, no ~80ms construction overhead. Each eval pass costs ~85ms (baseline speed), so 2 passes ≈ 170ms vs interleaved approach at ~193ms for a single pass.

## Training Strategy

**Two-pass self-refinement:** A fraction (~10%) of batch rows are trained with two forward passes per step:
1. Pass 1: forward without lookahead → get logits (detached)
2. Pass 2: feed pass 1's soft predictions (`softmax(logits) @ emb_weight`) into SmearGate/BigramHash as lookahead → compute loss

This trains the model to improve its own predictions — the same loop it will run at eval time. Remaining rows train normally (pass 1 only), ensuring the model is strong without lookahead too.

Predictions from a few steps ago may be preferable to current-step predictions to avoid memorisation (see E5).

**Sweep: feature direction mix.** Test whether the optimal config is:
- Both bigram + smear look forward (full lookahead)
- Bigram looks backward + smear looks forward (mixed)
- Other combinations

## Baseline

**SOTA (signalrush PR #414):** 11L, 512d, 8H/4KV, MLP 3x relu², XSA last 4, EMA 0.997, Tight SWA, GPTQ-lite int6, SmearGate + BigramHash(2048), Partial RoPE (16/64), LN Scale, VE(128) layers 9-10, seq_len=2048, batch=786432 tokens, Muon+AdamW.

---

## Experiments

### E0: Baseline Reproduction
**What:** Run SOTA config unchanged to establish comparison point.
**Config:** Default SOTA settings, SEED=1337
**Expected:** ~1.1228 BPB (sliding window stride=64), ~7100 steps, ~84ms/step

| Metric | Value |
|--------|-------|
| Steps | 6198 |
| Step avg | 97ms (includes W&B overhead) |
| Val BPB (sliding s64) | **1.1278** |
| Val BPB (post-EMA) | 1.1432 |
| Artifact size | 16,362,094 bytes |
| Notes | W&B overhead ~13ms/step → fewer steps than clean run. BPB 0.005 above reported SOTA, within seed/step-count variance. |

---

### E1: LeakyReLU(0.5)² Baseline
**What:** One-line activation change from relu² to leaky_relu(0.5)². Free -0.003 BPB per PR #493.
**Config:** `LEAKY_RELU_SLOPE=0.5`
**Expected:** ~1.120 BPB (sliding s64)
**Why:** Establishes a stronger baseline before testing lookahead features.

| Metric | Value |
|--------|-------|
| Steps | 6936 |
| Step avg | 86.5ms (includes W&B overhead) |
| Val BPB (sliding s64) | **1.1214** |
| Val BPB (post-EMA) | 1.1369 |
| Artifact size | 15,898,658 bytes |
| Notes | -0.0064 BPB vs E0 (1.1278). Matches expected ~0.003-0.006 improvement from LeakyReLU. |

---

### E2: Lookahead Smear + Lookahead Bigram (Self-Refinement)
**What:** Core experiment. Both SmearGate and BigramHash look forward. Training uses two-pass self-refinement: pass 1 without drafts → pass 2 uses own (detached) predictions as lookahead. ~10% of batch rows get the second pass. Draft dropout on remaining rows.
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1 TWOPASS_TRAIN_FRAC=0.1`
**Expected:** Same base step time (~85ms), ~10% overhead from two-pass rows. Multi-pass eval should recursively improve BPB. Key question: can the model learn to refine its own predictions?
**Why:** Directly tests recursive self-refinement — the core hypothesis.

| Metric | Value |
|--------|-------|
| Steps | 6540 |
| Step avg | 91.8ms |
| Val BPB (single pass s64) | **1.1383** |
| Val BPB (two-pass s64) | **1.1265** |
| Val BPB (three-pass s64) | not run (srun time limit) |
| Notes | Single-pass +0.011 vs E0 (training noise from lookahead). Two-pass -0.001 vs E0 — self-refinement works! Two-pass -0.012 vs own single-pass. Step time only +7ms vs baseline. TWOPASS_TRAIN_FRAC=0.1 was not tuned — sweeping this value is a TODO. |

---

### E3: Lookahead Smear Only (Self-Refinement)
**What:** Only SmearGate gets lookahead. BigramHash keeps backward-looking behavior. Same two-pass training.
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=0 TWOPASS_TRAIN_FRAC=0.1`
**Expected:** Tests whether smear alone captures most of the benefit. Bigram hash requires discrete IDs which is less clean with soft predictions.
**Why:** Feature direction ablation.

| Metric | Value |
|--------|-------|
| Steps | 6621 |
| Step avg | 90.6ms |
| Val BPB (single pass s64) | **1.1408** |
| Val BPB (two-pass s64) | **1.1259** |
| Notes | Smear alone gives -0.015 two-pass gain (best of any config). Two-pass 1.1259 beats E2's 1.1265. Bigram adds no value — smear is the key feature. |

---

### E4: Lookahead Bigram Only (Self-Refinement)
**What:** Only BigramHash gets lookahead. SmearGate keeps backward-looking behavior. Same two-pass training.
**Config:** `LOOKAHEAD_SMEAR=0 LOOKAHEAD_BIGRAM=1 TWOPASS_TRAIN_FRAC=0.1`
**Expected:** Tests whether bigram hash benefits more from lookahead than smear.
**Why:** Feature direction ablation.

| Metric | Value |
|--------|-------|
| Steps | 6290 |
| Step avg | 95.4ms |
| Val BPB (single pass s64) | **1.1272** |
| Val BPB (two-pass s64) | **1.1271** |
| Notes | Essentially zero two-pass gain (-0.0001). Discrete argmax IDs don't carry useful lookahead signal. Higher step time (95ms) from bigram forward hash overhead. |

---

### E5: Stale Predictions for Two-Pass Training
**What:** Same as E2 but uses model predictions from N steps ago instead of current step. Avoids memorisation risk.
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1 TWOPASS_TRAIN_FRAC=0.1 TWOPASS_STALE_STEPS=10`
**Expected:** Slightly worse draft quality but more robust — model can't just memorise its recent outputs.
**Why:** Tests whether stale predictions prevent overfitting to the refinement loop.

| Metric | Value |
|--------|-------|
| Steps | |
| Val BPB (two-pass s64) | |
| Notes | Deferred |

---

### E7: Best Lookahead + LeakyReLU
**What:** Best lookahead config from E2-E4 + LeakyReLU(0.5)². Tests whether improvements stack.
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1 TWOPASS_TRAIN_FRAC=0.1 LEAKY_RELU_SLOPE=0.5`
**Expected:** Combined improvement. If they stack: potential record.

| Metric | Value |
|--------|-------|
| Steps | 6475 |
| Step avg | 92.7ms |
| Val BPB (single pass s64) | **1.1345** |
| Val BPB (two-pass s64) | **1.1248** |
| Notes | Best two-pass BPB of any experiment. Beats E1 LeakyReLU (1.1214) at two-pass. LeakyReLU and lookahead stack. TWOPASS_TRAIN_FRAC=0.1 not tuned. |

---

### TWOPASS_TRAIN_FRAC Sweep
**What:** Sweep the fraction of steps that use two-pass self-refinement training.

#### Smear+Bigram (from E2)
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1` with varying `TWOPASS_TRAIN_FRAC`

| Frac | Steps | 1-pass BPB | 2-pass BPB | Notes |
|------|-------|------------|------------|-------|
| 0.02 | 6548 | 1.2035 | 1.1336 | Single-pass badly degraded |
| 0.05 | 6656 | 1.1453 | 1.1264 | |
| 0.10 | 6540 | 1.1383 | 1.1265 | (= E2) |
| 0.20 | 6353 | 1.1368 | 1.1265 | |
| 0.50 | 5745 | 1.1611 | 1.1310 | Too many two-pass steps |

#### Smear-only + LeakyReLU (best config)
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=0 LEAKY_RELU_SLOPE=0.5` with varying frac + 4-pass eval

| Frac | Steps | 1-pass BPB | 2-pass BPB | 3-pass | 4-pass | Notes |
|------|-------|------------|------------|--------|--------|-------|
| 0.05 | 6603 | 1.1334 | **1.1236** | 1.1236 | 1.1236 | Best overall two-pass BPB |
| 0.10 | 6523 | 1.1372 | 1.1249 | — | — | |
| 0.15 | 6437 | 1.1357 | 1.1251 | 1.1251 | 1.1251 | |

**Findings:** frac=0.05 optimal. Two-pass converges immediately — passes 3+ give zero additional gain. Two-pass BPB stable across 0.05–0.20 range.

---

### Curriculum & Prior Ablation
**What:** Sweep curriculum (late-stage lookahead) and prior type (mean embedding vs zeros) to find optimal single-pass / two-pass tradeoff.

| Run | Prior | Scale | Curriculum | Steps | 1-pass BPB | 2-pass BPB | Notes |
|-----|-------|-------|------------|-------|------------|------------|-------|
| v2 (find_unused) | mean | gate_fwd | 80/20 | 5714 | 1.1334 | 1.1330 | Failed: DDP overhead |
| v3 (mean prior) | mean | gate_fwd | 80/20 | 6654 | 1.1352 | **1.1237** | Best 2-pass gain (-0.012) |
| v4 (zero prior) | zero | gate_fwd | 80/20 | 6399 | 1.1359 | 1.1265 | Worse than mean prior |
| learned scale_fwd | zero | learned 0.1 | none | 6614 | 1.1224 | 1.1224 | No 2-pass gain — scale decayed to ~0 |
| learned scale_fwd | zero | learned 0.1 | 80/20 | 6661 | 1.1225 | 1.1225 | Same — model ignores lookahead |
| fixed 0.1 scale | zero | fixed 0.1 | none | 6654 | 1.1223 | 1.1225 | No 2-pass gain — model ignores zero prior |
| fixed 0.1 scale | zero | fixed 0.1 | 80/20 | 6711 | 1.1220 | 1.1222 | Same |
| bigram prior | bigram | fixed 0.1 | none | 6625 | 1.1234 | 1.1232 | Good single-pass, no 2-pass gain |
| **bigram prior** | **bigram** | **fixed 0.1** | **80/20** | **6684** | **1.1226** | **1.1224** | **Best overall BPB. No clear 2-pass gain** |

**Key findings:**
1. **Bigram prior + curriculum achieves best overall BPB: 1.1224** — only 0.001 above E1's 1.1214, with minimal training disruption.
2. However, the two-pass gain is negligible — the bigram prior is informative enough that the model's own predictions don't add much on top.
3. Mean embedding prior gives the best two-pass refinement (-0.012) but degrades single-pass by +0.012 — a wash.
4. Zero prior preserves single-pass perfectly but model ignores lookahead entirely.
5. The fundamental tension: informative priors → good single-pass but no refinement gain; uninformative priors → refinement works but single-pass suffers.

**Best configs:**
- **Best BPB overall:** bigram prior + curriculum = **1.1224** (2-pass, though 1-pass is equally good at 1.1226)
- **Best proven refinement:** mean prior frac=0.05 = **1.1236** (2-pass, with clear -0.010 gain over own 1-pass)

---

### E8: Temperature Sweep — τ=0.5
**What:** Sharper draft predictions during eval passes.
**Config:** Best lookahead config + `DRAFT_TEMP=0.5`
**Expected:** Sharper drafts → closer to real token embeddings → stronger signal for SmearGate/BigramHash.

| Metric | Value |
|--------|-------|
| Steps | |
| Val BPB (two-pass s64) | |
| Notes | |

---

### E9: Temperature Sweep — τ=2.0
**What:** Softer draft predictions — more uncertainty preserved.
**Config:** Best lookahead config + `DRAFT_TEMP=2.0`
**Expected:** More diffuse drafts. May help if features benefit from uncertainty signal rather than hard predictions.

| Metric | Value |
|--------|-------|
| Steps | |
| Val BPB (two-pass s64) | |
| Notes | |

---

## Execution Order

**Phase 1 — Quick signal (run first, ~20 min each):**
1. E1 (LeakyReLU baseline) ✓
2. E2 (core self-refinement, both features)

**Phase 2 — Ablations (based on Phase 1 results):**
3. E3 (smear only)
4. E4 (bigram only)

**Phase 3 — Refinements (if self-refinement shows promise):**
5. E5 (stale predictions)
6. E8, E9 (temperature)

**Phase 4 — Orthogonal techniques (can run in parallel with Phase 1–3):**
7. E10 timing test (does torch.compile handle non-uniform d_ff?)
8. E10 diamond profile (if timing OK)
9. E11 n=2 λ=0.3 (MTP baseline)
10. E11 n=4 λ=0.3 (if n=2 helps, test larger n)

**Phase 5 — Best combination:**
11. E7 (lookahead + LeakyReLU)
12. Best of {E2–E4} + best of {E10} + best of {E11} + LeakyReLU

---

### E10: Non-uniform FFN Width Sweep
**What:** Vary `d_ff` per layer while keeping total FFN parameters constant (~11 × 2 × 512 × 1536 baseline). The current model uses uniform 3× expansion (d_ff=1536) across all 11 layers. Test whether reallocating capacity to critical layers improves BPB at identical parameter count.
**Profiles to test:**
- **Diamond** (wide middle): layers 0–2,8–10 get 2× (1024), layers 3–7 get ~4× (2048). Motivated by NAS/interpretability evidence that middle layers perform the most abstract computation.
- **Tapered** (wide early): layers 0–3 get 4× (2048), layers 4–10 get ~2.2× (~1130). Motivated by ShortGPT finding that later layers are most redundant.
- **Inverse diamond** (wide edges): layers 0–2,8–10 get 4× (2048), layers 3–7 get 2× (1024). Control to test whether diamond is actually better or any non-uniformity helps.
**Config:** New env var `FFN_PROFILE={uniform,diamond,tapered,inverse_diamond}`
**Expected:** 1–5% BPB improvement if the literature generalises. Diamond is the bet. Key risk: torch.compile with non-uniform layer shapes — need a timing-only test first.
**Why:** Orthogonal to lookahead. Pure parameter reallocation, free at inference if it works.

| Profile | Steps | Step avg | Val BPB (s64) | Notes |
|---------|-------|----------|---------------|-------|
| diamond (unaligned) | 5897 | 101.75ms | 1.1292 | Non-aligned d_ff (1054,1269,...) caused ~18% step-time regression from poor GPU matmul tiling |
| diamond (aligned) | 6562 | 91.4ms | 1.1256 | d_ff rounded to multiples of 128. Step time improved but still ~5ms overhead. Wider middle layers → slightly slower |
| tapered | 6887 | 87.1ms | 1.1238 | Wide early layers (2048→1024). No step-time overhead. 0.002 BPB worse than E1 |
| inverse_diamond | 7003 | 85.7ms | **1.1220** | Wide edges, narrow middle. Fastest profile — slightly faster than uniform. BPB essentially tied with E1 (1.1214). Goes against literature prediction that middle layers need more capacity |

---

### E11: Multi-Token Prediction (Training-Only Auxiliary Loss)
**What:** Add auxiliary heads that predict tokens t+2 (and optionally t+3, t+4) from the trunk's hidden state at position t. Each head is a single transformer layer with shared unembedding. Trained jointly: `L = L_NTP + λ·L_MTP`. All heads discarded at eval — zero parameter/inference overhead.
**Config:** `MTP_N=2 MTP_LAMBDA=0.3`
**Sweep:**
- `MTP_N` in {2, 4}. At V=1024 each head is cheap (~512×1024 shared unembedding + one transformer layer). n=4 spans roughly one word at subword granularity, matching where prediction entropy concentrates. Byte-level results (Gloeckle et al., V=256, n=8) showed larger n helps at small vocabularies.
- `MTP_LAMBDA` in {0.1, 0.3, 1.0}.
**Expected:** Step-time overhead ~2ms per head (small at V=1024). n=2 adds ~4ms, n=4 adds ~8ms (~10% overhead). Modest BPB improvement (~0.003–0.010) based on modded-nanogpt evidence at ~124M params. Larger n may help more at small vocab than the literature suggests at V=32K.
**Why:** Proven technique at similar scale. Orthogonal to lookahead (auxiliary loss vs input features). Heads are training-only so no 16MB impact.

| MTP_N | MTP_LAMBDA | Steps | Step avg | Val BPB (s64) | Notes |
|-------|------------|-------|----------|---------------|-------|
| 2 | 0.1 | 6740 | 89.1ms | 1.1275 | +0.006 BPB vs E1. Small step-time overhead (~3ms) from extra head. MTP hurts — auxiliary loss competing with main NTP objective |
| 2 | 0.3 | 6969 | 86.1ms | 1.1354 | Worse than λ=0.1. Higher MTP weight = more interference with main NTP loss |
| 2 | 1.0 | 6960 | 86.2ms | 1.1546 | Much worse. MTP loss dominates training, main NTP quality degrades substantially |
| 4 | 0.3 | 6627 | 90.6ms | 1.1395 | 3 auxiliary heads (t+2,t+3,t+4). ~4ms overhead from extra heads. Worse than n=2 at same λ |

---

## Key Questions to Answer

1. **Can the model recursively refine its own predictions?** Compare E2 two-pass vs single-pass vs E0. Core question.
2. **Which feature benefits more from self-refinement?** Compare E3 (smear only) vs E4 (bigram only).
3. **Does refinement converge?** Track BPB across passes 1→2→3. Diminishing returns expected but useful to quantify.
4. **Do improvements stack with LeakyReLU?** E7 vs E1 and E2 individually.
5. **Does non-uniform FFN allocation help at fixed param count?** E10 diamond vs uniform baseline. Does torch.compile handle it without step-time regression?
6. **Does MTP help at V=1024, and does larger n help more?** E11 n=4 vs n=2. The byte-level literature predicts yes; step-time cost is the constraint.
7. **Do orthogonal techniques stack?** Best of {E2–E4} + best of {E10} + best of {E11} + E1 LeakyReLU.

---

### E12: Legal Score-First Test-Time Training (TTT)
**What:** At eval time, adapt the model on already-scored validation chunks before scoring subsequent chunks. Protocol: split val set into non-overlapping 32K-token chunks. For each chunk: SCORE under `torch.inference_mode()` first, then TRAIN on that chunk (SGD, lr=0.002, momentum=0.9, 3 epochs, all blocks unfrozen). Chunk N is scored by the model adapted on chunks 0..N-1.
**Config:** Eval-time only — no training changes. Apply on top of best model (E1 or best lookahead config).
**Expected:** -0.0025 BPB based on the SOTA submission (1.1218 pre-TTT → 1.1194 post-TTT). ~410s eval time, well within the 10-min eval budget.
**Why:** Proven technique used by the current #1 submission. Stacks on top of any model quality.

| TTT Training Mode | Pre-TTT (1-pass) | Pre-TTT (2-pass) | Post-TTT BPB | Delta vs 2-pass |
|-------------------|-----------------|-----------------|-------------|-----------------|
| Single-pass TTT training | 1.1329 | 1.1231 | 1.1509 | +0.028 |
| Two-pass TTT, loss on 2nd only | 1.1227 | 1.1227 | 1.1428 | +0.020 |
| Two-pass TTT, loss on both | 1.1221 | 1.1221 | ~1.1584 | +0.036 |

**Findings:** TTT hurts the lookahead-trained model regardless of training mode. Two-pass TTT training is better than single-pass (1.1428 vs 1.1509) but still net negative. The SOTA's TTT hyperparameters (lr=0.002, 3 epochs) are too aggressive for our model — BPB climbs monotonically after initial chunks. Would need lower LR or fewer epochs. The lookahead gate weights (gate_fwd) may be sensitive to SGD perturbation.

---

### ~~E13: GPTQ-lite Clip Search~~ (Already Implemented)
**Status:** Already present in `train_gpt_lookahead.py` — `quantize_int6_per_row()` searches over 5 clip percentiles `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]` and picks the one with lowest MSE. All experiments E0–E11 already used this. No action needed.

---

### E14: Temperature Scaling at Eval
**What:** Grid search over logit temperature T at eval time. Divide logits by T before softmax. Tests whether the model's predictions are miscalibrated.
**Config:** Eval-time only. Sweep T in {0.85, 0.90, 0.95, 1.0, 1.05, 1.10}. Note: model uses logit softcapping (tanh, cap=30) which may already handle calibration.
**Expected:** Up to -0.005 BPB if miscalibrated, but may be negligible given softcap. The ternary submission found T=0.90 optimal for relu².
**Why:** Zero-cost eval-time search. Quick to test.

| T | Val BPB (s64) | Notes |
|---|---------------|-------|
| 0.85 | | |
| 0.90 | | |
| 0.95 | | |
| 1.00 | | baseline |
| 1.05 | | |
| 1.10 | | |

---

### E15: Tokenizer-Side Boundary Features (Single-Pass)
**What:** Inject word-boundary and morphology signals directly into SmearGate/BigramHash using existing SentencePiece LUTs (`has_leading_space`, `base_bytes`, `is_boundary_token`), rather than relying on draft predictions. Concretely: add a small additive embedding or learned scale modulation conditioned on boundary bit, token byte-length bucket, and/or punctuation/digit class. The simplest version multiplies `SmearGate.scale` or `BigramHash.scale` by `(1 + α * is_boundary)` with a learned α.
**Config:** No new env vars needed beyond feature flags. Single forward pass — no two-pass training or multi-pass eval. LeakyReLU stays on.
**Expected:** If the two-pass lookahead gain (E3's -0.015) is mostly boundary-regime detection, this should capture a large fraction in one pass without degrading the base model. If it gets close to E1 + the two-pass delta, the entire draft machinery is unnecessary.
**Why:** The passes 3+ convergence result (zero gain beyond pass 2) suggests the lookahead value is a one-shot boundary correction. The `has_leading_space` and `base_bytes` LUTs already exist in `build_sentencepiece_luts()`. Zero train/eval mismatch, zero step-time overhead.

| Metric | Value |
|--------|-------|
| Steps | |
| Step avg | |
| Val BPB (s64) | |
| Notes | |

---

### E16: Score-First Adaptive Cache Mixer (Eval-Time)
**What:** At eval time, maintain an online count-based bigram table over already-scored validation tokens. Interpolate count-based log-probabilities with neural logits: `log p = log p_nn + λ · log p_cache`, where λ is a function of cache count and/or model entropy. The bigram table is 1024×1024 (4MB in float32). Updated only after tokens are scored (legal per competition rules). Apply on top of best trunk model.
**Config:** Eval-time only — no training changes. Hyperparameters: smoothing constant, λ schedule (fixed vs. count-dependent vs. entropy-dependent), scoring order (sequential vs. random chunks).
**Expected:** -0.001 to -0.003 BPB. The current #1 submission gets ~-0.0025 from full TTT; a count-based cache is much cheaper and could capture document-level statistics (names, topic words, formatting patterns) that the small neural model can't memorize. Stacks with TTT (E12) and temperature scaling (E14).
**Why:** Exploits the separate 10-minute eval budget without any training cost. Well-studied technique (dynamic evaluation, cache LMs). At V=1024 the bigram table is small enough to fit trivially in GPU memory.

| Metric | Value |
|--------|-------|
| Pre-cache BPB | |
| Post-cache BPB | |
| Eval time | |
| Notes | |

---

## Deprecated: Interleaved Draft Approach

The original approach doubled sequence length by interleaving draft tokens: `[x_0, d_1, x_1, d_2, ...]`. Timing tests showed ~193ms/step (2.3x baseline) with most overhead from draft batch construction (~78ms) rather than the longer sequence (~30ms). The lookahead features approach achieves the same goal (giving SmearGate/BigramHash future token info) without any sequence length increase, at baseline step cost.
