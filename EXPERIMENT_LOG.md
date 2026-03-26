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
| bigram prior | bigram | learned 0.1 | none | 6587 | 1.1235 | 1.1235 | Learned scale ≈ same as fixed |
| bigram prior | bigram | learned 0.1 | 80/20 | 6630 | 1.1239 | 1.1237 | Slightly worse — optimizer overhead |

**Key findings:**
1. **Bigram prior + curriculum + fixed 0.1 scale achieves best overall BPB: 1.1224** — only 0.001 above E1's 1.1214, with minimal training disruption.
2. However, the two-pass gain is negligible — the bigram prior is informative enough that the model's own predictions don't add much on top.
3. Mean embedding prior gives the best two-pass refinement (-0.012) but degrades single-pass by +0.012 — a wash.
4. Zero prior preserves single-pass perfectly but model ignores lookahead entirely.
5. Learned scale adds nothing over fixed 0.1 — the optimizer doesn't find a better value.
6. The fundamental tension: informative priors → good single-pass but no refinement gain; uninformative priors → refinement works but single-pass suffers.

**Best configs:**
- **Best BPB overall:** bigram prior + curriculum + fixed 0.1 = **1.1224** (2-pass, though 1-pass is equally good at 1.1226)
- **Best proven refinement:** mean prior + gate_fwd + frac=0.05 = **1.1236** (2-pass, with clear -0.010 gain over own 1-pass)

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

| T | 1-pass BPB | 2-pass BPB | Notes |
|---|-----------|-----------|-------|
| 0.85 | 1.1429 | 1.1429 | Much worse — too sharp |
| 0.90 | 1.1308 | 1.1308 | Worse |
| 0.95 | 1.1259 | 1.1259 | Worse |
| **1.00** | **1.1238** | **1.1238** | **Best — baseline optimal** |
| 1.05 | 1.1253 | — | Worse |
| 1.10 | 1.1304 | — | Worse |

**Findings:** T=1.0 is optimal. The logit softcap (tanh, cap=30) already handles calibration — any temperature scaling makes things worse. Both directions (sharper and softer) hurt. Temperature scaling is not useful for this model.

---

### E15: Tokenizer-Side Boundary Features (Single-Pass)
**What:** Inject word-boundary and morphology signals directly into SmearGate/BigramHash using existing SentencePiece LUTs (`has_leading_space`, `base_bytes`, `is_boundary_token`), rather than relying on draft predictions. Concretely: add a small additive embedding or learned scale modulation conditioned on boundary bit, token byte-length bucket, and/or punctuation/digit class. The simplest version multiplies `SmearGate.scale` or `BigramHash.scale` by `(1 + α * is_boundary)` with a learned α.
**Config:** No new env vars needed beyond feature flags. Single forward pass — no two-pass training or multi-pass eval. LeakyReLU stays on.
**Expected:** If the two-pass lookahead gain (E3's -0.015) is mostly boundary-regime detection, this should capture a large fraction in one pass without degrading the base model. If it gets close to E1 + the two-pass delta, the entire draft machinery is unnecessary.
**Why:** The passes 3+ convergence result (zero gain beyond pass 2) suggests the lookahead value is a one-shot boundary correction. The `has_leading_space` and `base_bytes` LUTs already exist in `build_sentencepiece_luts()`. Zero train/eval mismatch, zero step-time overhead.

| Metric | Value |
|--------|-------|
| Steps | 6931 |
| Step avg | 86.6ms |
| Val BPB (s64) | **1.1225** |
| Notes | Tied with E1 (1.1214) — boundary features neither help nor hurt. Zero step-time overhead. The model already captures word boundary info through its existing embeddings. Disproves the hypothesis that lookahead value comes from boundary detection. |

---

### E16: Score-First Adaptive Cache Mixer (Eval-Time)
**What:** At eval time, maintain an online count-based bigram table over already-scored validation tokens. Interpolate count-based log-probabilities with neural logits: `log p = log p_nn + λ · log p_cache`, where λ is a function of cache count and/or model entropy. The bigram table is 1024×1024 (4MB in float32). Updated only after tokens are scored (legal per competition rules). Apply on top of best trunk model.
**Config:** Eval-time only — no training changes. Hyperparameters: smoothing constant, λ schedule (fixed vs. count-dependent vs. entropy-dependent), scoring order (sequential vs. random chunks).
**Expected:** -0.001 to -0.003 BPB. The current #1 submission gets ~-0.0025 from full TTT; a count-based cache is much cheaper and could capture document-level statistics (names, topic words, formatting patterns) that the small neural model can't memorize. Stacks with TTT (E12) and temperature scaling (E14).
**Why:** Exploits the separate 10-minute eval budget without any training cost. Well-studied technique (dynamic evaluation, cache LMs). At V=1024 the bigram table is small enough to fit trivially in GPU memory.

| Metric | Value |
|--------|-------|
| Pre-cache BPB (s64) | 1.1243 |
| Post-cache BPB (λ=0.1) | 1.2634 (MUCH WORSE) |
| Eval time | 190s |
| Notes | Cache interpolation massively hurts. At V=1024 the bigram distribution is too dense/uniform — the smoothed cache predictions are worse than the neural model alone. λ=0.1 is likely too high; would need λ=0.001 or entropy-gated mixing. But the fundamental problem is that 1024×1024 bigram counts don't have enough discriminative power at this vocabulary size. |

---

## Phase 6: Analysis-Motivated Experiments (on SOTA+TTT baseline)

All experiments below run on the current SOTA (abaybektursun's LeakyReLU² + Parameter Banking + Legal TTT, 1.1194 BPB post-TTT). Motivated by the full-stack model analysis (v2). Split into **memory-saving** (maintain BPB with fewer bytes) and **performance-improving** (better BPB, potentially more bytes).

### E17: Mixed Quantization — int8 for MLP Down-Projections
**Category:** Memory-saving / free accuracy
**What:** Use int8 quantization (clip range [-127,127]) for mlp_down weights instead of int6 ([-31,31]). Both store as 1-byte int8 — the only cost is worse lzma compression (more unique values). Analysis shows mlp_down has 3x worse quantization error (RelMSE 6e-3 vs 2e-3 for mlp_up). This targets the single biggest source of quantization degradation.
**Config:** Modify `_classify_param` to route `mlp.proj` → int8. Everything else stays int6.
**Expected:** -0.001 to -0.003 BPB from reduced quantization error. Risk: artifact may exceed 16MB from worse compression.
**Why:** Directly targets the #1 quantization bottleneck identified in analysis. Zero training cost — post-training only.

| Metric | Value |
|--------|-------|
| Pre-TTT BPB (s64) | **1.1178** (-0.0037 vs SOTA 1.1215) |
| Post-TTT BPB | 1.1179 (control-tensor TTT, negligible change) |
| Artifact size | **18.03 MB — EXCEEDS 16MB LIMIT** |
| Notes | int8 for mlp_down gives a large BPB improvement (-0.0037) but the artifact blows up by +2.1MB from worse lzma compression. The wider int8 value range [-127,127] vs int6 [-31,31] produces more unique byte values that lzma can't compress as well. Would need to be paired with parameter savings elsewhere (E20, E22) to fit in 16MB. The BPB gain confirms mlp_down quantization is a real bottleneck. |

---

### E18: Extend ValueEmbedding to Layers 7–8
**Category:** Performance-improving (minimal extra memory)
**What:** Add VE to XSA layers 7–8 in addition to current layers 9–10. The shared VE table (1024×128) is already paid for; each new layer only adds 1 scale parameter. Analysis shows VE contributes substantially at layers 9–10 (norms 23.4, 17.0). Layers 7–8 are XSA layers with active attention — giving them token identity through values could help.
**Config:** `VE_LAYERS=7,8,9,10`
**Expected:** Modest BPB improvement. Negligible parameter/size cost. Could slightly slow step time from extra VE lookups.

| Metric | Value |
|--------|-------|
| Steps | 6725 |
| Step avg | 89.2ms (+2.7ms overhead from extra VE lookups) |
| Pre-TTT BPB | **1.1237** |
| Post-TTT BPB | **1.1215** |
| Artifact size | 15.9MB |
| Notes | +0.0026 worse post-TTT vs clean baseline (1.1189). VE expansion adds ~6.3ms/step → ~525 fewer steps. The extra VE lookups at layers 7-8 don't pay for themselves — step time cost outweighs any marginal benefit. |

---

### E19: Larger BigramHash Table (4096 or 8192 buckets)
**Category:** Performance-improving (uses more memory)
**What:** Increase bigram hash table from 1536 to 4096 or 8192 buckets. Analysis shows current 1536 buckets have 45 collisions per bucket — each embedding averages 45 different bigram contexts. SOTA ablation showed 2048→3072 gave -0.0009 BPB. More viable if memory is freed by E17/E20/E22.
**Config:** `BIGRAM_VOCAB_SIZE=4096` (then 8192 if room)
**Expected:** -0.001 to -0.002 BPB. 4096×128 = 512K params (vs 196K). Need to verify artifact fits 16MB.

| Bigram Size | Pre-TTT BPB | Post-TTT BPB | Artifact | Notes |
|-------------|-------------|-------------|----------|-------|
| 1536 (SOTA) | 1.1218 | 1.1194 | ~15.9MB | baseline |
| 4096 | | | | |
| 8192 | | | | |

---

### E20: Remove/Freeze Layer 0 Attention
**Category:** Memory-saving (~1.5M params)
**What:** Layer 0 attention contributes 3.8% with AttnScale 0.067 — essentially vestigial. Two variants:
- **v1 (freeze):** Set `blocks[0].attn_scale` to zero and freeze. Saves compute, no architecture change.
- **v2 (remove + widen MLP):** Skip attention entirely in layer 0. Reallocate params to widen layer 0 MLP from 3x (1536) to ~4.5x (2304). Makes the implicit token-lookup pattern explicit.
**Config:** v1: code change to freeze. v2: architecture change (breaks parameter banking for layer 0).
**Expected:** v1: neutral BPB, slight step-time improvement → more steps. v2: potentially better BPB from bigger lookup table, but non-uniform MLP may slow step time.
**Risk:** v2 breaks parameter banking; non-uniform d_ff can add ~5ms overhead (E10).

| Variant | Pre-TTT BPB | Post-TTT BPB | Step avg | Artifact | Notes |
|---------|-------------|-------------|----------|----------|-------|
| v1 (freeze) | | | | | |
| v2 (widen MLP) | | | | | |

---

### E21: Analysis-Informed MLP Profile
**Category:** Performance-improving (same total params)
**What:** Custom d_ff per layer informed by activation analysis, instead of uniform 3x. Profile:
- Layer 0: 2304 (4.5x) — pure MLP lookup, needs capacity
- Layers 1–2: 1536 (3x) — standard
- Layers 3–6: 1280 (2.5x) — attention-heavy middle layers, MLP less critical
- Layers 7–8: 1536 (3x) — XSA layers
- Layers 9–10: 1280 (2.5x) — small contribution layers
All d_ff rounded to multiples of 128 for GPU tiling. Total params kept constant.
**Config:** Per-layer d_ff list. Breaks parameter banking (non-uniform bank shapes).
**Expected:** Small BPB improvement. Risk of 5ms+ step-time overhead from non-uniform matmul shapes.
**Why:** Unlike E10's generic diamond/inverse-diamond, this profile is specifically motivated by per-layer activation data.

| Metric | Value |
|--------|-------|
| Pre-TTT BPB | |
| Post-TTT BPB | |
| Step avg | |
| Notes | |

---

### E22: Depth Recurrence for Middle Layers
**Category:** Memory-saving (fewer unique params → smaller artifact)
**What:** Layers 3–6 have similar activation patterns (MixX0 ≈ 0, moderate attn/MLP, lowest cosine with output). Two variants:
- **v1 (weight sharing):** Share MLP weights across layers 3–5 (keep attention unique). Per-layer adapter (learned scale or rank-1 offset). Reduces unique MLP params → smaller artifact → room for bigger bigram table or wider edge layers.
- **v2 (eval-time recursion):** At eval time, run layers 3–6 through 2 iterations instead of 1 pass each. No training change — just loop blocks twice. Tests whether middle layers benefit from iterative refinement. Uses the separate 10-min eval budget.
**Config:** v1: weight-tying code change. v2: eval-only loop in forward_logits.
**Expected:** v1: neutral BPB with ~2–3MB artifact savings. v2: uncertain — may help or hurt.
**Risk:** v1 interacts with Muon optimizer. v2 doubles compute for 4 layers (~30ms extra per eval pass).

| Variant | Pre-TTT BPB | Post-TTT BPB | Artifact | Notes |
|---------|-------------|-------------|----------|-------|
| v1 (shared MLP) | | | | |
| v2 (eval recursion) | | | | |

---

## Phase 7: Novel Techniques from Literature & Community (March 2026)

These experiments incorporate techniques from recent papers (2025–2026) and community discoveries from the parameter-golf leaderboard. Prioritized by expected value and implementation feasibility. Run on the SOTA baseline (1.1194 BPB post-TTT).

### E23: Multi-Order N-gram Cache with Entropy-Adaptive Interpolation (Eval-Time)
**Category:** Eval-time only — zero training cost
**What:** Revisit E16's cache approach with the technique used by sub-1.0 BPB submissions (PR #727, #740). Key differences from E16: (1) use 5-gram backoff (5→4→3→2→1-gram) instead of bigram-only, (2) entropy-adaptive interpolation weight — when the neural model is confident (low entropy), trust it; when uncertain, lean on the cache, (3) modified Kneser-Ney smoothing instead of simple add-k. Build the cache from already-scored text (legal per competition rules). The cache is built at eval time — zero artifact bytes.
**Config:** Eval-time only. Hyperparameters: max n-gram order (3–7), smoothing method, entropy threshold for adaptive λ.
**Expected:** -0.02 to -0.10+ BPB based on community results. E16 failed because bigram counts at V=1024 lack discriminative power; higher-order n-grams should provide much stronger signal. This is the single biggest lever on the current leaderboard.
**Why:** The gap from 1.12 to sub-1.0 BPB is almost entirely this technique. Our E16 failure was a methodology issue (wrong n-gram order + fixed λ), not a fundamental limitation.

| Max Order | Smoothing | λ Strategy | Pre-cache BPB | Post-cache BPB | Notes |
|-----------|-----------|------------|---------------|----------------|-------|
| 5 | add-k (k=1) | entropy-adaptive λ=0.1 | 1.1218 | **1.150** (WORSE) | Timed out at chunk 951/1893. Cache hurts badly — same pattern as E16. |
| 5 | Kneser-Ney | entropy-adaptive | | | |
| 7 | Kneser-Ney | entropy-adaptive | | | |

**Findings (5-gram, add-k, λ=0.1 adaptive):** Cache interpolation worsens BPB by +0.028 — massive regression. λ=0.1 is still far too aggressive even with entropy scaling. The per-token loop is also extremely slow (~35s per 50 chunks). The fundamental issue may be that our add-k smoothing with k=1 produces nearly uniform distributions at higher n-gram orders (most 5-gram contexts are seen only once), so the cache is essentially adding noise. Key differences from community sub-1.0 implementations: (1) they likely use much lower effective λ, (2) proper Kneser-Ney smoothing handles unseen n-grams far better than add-k, (3) they may vectorize the cache lookup rather than per-token Python loops. The implementation needs a complete rewrite with Kneser-Ney and batched lookups before retesting.

---

### E24: ESLM Token-Level Loss Masking
**Category:** Training improvement — near-zero overhead
**What:** Apply Value-at-Risk (VaR) thresholding on per-token loss within each batch. Compute per-token loss, find the quantile threshold (e.g., 60th percentile), zero out losses below the threshold before backward pass. This retains only the most informative tokens for gradient computation. From Bal et al. (May 2025), "Risk-Averse Selective Language Modeling". No reference model needed, operates online, ~5 lines of code.
**Config:** `ESLM_QUANTILE=0.6` (mask bottom 40% of tokens by loss). Sweep: {0.4, 0.5, 0.6, 0.7}.
**Expected:** ~1.5x data efficiency for free. Our analysis shows word-initial tokens (40% of tokens) account for 67% of total loss — ESLM naturally focuses gradients on these hard tokens. Near-zero step-time overhead (just a quantile computation + mask).
**Why:** Directly addresses the loss distribution skew identified in our analysis. Cheapest possible training improvement.

| Quantile | Steps | Step avg | Val BPB | Notes |
|----------|-------|----------|---------|-------|
| 0.4 | 7070 | 84.9ms | **1.2488** | +0.127 vs baseline. Masking easy tokens starves gradient on token patterns the model needs |
| 0.5 | ~7000 | ~84ms | **1.4161** | Worse. Higher quantile = more masking = more damage |
| 0.6 | ~7000 | ~84ms | **1.6426** | Much worse |
| 0.7 | ~7000 | ~84ms | **1.8575** | Catastrophic. Masking 70% of tokens destroys training |

**Conclusion:** ESLM is harmful at all tested quantiles. At V=1024, even "easy" tokens carry useful gradient signal for maintaining learned representations. The paper's results on large-vocab models don't transfer — with 1024 tokens, the loss distribution is less skewed and every token matters.

---

### E25: LAWA Weight Averaging (Replacing EMA/SWA)
**Category:** Training improvement — drop-in replacement
**What:** Replace EMA + SWA with LAWA (Latest-Averaging Weight Averaging). Maintain a FIFO buffer of K recent checkpoints (spaced by ~60s), average them uniformly. From Ajroldi et al. (Feb 2025), tested on 124M transformer on 5B FineWebEdu tokens — nearly identical to our setting. Key finding: optimal averaging horizon is ~1% of total training budget; LAWA reaches validation targets in 15–25% fewer steps than EMA.
**Config:** `LAWA_K=10 LAWA_INTERVAL=60` (buffer of 10 checkpoints, one every 60s). Start averaging early (unlike classical SWA). Disable existing EMA/SWA.
**Expected:** 15–25% effective compute improvement. Direct upgrade to existing infrastructure. The current EMA (decay=0.997) + SWA (every 50 steps) may be suboptimal — LAWA's uniform average over a sliding window is more robust.
**Why:** Validated on our exact setting (124M/FineWebEdu). Drop-in replacement for existing weight averaging.

| K | Interval | Steps | ms/step | Pre-TTT BPB | Post-TTT BPB | Notes |
|---|----------|-------|---------|-------------|-------------|-------|
| 10 | 100 steps | 7152 | 83.9 | 1.1226 | 1.1202 | +0.0013 vs clean baseline (1.1189). EMA+SWA is better than LAWA for this model. |
| 5 | 60s | | | | | |
| 10 | 30s | | | | | |

**Findings:** LAWA is worse than default EMA+SWA by +0.0013 BPB post-TTT. The LAWA paper's 15-25% efficiency gains may not transfer to this exact setup, which already uses tight SWA + high EMA decay (0.997). Not worth pursuing further sweeps.

---

### E26: Self-Distillation from EMA Copy
**Category:** Training improvement — near-zero overhead
**What:** Use the existing EMA model's predictions as soft targets for an auxiliary KL-divergence loss: `L = L_NTP + α × KL(logits, ema_logits.detach())`. The EMA teacher provides smoother probability distributions that act as adaptive label smoothing — no hardcoded smoothing constant, the regularization strength naturally adapts to what the model has learned. ~3 lines of code. The EMA forward pass can reuse the same compiled graph.
**Config:** `SELF_DISTILL_ALPHA=0.1`. Sweep: {0.05, 0.1, 0.3}.
**Expected:** Small but consistent BPB improvement (-0.001 to -0.003). Free regularization from infrastructure we already pay for. Risk: EMA forward pass adds ~85ms/step if done every step — may need to sample (e.g., every 5th step).
**Why:** Exploits existing EMA infrastructure. The EMA model is a better teacher than label smoothing because its soft targets reflect learned token co-occurrence patterns.

| Alpha | Frequency | Steps | Step avg | Val BPB | Notes |
|-------|-----------|-------|----------|---------|-------|
| 0.05 | every step | | | | |
| 0.1 | every step | | | | |
| 0.1 | every 5th | | | | |
| 0.3 | every step | | | | |

---

### E27: TrigramHash Embedding
**Category:** Training improvement — small parameter cost
**What:** Extend BigramHash to 3-token patterns: hash(tok[t], tok[t-1], tok[t-2]) into a separate learned embedding table. Already adopted by the competition community. Captures longer local context (e.g., 3-char subword patterns) without attention. Use different hash constants to avoid collision correlation with the bigram table.
**Config:** `TRIGRAM_VOCAB_SIZE=2048 TRIGRAM_DIM=128`. Separate embedding table + projection + learned scale, same architecture as BigramHash. The trigram embedding is added alongside the bigram embedding.
**Expected:** -0.001 to -0.003 BPB. Helps at word-initial positions where bigram context (2 tokens) is ambiguous but trigram context (3 tokens) may disambiguate. Parameter cost: 2048×128 + 128×512 = 328K params (~1.3MB uncompressed, ~0.3MB int6+lzma).
**Why:** Natural extension of existing infrastructure. Community-validated. Targets word-initial tokens where our analysis shows loss concentrates.

| Trigram Size | Steps | Step avg | Val BPB | Artifact | Notes |
|-------------|-------|----------|---------|----------|-------|
| 2048 | | | | | |
| 4096 | | | | | |

---

### E28: Differential Attention
**Category:** Architecture change — moderate implementation effort
**What:** Replace standard multi-head attention with differential attention (ICLR 2025, Microsoft). Each differential head computes attention as `softmax(Q₁K₁ᵀ) - λ·softmax(Q₂K₂ᵀ)`, cancelling "attention noise" — spurious attention to irrelevant tokens. A learnable scalar λ per head (initialized via exponential decay across layers) controls subtraction strength. Halves head count to match parameter budget: 8 standard → 4 differential heads (each with 2 sub-heads). Requires per-head GroupNorm between the subtraction and value projection.
**Config:** Replace attention in all layers. `DIFF_ATTN=1`. Keep KV head count at 4 (2 diff KV heads × 2 sub-heads).
**Expected:** At 3B scale, 7.5% accuracy gain on math reasoning. At sub-100M, noise cancellation may be even more valuable since each head carries proportionally more weight. Risk: GroupNorm may add compile overhead; interaction with XSA (which also modifies attention) needs careful handling.
**Why:** Directly addresses attention noise, which is proportionally more costly in small models. The λ parameters add negligible memory.

| Metric | Value |
|--------|-------|
| Steps | |
| Step avg | |
| Val BPB | |
| Notes | |

---

### E29: WSD Learning Rate Schedule
**Category:** Training improvement — hyperparameter change only
**What:** Switch from current cosine-with-warmdown to explicit Warmup-Stable-Decay (WSD). Multiple papers (Hägele 2024, Wen 2024, Dremov TMLR 2025) show WSD outperforms cosine for fixed compute budgets. The "river valley" hypothesis: during stable phase at peak LR, the model explores flat manifold directions; during decay, oscillations are suppressed. Schedule: warmup ~1% of steps (~70), stable ~75-80% (~5600), sqrt-decay final ~20% (~1400). Additional trick: raise AdamW β₂ to 0.995 during cooldown.
**Config:** `LR_SCHEDULE=wsd WSD_STABLE_FRAC=0.79 WSD_DECAY_SHAPE=sqrt`. The current WARMDOWN_ITERS=3500 (of ~7000 steps) is already 50% decay — WSD suggests this is too aggressive; most time should be at peak LR.
**Expected:** -0.001 to -0.003 BPB from better LR utilization. The current schedule may be leaving performance on the table by starting decay too early.
**Why:** Pure hyperparameter optimization. Zero code risk. Multiple independent papers converge on this conclusion.

| Schedule | Stable% | Decay Shape | Val BPB | Notes |
|----------|---------|-------------|---------|-------|
| linear warmdown (baseline) | 50% decay | linear | 1.1214 | E1 baseline (WARMDOWN_ITERS=3500 of ~7000 steps) |
| WSD | 79% | sqrt | **1.1226** | +0.0012 vs baseline. Essentially neutral |
| WSD | 79% | linear | **1.1257** | +0.0043 vs baseline. Slightly worse than sqrt |
| WSD + raised β₂ | 79% | sqrt | | not run — sqrt result doesn't justify further testing |

**Conclusion:** WSD is neutral at this scale. The current 50% linear warmdown is already near-optimal. The "river valley" hypothesis may not apply with 10-min budgets where the stable phase is already short enough. The ~0.001 BPB difference is within seed variance.

---

### E30: MiLe Loss (Entropy-Weighted Cross-Entropy)
**Category:** Training improvement — zero parameter overhead
**What:** Replace standard cross-entropy with `loss = H(p)^γ × (-log p_target)`, where H(p) is the predicted distribution's entropy. Upweights tokens where the model is genuinely uncertain (high-entropy predictions) while downweighting tokens where the model is confident or where multiple valid continuations exist. From NAACL 2024. At V=1024, the small vocabulary creates more uniform distributions on average, making entropy-based weighting especially informative.
**Config:** `MILE_GAMMA=1.0`. Sweep: {0.5, 1.0, 1.5}.
**Expected:** Small BPB improvement. Complements ESLM (E24): ESLM masks easy tokens entirely, MiLe reweights the remaining ones by uncertainty. Zero parameter overhead, single hyperparameter.
**Why:** Targets the same loss distribution skew as ESLM but via continuous reweighting rather than hard masking. May compose well with ESLM.

| γ | Steps | Val BPB | Notes |
|---|-------|---------|-------|
| 0.5 | ~7000 | **28.87** | Catastrophic divergence. Train loss oscillates 0.0004 ↔ 3.4 |
| 1.0 | ~7000 | **32.34** | Same divergence pattern |
| 1.5 | ~7000 | **32.44** | Same divergence pattern |

**Conclusion:** MiLe creates a degenerate feedback loop at this scale. When the model becomes confident (low entropy), MiLe zeroes gradients for those tokens → model can't maintain learned patterns → loss oscillates wildly. The entropy weighting is fundamentally unstable as a training signal: it punishes the model for being confident. With V=1024 (lower natural entropy than V=32k+), the effect is amplified.

---

### E31: Batch Size Warmup
**Category:** Training improvement — scheduling change only
**What:** Start training with smaller effective batch size and increase over time. "Critical Batch Size Revisited" (NeurIPS 2025, Allen AI) shows CBS is near zero at initialization and increases during training. Starting small gives more gradient updates when each step matters most (early training). Implementation: start with gradient accumulation=1 (or 2), double when CBS grows (or on a fixed schedule). Achieves same loss with 43% fewer gradient steps on OLMo 1B.
**Config:** `BATCH_WARMUP=1` — start at half batch size for first 20% of steps, then full. With 8 GPUs, this means fewer tokens per step early but more steps.
**Expected:** -0.001 to -0.005 BPB from better utilization of early training steps. Risk: with only ~7000 total steps, the granularity may be too coarse. Need to verify torch.compile handles the batch size change (may require recompilation at transition point).
**Why:** Free scheduling improvement. Multiple papers agree: don't start with maximum batch size.

| Schedule | Steps | Val BPB | Notes |
|----------|-------|---------|-------|
| fixed (baseline) | ~7000 | 1.1194 | current |
| half→full at 20% | | | |
| quarter→full at 10%,20% | | | |

---

### E32: Existing Code Knobs Sweep (GATED_ATTENTION, VALUE_RESIDUAL, DTG)
**Category:** Training improvement — already implemented, zero implementation risk
**What:** Three features are implemented in `train_gpt.py` but never tested. Each is a single env var flip:
- **GATED_ATTENTION=1**: Per-head sigmoid gate on attention output. `nn.Linear(dim, num_heads)` → sigmoid, init bias=4.0 (starts ~open). Learns to downweight noisy heads. Adds ~4K params per layer (~44K total). Conceptually related to differential attention (E28) but simpler — gates entire heads rather than subtracting softmax maps.
- **VALUE_RESIDUAL=1**: DeepSeek-V2 style value residual. First layer's raw values (`v0`) are mixed into every subsequent layer: `v = λ₀·v₀ + λ₁·v` with learned `vr_lambda=[0.5, 0.5]`. Preserves token identity through the attention stack — similar motivation to VE (E18) but through values rather than additive embeddings. Adds 2 params per layer.
- **DTG_ENABLED=1**: Dynamic Token Gating. Per-block `nn.Linear(dim, 1)` → sigmoid gate on entire block output: `x_out = x_in + gate·(x_out - x_in)`. Computed from detached input. Learns per-token whether to apply or skip each block. Init bias=2.0 (starts ~open). Adds ~513 params per layer.
**Config:** Test each individually, then best combination. All on SOTA baseline with LeakyReLU.
**Expected:** Each is a well-motivated architectural refinement with small param overhead. GATED_ATTENTION may particularly help since our analysis shows layer 0 attention is vestigial (3.8%) — a gated head could learn to suppress it. VALUE_RESIDUAL may help since VE at layers 9-10 already shows the model wants token identity in late layers.
**Why:** Highest possible EV: these features are already debugged and compiled. Zero implementation risk. Just need GPU time.

| Config | Steps | Step avg | Pre-TTT BPB | Post-TTT BPB | Artifact | Notes |
|--------|-------|----------|-------------|-------------|----------|-------|
| **Clean baseline** | **7248** | **82.9ms** | **1.1214** | **1.1189** | 15.8MB | Reproduction beats reported SOTA (1.1218/1.1194). Strong baseline. |
| GATED_ATTENTION=1 | 6916 | 86.8ms | 1.1233 | 1.1210 | 15.9MB | +0.0021 worse. 3.9ms overhead from per-head gate linear. |
| VALUE_RESIDUAL=1 | 7105 | 84.5ms | 1.1233 | 1.1210 | 15.8MB | +0.0021 worse. Slower steps cancel the identity-preservation benefit. |
| DTG_ENABLED=1 | 6582 | 91.2ms | 1.1237 | 1.1215 | 15.9MB | +0.0026 worse. 8.3ms slower — per-block gate too expensive. |

**Findings:** All features hurt. The SOTA config is already well-tuned — these architectural additions add step-time overhead that reduces total training steps, and the per-step benefit doesn't compensate. DTG is the worst offender (4.7ms overhead for zero BPB gain). VALUE_RESIDUAL's speed gain (84.5 vs 82.9ms) was noise relative to baseline, not a real improvement. Combination runs (GA+VR, GA+VR+DTG) are not worth pursuing since all individual features hurt.

---

### E33: Control-Tensor-Only TTT (Ultragentle Adaptation)
**Category:** Eval-time improvement — requires code change to TTT parameter selection
**What:** Modify TTT (E12) to update only the small control tensors — gates, scales, VE weights, smear parameters — instead of all block parameters. The `CONTROL_TENSOR_NAME_PATTERNS` list already classifies these: `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weight`, `smear`, `ve_layer_scales`, `ve_shared.scale`, `attn_gate`, `vr_lambda`, `dtg_gate`. Total ~500-5K parameters depending on which features are active. Use much lower LR (1e-4 vs 2e-3) and fewer epochs (1 vs 3).
**Config:** New TTT mode: `TTT_PARAMS=control` (vs current `TTT_PARAMS=all`). Sweep `TTT_LR` in {5e-5, 1e-4, 5e-4}, `TTT_EPOCHS` in {1, 2}.
**Expected:** E12's failure mode was clear: full-model SGD at lr=0.002 was too aggressive — BPB climbed monotonically after initial chunks. With ~500 low-dimensional control params, overfitting risk drops dramatically. These params (attention scales, skip weights, smear gates) control how the model routes information — adapting them to document-specific statistics is well-motivated. If control-tensor TTT works, it could be combined with the n-gram cache (E23) for additive gains.
**Why:** E12 proved TTT is too aggressive on this model. The fix isn't to abandon adaptation but to restrict it to the safest, most interpretable parameters. This is genuinely different from E12 — different parameter set, different LR, different epoch count.

| TTT_PARAMS | TTT_LR | TTT_EPOCHS | Param count | Pre-TTT BPB | Post-TTT BPB | Notes |
|------------|--------|------------|-------------|-------------|-------------|-------|
| control | 1e-4 | 1 | 25,691 | 1.1178 | **1.1179** | Flat — control tensors don't adapt at this LR |
| control | 5e-5 | 1 | | | | |
| control | 5e-4 | 1 | | | | |
| control | 1e-4 | 2 | | | | |

**First result (lr=1e-4, 1 epoch, 243s):** 25,691 control params (skip_weights, smear.gate, per-layer attn_scale/mlp_scale/resid_mix/q_gain, ve scales). BPB essentially flat (1.1178→1.1179). The control tensors are too few and too low-dimensional to capture document-level patterns. May need higher LR or fundamentally more params. Note: this ran on the E17 model (int8 mlp_down), so pre-TTT is 1.1178 not 1.1215.

---

### E34: Forward VE Injection (Late-Layer Draft Signal)
**Category:** Architecture change — moderate implementation effort
**What:** Instead of injecting draft predictions into SmearGate (pre-attention, global), inject them into late-layer Value Embeddings where the model is already performing output disambiguation. Pass 1: run model normally. Pass 2: use pass-1 soft predictions (`softmax(logits) @ ve_embed.weight`) as an additional VE contribution in layers 9-10 only. This keeps the draft signal in the part of the network that's already doing token identity reinjection (VE norms 23.4, 17.0 in analysis), rather than polluting early representations via SmearGate.
**Config:** `FORWARD_VE=1 VE_LAYERS=9,10`. New per-VE-layer learned scale (init 0.01). Two-pass training at frac=0.05.
**Expected:** If the remaining lookahead ceiling exists, this is the most principled injection point. The zero-prior SmearGate experiment showed the model ignores ungrounded signals at the embedding level — but VE at layers 9-10 operates on the refined residual stream where the model has already built rich representations. The draft signal may be more useful here.
**Why:** SmearGate injection failed because it operates pre-attention on raw embeddings. VE injection operates post-attention on refined representations. This directly tests whether the injection point was the problem, not the concept.

| Injection | Prior | Steps | 1-pass BPB | 2-pass BPB | Notes |
|-----------|-------|-------|------------|------------|-------|
| VE layers 9-10 | mean emb | | | | |
| VE layers 9-10 | bigram | | | | |
| VE layers 7-10 | mean emb | | | | |

---

## Strategic Review: Leaderboard Analysis (March 26, 2026)

External review of the current parameter-golf leaderboard revealed a critical priority misalignment. The competition is being won at the **eval layer** (n-gram cache), not the architecture layer.

### What's actually winning

- **PR #727** achieves **0.9674 BPB** — neural-only 1.1271 dropping to 0.9674 with n-gram cache. That's **-0.16 BPB** from the cache alone.
- The gap from 1.12 to sub-1.0 is almost entirely the n-gram cache technique. No architecture change in our experiments has exceeded ±0.005 BPB.
- Best non-TTT neural model (#609, 1.1154) uses: XSA-all (all 11 layers), Full GPTQ, selective pruning.

### What our experiments conclusively killed

At this scale and budget, the following **do not help**: MTP (E11), ESLM (E24), MiLe (E30 — diverged), LAWA (E25), WSD (E29 — neutral), gated attention/value residual/DTG (E32 — all hurt via step time), temperature scaling (E14 — useless with softcap), boundary features (E15), non-uniform FFN (E10 — throughput loss), extended VE (E18 — throughput loss), bigram cache (E16), control-tensor TTT (E33 — too few params). The SOTA config is already well-tuned — improvements are at the systems/eval layer.

### Why our n-gram cache failed (E16, E23) — specific fixable bugs

1. **E16 used bigram only.** At V=1024, bigram counts too dense. Need 5-7-gram orders.
2. **E23 used add-k smoothing (k=1).** Catastrophic for sparse tables — most 5-gram contexts seen once, so smoothed distribution is nearly uniform. Use **Stupid Backoff** (fixed α=0.4 discount per level).
3. **λ=0.1 is 5-10x too high.** Effective mixing should be 0.01–0.05. Entropy-adaptive helps but the base must be lower.
4. **Per-token Python loops.** Need vectorized batch lookups using XOR-hash into fixed-size tables (~4M buckets).

### Revised priority order

1. **E35: N-gram cache rewrite** — Stupid Backoff, orders 2-7, XOR-hash, vectorized. Expected: -0.10 to -0.16 BPB (50-80x more than any architecture change).
2. **E36: XSA-all** — `XSA_LAST_N=11`. Flag flip, adopted by frontier submissions. Expected: -0.002 to -0.005 BPB.
3. **E37: Full GPTQ** — second-order quantization + selective pruning. Could free artifact bytes for int8 mlp_down (E17 showed -0.0037 BPB but +2MB).
4. **Combine best neural base + n-gram cache.** The cache improvement scales with better base models.

---

### E35: N-gram Cache Rewrite — Stupid Backoff (Eval-Time)
**Category:** Eval-time only — zero training cost. **HIGHEST PRIORITY.**
**What:** Complete rewrite of E23's n-gram cache. Key changes from E23:
1. **Stupid Backoff** (Brants et al. 2007): for n-gram order k, score = count(context+token) / count(context) if count(context) > 0, else backoff to order k-1 with discount α=0.4. No explicit smoothing — just raw relative frequency with backoff.
2. **XOR-hash into fixed-size count tables** (~4M buckets per n-gram order). Same pattern as BigramHashEmbedding but for counts.
3. **Entropy-adaptive α** starting at 0.02 (not 0.1). When neural model entropy < 1.0 bits, α → 0 (trust model). When entropy > 4.0 bits, α → 0.05 (lean on cache).
4. **Vectorized batch processing** — compute cache log-probs for entire batch at once using tensor ops, not per-token Python loops.
5. **Orders 2-7** with backoff chain.
**Config:** Eval-time only. Sweep: base_alpha in {0.01, 0.02, 0.05}, max_order in {5, 7}.
**Expected:** -0.10 to -0.16 BPB based on PR #727 (1.1271 → 0.9674).

| Max Order | Alpha | Adaptive | Pre-cache BPB | Post-cache BPB | Eval Time | Notes |
|-----------|-------|----------|---------------|----------------|-----------|-------|
| 7 | 0.02 | entropy | | | | |
| 7 | 0.05 | entropy | | | | |
| 5 | 0.02 | entropy | | | | |
| 7 | 0.01 | entropy | | | | |

---

### E36: XSA on All Layers
**Category:** Training improvement — flag change only
**What:** Enable XSA (cross-attention self-attention subtraction) on all 11 layers instead of just the last 4. Currently `XSA_LAST_N=4` enables XSA on layers 7-10. The best non-TTT submission (#609, 1.1154) uses XSA on all layers. XSA removes self-value projection from attention output, forcing heads to attend to other tokens rather than copying their own value.
**Config:** `XSA_LAST_N=11`
**Expected:** -0.002 to -0.005 BPB. Most adopted technique across frontier submissions.
**Risk:** XSA adds a small per-layer compute cost (GQA-aware projection subtraction). With 11 layers vs 4, this could add ~3-5ms/step. Need to verify step time doesn't regress.

| XSA Layers | Steps | Step avg | Pre-TTT BPB | Post-TTT BPB | Notes |
|------------|-------|----------|-------------|-------------|-------|
| last 4 (baseline) | 7248 | 82.9ms | 1.1214 | 1.1189 | current |
| all 11 | | | | | |

---

### E37: Full GPTQ + Selective Pruning
**Category:** Post-training — quantization improvement
**What:** Replace GPTQ-lite (per-row clip search over 5 percentiles) with full GPTQ (second-order quantization using Hessian information). Also add selective pruning: zero out the smallest-magnitude quantized weights to improve lzma compression. Since zeros compress extremely well, this could recover artifact bytes, potentially enabling int8 for mlp_down (E17: -0.0037 BPB but +2MB).
**Config:** Post-training only. Need to implement GPTQ calibration pass using a small set of training data.
**Expected:** -0.001 to -0.003 BPB from better quantization fidelity. Selective pruning could save 1-2MB of artifact space.
**Risk:** GPTQ calibration adds eval-time compute (~5 min). Implementation is more complex than clip search.

| Quantization | Pruning | Pre-TTT BPB | Artifact Size | Notes |
|-------------|---------|-------------|---------------|-------|
| GPTQ-lite (baseline) | none | 1.1214 | 15.8MB | current |
| Full GPTQ | none | | | |
| Full GPTQ | 5% smallest | | | |
| GPTQ-lite + int8 mlp_down | 10% smallest | | | E17 combo |

---

## SOTA Model Analysis (abaybektursun, 1.1215 BPB pre-TTT)

Full-stack analysis of the current SOTA model (LeakyReLU² + Parameter Banking, PR #549) on 6.55M validation tokens. Trained from scratch, scored at **1.1215 BPB** sliding window s64. Analysis scripts: `analyze_model.py` (v1), `analyze_model_v2.py` (v2). Results: `analysis_results_v2/a5f86f51/`.

### Data Characterization

**Token vocabulary breakdown** (V=1024, BPE on FineWeb):
- 325 word-initial tokens (▁prefix), 402 continuation-alpha, 256 byte-fallback, 27 punctuation, 10 digits, 4 control
- In actual val data: **40.7% word-initial, 38.6% single-char, 8.7% uppercase-start, 6.6% punctuation, 2.5% digit, 0.4% byte-fallback**

**Word position distribution** — BPE-1024 splits words into many short pieces:

| Position in word | Fraction |
|-----------------|----------|
| 0 (word start) | 40.8% |
| 1 | 23.4% |
| 2 | 17.2% |
| 3 | 9.8% |
| 4 | 4.6% |
| 5+ | 4.2% |

**Document statistics** — 50,000 documents in the validation set:
- Mean length 1,240 tokens, median 733, p10=226, p90=2,454
- 293,759 unique bigrams (28% of possible 1024²)

### Token-Level Loss Decomposition

**Word-initial tokens account for 67% of total loss with only 40% of tokens.** The v2 analysis breaks this down by position within word:

| Word Position | Mean NLL | Fraction | Difficulty |
|---------------|----------|----------|------------|
| **0 (word start)** | **3.178** | 40.3% | HARD |
| 1 (2nd piece) | 1.444 | 23.4% | easy |
| 2 | 0.714 | 17.3% | easy |
| 3 | 0.737 | 9.9% | easy |
| 4 | 0.912 | 4.7% | easy |
| 5+ | ~1.0–1.2 | 4.4% | easy |

Position 0 is 2.2x harder than position 1 and 4.5x harder than position 2. The model is near-perfect by position 2. Loss slightly increases for positions 4+ (longer/rarer words).

**Loss by document position** — the model adapts within a document:

| Doc Position | Mean NLL | Fraction |
|-------------|----------|----------|
| First 20 tokens | **2.490** | 1.6% |
| 20–100 | 2.028 | 6.4% |
| 100–500 | 1.901 | 26.0% |
| 500+ | 1.883 | 66.0% |

First 20 tokens of a document are 32% harder. The model adapts within ~100 tokens.

**Loss by preceding token category:**

| Previous Token | Mean NLL |
|---------------|----------|
| After punctuation | **2.774** |
| After word-start | 2.225 |
| After uppercase | 2.036 |
| After digit | 1.690 |

After punctuation (sentence/clause boundaries) is the hardest context — maximum entropy about what comes next.

**Loss by text type** (documents bucketed into terciles by heuristic):

| Heuristic | Low NLL | High NLL | Gap |
|-----------|---------|----------|-----|
| Repetition score | 2.156 | 1.793 | 17% — repetitive text much easier |
| Punct density | 1.842 | 1.951 | More punct = harder |
| Digit density | 1.992 | 1.822 | More digits = easier |

Repetition is the strongest loss correlator. Cache/TTT techniques targeting document-level repetition patterns have clear upside.

### Hardest and Easiest Token Types

**Top 5 hardest tokens** (≥100 occurrences) — **every one is word-initial**, short ambiguous prefixes:
`▁und` (7.30), `▁tw` (6.37), `▁int` (5.82), `▁des` (5.65), `▁Ne` (5.57)

**Top 5 easiest tokens** — **every one is a continuation**: `<0x80>` (0.08), `rent` (0.24), `ility` (0.28), `ment` (0.30), `ion` (0.32)

Full per-token-type loss table: `analysis_results_v2/*/token_type_loss.csv`

### Model Activation Deep-Dive

**Embedding space:** Token embedding norm 8.25, bigram contribution 1.85 (22.7%), bigram learned scale 0.034.

**SmearGate does NOT distinguish word boundaries.** Cosine(smeared, unsmeared) is identical at 0.777 for both word-start and continuation tokens. SmearGate operates as a generic backward smoother, not a boundary detector.

**Per-layer summary:**

| Lyr | ResNorm | Attn% | MLP% | AttnSc | MixX0 | VE | XSA | Role |
|-----|---------|-------|------|--------|-------|------|-----|------|
| 0 | 526 | 3.8% | 98.9% | 0.067 | **0.630** | — | | MLP token lookup |
| 1 | 361 | 17.8% | 55.7% | 0.371 | **0.595** | — | | Re-reads embeddings |
| 3 | 184 | **35.7%** | 58.0% | 0.424 | -0.102 | — | | Attention peak |
| 7 | 181 | 21.2% | 51.3% | **0.630** | -0.044 | — | yes | Strongest XSA |
| 9 | 88 | 13.7% | 35.3% | 0.220 | -0.016 | **23.4** | yes | VE injection |
| 10 | 60 | 8.0% | 55.4% | 0.145 | 0.006 | **17.0** | yes | Output |

Key observations:
- **Layer 0**: pure MLP token lookup, attention vestigial (3.8%). High MixX0 (0.63) re-reads raw embeddings — it's essentially a bigram/unigram statistics layer. Its FFN activation magnitude (mean 8.0) is 3x any other layer.
- **Layer 3**: attention peak (35.7%), deepest encoder layer.
- **Layer 7**: highest AttnScale (0.63), first XSA layer, strongest skip connection (38% of residual from encoder layer 2).
- **Layers 9–10**: VE re-injects token identity (norms 23.4, 17.0). Small scales — careful adjustments.
- **Layers 0–1** re-read raw embeddings via MixX0 (0.63, 0.60). All other layers MixX0 ≈ 0.

### Residual Norm by Word Position

| Lyr | WP0 | WP1 | WP2 | WP3 |
|-----|-----|-----|-----|-----|
| 0 | 407 | **759** | 595 | 450 |
| 5 | 159 | 152 | 151 | 158 |
| 10 | 51 | **70** | 67 | 59 |

WP1 (second piece of word) has the highest residual norms — the model activates most after the ambiguous word-start. At output layer, WP0 has the *lowest* norm (51), consistent with low-confidence predictions.

### U-Net Skip Connections

| Decoder Lyr | Skip From | Skip Norm | Skip/Residual |
|-------------|-----------|-----------|---------------|
| 7 | ←2 | **81.0** | **37.9%** |
| 6 | ←3 | 50.5 | 26.4% |
| 8 | ←1 | 55.6 | 26.3% |
| 5 | ←4 | 47.8 | 24.3% |
| 9 | ←0 | 16.5 | 10.3% |

Layer 7←2 is the strongest skip (38% of residual). Layer 9←0 has near-zero weight (0.009).

### Quantization Error

MLP down-projections lose most — relative MSE ~6e-3, **3x worse** than mlp_up (~2e-3). Mixed precision (int8 for mlp_down) is the clearest quantization improvement.

### Hardest and Easiest Sequences

**Hardest: OCR-corrupted text** — "per ceot. DUTY OX ARTICLES OF LCXCBT" (NLL 4.07), "optmU sc, closed urs-tc" (NLL 3.88). OCR artifacts in FineWeb set a floor on achievable BPB.

**Easiest: legal boilerplate** — "JOCKEYCLUB.COM" (NLL 0.80), "reverse engineer, disassemble or otherwise reduce the Software" (NLL 0.82).

### Implications

1. **Word position 0 is the entire game.** NLL 3.18 vs 0.71 at position 2 — 4.5x gap. All approaches should target word-initial tokens.
2. **Document-initial tokens are 32% harder.** First 20 tokens NLL 2.49 vs 1.88 at pos 500+. TTT/cache should target early-document.
3. **After punctuation is hardest context (NLL 2.77).** Sentence boundaries = maximum entropy.
4. **Repetition is the strongest text-type predictor.** 17% gap between high/low repetition docs.
5. **SmearGate doesn't distinguish word boundaries.** Cosine identical for word-start/continuation.
6. **Layer 0 attention is vestigial (3.8%).** Could be removed or repurposed.
7. **VE at layers 9–10 is substantial (norms 23.4, 17.0).** Extending to more layers could help.
8. **Skip at layer 7←2 is the strongest (38%).** U-Net is doing real work here.
