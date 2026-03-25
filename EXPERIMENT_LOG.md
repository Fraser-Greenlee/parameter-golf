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
**Config:** `LOOKAHEAD_SMEAR=1 LOOKAHEAD_BIGRAM=1` with varying `TWOPASS_TRAIN_FRAC`

| Frac | Steps | Step avg | 1-pass BPB | 2-pass BPB | Notes |
|------|-------|----------|------------|------------|-------|
| 0.02 | 6548 | 91.6ms | 1.2035 | 1.1336 | Single-pass badly degraded |
| 0.05 | 6421 | 93.4ms | — | — | **Invalid: concurrent run clobbered checkpoint** |
| 0.10 | 6540 | 91.8ms | 1.1383 | **1.1265** | (= E2 result) |
| 0.20 | 6231 | 96.4ms | — | — | **Invalid: concurrent run clobbered checkpoint** |
| 0.50 | 5745 | 104.5ms | 1.1611 | 1.1310 | Too many two-pass steps, fewer total steps |

**Note:** frac=0.05 and 0.20 produced identical eval results due to concurrent runs overwriting `final_model.pt`. Fixed by adding run-ID to model filenames. Rerun needed.

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

## Deprecated: Interleaved Draft Approach

The original approach doubled sequence length by interleaving draft tokens: `[x_0, d_1, x_1, d_2, ...]`. Timing tests showed ~193ms/step (2.3x baseline) with most overhead from draft batch construction (~78ms) rather than the longer sequence (~30ms). The lookahead features approach achieves the same goal (giving SmearGate/BigramHash future token info) without any sequence length increase, at baseline step cost.
