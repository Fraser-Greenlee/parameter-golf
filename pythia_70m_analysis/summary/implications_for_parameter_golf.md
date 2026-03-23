# Implications for Parameter Golf Init Strategy

## What Pythia-70M Teaches Us About Transformer Structure

### 1. The Induction Circuit Architecture

The dominant learned circuit is:
```
L2_H1 (98% prev-token) --> L3_H0, L3_H6 (induction heads)
                        --> L4_H1, L4_H2, L4_H4 (refined induction)
```

**Observation**: The model devotes 1 head to pure previous-token attention and 5 heads to induction across 2 layers. This is the single most important circuit -- it enables in-context learning by predicting "what followed this token last time it appeared."

**Implication for our 9-layer model**: We should dedicate:
- 1 head in layers 1-2 as a hard previous-token head (our current identity_qk init partially does this)
- 2-3 heads in layers 3-5 biased toward induction behavior

**How to init induction**: The previous-token head needs QK circuit that attends to i-1 (positional, via RoPE). Our earlier P2 RoPE-offset construction was correct in principle but hurt because it replaced too many heads. Limit it to 1 head per KV group in early layers.

### 2. BOS-Sink Heads are Induction Heads in Disguise

**Critical finding**: Many heads classified as "BOS-sinks" (L3_H0, L3_H6, L4_H1, L4_H2) are actually induction heads that default to BOS when no matching context exists. BOS is the "null attention" target.

**Implication**: Our full_circuit init with identity QK (attend to similar tokens) may accidentally prevent this BOS-fallback behavior. We should ensure some heads can attend to position 0 when no good content match exists. This could mean:
- Not forcing all heads toward identity QK
- Leaving some heads with low-norm QK weights (so they naturally produce flat/BOS-biased attention)

### 3. Layer 5 is All Copy Heads

All 8 heads in Pythia-70M's final layer (L5) have OV identity cosine 0.70-0.96 and very low entropy (0.17-0.36). They function purely as "copy what I attend to" output heads.

**Implication**: Our `copy_ov` init component is validated for the final 2-3 layers. But we should make it **stronger** in the last layer -- Pythia-70M's L5 heads have OV identity up to 0.96, much higher than our current init of ~0.4.

**Specific change**: In our `full_circuit` init, ramp OV identity from 0.0 in early layers to 0.8-0.9 in the last layer (quadratic ramp). Our earlier depth_circuit experiment tried this but also reduced beta_qk, which hurt. Keep beta_qk at 0.3 (identity QK) AND increase OV identity in later layers.

### 4. Entropy Decreases Sharply Through Layers

| Layer | Mean Entropy |
|-------|-------------|
| 0 | 2.16 |
| 1 | 2.09 |
| 2 | 1.92 |
| 3 | 1.36 |
| 4 | 0.60 |
| 5 | 0.24 |

**Implication**: Early layers should have distributed attention (explore many positions), late layers should have sharp attention (commit to one position). This could be encoded at init:
- Early layers: smaller QK dot products (softer attention distribution)
- Late layers: larger QK dot products (sharper attention via temperature effect)

**Specific change**: Scale the `q_gain` parameter by layer. Current full_circuit uses uniform gain. Use `q_gain = 1.0 + 2.0 * (layer / n_layers)^2` to produce softer early attention and sharper late attention.

### 5. Suppression Heads are Important

L1_H0 (ov_id=-0.40), L2_H2 (-0.53), L2_H6 (-0.70) are **suppression heads** that actively reduce the probability of specific tokens. They attend to distant/BOS positions and write negative OV contributions.

**Implication**: Our current init makes all OV circuits positive (identity-like copy). We should init some heads with **negative** OV identity (suppression) in layers 1-3. Specifically:
- 1-2 heads per layer in layers 1-3 with OV ~ -0.3 * I (suppression)
- These should have high-entropy QK (attend broadly) since they're suppressing common predictions

### 6. FFN Needs No Special Init

All 2048 neurons per layer are active (no dead neurons). Activation magnitudes increase 7x from L0 to L5. The FFN learns basic token detection in L0 and output shaping in L5 automatically.

**Implication**: FFN weight init is less important than attention init. The standard Kaiming/He initialization is fine. The model quickly learns token detectors in L0 regardless of init.

### 7. Previous-Token Signal is Provided Redundantly

6 heads across layers 0-2 provide previous-token information (L0_H1 72%, L0_H7 43%, L2_H1 98%, etc.). The model is very redundant in establishing this basic signal.

**Implication**: Our single hard previous-token head init is sufficient -- the model will develop additional soft previous-token heads during training as needed. Don't over-allocate heads to this.

## Recommended Init Changes

### Proposed `enhanced_circuit` init

Based on these findings, modify the full_circuit recipe:

1. **Keep**: Identity QK bias (beta_qk=0.3) for most heads
2. **Keep**: Orthogonal subspaces, symmetry-breaking noise
3. **Change**: Ramp OV identity by depth:
   - Layers 0-2: `ov_identity = 0.0` (let these develop freely)
   - Layers 3-5: `ov_identity = 0.2`
   - Layers 6-8: `ov_identity = 0.6` (strong copy in final layers)
4. **Add**: Q-gain scaling by depth:
   - `q_gain = 1.0 + 2.0 * (layer/n_layers)^2`
   - This produces softer early attention, sharper late attention
5. **Add**: 1-2 suppression heads per layer in layers 1-4:
   - `ov_identity = -0.3` for these heads
   - Leave QK at default (not identity-biased)
6. **Keep identity QK**: Despite Pythia-70M having near-zero QK identity cosine, our 200-step experiments showed identity QK helps at our training duration. The model doesn't have time to learn content-specific QK circuits in 200 steps, so the identity bias is still useful.

### Priority Order

1. **OV depth ramp** (most impactful -- matches the strongest signal in Pythia-70M)
2. **Q-gain scaling** (encodes the entropy progression)
3. **Suppression heads** (novel, unvalidated, test separately)
4. Keep bigram SVD embeddings (our biggest single win, orthogonal to attention init)
