# Pythia-70M FFN Layer Analysis

Based on post-GELU activation analysis across 30 text samples (2200 token positions).

## Layer-by-Layer FFN Statistics

| Layer | Active (>10%) | Sparse (1-5%) | Dead (<1%) | Mean Fire Rate | Max Activation |
|-------|---------------|---------------|------------|----------------|----------------|
| 0 | **2048/2048** | 0 | 0 | 45.1% | 4.25 |
| 1 | 2048/2048 | 0 | 0 | 53.1% | ~6 |
| 2 | 2047/2048 | 0 | 0 | 55.2% | ~8 |
| 3 | 2041/2048 | 1 | 0 | 67.5% | 6.31 |
| 4 | 2043/2048 | 1 | 0 | 61.8% | ~8 |
| 5 | 2043/2048 | 2 | 0 | **66.3%** | **29.11** |

**Key trend**: Activations get dramatically stronger in later layers. L5 max activations (25-29) are 7x larger than L0 max (4). This makes sense -- later layers produce the final logit contributions and need strong signals.

**Sparsity**: Almost no dead neurons across any layer. The model fully utilizes its 2048 FFN neurons per layer. This contrasts with larger models that often have 5-10% dead neurons.

## Notable Neuron Types

### Type 1: Token Detectors (fire on specific tokens)

**Layer 0** -- basic token recognizers:
- N1518: "in" detector (fires max 3.24 on " in")
- N447: "is" detector (fires max 3.77 on " is")
- N306: Comma detector (fires max 3.08 on ",")
- N1561: Parenthesis detector (fires max 4.09 on " (")

**Layer 5** -- sentence boundary detectors:
- N297: Period/end-of-sentence (fires 29.11 on ".", 100% fire rate)
- N348: Day/time nouns (fires 27.65 on " day", " series")

### Type 2: Semantic Category Detectors

**Layer 0**:
- N780: **Plural nouns** (fires on: molecules, objects, communities, regions, envelopes, areas, phases, ships). Output boosts: word-initial subwords (hip, ource, ystem).
- N162: **Technical nouns** (fires on: cache, leak, poles, input, size, queries)
- N1277: **Numbers** (fires on: 27, 10, 120, 350, 15, 30)

**Layer 3**:
- N1076: **Code syntax** (fires on: ), =, ;, (, "). Code structure detector.
- N1348: **Motion verbs** (fires on: moves, weaken, guiding, read)

**Layer 5**:
- N73: **Adjectives/restarts** (fires on: quick, The, Another, is, and)
- N1304: **Content nouns** (fires on: log, circular, mat, bench, some)

### Type 3: Structural/Positional Neurons

**Layer 0**:
- N1511: **Newline detector** (fires max 1.69 on newline characters, 91% rate)
- N1530: **Conjunction detector** (fires on: or, and). Output boosts discourse connectives (hence, consequently)

**Layer 3**:
- N1745: **Indentation/whitespace** (fires on: "        ", right, next)
- N1309: **Code structure** (fires on: indentation, plt, newlines, parentheses)

### Type 4: Context-Dependent Neurons

**Layer 5**:
- N98: **Sentence completion** (fires 100%, max 25.14 on " replied", ');', " reason", " said"). Fires strongest at sentence/clause boundaries.
- N407: **Prepositional context** (fires max 13.31 on " with", ",", " period", " of")
- N1516: **Reported speech / volition** (fires on: warned, agree, roll, would, signaled)

## FFN Output Effects

The output boost/suppress patterns show what each neuron does to the logit distribution:

**Common pattern**: Most neurons suppress whitespace/padding tokens (the "output suppresses" column is dominated by whitespace tokens in later layers). This makes sense -- the default high-probability prediction is common tokens, and specialized neurons suppress those to make room for the specific tokens they detect.

**Layer 0 examples**:
- N780 (plural nouns): When it fires, it boosts subword-initial tokens like "hip" (+0.233), "ource" (+0.214), "ystem" (+0.211). These are continuations of compound words.
- N1518 (" in"): Suppresses " by" (-0.195), "ins" (-0.194), "ination" (-0.191). When "in" is detected, it suppresses alternative prepositions and "in-" prefix words.

**Layer 5 examples**:
- N297 (sentence end): fires on "." with activation 29.11. At sentence boundaries, this neuron massively shifts the logit distribution.
- N404 (plural/action): fires on "spaces", "issues", ".". Boosts continuation tokens like " should" (+1.229).

## Implications

1. **No dead neurons** at 70M scale -- the model needs all 2048 neurons per layer. At our competition scale (~3M params, 1024 FFN neurons), we should expect similar full utilization.

2. **Activation magnitude increases dramatically through layers** (4x at L0, 29x at L5). This suggests the FFN's role shifts from feature detection (L0) to output distribution shaping (L5).

3. **Layer 0 FFN learns basic token identity** (comma, period, "in", "is", numbers, plural nouns). These are the features that the attention heads in layers 1+ will read from.

4. **Layer 5 FFN acts as the final logit adjustment** with very strong activations. It's essentially doing the heavy lifting of converting attended information into the right output distribution.

5. **Pythia-70M's parallel residual architecture** means attention and FFN at the same layer DON'T see each other's output. Our competition model uses serial residual (FFN after attention), which means our FFN can condition on attention output -- a potential advantage.
