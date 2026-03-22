# Universality and Formula Analysis Report

## Executive Summary

Structures ranked by universality score (higher = more compressible):

| Rank | Structure | Score | Prevalence | Consistency | Pattern Sim | Code Bytes |
|---|---|---|---|---|---|---|
| 1 | generic_spectral_template | 0.374 | 0.500 | 0.500 | 0.843 | 564 |
| 2 | mlp_spectral_init | 0.153 | 0.500 | 0.500 | 0.500 | 816 |
| 3 | copy_head_ov | 0.128 | 0.392 | 0.116 | 0.482 | 171 |
| 4 | previous_token_head | 0.037 | 0.061 | 0.440 | 0.567 | 411 |
| 5 | induction_head | 0.008 | 0.026 | 0.264 | 0.377 | 345 |

## Head Type Universality

### content

- **Prevalence**: mean=0.283, std=0.064
  - Per model: {'SmolLM2-135M': 0.2815, 'Pythia-70M': 0.3542, 'Qwen2.5-0.5B': 0.3155, 'GPT-2': 0.1806}
- **Layer position**:
  - SmolLM2-135M: {'early': 0.434, 'middle': 0.224, 'late': 0.342}
  - Pythia-70M: {'early': 0.588, 'middle': 0.294, 'late': 0.118}
  - Qwen2.5-0.5B: {'early': 0.387, 'middle': 0.189, 'late': 0.425}
  - GPT-2: {'early': 0.385, 'middle': 0.269, 'late': 0.346}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.2795, 'ov_decay_rate': 0.0878, 'qk_id_cos': 1.6261, 'ov_id_cos': 1.6121}
  - Pythia-70M: {'qk_decay_rate': 0.1873, 'ov_decay_rate': 0.0609, 'qk_id_cos': 5.5148, 'ov_id_cos': 1.2778}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.2394, 'ov_decay_rate': 0.1384, 'qk_id_cos': 2.0924, 'ov_id_cos': 2.4747}
  - GPT-2: {'qk_decay_rate': 0.224, 'ov_decay_rate': 0.1152, 'qk_id_cos': 6.7433, 'ov_id_cos': 1.6475}

### copy

- **Prevalence**: mean=0.392, std=0.057
  - Per model: {'SmolLM2-135M': 0.4222, 'Pythia-70M': 0.3125, 'Qwen2.5-0.5B': 0.369, 'GPT-2': 0.4653}
- **Layer position**:
  - SmolLM2-135M: {'early': 0.184, 'middle': 0.421, 'late': 0.395}
  - Pythia-70M: {'early': 0.067, 'middle': 0.267, 'late': 0.667}
  - Qwen2.5-0.5B: {'early': 0.29, 'middle': 0.444, 'late': 0.266}
  - GPT-2: {'early': 0.134, 'middle': 0.388, 'late': 0.478}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.2249, 'ov_decay_rate': 0.2952, 'qk_id_cos': 2.0353, 'ov_id_cos': 0.2841}
  - Pythia-70M: {'qk_decay_rate': 0.1604, 'ov_decay_rate': 0.3005, 'qk_id_cos': 104.0, 'ov_id_cos': 0.3257}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.1656, 'ov_decay_rate': 0.3319, 'qk_id_cos': 1.8101, 'ov_id_cos': 0.3004}
  - GPT-2: {'qk_decay_rate': 0.0923, 'ov_decay_rate': 0.3619, 'qk_id_cos': 11.3321, 'ov_id_cos': 0.3157}

### induction

- **Prevalence**: mean=0.026, std=0.033
  - Per model: {'SmolLM2-135M': 0.0037, 'Pythia-70M': 0.0833, 'Qwen2.5-0.5B': 0.0119, 'GPT-2': 0.0069}
- **Layer position**:
  - SmolLM2-135M: {'middle': 1.0}
  - Pythia-70M: {'middle': 0.5, 'late': 0.5}
  - Qwen2.5-0.5B: {'middle': 0.75, 'late': 0.25}
  - GPT-2: {'middle': 1.0}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.0, 'ov_decay_rate': 0.0, 'qk_id_cos': 0.0, 'ov_id_cos': 0.0}
  - Pythia-70M: {'qk_decay_rate': 0.0552, 'ov_decay_rate': 0.0409, 'qk_id_cos': 0.6533, 'ov_id_cos': 29.9118}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.0732, 'ov_decay_rate': 0.0522, 'qk_id_cos': 0.5136, 'ov_id_cos': 13.4225}
  - GPT-2: {'qk_decay_rate': 0.0, 'ov_decay_rate': 0.0, 'qk_id_cos': 0.0, 'ov_id_cos': 0.0}

### positional

- **Prevalence**: mean=0.028, std=0.017
  - Per model: {'SmolLM2-135M': 0.0037, 'Pythia-70M': 0.0208, 'Qwen2.5-0.5B': 0.0476, 'GPT-2': 0.0417}
- **Layer position**:
  - SmolLM2-135M: {'middle': 1.0}
  - Pythia-70M: {'early': 1.0}
  - Qwen2.5-0.5B: {'early': 0.312, 'middle': 0.438, 'late': 0.25}
  - GPT-2: {'early': 0.833, 'middle': 0.167}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.0, 'ov_decay_rate': 0.0, 'qk_id_cos': 0.0, 'ov_id_cos': 0.0}
  - Pythia-70M: {'qk_decay_rate': 0.0, 'ov_decay_rate': 0.0, 'qk_id_cos': 0.0, 'ov_id_cos': 0.0}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.2829, 'ov_decay_rate': 0.8777, 'qk_id_cos': 0.4915, 'ov_id_cos': 0.2634}
  - GPT-2: {'qk_decay_rate': 0.2105, 'ov_decay_rate': 0.5012, 'qk_id_cos': 0.2254, 'ov_id_cos': 5.7765}

### previous_token

- **Prevalence**: mean=0.061, std=0.027
  - Per model: {'SmolLM2-135M': 0.0333, 'Pythia-70M': 0.1042, 'Qwen2.5-0.5B': 0.0446, 'GPT-2': 0.0625}
- **Layer position**:
  - SmolLM2-135M: {'early': 0.222, 'middle': 0.667, 'late': 0.111}
  - Pythia-70M: {'early': 0.2, 'middle': 0.4, 'late': 0.4}
  - Qwen2.5-0.5B: {'early': 0.6, 'middle': 0.267, 'late': 0.133}
  - GPT-2: {'early': 0.778, 'middle': 0.222}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.0708, 'ov_decay_rate': 0.1233, 'qk_id_cos': 0.9926, 'ov_id_cos': 1.3291}
  - Pythia-70M: {'qk_decay_rate': 0.2139, 'ov_decay_rate': 0.23, 'qk_id_cos': 1.4637, 'ov_id_cos': 1.0963}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.3217, 'ov_decay_rate': 0.1673, 'qk_id_cos': 4.2558, 'ov_id_cos': 3.2119}
  - GPT-2: {'qk_decay_rate': 0.2455, 'ov_decay_rate': 0.2405, 'qk_id_cos': 0.5914, 'ov_id_cos': 5.8469}

### suppression

- **Prevalence**: mean=0.209, std=0.051
  - Per model: {'SmolLM2-135M': 0.2556, 'Pythia-70M': 0.125, 'Qwen2.5-0.5B': 0.2113, 'GPT-2': 0.2431}
- **Layer position**:
  - SmolLM2-135M: {'early': 0.493, 'middle': 0.246, 'late': 0.261}
  - Pythia-70M: {'early': 0.5, 'middle': 0.5}
  - Qwen2.5-0.5B: {'early': 0.296, 'middle': 0.324, 'late': 0.38}
  - GPT-2: {'early': 0.486, 'middle': 0.314, 'late': 0.2}
- **Weight consistency (CV)**:
  - SmolLM2-135M: {'qk_decay_rate': 0.2165, 'ov_decay_rate': 0.2927, 'qk_id_cos': 1.6298, 'ov_id_cos': 0.5601}
  - Pythia-70M: {'qk_decay_rate': 0.0944, 'ov_decay_rate': 0.1319, 'qk_id_cos': 2.9673, 'ov_id_cos': 0.5083}
  - Qwen2.5-0.5B: {'qk_decay_rate': 0.1484, 'ov_decay_rate': 0.3205, 'qk_id_cos': 1.5162, 'ov_id_cos': 0.4731}
  - GPT-2: {'qk_decay_rate': 0.2724, 'ov_decay_rate': 0.2571, 'qk_id_cos': 18.0933, 'ov_id_cos': 0.4342}

## MLP Structure Universality

| Model | Up Rank | Up Slope | Prod Rank | Prod Slope |
|---|---|---|---|---|
| SmolLM2-135M | 532.4 | -17.4 | 414.2 | -15.5 |
| Pythia-70M | 469.2 | -23.2 | 356.7 | -59.4 |
| Qwen2.5-0.5B | 859.1 | -19.4 | 681.9 | -41.0 |
| GPT-2 | 700.9 | -2.5 | 511.3 | 42.6 |

## Parametric Formulas

### previous_token_head

**Previous-token head: QK biased toward shifted-diagonal, OV near identity with decay**

Parameters: {
  "qk_id_cos": -0.1814,
  "qk_norm_scale": 52.527,
  "ov_id_cos": 0.1982,
  "ov_decay_rate": 0.0522
}

```python
def init_previous_token_head(hd, qk_scale=52.5270, ov_decay=0.0522):
    # QK circuit: shifted diagonal (attend to i-1)
    qk = torch.zeros(hd, hd)
    for i in range(1, hd):
        qk[i, i-1] = 1.0
    qk = qk * qk_scale / (hd ** 0.5)
    # OV circuit: identity with exponential SV decay
    sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
    ov = torch.diag(sigmas)
    return qk, ov
```

### induction_head

**Induction head: QK near identity (match tokens), OV near identity (copy)**

Parameters: {
  "qk_id_cos": 0.3106,
  "qk_norm_scale": 54.6861,
  "ov_id_cos": 0.083,
  "ov_decay_rate": 0.0572
}

```python
def init_induction_head(hd, beta=54.6861, gamma=0.0830, ov_decay=0.0572):
    # QK circuit: identity-like (attend to matching tokens)
    qk = beta * torch.eye(hd)
    # OV circuit: identity with decay (copy content)
    sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
    ov = gamma * torch.diag(sigmas)
    return qk, ov
```

### copy_head_ov

**Copy head: OV circuit is scaled identity with exponential SV decay**

Parameters: {
  "ov_scale": 0.6448,
  "ov_decay_rate": 0.0447
}

```python
def init_copy_ov(hd, delta=0.6448, decay=0.0447):
    sigmas = torch.exp(-decay * torch.arange(hd, dtype=torch.float32))
    ov = delta * torch.diag(sigmas)
    return ov
```

### generic_spectral_template

**Generic head: exponential SV decay for both QK and OV circuits**

Parameters: {
  "qk_decay_rate": 0.053,
  "ov_decay_rate": 0.0379,
  "head_dim": 64
}

```python
def init_generic_head(hd, d_model, r_qk=0.0530, r_ov=0.0379):
    # Random orthogonal bases
    U_qk = torch.linalg.qr(torch.randn(hd, hd))[0]
    V_qk = torch.linalg.qr(torch.randn(hd, hd))[0]
    U_ov = torch.linalg.qr(torch.randn(hd, hd))[0]
    V_ov = torch.linalg.qr(torch.randn(hd, hd))[0]
    # Exponential SV profiles
    s_qk = torch.exp(-r_qk * torch.arange(hd, dtype=torch.float32))
    s_ov = torch.exp(-r_ov * torch.arange(hd, dtype=torch.float32))
    qk = U_qk @ torch.diag(s_qk) @ V_qk.T
    ov = U_ov @ torch.diag(s_ov) @ V_ov.T
    return qk, ov
```

### mlp_spectral_init

**MLP initialization with matched spectral profile**

Parameters: {
  "mean_neuron_gini": 0.0578,
  "mean_up_eff_rank": 664.13
}

```python
def init_mlp_spectral(d_model, hidden_dim, eff_rank=664.1):
    # Initialize with target effective rank via SV shaping
    # Generate random matrix and reshape its spectrum
    U = torch.linalg.qr(torch.randn(hidden_dim, min(hidden_dim, d_model)))[0]
    V = torch.linalg.qr(torch.randn(d_model, min(hidden_dim, d_model)))[0]
    k = min(hidden_dim, d_model)
    # Decay rate chosen to match target effective rank
    # eff_rank = sum(exp(-r*i))^2 / sum(exp(-2r*i)) ~ 1/(1-exp(-r)) for large k
    # Solve: r ~ log(k / eff_rank) / k
    r = max(0.001, np.log(k / eff_rank) / k)
    sigmas = torch.exp(-r * torch.arange(k, dtype=torch.float32))
    sigmas = sigmas * (d_model ** -0.5)  # standard scaling
    W_up = (U @ torch.diag(sigmas) @ V.T).T  # (hidden_dim, d_model) -> (d_model, hidden_dim).T
    return W_up
```

## Validation Results (Pythia-70M)

| Formula | Attn Cosine Sim | Logit KL Div |
|---|---|---|
| previous_token_head | 0.5668 | 1.840721 |
| induction_head | 0.3772 | 0.319377 |
| copy_head_ov | 0.4820 | 1.469702 |
| generic_spectral_template | 0.8428 | 1.403897 |

### previous_token_head
- Test layer: 1, heads: [0, 1, 2, 3]
- Per-head attention cosine similarity: [0.5979, 0.5688, 0.5241, 0.5765]

### induction_head
- Test layer: 1, heads: [0, 1, 2, 3]
- Per-head attention cosine similarity: [0.3353, 0.4068, 0.3208, 0.4459]

### copy_head_ov
- Test layer: 1, heads: [0, 1, 2, 3]
- Per-head attention cosine similarity: [0.478, 0.5175, 0.4075, 0.5252]

### generic_spectral_template
- Test layer: 1, heads: [0, 1, 2, 3]
- Per-head attention cosine similarity: [0.7779, 0.8344, 0.868, 0.8908]

## Recommended Initialization for Parameter-Golf

Based on the analysis, the recommended init scheme combines:

1. **generic_spectral_template** (score=0.374): Generic head: exponential SV decay for both QK and OV circuits
2. **mlp_spectral_init** (score=0.153): MLP initialization with matched spectral profile
3. **copy_head_ov** (score=0.128): Copy head: OV circuit is scaled identity with exponential SV decay

Key parameters for `train_gpt.py` circuit-aware init:
```python
# Derived from cross-model analysis
# previous_token_head.qk_id_cos = -0.1814
# previous_token_head.qk_norm_scale = 52.527
# previous_token_head.ov_id_cos = 0.1982
# previous_token_head.ov_decay_rate = 0.0522
# induction_head.qk_id_cos = 0.3106
# induction_head.qk_norm_scale = 54.6861
# induction_head.ov_id_cos = 0.083
# induction_head.ov_decay_rate = 0.0572
# copy_head_ov.ov_scale = 0.6448
# copy_head_ov.ov_decay_rate = 0.0447
# generic_spectral_template.qk_decay_rate = 0.053
# generic_spectral_template.ov_decay_rate = 0.0379
# generic_spectral_template.head_dim = 64
# mlp_spectral_init.mean_neuron_gini = 0.0578
# mlp_spectral_init.mean_up_eff_rank = 664.13
```
