# Head Behavior Analysis Report

## Head Type Distribution Across Models

| Model | content | copy | induction | positional | previous_token | suppression | Total |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 76 | 114 | 1 | 1 | 9 | 69 | 270 |
| Pythia-70M | 17 | 15 | 4 | 1 | 5 | 6 | 48 |
| Qwen2.5-0.5B | 106 | 124 | 4 | 16 | 15 | 71 | 336 |
| GPT-2 | 26 | 67 | 1 | 6 | 9 | 35 | 144 |

## SmolLM2-135M

- Layers: 30, Heads: 9
- Total heads: 270

### Top Induction Heads
- L17H5: induction=0.439, prev_token=0.059, type=induction
- L17H4: induction=0.262, prev_token=0.021, type=content
- L18H6: induction=0.261, prev_token=0.021, type=copy
- L20H6: induction=0.210, prev_token=0.028, type=suppression
- L23H0: induction=0.194, prev_token=0.049, type=suppression

### Top Previous-Token Heads
- L16H0: prev_token=0.790, positional=0.919, type=previous_token
- L11H1: prev_token=0.740, positional=0.891, type=previous_token
- L2H1: prev_token=0.607, positional=0.822, type=previous_token
- L1H2: prev_token=0.606, positional=0.800, type=previous_token
- L19H3: prev_token=0.503, positional=0.721, type=previous_token

### Top Copy Heads (high OV identity cosine)
- L27H1: ov_id_cos=0.971, ov_eig_pos=1.000, type=copy
- L28H5: ov_id_cos=0.951, ov_eig_pos=1.000, type=copy
- L29H1: ov_id_cos=0.951, ov_eig_pos=1.000, type=copy
- L29H2: ov_id_cos=0.950, ov_eig_pos=1.000, type=copy
- L26H4: ov_id_cos=0.944, ov_eig_pos=1.000, type=copy

### Score Statistics
- prev_token_score: mean=0.099, std=0.100, min=0.020, max=0.790
- induction_score: mean=0.036, std=0.045, min=0.000, max=0.439
- positional_score: mean=0.133, std=0.164, min=0.018, max=0.934
- entropy: mean=1.399, std=0.531, min=0.000, max=2.804
- ov_identity_cos: mean=0.179, std=0.500, min=-0.972, max=0.971
- ov_eig_pos_frac: mean=0.615, std=0.307, min=0.000, max=1.000

## Pythia-70M

- Layers: 6, Heads: 8
- Total heads: 48

### Top Induction Heads
- L4H2: induction=0.350, prev_token=0.037, type=induction
- L3H6: induction=0.335, prev_token=0.036, type=induction
- L4H1: induction=0.332, prev_token=0.069, type=induction
- L3H0: induction=0.305, prev_token=0.053, type=induction
- L4H7: induction=0.245, prev_token=0.036, type=copy

### Top Previous-Token Heads
- L2H1: prev_token=0.778, positional=0.921, type=previous_token
- L5H7: prev_token=0.411, positional=0.402, type=previous_token
- L0H1: prev_token=0.398, positional=0.788, type=previous_token
- L2H4: prev_token=0.339, positional=0.463, type=previous_token
- L5H1: prev_token=0.320, positional=0.200, type=previous_token

### Top Copy Heads (high OV identity cosine)
- L5H3: ov_id_cos=0.956, ov_eig_pos=0.984, type=copy
- L5H4: ov_id_cos=0.928, ov_eig_pos=1.000, type=copy
- L5H2: ov_id_cos=0.880, ov_eig_pos=0.984, type=copy
- L5H6: ov_id_cos=0.877, ov_eig_pos=0.969, type=copy
- L5H1: ov_id_cos=0.777, ov_eig_pos=0.984, type=previous_token

### Score Statistics
- prev_token_score: mean=0.154, std=0.138, min=0.029, max=0.778
- induction_score: mean=0.073, std=0.098, min=0.000, max=0.350
- positional_score: mean=0.252, std=0.200, min=0.025, max=0.921
- entropy: mean=1.352, std=0.789, min=0.241, max=2.693
- ov_identity_cos: mean=0.210, std=0.388, min=-0.703, max=0.956
- ov_eig_pos_frac: mean=0.634, std=0.232, min=0.016, max=1.000

## Qwen2.5-0.5B

- Layers: 24, Heads: 14
- Total heads: 336

### Top Induction Heads
- L9H13: induction=0.439, prev_token=0.022, type=induction
- L11H12: induction=0.370, prev_token=0.021, type=induction
- L11H10: induction=0.341, prev_token=0.025, type=induction
- L16H2: induction=0.314, prev_token=0.046, type=induction
- L9H10: induction=0.286, prev_token=0.020, type=content

### Top Previous-Token Heads
- L8H7: prev_token=0.881, positional=0.957, type=previous_token
- L1H9: prev_token=0.745, positional=0.904, type=previous_token
- L15H2: prev_token=0.729, positional=0.866, type=previous_token
- L0H11: prev_token=0.657, positional=0.884, type=previous_token
- L0H13: prev_token=0.647, positional=0.879, type=previous_token

### Top Copy Heads (high OV identity cosine)
- L14H7: ov_id_cos=0.969, ov_eig_pos=1.000, type=copy
- L18H12: ov_id_cos=0.967, ov_eig_pos=1.000, type=copy
- L14H9: ov_id_cos=0.956, ov_eig_pos=1.000, type=copy
- L5H8: ov_id_cos=0.948, ov_eig_pos=1.000, type=copy
- L5H12: ov_id_cos=0.944, ov_eig_pos=0.984, type=copy

### Score Statistics
- prev_token_score: mean=0.099, std=0.113, min=0.001, max=0.881
- induction_score: mean=0.044, std=0.052, min=0.000, max=0.439
- positional_score: mean=0.186, std=0.235, min=0.018, max=1.000
- entropy: mean=1.494, std=0.628, min=0.000, max=2.849
- ov_identity_cos: mean=0.113, std=0.533, min=-0.992, max=0.969
- ov_eig_pos_frac: mean=0.578, std=0.315, min=0.000, max=1.000

## GPT-2

- Layers: 12, Heads: 12
- Total heads: 144

### Top Induction Heads
- L5H1: induction=0.312, prev_token=0.020, type=induction
- L5H5: induction=0.292, prev_token=0.025, type=copy
- L5H0: induction=0.244, prev_token=0.044, type=copy
- L7H10: induction=0.206, prev_token=0.023, type=copy
- L5H8: induction=0.205, prev_token=0.044, type=suppression

### Top Previous-Token Heads
- L4H11: prev_token=0.996, positional=0.980, type=previous_token
- L2H2: prev_token=0.529, positional=0.744, type=previous_token
- L3H7: prev_token=0.415, positional=0.634, type=previous_token
- L3H2: prev_token=0.396, positional=0.607, type=previous_token
- L2H9: prev_token=0.359, positional=0.692, type=previous_token

### Top Copy Heads (high OV identity cosine)
- L11H3: ov_id_cos=0.973, ov_eig_pos=1.000, type=copy
- L11H10: ov_id_cos=0.959, ov_eig_pos=1.000, type=copy
- L9H10: ov_id_cos=0.949, ov_eig_pos=1.000, type=copy
- L11H11: ov_id_cos=0.947, ov_eig_pos=0.984, type=copy
- L10H9: ov_id_cos=0.943, ov_eig_pos=1.000, type=copy

### Score Statistics
- prev_token_score: mean=0.108, std=0.120, min=0.003, max=0.996
- induction_score: mean=0.043, std=0.051, min=0.000, max=0.312
- positional_score: mean=0.177, std=0.224, min=0.016, max=0.980
- entropy: mean=1.489, std=0.614, min=0.016, max=2.938
- ov_identity_cos: mean=0.192, std=0.556, min=-0.973, max=0.973
- ov_eig_pos_frac: mean=0.613, std=0.335, min=0.000, max=1.000

## Cross-Model Comparison

### Induction Score by Layer Position (normalized)

- **SmolLM2-135M**: peak induction at layer 17/30 (pos=0.59, score=0.120)
- **Pythia-70M**: peak induction at layer 4/6 (pos=0.80, score=0.204)
- **Qwen2.5-0.5B**: peak induction at layer 16/24 (pos=0.70, score=0.107)
- **GPT-2**: peak induction at layer 5/12 (pos=0.45, score=0.101)
