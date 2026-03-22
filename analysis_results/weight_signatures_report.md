# Weight Signature Analysis Report

## Canonical Spectral Signatures by Head Type

### content

**SmolLM2-135M** (76 heads):
- QK: decay=0.086+-0.024, eff_rank=36.3, id_cos=0.125, eig_pos=0.532
- OV: decay=0.058+-0.005, eff_rank=46.5, id_cos=0.087, eig_pos=0.564
- Layer position: 0.45+-0.32

**Pythia-70M** (17 heads):
- QK: decay=0.068+-0.013, eff_rank=41.8, id_cos=0.054, eig_pos=0.545
- OV: decay=0.054+-0.003, eff_rank=48.8, id_cos=0.092, eig_pos=0.557
- Layer position: 0.28+-0.29

**Qwen2.5-0.5B** (106 heads):
- QK: decay=0.087+-0.021, eff_rank=34.8, id_cos=0.105, eig_pos=0.544
- OV: decay=0.061+-0.008, eff_rank=45.2, id_cos=0.063, eig_pos=0.551
- Layer position: 0.52+-0.37

**GPT-2** (26 heads):
- QK: decay=0.049+-0.011, eff_rank=50.5, id_cos=0.064, eig_pos=0.530
- OV: decay=0.055+-0.006, eff_rank=47.6, id_cos=0.080, eig_pos=0.570
- Layer position: 0.47+-0.34

### copy

**SmolLM2-135M** (114 heads):
- QK: decay=0.076+-0.017, eff_rank=39.9, id_cos=0.094, eig_pos=0.538
- OV: decay=0.048+-0.014, eff_rank=50.8, id_cos=0.643, eig_pos=0.898
- Layer position: 0.59+-0.27

**Pythia-70M** (15 heads):
- QK: decay=0.080+-0.013, eff_rank=31.2, id_cos=0.001, eig_pos=0.517
- OV: decay=0.043+-0.013, eff_rank=53.4, id_cos=0.628, eig_pos=0.884
- Layer position: 0.77+-0.24

**Qwen2.5-0.5B** (124 heads):
- QK: decay=0.077+-0.013, eff_rank=39.9, id_cos=0.105, eig_pos=0.543
- OV: decay=0.047+-0.016, eff_rank=51.3, id_cos=0.649, eig_pos=0.899
- Layer position: 0.50+-0.24

**GPT-2** (67 heads):
- QK: decay=0.051+-0.005, eff_rank=50.3, id_cos=0.028, eig_pos=0.510
- OV: decay=0.041+-0.015, eff_rank=53.8, id_cos=0.659, eig_pos=0.888
- Layer position: 0.64+-0.28

### induction

**SmolLM2-135M** (1 heads):
- QK: decay=0.067+-0.000, eff_rank=43.4, id_cos=0.228, eig_pos=0.609
- OV: decay=0.056+-0.000, eff_rank=47.1, id_cos=0.050, eig_pos=0.562
- Layer position: 0.59+-0.00

**Pythia-70M** (4 heads):
- QK: decay=0.085+-0.005, eff_rank=28.7, id_cos=0.115, eig_pos=0.680
- OV: decay=0.056+-0.002, eff_rank=47.3, id_cos=0.003, eig_pos=0.516
- Layer position: 0.70+-0.10

**Qwen2.5-0.5B** (4 heads):
- QK: decay=0.083+-0.006, eff_rank=36.0, id_cos=0.169, eig_pos=0.562
- OV: decay=0.059+-0.003, eff_rank=46.6, id_cos=0.014, eig_pos=0.508
- Layer position: 0.51+-0.11

**GPT-2** (1 heads):
- QK: decay=0.043+-0.000, eff_rank=53.8, id_cos=0.731, eig_pos=0.938
- OV: decay=0.057+-0.000, eff_rank=49.3, id_cos=0.265, eig_pos=0.672
- Layer position: 0.45+-0.00

### positional

**SmolLM2-135M** (1 heads):
- QK: decay=0.045+-0.000, eff_rank=50.9, id_cos=0.635, eig_pos=1.000
- OV: decay=0.014+-0.000, eff_rank=62.1, id_cos=-0.961, eig_pos=0.000
- Layer position: 0.38+-0.00

**Pythia-70M** (1 heads):
- QK: decay=0.048+-0.000, eff_rank=50.7, id_cos=-0.523, eig_pos=0.109
- OV: decay=0.060+-0.000, eff_rank=46.5, id_cos=-0.335, eig_pos=0.328
- Layer position: 0.20+-0.00

**Qwen2.5-0.5B** (16 heads):
- QK: decay=0.071+-0.020, eff_rank=39.3, id_cos=0.359, eig_pos=0.851
- OV: decay=0.023+-0.020, eff_rank=58.2, id_cos=-0.854, eig_pos=0.042
- Layer position: 0.51+-0.31

**GPT-2** (6 heads):
- QK: decay=0.030+-0.006, eff_rank=55.3, id_cos=0.733, eig_pos=0.971
- OV: decay=0.043+-0.021, eff_rank=52.8, id_cos=-0.106, eig_pos=0.474
- Layer position: 0.08+-0.13

### previous_token

**SmolLM2-135M** (9 heads):
- QK: decay=0.068+-0.005, eff_rank=40.5, id_cos=-0.068, eig_pos=0.424
- OV: decay=0.058+-0.007, eff_rank=47.1, id_cos=0.245, eig_pos=0.698
- Layer position: 0.45+-0.24

**Pythia-70M** (5 heads):
- QK: decay=0.066+-0.014, eff_rank=36.8, id_cos=-0.157, eig_pos=0.369
- OV: decay=0.047+-0.011, eff_rank=51.0, id_cos=0.331, eig_pos=0.734
- Layer position: 0.56+-0.39

**Qwen2.5-0.5B** (15 heads):
- QK: decay=0.077+-0.025, eff_rank=38.4, id_cos=-0.047, eig_pos=0.474
- OV: decay=0.056+-0.009, eff_rank=48.0, id_cos=0.126, eig_pos=0.598
- Layer position: 0.29+-0.29

**GPT-2** (9 heads):
- QK: decay=0.039+-0.009, eff_rank=54.1, id_cos=-0.453, eig_pos=0.219
- OV: decay=0.047+-0.011, eff_rank=51.8, id_cos=0.091, eig_pos=0.557
- Layer position: 0.27+-0.11

### suppression

**SmolLM2-135M** (69 heads):
- QK: decay=0.072+-0.015, eff_rank=42.4, id_cos=0.175, eig_pos=0.617
- OV: decay=0.049+-0.014, eff_rank=50.5, id_cos=-0.478, eig_pos=0.203
- Layer position: 0.41+-0.29

**Pythia-70M** (6 heads):
- QK: decay=0.057+-0.005, eff_rank=47.5, id_cos=0.116, eig_pos=0.581
- OV: decay=0.052+-0.007, eff_rank=49.0, id_cos=-0.375, eig_pos=0.271
- Layer position: 0.30+-0.10

**Qwen2.5-0.5B** (71 heads):
- QK: decay=0.077+-0.011, eff_rank=40.4, id_cos=0.186, eig_pos=0.603
- OV: decay=0.049+-0.016, eff_rank=50.3, id_cos=-0.529, eig_pos=0.180
- Layer position: 0.51+-0.27

**GPT-2** (35 heads):
- QK: decay=0.056+-0.015, eff_rank=46.3, id_cos=0.019, eig_pos=0.500
- OV: decay=0.046+-0.012, eff_rank=51.6, id_cos=-0.543, eig_pos=0.153
- Layer position: 0.38+-0.26

## NMF Factorization Results

### SmolLM2-135M

**content**:
- QK recon error: 0.510
- OV recon error: 0.532
- QK component importance: [56.2436, 56.1248, 47.8752, 71.2223]
- OV component importance: [20.1962, 19.1324, 19.9798, 20.1426]

**suppression**:
- QK recon error: 0.587
- OV recon error: 0.665
- QK component importance: [31.8777, 34.8283, 32.7251, 33.1752]
- OV component importance: [18.232, 17.7982, 18.1588, 18.9595]

**copy**:
- QK recon error: 0.550
- OV recon error: 0.693
- QK component importance: [30.1646, 30.8106, 32.1474, 30.363]
- OV component importance: [25.8202, 25.2517, 26.1118, 26.0005]

**previous_token**:
- QK recon error: 0.484
- OV recon error: 0.582
- QK component importance: [17.6788, 18.0781, 20.6616, 17.659]
- OV component importance: [13.3134, 13.1582, 12.7554, 15.264]

### Pythia-70M

**content**:
- QK recon error: 0.494
- OV recon error: 0.542
- QK component importance: [1.8895, 1.6429, 2.1517, 2.065]
- OV component importance: [0.2236, 0.2149, 0.2306, 0.232]

**suppression**:
- QK recon error: 0.549
- OV recon error: 0.567
- QK component importance: [0.9058, 0.8559, 0.9841, 0.7995]
- OV component importance: [0.2923, 0.3117, 0.3587, 0.3199]

**previous_token**:
- QK recon error: 0.506
- OV recon error: 0.547
- QK component importance: [0.8702, 1.0872, 0.992, 0.9923]
- OV component importance: [0.2454, 0.2202, 0.258, 0.2402]

**induction**:
- QK recon error: 0.426
- OV recon error: 0.510
- QK component importance: [2.2932, 2.7658, 3.5112, 2.3661]
- OV component importance: [0.247, 0.2477, 0.2674, 0.2791]

**copy**:
- QK recon error: 0.348
- OV recon error: 0.737
- QK component importance: [4.7842, 7.2302, 7.6533, 5.9505]
- OV component importance: [0.2553, 0.2467, 0.2601, 0.2518]

### Qwen2.5-0.5B

**content**:
- QK recon error: 0.515
- OV recon error: 0.532
- QK component importance: [2.9302, 3.2884, 3.2371, 3.5933]
- OV component importance: [0.2666, 0.2689, 0.2759, 0.2707]

**previous_token**:
- QK recon error: 0.575
- OV recon error: 0.608
- QK component importance: [0.7597, 0.661, 0.5938, 0.6003]
- OV component importance: [0.1712, 0.1873, 0.1948, 0.1831]

**copy**:
- QK recon error: 0.550
- OV recon error: 0.689
- QK component importance: [0.5848, 0.5596, 0.575, 0.5929]
- OV component importance: [0.3136, 0.3095, 0.3205, 0.3225]

**positional**:
- QK recon error: 0.491
- OV recon error: 0.822
- QK component importance: [0.5704, 0.6649, 0.6914, 0.5246]
- OV component importance: [0.3982, 0.37, 0.3914, 0.3459]

**suppression**:
- QK recon error: 0.568
- OV recon error: 0.628
- QK component importance: [0.4857, 0.5023, 0.5202, 0.4844]
- OV component importance: [0.2599, 0.2699, 0.2677, 0.2712]

**induction**:
- QK recon error: 0.535
- OV recon error: 0.557
- QK component importance: [0.4526, 0.4174, 0.3739, 0.3561]
- OV component importance: [0.2103, 0.2247, 0.203, 0.2053]

### GPT-2

**copy**:
- QK recon error: 0.558
- OV recon error: 0.708
- QK component importance: [16.7424, 16.8154, 16.4339, 17.5247]
- OV component importance: [9.9653, 10.0098, 10.3425, 10.4516]

**positional**:
- QK recon error: 0.595
- OV recon error: 0.627
- QK component importance: [92.2791, 85.6066, 96.3033, 93.8781]
- OV component importance: [4.7936, 4.7104, 4.7104, 4.7078]

**content**:
- QK recon error: 0.572
- OV recon error: 0.461
- QK component importance: [18.6387, 18.9572, 19.9397, 21.0571]
- OV component importance: [17.7592, 14.4328, 17.5799, 16.6305]

**suppression**:
- QK recon error: 0.557
- OV recon error: 0.640
- QK component importance: [19.2655, 18.584, 21.3187, 20.1867]
- OV component importance: [8.1643, 8.5797, 8.3014, 8.8376]

**previous_token**:
- QK recon error: 0.592
- OV recon error: 0.744
- QK component importance: [16.3897, 15.254, 15.4858, 16.6837]
- OV component importance: [6.0792, 6.2611, 6.7651, 6.5472]

## Weight-Behavior Correlations

### SmolLM2-135M

| Correlation | r | p-value |
|---|---|---|
| qk_norm_vs_entropy | 0.389*** | 0.0e+00 |
| qk_id_cos_vs_prev_token_score | -0.354*** | 0.0e+00 |
| qk_decay_rate_vs_prev_token_score | -0.266*** | 9.0e-06 |
| qk_decay_rate_vs_induction_score | 0.223*** | 2.3e-04 |
| qk_id_cos_vs_induction_score | 0.190** | 1.7e-03 |
| qk_norm_vs_positional_score | 0.190** | 1.8e-03 |
| qk_decay_rate_vs_positional_score | -0.176** | 3.6e-03 |
| qk_eff_rank_vs_entropy | -0.176** | 3.8e-03 |
| ov_decay_rate_vs_induction_score | 0.167** | 6.0e-03 |
| ov_norm_vs_prev_token_score | -0.165** | 6.5e-03 |
| qk_id_cos_vs_entropy | -0.160** | 8.6e-03 |
| qk_eff_rank_vs_prev_token_score | 0.150* | 1.4e-02 |
| qk_eff_rank_vs_induction_score | -0.145* | 1.7e-02 |
| ov_decay_rate_vs_positional_score | -0.143* | 1.9e-02 |
| ov_norm_vs_entropy | -0.140* | 2.2e-02 |

### Pythia-70M

| Correlation | r | p-value |
|---|---|---|
| qk_eff_rank_vs_entropy | 0.775*** | 0.0e+00 |
| qk_decay_rate_vs_entropy | -0.756*** | 0.0e+00 |
| qk_norm_vs_entropy | -0.750*** | 0.0e+00 |
| qk_decay_rate_vs_positional_score | -0.635*** | 1.0e-06 |
| qk_eff_rank_vs_positional_score | 0.545*** | 6.2e-05 |
| ov_id_cos_vs_entropy | -0.534*** | 9.4e-05 |
| qk_decay_rate_vs_induction_score | 0.496*** | 3.3e-04 |
| qk_norm_vs_positional_score | -0.487*** | 4.4e-04 |
| ov_decay_rate_vs_entropy | 0.462*** | 9.5e-04 |
| ov_eff_rank_vs_entropy | -0.451** | 1.3e-03 |
| qk_eff_rank_vs_induction_score | -0.441** | 1.7e-03 |
| ov_norm_vs_entropy | 0.421** | 2.9e-03 |
| qk_id_cos_vs_prev_token_score | -0.365* | 1.1e-02 |
| qk_decay_rate_vs_prev_token_score | -0.360* | 1.2e-02 |
| ov_decay_rate_vs_induction_score | 0.347* | 1.6e-02 |

### Qwen2.5-0.5B

| Correlation | r | p-value |
|---|---|---|
| ov_id_cos_vs_positional_score | -0.365*** | 0.0e+00 |
| qk_norm_vs_positional_score | 0.278*** | 0.0e+00 |
| qk_id_cos_vs_prev_token_score | -0.267*** | 1.0e-06 |
| ov_decay_rate_vs_positional_score | -0.260*** | 1.0e-06 |
| ov_norm_vs_prev_token_score | -0.211*** | 9.4e-05 |
| ov_eff_rank_vs_positional_score | 0.189*** | 4.9e-04 |
| qk_norm_vs_prev_token_score | 0.178** | 1.0e-03 |
| ov_norm_vs_induction_score | -0.174** | 1.4e-03 |
| ov_decay_rate_vs_induction_score | 0.172** | 1.6e-03 |
| ov_id_cos_vs_entropy | 0.154** | 4.6e-03 |
| qk_id_cos_vs_entropy | -0.153** | 5.0e-03 |
| qk_id_cos_vs_positional_score | 0.146** | 7.3e-03 |
| ov_eff_rank_vs_induction_score | -0.133* | 1.5e-02 |
| ov_id_cos_vs_prev_token_score | 0.118* | 3.0e-02 |
| ov_eff_rank_vs_entropy | -0.113* | 3.8e-02 |

### GPT-2

| Correlation | r | p-value |
|---|---|---|
| qk_norm_vs_positional_score | 0.696*** | 0.0e+00 |
| qk_id_cos_vs_prev_token_score | -0.459*** | 0.0e+00 |
| qk_decay_rate_vs_positional_score | -0.448*** | 0.0e+00 |
| qk_id_cos_vs_induction_score | 0.413*** | 0.0e+00 |
| qk_id_cos_vs_entropy | -0.402*** | 1.0e-06 |
| qk_eff_rank_vs_entropy | -0.380*** | 3.0e-06 |
| ov_norm_vs_positional_score | -0.356*** | 1.2e-05 |
| ov_eff_rank_vs_entropy | -0.344*** | 2.4e-05 |
| ov_norm_vs_prev_token_score | -0.310*** | 1.6e-04 |
| ov_id_cos_vs_entropy | -0.308*** | 1.8e-04 |
| qk_decay_rate_vs_entropy | 0.270** | 1.1e-03 |
| qk_norm_vs_prev_token_score | 0.245** | 3.0e-03 |
| ov_decay_rate_vs_induction_score | 0.232** | 5.2e-03 |
| qk_decay_rate_vs_prev_token_score | -0.213* | 1.1e-02 |
| ov_id_cos_vs_positional_score | -0.184* | 2.7e-02 |
