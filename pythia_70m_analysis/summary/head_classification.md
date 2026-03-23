# Pythia-70M Attention Head Classification

Based on empirical attention patterns across 30 diverse text samples (2200 token positions total), plus weight-space OV/QK circuit analysis on 500-token vocabulary subset.

## Layer-by-Layer Classification

### Layer 0 -- High Entropy Feature Extraction

All heads have high entropy (1.6-3.0) with distributed attention. No sharp specialization yet.

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | Notes |
|------|------|---------|-------|-------|------|-------------|-------|
| H0 | Content-mixed | 2.93 | 14% | 23% | 7% | 23% | Broadly distributed, no clear bias |
| H1 | **Soft prev-token** | 1.64 | **72%** | 17% | 2% | 0% | Strongest prev-token head in L0 |
| H2 | Distant content | 2.57 | 6% | 17% | 7% | 57% | Attends to distant semantically-related tokens |
| H3 | Self-attention | 1.92 | 4% | **37%** | 0% | 42% | QK has 96% self-preference (identity circuit) |
| H4 | Local positional | 1.69 | 34% | 15% | 3% | 12% | Blends prev-token and near (2-5) |
| H5 | Mixed local | 1.80 | 26% | 23% | 2% | 21% | Balanced self/prev/near |
| H6 | Broad content | 2.97 | 13% | 24% | 6% | 28% | Highest entropy head in model |
| H7 | **Soft prev-token** | 1.75 | **43%** | 22% | 3% | 10% | Second prev-token head |

**Summary**: L0 provides basic previous-token signal (H1, H7), self-identity (H3), and distributed content (H0, H2, H6).

### Layer 1 -- Refinement and Local Context

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | Notes |
|------|------|---------|-------|-------|------|-------------|-------|
| H0 | Content-mixed | 2.39 | 24% | 22% | 2% | 27% | Balanced across all distances |
| H1 | Self + prev | 1.54 | 37% | 31% | 1% | 19% | Strong OV copy (ov_id=0.47) |
| H2 | Prev + distant | 1.64 | 32% | 16% | 3% | 37% | Routes information from prev to distant |
| H3 | **Near-positional** | 2.09 | **43%** | 17% | 2% | 1% | 39% attend 2-5 back; local window |
| H4 | Distant content | 2.85 | 5% | 8% | 5% | 70% | Long-range content retrieval |
| H5 | Local mixed | 1.97 | 27% | 22% | 2% | 21% | Suppression head (ov_id=-0.21) |
| H6 | Content-mixed | 2.24 | 19% | 23% | 1% | 37% | OV copy emerging (ov_id=0.19) |
| H7 | Distant content | 2.07 | 18% | 19% | 3% | 42% | OV copy emerging (ov_id=0.22) |

**Summary**: L1 adds near-positional context (H3), long-range retrieval (H4), and begins developing OV copy/suppression circuits.

### Layer 2 -- Specialization and Key Previous-Token Head

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | Notes |
|------|------|---------|-------|-------|------|-------------|-------|
| H0 | Near-positional | 2.19 | 20% | 8% | 6% | 24% | 49% attend 2-5 positions back |
| **H1** | **PURE PREV-TOKEN** | **0.81** | **98%** | **2%** | 1% | 0% | **Textbook previous-token head**. Critical for induction circuit. |
| H2 | Distant + BOS | 2.73 | 5% | 5% | 16% | 52% | Suppression head (ov_id=-0.53) |
| H3 | Self + prev | 1.96 | 25% | 29% | 2% | 19% | OV copy emerging (ov_id=0.48) |
| H4 | Local prev-token | 1.13 | 36% | 26% | 2% | 3% | Secondary prev-token head |
| H5 | Distant content | 2.84 | 4% | 9% | 4% | 74% | Long-range content, high entropy |
| H6 | Distant + BOS | 1.73 | 2% | 6% | 14% | 80% | Strong suppression (ov_id=-0.70) |
| H7 | Near-positional | 1.97 | 26% | 2% | 4% | 11% | 61% attend 2-5 back |

**Summary**: L2 produces the model's primary previous-token head (H1, 98% prev-token attention), which is the foundation for induction. Also develops suppression heads (H2, H6) and near-positional patterns (H0, H7).

### Layer 3 -- Induction Heads Emerge

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | Induction Evidence |
|------|------|---------|-------|-------|------|-------------|-------------------|
| **H0** | **INDUCTION** | 0.86 | 5% | 2% | 19% | 79% | sat@9->on@3 (0.73), on@10->the@4 (0.99), .@13->The@7 (0.90) |
| H1 | BOS-sink | 0.60 | 3% | 1% | 20% | 85% | No induction signal |
| H2 | Content-mixed | 2.37 | 16% | 10% | 11% | 37% | Weak/no induction |
| H3 | BOS-sink | 2.26 | 3% | 2% | 23% | 84% | No induction signal |
| H4 | Prev-token + mixed | 1.72 | 46% | 16% | 3% | 18% | Partial: some matching on repetitive |
| H5 | BOS-sink | 0.81 | 2% | 2% | 23% | 86% | No induction signal |
| **H6** | **INDUCTION** | 0.55 | 3% | 2% | 19% | 84% | sat@9->on@3 (1.00), on@10->the@4 (0.98), the@11->mat@5 (0.55) |
| H7 | Content-mixed + ind | 1.89 | 13% | 5% | 14% | 53% | Some induction on repetitive |

**Key insight**: L3_H0 and L3_H6 appear as BOS-sinks on non-repetitive text but switch to strong induction on repetitive sequences. This is the expected behavior -- induction heads attend to "position after prior occurrence of current token", and when no such occurrence exists, attention defaults to BOS as a safe fallback.

### Layer 4 -- Refined Induction and BOS Sinks

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | Induction Evidence |
|------|------|---------|-------|-------|------|-------------|-------------------|
| H0 | Content/distant | 1.18 | 12% | 3% | 12% | 55% | Partial matching on repetitive |
| **H1** | **INDUCTION** | 0.97 | 7% | 3% | 17% | 71% | The@7->cat@1 (0.70), the@11->mat@5 (0.93), The@14->dog@8 (0.67) |
| **H2** | **INDUCTION** | 0.58 | 3% | 2% | 21% | 79% | sat@9->on@3 (0.75), on@10->the@4 (1.00), .@13->The@7 (0.97) |
| H3 | **Strong BOS-sink** | 0.39 | 2% | 1% | **47%** | 87% | Pure BOS attention, no induction |
| **H4** | **INDUCTION** | 0.42 | 2% | 2% | 22% | 84% | The@7->cat@1 (0.81), the@11->mat@5 (1.00), The@14->cat@1 (0.69) |
| H5 | Content matching | 0.51 | 8% | 2% | 17% | 66% | Some content-based matching |
| H6 | Content focused | 0.35 | 6% | 6% | 14% | 70% | The@7->cat@1 (1.00), subject tracking |
| H7 | BOS-sink | 0.37 | 4% | 2% | 23% | 82% | No clear induction |

**Summary**: L4 has 3 clear induction heads (H1, H2, H4) plus content-matching heads. Entropy is very low across the board (0.35-1.18), indicating sharp, focused attention.

### Layer 5 -- Output Routing (Very Sharp Attention)

| Head | Type | Entropy | Prev% | Self% | BOS% | Offset 6+% | OV Identity |
|------|------|---------|-------|-------|------|-------------|-------------|
| H0 | Self + prev copy | 0.26 | 20% | 19% | 12% | 38% | 0.72 |
| H1 | **Prev-token + near** | 0.27 | **33%** | 8% | 3% | 13% | **0.78** |
| H2 | **Self-attention dominant** | 0.17 | 19% | **44%** | 2% | 20% | **0.88** |
| H3 | BOS-sink + focused | 0.20 | 9% | 9% | 21% | 63% | **0.96** |
| H4 | BOS-sink + focused | 0.20 | 15% | 7% | 20% | 57% | **0.93** |
| H5 | Distant content | 0.23 | 13% | 11% | 12% | 52% | 0.70 |
| H6 | BOS-sink | 0.17 | 4% | 4% | 24% | 76% | **0.88** |
| H7 | **Self + prev-token** | 0.36 | **37%** | **44%** | 2% | 4% | **0.77** |

**Summary**: All L5 heads have very low entropy (0.17-0.36) and very high OV identity cosine (0.70-0.96). These are the model's primary **output copy heads** -- they identify which token to copy to the output logits. H2 and H7 prefer self-attention (44% self), H1 and H7 prefer previous-token, and H3/H4/H6 attend to BOS/distant positions.

## Identified Circuits

### 1. Induction Circuit (Primary)

```
L2_H1 (98% prev-token) --K-composition--> L3_H0, L3_H6 (induction)
                                       --> L4_H1, L4_H2, L4_H4 (refined induction)
```

L2_H1 writes "what token preceded me" into the residual stream. Layers 3-4 induction heads read this via K-composition: their keys encode "what preceded me" (written by L2_H1), and their queries encode "what am I". When query matches a key (same token appeared before), attention routes to the position after that prior occurrence, enabling prediction of what comes next.

**On repetitive text**: The circuit fires strongly, showing textbook induction (sat@9->on@3, on@10->the@4).

**On non-repetitive text**: No matching prior occurrences exist, so these heads default to BOS-sink behavior (dumping attention on position 0).

### 2. Additional Previous-Token Providers

L0_H1 (72% prev), L0_H7 (43%), L2_H4 (36%) also provide previous-token information to the residual stream. These softer previous-token heads may contribute to induction via earlier layers or provide redundancy.

### 3. Copy/Output Circuit (Layer 5)

All 8 L5 heads have OV identity cosine 0.70-0.96, meaning their OV circuits copy the attended token's embedding toward the output logits. Combined with their very sharp attention (entropy 0.17-0.36), they function as the model's primary output mechanism:

- **H2, H7**: Copy self (what am I?) -- reinforces current token's logit
- **H1**: Copy previous token -- bigram prediction
- **H3, H4, H6**: Copy from BOS/distant -- may carry sentence-level information

### 4. Suppression Circuit (Layer 2)

L2_H2 (ov_id=-0.53) and L2_H6 (ov_id=-0.70) actively suppress token logits. These heads attend to distant/BOS positions and write negative contributions -- suppressing tokens that would otherwise be over-predicted.

## Aggregate Statistics

| Metric | Count | Heads |
|--------|-------|-------|
| Strong prev-token (>30%) | 6 | L0_H1, L0_H7, L2_H1, L3_H4, L5_H1, L5_H7 |
| Induction (verified on repetitive) | 5 | L3_H0, L3_H6, L4_H1, L4_H2, L4_H4 |
| BOS-sink (>15% BOS, on normal text) | 14 | L3_H0,H1,H3,H5,H6; L4_H1-H4,H7; L5_H3,H4,H6 |
| Strong copy (OV identity >0.4) | 13 | L1_H1; L2_H3; L3_H4; L4_H0; all L5 |
| Suppression (OV identity <-0.2) | 5 | L1_H0, L1_H3, L1_H5, L2_H2, L2_H6 |
| High entropy content (>2.5) | 7 | L0_H0,H2,H6; L1_H4; L2_H2,H5; L3_H2 |

## Entropy Progression

| Layer | Mean Entropy | Min | Max |
|-------|-------------|-----|-----|
| 0 | 2.16 | 1.64 | 2.97 |
| 1 | 2.09 | 1.54 | 2.85 |
| 2 | 1.92 | 0.81 | 2.84 |
| 3 | 1.36 | 0.55 | 2.37 |
| 4 | 0.60 | 0.35 | 1.18 |
| 5 | 0.24 | 0.17 | 0.36 |

Clear trend: attention becomes sharper through the layers. By L5, all heads have near-zero entropy (sharp, confident attention decisions).
