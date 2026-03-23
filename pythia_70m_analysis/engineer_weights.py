"""Engineer Pythia-70M attention weights from first principles.

Constructs weight matrices from code (no pre-trained weights loaded) to reproduce
the key circuits discovered in our analysis:
  1. Previous-token head (L2_H1: 98% prev-token via RoPE offset)
  2. Induction heads (L3_H0, L3_H6: K-composition with prev-token head)
  3. Copy heads (L5: high OV identity for output routing)
  4. Suppression heads (L2_H2, L2_H6: negative OV)
  5. BOS-sink fallback (natural when no induction match exists)

Architecture: Pythia-70M = GPT-NeoX, 6 layers, 8 heads, d_model=512, head_dim=64
  - RoPE: rotary_pct=0.25 -> 16 rotary dims (8 freq pairs), 48 content dims
  - Parallel residual: attention + FFN run simultaneously
  - Separate embed_in (50304, 512) and embed_out (50304, 512)
  - QKV fused: [1536, 512] interleaved [Q0,K0,V0, Q1,K1,V1, ...]
  - Dense out: [512, 512], head h -> columns [h*64:(h+1)*64]
"""
import sys, time, math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, str(Path(__file__).parent))
from text_samples import TEXT_SAMPLES, SAMPLE_LABELS

N_LAYERS = 6
N_HEADS = 8
D_MODEL = 512
HEAD_DIM = 64
ROTARY_DIM = 16  # dims 0..15 get RoPE
CONTENT_DIM = 48  # dims 16..63 are pure content
ROPE_BASE = 10000

# RoPE frequencies: inv_freq[j] = 1 / (base^(2j/rotary_dim)) for j=0..7
INV_FREQ = 1.0 / (ROPE_BASE ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))

# --- Subspace allocation ---
# We use a shared orthogonal basis for the residual stream.
# Each head gets a 64-dim subspace. With 8 heads that's 512 dims = full space.
# For cross-layer composition (induction circuit), we designate specific subspaces.

torch.manual_seed(42)
FULL_BASIS = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL))[0]  # [512, 512] orthogonal


def get_subspace(head_idx):
    """Get the orthogonal basis vectors for a head's 64-dim subspace."""
    return FULL_BASIS[:, head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM]  # [512, 64]


def set_qkv_for_head(qkv_weight, qkv_bias, layer, head, W_Q, W_K, W_V,
                     b_Q=None, b_K=None, b_V=None):
    """Write Q, K, V weights and biases for a specific head into the fused QKV matrix.

    qkv_weight: [1536, 512], interleaved [Q0,K0,V0, Q1,K1,V1, ...]
    W_Q, W_K, W_V: each [64, 512]
    b_Q, b_K, b_V: each [64] (optional biases)
    """
    base = head * 3 * HEAD_DIM
    qkv_weight.data[base:base + HEAD_DIM, :] = W_Q
    qkv_weight.data[base + HEAD_DIM:base + 2 * HEAD_DIM, :] = W_K
    qkv_weight.data[base + 2 * HEAD_DIM:base + 3 * HEAD_DIM, :] = W_V
    qkv_bias.data[base:base + HEAD_DIM] = b_Q if b_Q is not None else 0
    qkv_bias.data[base + HEAD_DIM:base + 2 * HEAD_DIM] = b_K if b_K is not None else 0
    qkv_bias.data[base + 2 * HEAD_DIM:base + 3 * HEAD_DIM] = b_V if b_V is not None else 0


def set_output_for_head(dense_weight, dense_bias, head, W_O):
    """Write output projection for a specific head.

    dense_weight: [512, 512], head h -> columns [h*64:(h+1)*64]
    W_O: [512, 64]
    """
    dense_weight.data[:, head * HEAD_DIM:(head + 1) * HEAD_DIM] = W_O
    # Bias is shared across heads, zero it once
    dense_bias.data.zero_()


# ============================================================
# Circuit 1: Previous-Token Head
# ============================================================

def make_prev_token_head(head_subspace, alpha=10.0):
    """Construct Q, K, V, O matrices for a previous-token head.

    Key insight: Use QKV BIAS (not weights) for the rotary dims so the
    positional signal is CONTENT-INDEPENDENT. This prevents self-attention
    from overwhelming the previous-token signal.

    For each frequency pair j (theta_j = inv_freq[j]):
      Q_bias at (2j, 2j+1) = (alpha, 0)       <- constant, no content dependence
      K_bias at (2j, 2j+1) = (alpha*cos(theta_j), alpha*sin(theta_j))  <- pre-rotated by +1

    After RoPE: dot_j = alpha^2 * cos(theta_j * (delta - 1))
    Peak at delta = 1 (previous token). Purely positional, no content noise.

    V and O: Copy the attended token's content into the residual stream
    so that later induction heads can read "what token preceded me".
    """
    P = head_subspace  # [512, 64] orthogonal columns

    # WEIGHTS: zero for rotary dims (positional signal comes from bias)
    W_Q = torch.zeros(HEAD_DIM, D_MODEL)
    W_K = torch.zeros(HEAD_DIM, D_MODEL)

    # BIAS: constant positional signal in rotary dims
    # GPT-NeoX uses HALF-HALF layout: dims [0..7] are real parts of 8 freq pairs,
    # dims [8..15] are imaginary parts. NOT interleaved!
    HALF_ROT = ROTARY_DIM // 2  # 8
    b_Q = torch.zeros(HEAD_DIM)
    b_K = torch.zeros(HEAD_DIM)

    for j in range(HALF_ROT):
        theta_j = float(INV_FREQ[j])
        # Q: constant (alpha, 0) per pair -> purely positional
        b_Q[j] = alpha              # real part of pair j
        b_Q[HALF_ROT + j] = 0       # imaginary part of pair j
        # K: pre-rotated by +theta_j -> peaks at delta=1 after RoPE
        b_K[j] = alpha * math.cos(theta_j)           # real
        b_K[HALF_ROT + j] = alpha * math.sin(theta_j)  # imaginary

    # V: project from residual stream into head space (for OV copying)
    # O: project back to residual stream
    # OV = O @ V writes a copy of the attended token's representation
    # Large scale so the prev-token signal dominates the induction heads' K projection
    v_scale = 2.0
    W_V = v_scale * P.T  # [64, 512]
    W_O = v_scale * P    # [512, 64]
    b_V = torch.zeros(HEAD_DIM)

    return W_Q, W_K, W_V, W_O, b_Q, b_K, b_V


# ============================================================
# Circuit 2: Induction Head (K-composition)
# ============================================================

def make_induction_head(own_subspace, prev_token_subspace, alpha=4.0, content_alpha=2.0):
    """Construct an induction head that composes with a previous-token head.

    The induction pattern [A][B]...[A] -> predict B requires:
    - K reads from the subspace where the prev-token head wrote (so K encodes
      "what token preceded me" = the prev-token head's OV output)
    - Q reads from the content embedding (so Q encodes "what am I")
    - When Q matches K (same token A appeared before, preceded by the same context),
      attention routes to that position, and OV copies the next token B.

    Strategy:
    - K projects from the prev-token head's output subspace (content dims only)
    - Q projects from the model's embedding subspace (content dims only)
    - We use the content dims (16..63) for this matching, leaving rotary dims for
      a mild positional bias toward recent positions.
    """
    P_own = own_subspace       # [512, 64]
    P_prev = prev_token_subspace  # [512, 64] subspace where prev-token head writes

    W_Q = torch.zeros(HEAD_DIM, D_MODEL)
    W_K = torch.zeros(HEAD_DIM, D_MODEL)

    # Content dims (16..63): BOTH Q and K project from the prev-token head's subspace.
    # Why: The prev-token head's OV writes into P_prev columns of the residual stream.
    # K reads from P_prev -> K encodes "what the prev-token head wrote at this position"
    #   = content of the token that preceded this position
    # Q ALSO reads from P_prev -> Q encodes "what the prev-token head wrote at this position"
    #   = content of the token that preceded this position
    # Match: Q at pos q matches K at pos k when prev(q) == prev(k),
    #   i.e., when the same token appears before both positions -> induction!
    for d in range(CONTENT_DIM):
        dim_idx = ROTARY_DIM + d  # head-space dim index
        W_Q[dim_idx, :] = content_alpha * P_prev[:, d]
        W_K[dim_idx, :] = content_alpha * P_prev[:, d]

    # Rotary dims: mild recency bias (not pre-rotated, so peaks at delta=0/self)
    # Half-half layout: dims [0..7] = real, dims [8..15] = imaginary
    HALF_ROT = ROTARY_DIM // 2
    for j in range(HALF_ROT):
        u = P_own[:, CONTENT_DIM + j] if CONTENT_DIM + j < HEAD_DIM else P_own[:, j]
        W_Q[j, :] = 0.3 * u         # real part
        W_K[j, :] = 0.3 * u
        W_Q[HALF_ROT + j, :] = 0    # imaginary part
        W_K[HALF_ROT + j, :] = 0

    # V and O: copy attended token's content toward output
    v_scale = 0.3
    W_V = v_scale * P_own.T  # [64, 512]
    W_O = v_scale * P_own    # [512, 64]

    return W_Q, W_K, W_V, W_O, None, None, None


# ============================================================
# Circuit 3: Copy Head (high OV identity)
# ============================================================

def make_copy_head(head_subspace, ov_strength=0.8, qk_content_strength=1.5):
    """Construct a copy head with strong OV identity circuit.

    OV circuit ≈ scaled identity: when attending to a token, boost that token's
    logit in the output. Combined with sharp content-based attention (attend to
    content-similar tokens), this implements "copy what I'm looking at."

    Layer 5 heads in Pythia-70M have:
    - Very low entropy (0.17-0.36): sharp attention
    - High OV identity (0.70-0.96): strong copying
    - Mix of self-attention and previous-token patterns
    """
    P = head_subspace  # [512, 64]

    # QK: content-based attention (identity-like: attend to similar tokens)
    W_Q = torch.zeros(HEAD_DIM, D_MODEL)
    W_K = torch.zeros(HEAD_DIM, D_MODEL)

    # Content dims: identity-like QK (tokens attend to content-similar tokens)
    for d in range(CONTENT_DIM):
        dim_idx = ROTARY_DIM + d
        W_Q[dim_idx, :] = qk_content_strength * P[:, d]
        W_K[dim_idx, :] = qk_content_strength * P[:, d]

    # Mild self/prev bias via rotary dims (half-half layout)
    HALF_ROT = ROTARY_DIM // 2
    for j in range(HALF_ROT):
        u = P[:, CONTENT_DIM + j] if CONTENT_DIM + j < HEAD_DIM else P[:, j]
        W_Q[j, :] = 0.5 * u
        W_K[j, :] = 0.5 * u

    # OV: near-identity (copy circuit) with exponential SV decay
    sigmas = torch.exp(-0.04 * torch.arange(HEAD_DIM).float())
    sigmas *= ov_strength
    W_V = (P * sigmas.unsqueeze(0)).T  # [64, 512]
    W_O = P * sigmas.unsqueeze(0)      # [512, 64]

    return W_Q, W_K, W_V, W_O, None, None, None


# ============================================================
# Circuit 4: Suppression Head (negative OV)
# ============================================================

def make_suppression_head(head_subspace, ov_strength=-0.4, qk_entropy="high"):
    """Construct a suppression head with negative OV identity.

    These heads attend broadly (high entropy) and write negative contributions,
    suppressing tokens that would otherwise be over-predicted.
    """
    P = head_subspace  # [512, 64]

    # QK: broad attention (small weights -> high entropy / flat distribution)
    qk_scale = 0.3  # small scale -> softer attention
    W_Q = torch.zeros(HEAD_DIM, D_MODEL)
    W_K = torch.zeros(HEAD_DIM, D_MODEL)

    for d in range(CONTENT_DIM):
        dim_idx = ROTARY_DIM + d
        W_Q[dim_idx, :] = qk_scale * P[:, d]
        W_K[dim_idx, :] = qk_scale * P[:, d]

    # OV: negative identity (suppression)
    sigmas = torch.exp(-0.04 * torch.arange(HEAD_DIM).float())
    sigmas *= ov_strength  # negative!
    W_V = (P * sigmas.unsqueeze(0)).T
    W_O = P * sigmas.unsqueeze(0)

    return W_Q, W_K, W_V, W_O, None, None, None


# ============================================================
# Circuit 5: Generic content head (default for unassigned heads)
# ============================================================

def make_content_head(head_subspace, scale=0.5):
    """Default content-based head with moderate identity QK and weak OV."""
    P = head_subspace

    W_Q = torch.zeros(HEAD_DIM, D_MODEL)
    W_K = torch.zeros(HEAD_DIM, D_MODEL)

    for d in range(CONTENT_DIM):
        dim_idx = ROTARY_DIM + d
        W_Q[dim_idx, :] = scale * P[:, d]
        W_K[dim_idx, :] = scale * P[:, d]

    # Small noise in rotary dims for mild positional diversity (half-half layout)
    HALF_ROT = ROTARY_DIM // 2
    for j in range(HALF_ROT):
        u = P[:, j]
        W_Q[j, :] = 0.2 * u
        W_K[j, :] = 0.2 * u

    # Weak OV (let training determine)
    ov_scale = 0.1
    W_V = ov_scale * P.T
    W_O = ov_scale * P

    return W_Q, W_K, W_V, W_O, None, None, None


# ============================================================
# Main: Assemble the full model
# ============================================================

# Circuit assignment plan (matching Pythia-70M's discovered structure):
#
# Layer 0: All content heads (early feature extraction, high entropy)
# Layer 1: 2 suppression, 6 content (L1_H0, L1_H5 are suppression in Pythia)
# Layer 2: 1 prev-token (H1), 2 suppression (H2, H6), 5 content
# Layer 3: 2 induction (H0, H6), 6 content/BOS-sink
# Layer 4: 3 induction (H1, H2, H4), 5 content
# Layer 5: All copy heads (output routing)

CIRCUIT_PLAN = {
    # (layer, head) -> circuit_type
    # Layer 0
    (0, 0): "content", (0, 1): "content", (0, 2): "content", (0, 3): "content",
    (0, 4): "content", (0, 5): "content", (0, 6): "content", (0, 7): "content",
    # Layer 1
    (1, 0): "suppress", (1, 1): "content", (1, 2): "content", (1, 3): "content",
    (1, 4): "content",  (1, 5): "suppress", (1, 6): "content", (1, 7): "content",
    # Layer 2
    (2, 0): "content",  (2, 1): "prev_token", (2, 2): "suppress", (2, 3): "content",
    (2, 4): "content",  (2, 5): "content",    (2, 6): "suppress", (2, 7): "content",
    # Layer 3
    (3, 0): "induction", (3, 1): "content", (3, 2): "content", (3, 3): "content",
    (3, 4): "content",   (3, 5): "content", (3, 6): "induction", (3, 7): "content",
    # Layer 4
    (4, 0): "content",   (4, 1): "induction", (4, 2): "induction", (4, 3): "content",
    (4, 4): "induction", (4, 5): "content",   (4, 6): "content",   (4, 7): "content",
    # Layer 5
    (5, 0): "copy", (5, 1): "copy", (5, 2): "copy", (5, 3): "copy",
    (5, 4): "copy", (5, 5): "copy", (5, 6): "copy", (5, 7): "copy",
}


def populate_model(model):
    """Populate all attention weights with engineered circuits."""
    # Get the subspace for the previous-token head (L2_H1) -- induction heads need this
    prev_token_subspace = get_subspace(1)  # head 1's subspace

    for layer_idx in range(N_LAYERS):
        layer = model.gpt_neox.layers[layer_idx]
        qkv_w = layer.attention.query_key_value.weight
        qkv_b = layer.attention.query_key_value.bias
        dense_w = layer.attention.dense.weight
        dense_b = layer.attention.dense.bias

        for head_idx in range(N_HEADS):
            circuit = CIRCUIT_PLAN[(layer_idx, head_idx)]
            subspace = get_subspace(head_idx)

            if circuit == "prev_token":
                W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_prev_token_head(subspace, alpha=10.0)
            elif circuit == "induction":
                W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_induction_head(
                    subspace, prev_token_subspace, alpha=4.0, content_alpha=2.0
                )
            elif circuit == "copy":
                W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_copy_head(
                    subspace, ov_strength=0.8, qk_content_strength=1.5
                )
            elif circuit == "suppress":
                W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_suppression_head(subspace, ov_strength=-0.4)
            else:  # content
                W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_content_head(subspace, scale=0.5)

            set_qkv_for_head(qkv_w, qkv_b, layer_idx, head_idx, W_Q, W_K, W_V,
                             b_Q, b_K, b_V)
            set_output_for_head(dense_w, dense_b, head_idx, W_O)

    # Zero out FFN weights (let them be default / don't engineer FFN)
    # Actually, keep whatever random init the model has for FFN
    # Zero out layer norms to pass-through (set weight=1, bias=0)
    for layer_idx in range(N_LAYERS):
        layer = model.gpt_neox.layers[layer_idx]
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.input_layernorm.bias.data.zero_()
        layer.post_attention_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.bias.data.zero_()

    # Final layer norm
    model.gpt_neox.final_layer_norm.weight.data.fill_(1.0)
    model.gpt_neox.final_layer_norm.bias.data.zero_()


# ============================================================
# Verification: Run text samples and check attention patterns
# ============================================================

def verify_attention_patterns(model, tokenizer):
    """Run samples and report attention statistics for key engineered heads."""
    model.eval()

    key_heads = [
        (2, 1, "prev_token"),
        (3, 0, "induction"),
        (3, 6, "induction"),
        (4, 1, "induction"),
        (4, 2, "induction"),
        (5, 0, "copy"),
        (5, 3, "copy"),
        (1, 0, "suppress"),
        (2, 2, "suppress"),
    ]

    # Aggregate stats per head
    stats = {(l, h): {"prev": 0, "self": 0, "bos": 0, "total": 0, "entropies": []}
             for l, h, _ in key_heads}

    for s_idx, text in enumerate(TEXT_SAMPLES):
        inputs = tokenizer(text, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
        seq_len = len(tokens)

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        for l, h, _ in key_heads:
            attn = out.attentions[l][0, h].numpy()  # [seq, seq]
            for q in range(1, seq_len):
                row = attn[q, :q + 1]
                top1 = np.argmax(row)
                s = stats[(l, h)]
                s["total"] += 1
                if top1 == q:
                    s["self"] += 1
                if top1 == q - 1:
                    s["prev"] += 1
                if top1 == 0:
                    s["bos"] += 1
                row_safe = np.clip(row, 1e-10, 1.0)
                entropy = -(row_safe * np.log(row_safe)).sum()
                s["entropies"].append(entropy)

    # Print detailed results for repetitive sample
    print("\n=== Repetitive Sample Detail (Sample 25: 'The cat sat on the mat...') ===")
    inputs = tokenizer(TEXT_SAMPLES[25], return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    for l, h, name in key_heads:
        if name in ("induction", "prev_token"):
            attn = out.attentions[l][0, h].numpy()
            print(f"\n  L{l}_H{h} ({name}):")
            for q in range(1, min(15, len(tokens))):
                row = attn[q, :q + 1]
                top3 = np.argsort(-row)[:3]
                attn_str = "  ".join(f"'{tokens[i]}'@{i} ({row[i]:.2f})" for i in top3)
                print(f"    '{tokens[q]}'@{q} -> {attn_str}")

    return stats


def main():
    print("Creating fresh Pythia-70M model with engineered weights...")
    config = AutoConfig.from_pretrained("EleutherAI/pythia-70m")
    model = AutoModelForCausalLM.from_config(config)  # random init, no pretrained weights
    model.config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    print("Populating attention weights with engineered circuits...")
    populate_model(model)

    print("Running verification on 30 text samples...")
    stats = verify_attention_patterns(model, tokenizer)

    print("\n=== Aggregate Statistics ===")
    print(f"{'Head':<12} {'Type':<12} {'Prev%':>7} {'Self%':>7} {'BOS%':>7} {'Entropy':>8}")
    print("-" * 60)
    key_heads = [
        (2, 1, "prev_token"),
        (3, 0, "induction"),
        (3, 6, "induction"),
        (4, 1, "induction"),
        (4, 2, "induction"),
        (5, 0, "copy"),
        (5, 3, "copy"),
        (1, 0, "suppress"),
        (2, 2, "suppress"),
    ]
    for l, h, name in key_heads:
        s = stats[(l, h)]
        t = max(s["total"], 1)
        ent = np.mean(s["entropies"])
        print(f"L{l}_H{h:<6} {name:<12} {s['prev']/t:>6.1%} {s['self']/t:>6.1%} "
              f"{s['bos']/t:>6.1%} {ent:>8.3f}")

    # Compare with Pythia-70M trained targets
    print("\n=== Comparison with Trained Pythia-70M ===")
    targets = {
        (2, 1): {"prev": 98.4, "ent": 0.81, "name": "prev_token"},
        (3, 0): {"prev": 4.6,  "ent": 0.86, "name": "induction (BOS-sink on non-rep)"},
        (3, 6): {"prev": 3.1,  "ent": 0.55, "name": "induction (BOS-sink on non-rep)"},
        (5, 0): {"prev": 19.7, "ent": 0.26, "name": "copy (self+prev)"},
        (5, 3): {"prev": 9.0,  "ent": 0.20, "name": "copy (BOS+distant)"},
    }
    print(f"{'Head':<10} {'Metric':<12} {'Engineered':>12} {'Trained':>10} {'Match':>7}")
    print("-" * 55)
    for (l, h), tgt in targets.items():
        s = stats[(l, h)]
        t = max(s["total"], 1)
        eng_prev = s["prev"] / t * 100
        eng_ent = np.mean(s["entropies"])
        # Prev-token rate comparison
        prev_match = "OK" if abs(eng_prev - tgt["prev"]) < 20 else "MISS"
        print(f"L{l}_H{h:<6} {'prev%':<12} {eng_prev:>11.1f}% {tgt['prev']:>9.1f}% {prev_match:>7}")
        ent_match = "OK" if abs(eng_ent - tgt["ent"]) < 1.0 else "MISS"
        print(f"{'':10} {'entropy':<12} {eng_ent:>12.3f} {tgt['ent']:>10.3f} {ent_match:>7}")


if __name__ == "__main__":
    main()
