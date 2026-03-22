"""
Script 1: Behavioral Head Classification

Run each model on actual text and classify every attention head by function.
Computes per-head behavioral scores (previous-token, induction, positional, entropy,
copy/suppression) and assigns head types via thresholds + K-means clustering.

Usage:
    source .venv/bin/activate
    python analyze_head_behavior.py
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import shared model definitions
from analyze_attention_circuits import MODELS, ModelSpec, extract_attention_weights

OUT_DIR = Path("analysis_results")

# ─── Test inputs ──────────────────────────────────────────────────────────────

TEST_INPUTS = {
    "repeated_bigrams": (
        "The cat sat on the mat. The cat sat on the mat. "
        "The dog ran to the park. The dog ran to the park. "
        "A bird flew over the tree. A bird flew over the tree."
    ),
    "sequential": (
        "a b c d e f g h i j k l m n o p q r s t u v w x y z "
        "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    ),
    "repeated_entities": (
        "Alice met Bob at the store. Then Alice met Bob again at the park. "
        "Charlie called Diana on Monday. Later Charlie called Diana on Friday. "
        "Eve sent Frank a letter. Soon Eve sent Frank another letter."
    ),
    "natural_prose": (
        "The development of large language models has transformed natural language "
        "processing in recent years. These models learn to predict the next token in "
        "a sequence by training on vast amounts of text data. The attention mechanism "
        "allows each token to attend to all previous tokens in the context window, "
        "enabling the model to capture long-range dependencies. Researchers have found "
        "that specific attention heads specialize in particular functions, such as "
        "tracking syntactic relationships or copying tokens from earlier in the sequence."
    ),
}


# ─── Attention extraction via forward pass ────────────────────────────────────

def get_attention_patterns(model, tokenizer, text, spec):
    """Run forward pass and extract attention patterns for all heads."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return attentions, tokens, input_ids[0]


# ─── Behavioral metrics ──────────────────────────────────────────────────────

def compute_previous_token_score(attn):
    """Mean attention weight on position i-1 for each query position i."""
    # attn: (num_heads, seq_len, seq_len)
    seq_len = attn.shape[-1]
    if seq_len < 2:
        return np.zeros(attn.shape[0])

    scores = []
    for h in range(attn.shape[0]):
        total = 0.0
        count = 0
        for i in range(1, seq_len):
            total += float(attn[h, i, i - 1])
            count += 1
        scores.append(total / max(count, 1))
    return np.array(scores)


def compute_induction_score(attn, input_ids):
    """
    For repeated subsequences, measure diagonal attention from repeat_pos+k -> first_pos+k.
    Uses the Olsson et al. 2022 method: for each token that appeared before,
    check if the attention looks back to the token after the previous occurrence.
    """
    seq_len = attn.shape[-1]
    ids = input_ids.numpy() if isinstance(input_ids, torch.Tensor) else input_ids
    n_heads = attn.shape[0]
    scores = np.zeros(n_heads)
    count = 0

    # For each position i, find if ids[i] appeared at some earlier position j.
    # If so, an induction head at position i should attend to position j+1.
    for i in range(2, seq_len):
        token_i = ids[i]
        for j in range(0, i - 1):
            if ids[j] == token_i and j + 1 < seq_len:
                # Induction pattern: position i attends to j+1
                for h in range(n_heads):
                    scores[h] += float(attn[h, i, j + 1])
                count += 1

    if count > 0:
        scores /= count
    return scores


def compute_positional_score(attn):
    """
    R^2 of attention as function of |query - key| position distance.
    High score means attention is determined primarily by relative position.
    """
    n_heads = attn.shape[0]
    seq_len = attn.shape[-1]
    scores = np.zeros(n_heads)

    if seq_len < 3:
        return scores

    for h in range(n_heads):
        # Collect (distance, attention_weight) pairs
        distances = []
        weights = []
        for i in range(seq_len):
            for j in range(i + 1):  # causal: j <= i
                distances.append(i - j)
                weights.append(float(attn[h, i, j]))

        distances = np.array(distances)
        weights = np.array(weights)

        # R^2: how much variance in weights is explained by distance?
        if len(distances) < 3 or np.var(weights) < 1e-10:
            scores[h] = 0.0
            continue

        # Bin by distance and compute mean weight per distance
        max_dist = int(distances.max())
        dist_means = {}
        for d in range(max_dist + 1):
            mask = distances == d
            if mask.any():
                dist_means[d] = weights[mask].mean()

        # Predicted weights based on distance alone
        predicted = np.array([dist_means.get(int(d), 0) for d in distances])
        ss_res = np.sum((weights - predicted) ** 2)
        ss_tot = np.sum((weights - weights.mean()) ** 2)
        scores[h] = max(0, 1 - ss_res / (ss_tot + 1e-10))

    return scores


def compute_entropy(attn):
    """Mean entropy of attention distribution per query position."""
    n_heads = attn.shape[0]
    seq_len = attn.shape[-1]
    scores = np.zeros(n_heads)

    for h in range(n_heads):
        entropies = []
        for i in range(seq_len):
            p = attn[h, i, :i + 1]  # causal: only up to position i
            p = np.clip(p, 1e-10, 1.0)
            p = p / p.sum()
            ent = -np.sum(p * np.log(p))
            entropies.append(ent)
        scores[h] = np.mean(entropies)

    return scores


def compute_ov_eigenvalue_sign(layers):
    """
    For each head, compute fraction of OV eigenvalues with positive real part.
    Returns list of lists (per layer, per head).
    """
    result = []
    for layer in layers:
        n_heads = layer["n_heads"]
        n_kv = layer["n_kv"]
        group = n_heads // n_kv
        layer_fracs = []
        for h in range(n_heads):
            g = h // group
            ov_core = layer["v"][g] @ layer["o"][:, h, :]  # (hd, hd)
            eig = torch.linalg.eigvals(ov_core).numpy()
            frac_positive = float(np.mean(eig.real > 0))
            layer_fracs.append(frac_positive)
        result.append(layer_fracs)
    return result


def compute_ov_identity_cos(layers):
    """OV identity cosine per head. Returns list of lists."""
    result = []
    for layer in layers:
        n_heads = layer["n_heads"]
        n_kv = layer["n_kv"]
        group = n_heads // n_kv
        hd = layer["hd"]
        layer_cos = []
        for h in range(n_heads):
            g = h // group
            ov_core = layer["v"][g] @ layer["o"][:, h, :]
            ov_norm = float(torch.linalg.norm(ov_core, "fro"))
            ov_normed = ov_core / (ov_norm / (hd ** 0.5) + 1e-8)
            cos = float((ov_normed * torch.eye(hd)).sum() / hd)
            layer_cos.append(cos)
        result.append(layer_cos)
    return result


# ─── Head type classification ────────────────────────────────────────────────

def classify_heads(head_features):
    """
    Assign head types based on thresholds, then also do K-means clustering.
    head_features: list of dicts with all scores per head.
    Returns head_features with 'type' and 'cluster' fields added.
    """
    for hf in head_features:
        # Threshold-based classification
        if hf["induction_score"] > 0.3:
            hf["type"] = "induction"
        elif hf["prev_token_score"] > 0.3:
            hf["type"] = "previous_token"
        elif hf["positional_score"] > 0.7:
            hf["type"] = "positional"
        elif hf["ov_identity_cos"] > 0.3 and hf["ov_eig_pos_frac"] > 0.6:
            hf["type"] = "copy"
        elif hf["ov_eig_pos_frac"] < 0.4:
            hf["type"] = "suppression"
        else:
            hf["type"] = "content"

    # K-means clustering on feature vectors
    feature_keys = [
        "prev_token_score", "induction_score", "positional_score",
        "entropy", "ov_identity_cos", "ov_eig_pos_frac",
    ]
    X = np.array([[hf[k] for k in feature_keys] for hf in head_features])

    if len(X) >= 6:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=min(6, len(X)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        for i, hf in enumerate(head_features):
            hf["cluster"] = int(clusters[i])
    else:
        for hf in head_features:
            hf["cluster"] = 0

    return head_features


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_model(spec):
    """Run full behavioral analysis for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Analyzing {spec.name} ({spec.hf_id})...")
    print(f"{'=' * 60}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
        output_attentions=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)

    # Get config info
    cfg = model.config
    if spec.arch == "gpt2":
        n_layers = cfg.n_layer
        n_heads = cfg.n_head
    else:
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads

    print(f"  {n_layers} layers, {n_heads} heads")

    # Extract weight-based metrics
    print(f"  Extracting attention weights...")
    layers = extract_attention_weights(model, spec)
    ov_eig_signs = compute_ov_eigenvalue_sign(layers)
    ov_id_cos = compute_ov_identity_cos(layers)
    del layers

    # Run forward passes on test inputs
    print(f"  Running forward passes on test inputs...")
    all_attentions = {}
    all_tokens = {}
    all_ids = {}
    for name, text in TEST_INPUTS.items():
        attns, tokens, ids = get_attention_patterns(model, tokenizer, text, spec)
        # Convert to numpy
        all_attentions[name] = [a[0].numpy() for a in attns]  # remove batch dim
        all_tokens[name] = tokens
        all_ids[name] = ids.numpy()

    # Compute behavioral metrics per head
    print(f"  Computing behavioral metrics...")
    head_features = []

    for li in range(n_layers):
        # Aggregate scores across test inputs
        prev_scores_all = []
        induction_scores_all = []
        positional_scores_all = []
        entropy_scores_all = []

        for name in TEST_INPUTS:
            attn = all_attentions[name][li]  # (n_heads, seq, seq)
            prev_scores_all.append(compute_previous_token_score(attn))

            if name == "repeated_bigrams":
                induction_scores_all.append(
                    compute_induction_score(attn, all_ids[name])
                )

            positional_scores_all.append(compute_positional_score(attn))
            entropy_scores_all.append(compute_entropy(attn))

        prev_scores = np.mean(prev_scores_all, axis=0)
        induction_scores = (
            np.mean(induction_scores_all, axis=0)
            if induction_scores_all
            else np.zeros(n_heads)
        )
        positional_scores = np.mean(positional_scores_all, axis=0)
        entropy_scores = np.mean(entropy_scores_all, axis=0)

        for h in range(n_heads):
            head_features.append({
                "layer": li,
                "head": h,
                "prev_token_score": round(float(prev_scores[h]), 4),
                "induction_score": round(float(induction_scores[h]), 4),
                "positional_score": round(float(positional_scores[h]), 4),
                "entropy": round(float(entropy_scores[h]), 4),
                "ov_identity_cos": round(float(ov_id_cos[li][h]), 4),
                "ov_eig_pos_frac": round(float(ov_eig_signs[li][h]), 4),
            })

    # Classify heads
    print(f"  Classifying heads...")
    head_features = classify_heads(head_features)

    # Count types
    type_counts = {}
    for hf in head_features:
        t = hf["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"  Head type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c} ({100 * c / len(head_features):.1f}%)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": spec.name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "heads": head_features,
        "type_distribution": type_counts,
    }


# ─── Report generation ───────────────────────────────────────────────────────

def generate_report(all_results):
    """Generate markdown report."""
    lines = ["# Head Behavior Analysis Report\n"]

    # Summary table
    lines.append("## Head Type Distribution Across Models\n")
    all_types = set()
    for r in all_results:
        all_types.update(r["type_distribution"].keys())
    all_types = sorted(all_types)

    header = "| Model | " + " | ".join(all_types) + " | Total |"
    sep = "|" + "---|" * (len(all_types) + 2)
    lines.append(header)
    lines.append(sep)
    for r in all_results:
        total = r["n_layers"] * r["n_heads"]
        counts = [str(r["type_distribution"].get(t, 0)) for t in all_types]
        lines.append(f"| {r['model']} | " + " | ".join(counts) + f" | {total} |")
    lines.append("")

    # Per-model details
    for r in all_results:
        lines.append(f"## {r['model']}\n")
        lines.append(f"- Layers: {r['n_layers']}, Heads: {r['n_heads']}")
        lines.append(f"- Total heads: {r['n_layers'] * r['n_heads']}\n")

        # Notable heads
        heads = r["heads"]
        lines.append("### Top Induction Heads")
        top_ind = sorted(heads, key=lambda h: -h["induction_score"])[:5]
        for h in top_ind:
            lines.append(
                f"- L{h['layer']}H{h['head']}: induction={h['induction_score']:.3f}, "
                f"prev_token={h['prev_token_score']:.3f}, type={h['type']}"
            )
        lines.append("")

        lines.append("### Top Previous-Token Heads")
        top_prev = sorted(heads, key=lambda h: -h["prev_token_score"])[:5]
        for h in top_prev:
            lines.append(
                f"- L{h['layer']}H{h['head']}: prev_token={h['prev_token_score']:.3f}, "
                f"positional={h['positional_score']:.3f}, type={h['type']}"
            )
        lines.append("")

        lines.append("### Top Copy Heads (high OV identity cosine)")
        top_copy = sorted(heads, key=lambda h: -h["ov_identity_cos"])[:5]
        for h in top_copy:
            lines.append(
                f"- L{h['layer']}H{h['head']}: ov_id_cos={h['ov_identity_cos']:.3f}, "
                f"ov_eig_pos={h['ov_eig_pos_frac']:.3f}, type={h['type']}"
            )
        lines.append("")

        # Score distributions
        lines.append("### Score Statistics")
        for key in ["prev_token_score", "induction_score", "positional_score",
                     "entropy", "ov_identity_cos", "ov_eig_pos_frac"]:
            vals = [h[key] for h in heads]
            lines.append(
                f"- {key}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
                f"min={np.min(vals):.3f}, max={np.max(vals):.3f}"
            )
        lines.append("")

    # Cross-model comparison
    lines.append("## Cross-Model Comparison\n")
    lines.append("### Induction Score by Layer Position (normalized)\n")
    for r in all_results:
        heads = r["heads"]
        n_layers = r["n_layers"]
        layer_scores = []
        for li in range(n_layers):
            layer_heads = [h for h in heads if h["layer"] == li]
            layer_scores.append(np.mean([h["induction_score"] for h in layer_heads]))
        positions = np.linspace(0, 1, n_layers)
        top_layer = int(np.argmax(layer_scores))
        lines.append(
            f"- **{r['model']}**: peak induction at layer {top_layer}/{n_layers} "
            f"(pos={positions[top_layer]:.2f}, score={layer_scores[top_layer]:.3f})"
        )
    lines.append("")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_results = []

    for spec in MODELS:
        result = analyze_model(spec)
        all_results.append(result)

    # Save JSON
    json_path = OUT_DIR / "head_behavior.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved head behavior data to {json_path}")

    # Save report
    report = generate_report(all_results)
    report_path = OUT_DIR / "head_behavior_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
