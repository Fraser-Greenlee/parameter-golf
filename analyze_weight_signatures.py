"""
Script 2: Weight Structure Per Head Type

For each head type identified in Script 1, characterize QK and OV circuit
weight matrices: SVD decomposition, NMF factorization, layer position
dependence, per-type spectral templates, and weight-to-behavior correlation.

Depends on: analyze_head_behavior.py (head_behavior.json)

Usage:
    source .venv/bin/activate
    python analyze_weight_signatures.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import pearsonr

from analyze_attention_circuits import MODELS, ModelSpec, extract_attention_weights

OUT_DIR = Path("analysis_results")


# ─── Circuit extraction ──────────────────────────────────────────────────────

def extract_circuits(layers):
    """
    Extract QK and OV circuit matrices for every head.
    Returns list of (layer_idx, head_idx, qk_core, ov_core) tuples.
    """
    circuits = []
    for li, layer in enumerate(layers):
        n_heads = layer["n_heads"]
        n_kv = layer["n_kv"]
        group = n_heads // n_kv
        for h in range(n_heads):
            g = h // group
            qk = layer["q"][h] @ layer["k"][g].T   # (hd, hd)
            ov = layer["v"][g] @ layer["o"][:, h, :]  # (hd, hd)
            circuits.append((li, h, qk, ov))
    return circuits


# ─── Per-head circuit analysis ────────────────────────────────────────────────

def analyze_circuit_detailed(qk, ov):
    """Detailed SVD/eigenvalue analysis of a single head's circuits."""
    hd = qk.shape[0]

    # QK circuit
    qk_U, qk_S, qk_Vh = torch.linalg.svd(qk)
    qk_svs = qk_S.numpy()
    qk_eig = torch.linalg.eigvals(qk).numpy()
    qk_norm = float(torch.linalg.norm(qk, "fro"))

    # Effective rank
    qk_p = qk_svs / (qk_svs.sum() + 1e-10)
    qk_eff_rank = float(np.exp(-np.sum(qk_p * np.log(qk_p + 1e-10))))

    # SV decay rate (log-linear fit)
    indices = np.arange(len(qk_svs))
    normed_svs = qk_svs / (qk_svs[0] + 1e-10)
    valid = normed_svs > 1e-6
    if valid.sum() > 2:
        log_sv = np.log(normed_svs[valid])
        coeffs = np.polyfit(indices[valid], log_sv, 1)
        qk_decay = float(-coeffs[0])
    else:
        qk_decay = 0.0

    # OV circuit
    ov_U, ov_S, ov_Vh = torch.linalg.svd(ov)
    ov_svs = ov_S.numpy()
    ov_eig = torch.linalg.eigvals(ov).numpy()
    ov_norm = float(torch.linalg.norm(ov, "fro"))

    ov_p = ov_svs / (ov_svs.sum() + 1e-10)
    ov_eff_rank = float(np.exp(-np.sum(ov_p * np.log(ov_p + 1e-10))))

    normed_ov = ov_svs / (ov_svs[0] + 1e-10)
    valid_ov = normed_ov > 1e-6
    if valid_ov.sum() > 2:
        log_sv = np.log(normed_ov[valid_ov])
        coeffs = np.polyfit(indices[valid_ov], log_sv, 1)
        ov_decay = float(-coeffs[0])
    else:
        ov_decay = 0.0

    # Identity cosine
    qk_normed = qk / (qk_norm / (hd ** 0.5) + 1e-8)
    qk_id_cos = float((qk_normed * torch.eye(hd)).sum() / hd)

    ov_normed = ov / (ov_norm / (hd ** 0.5) + 1e-8)
    ov_id_cos = float((ov_normed * torch.eye(hd)).sum() / hd)

    # Eigenvalue statistics
    qk_eig_pos_frac = float(np.mean(qk_eig.real > 0))
    ov_eig_pos_frac = float(np.mean(ov_eig.real > 0))
    qk_eig_real_mean = float(np.mean(qk_eig.real))
    ov_eig_real_mean = float(np.mean(ov_eig.real))

    # Top-k SV concentration (what fraction of energy in top k SVs)
    total_energy = float(np.sum(qk_svs ** 2))
    qk_top1_frac = float(qk_svs[0] ** 2 / (total_energy + 1e-10))
    qk_top5_frac = float(np.sum(qk_svs[:5] ** 2) / (total_energy + 1e-10))

    total_energy_ov = float(np.sum(ov_svs ** 2))
    ov_top1_frac = float(ov_svs[0] ** 2 / (total_energy_ov + 1e-10))
    ov_top5_frac = float(np.sum(ov_svs[:5] ** 2) / (total_energy_ov + 1e-10))

    return {
        "qk_svs_normed": [round(float(x), 4) for x in normed_svs],
        "ov_svs_normed": [round(float(x), 4) for x in normed_ov],
        "qk_decay_rate": round(qk_decay, 4),
        "ov_decay_rate": round(ov_decay, 4),
        "qk_eff_rank": round(qk_eff_rank, 2),
        "ov_eff_rank": round(ov_eff_rank, 2),
        "qk_norm": round(qk_norm, 4),
        "ov_norm": round(ov_norm, 4),
        "qk_id_cos": round(qk_id_cos, 4),
        "ov_id_cos": round(ov_id_cos, 4),
        "qk_eig_pos_frac": round(qk_eig_pos_frac, 4),
        "ov_eig_pos_frac": round(ov_eig_pos_frac, 4),
        "qk_eig_real_mean": round(qk_eig_real_mean, 4),
        "ov_eig_real_mean": round(ov_eig_real_mean, 4),
        "qk_top1_frac": round(qk_top1_frac, 4),
        "qk_top5_frac": round(qk_top5_frac, 4),
        "ov_top1_frac": round(ov_top1_frac, 4),
        "ov_top5_frac": round(ov_top5_frac, 4),
    }


# ─── NMF factorization ───────────────────────────────────────────────────────

def nmf_factorize(matrix_np, n_components=4, max_iter=200):
    """
    Simple NMF on the absolute values of a matrix.
    Returns W, H such that |matrix| ~ W @ H.
    """
    M = np.abs(matrix_np).astype(np.float64)
    m, n = M.shape
    rng = np.random.RandomState(42)

    W = rng.rand(m, n_components).astype(np.float64) + 0.01
    H = rng.rand(n_components, n).astype(np.float64) + 0.01

    for _ in range(max_iter):
        # Multiplicative update rules
        WH = W @ H + 1e-10
        H *= (W.T @ M) / (W.T @ WH + 1e-10)
        WH = W @ H + 1e-10
        W *= (M @ H.T) / (WH @ H.T + 1e-10)

    reconstruction_error = float(np.linalg.norm(M - W @ H, "fro") / np.linalg.norm(M, "fro"))

    # Component importance (by column norm of W * row norm of H)
    component_importance = []
    for c in range(n_components):
        importance = float(np.linalg.norm(W[:, c]) * np.linalg.norm(H[c, :]))
        component_importance.append(round(importance, 4))

    return {
        "reconstruction_error": round(reconstruction_error, 4),
        "component_importance": component_importance,
        "n_components": n_components,
    }


# ─── Per-type aggregation ────────────────────────────────────────────────────

def aggregate_by_type(head_data):
    """
    Aggregate circuit statistics by head type.
    head_data: list of dicts with 'type' and circuit analysis fields.
    """
    types = defaultdict(list)
    for hd in head_data:
        types[hd["type"]].append(hd)

    aggregated = {}
    for htype, heads in types.items():
        n = len(heads)
        metrics = {}
        for key in ["qk_decay_rate", "ov_decay_rate", "qk_eff_rank", "ov_eff_rank",
                     "qk_norm", "ov_norm", "qk_id_cos", "ov_id_cos",
                     "qk_eig_pos_frac", "ov_eig_pos_frac",
                     "qk_eig_real_mean", "ov_eig_real_mean",
                     "qk_top1_frac", "qk_top5_frac",
                     "ov_top1_frac", "ov_top5_frac"]:
            vals = [h[key] for h in heads]
            metrics[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        # Average spectral template (normalized SVs)
        qk_profiles = np.array([h["qk_svs_normed"] for h in heads])
        ov_profiles = np.array([h["ov_svs_normed"] for h in heads])
        metrics["qk_spectral_template"] = [round(float(x), 4) for x in qk_profiles.mean(axis=0)]
        metrics["ov_spectral_template"] = [round(float(x), 4) for x in ov_profiles.mean(axis=0)]
        metrics["qk_spectral_std"] = [round(float(x), 4) for x in qk_profiles.std(axis=0)]
        metrics["ov_spectral_std"] = [round(float(x), 4) for x in ov_profiles.std(axis=0)]

        # Layer position distribution
        layer_positions = [h["layer_position"] for h in heads]
        metrics["layer_position_mean"] = round(float(np.mean(layer_positions)), 4)
        metrics["layer_position_std"] = round(float(np.std(layer_positions)), 4)

        aggregated[htype] = {"count": n, "metrics": metrics}

    return aggregated


# ─── Weight-to-behavior correlation ──────────────────────────────────────────

def compute_correlations(head_data):
    """Correlate weight-space metrics with behavioral scores."""
    weight_keys = ["qk_decay_rate", "ov_decay_rate", "qk_eff_rank", "ov_eff_rank",
                   "qk_id_cos", "ov_id_cos", "qk_norm", "ov_norm"]
    behavior_keys = ["prev_token_score", "induction_score", "positional_score", "entropy"]

    correlations = {}
    for wk in weight_keys:
        for bk in behavior_keys:
            w_vals = [h[wk] for h in head_data]
            b_vals = [h[bk] for h in head_data]
            if len(w_vals) > 3 and np.std(w_vals) > 1e-10 and np.std(b_vals) > 1e-10:
                r, p = pearsonr(w_vals, b_vals)
                correlations[f"{wk}_vs_{bk}"] = {
                    "r": round(float(r), 4),
                    "p": round(float(p), 6),
                }

    return correlations


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_model(spec, behavior_data):
    """Run weight signature analysis for one model."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'=' * 60}")
    print(f"Analyzing weight signatures: {spec.name}")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()

    # Extract circuits
    print(f"  Extracting attention weights and circuits...")
    layers = extract_attention_weights(model, spec)
    circuits = extract_circuits(layers)
    del model, layers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Look up head types from behavior data
    head_types = {}
    head_behavior = {}
    for hb in behavior_data["heads"]:
        key = (hb["layer"], hb["head"])
        head_types[key] = hb["type"]
        head_behavior[key] = hb

    n_layers = behavior_data["n_layers"]

    # Analyze each head
    print(f"  Analyzing {len(circuits)} head circuits...")
    head_data = []
    for li, h, qk, ov in circuits:
        analysis = analyze_circuit_detailed(qk, ov)
        key = (li, h)
        analysis["layer"] = li
        analysis["head"] = h
        analysis["layer_position"] = li / max(n_layers - 1, 1)
        analysis["type"] = head_types.get(key, "unknown")

        # Copy behavioral scores
        if key in head_behavior:
            for bk in ["prev_token_score", "induction_score", "positional_score", "entropy"]:
                analysis[bk] = head_behavior[key][bk]

        # NMF on QK and OV circuits (sample — do every 4th head to save time)
        if h % 4 == 0:
            qk_nmf = nmf_factorize(qk.numpy(), n_components=4)
            ov_nmf = nmf_factorize(ov.numpy(), n_components=4)
            analysis["qk_nmf"] = qk_nmf
            analysis["ov_nmf"] = ov_nmf

        head_data.append(analysis)

        if len(head_data) % 50 == 0:
            print(f"    {len(head_data)}/{len(circuits)} heads done")

    # Aggregate by type
    print(f"  Aggregating by head type...")
    type_aggregated = aggregate_by_type(head_data)

    # Correlations
    print(f"  Computing weight-behavior correlations...")
    correlations = compute_correlations(head_data)

    # NMF summary by type
    nmf_by_type = defaultdict(lambda: {"qk_errors": [], "ov_errors": [],
                                        "qk_importance": [], "ov_importance": []})
    for hd in head_data:
        if "qk_nmf" in hd:
            t = hd["type"]
            nmf_by_type[t]["qk_errors"].append(hd["qk_nmf"]["reconstruction_error"])
            nmf_by_type[t]["ov_errors"].append(hd["ov_nmf"]["reconstruction_error"])
            nmf_by_type[t]["qk_importance"].append(hd["qk_nmf"]["component_importance"])
            nmf_by_type[t]["ov_importance"].append(hd["ov_nmf"]["component_importance"])

    nmf_summary = {}
    for t, data in nmf_by_type.items():
        nmf_summary[t] = {
            "qk_recon_error_mean": round(float(np.mean(data["qk_errors"])), 4),
            "ov_recon_error_mean": round(float(np.mean(data["ov_errors"])), 4),
            "qk_component_importance_mean": [round(float(x), 4) for x in np.mean(data["qk_importance"], axis=0)],
            "ov_component_importance_mean": [round(float(x), 4) for x in np.mean(data["ov_importance"], axis=0)],
        }

    return {
        "model": spec.name,
        "type_aggregated": type_aggregated,
        "correlations": correlations,
        "nmf_summary": nmf_summary,
        "n_heads_analyzed": len(head_data),
    }


# ─── Report generation ───────────────────────────────────────────────────────

def generate_report(all_results):
    """Generate markdown report."""
    lines = ["# Weight Signature Analysis Report\n"]

    # Per-type canonical signatures across all models
    lines.append("## Canonical Spectral Signatures by Head Type\n")

    # Collect all types
    all_types = set()
    for r in all_results:
        all_types.update(r["type_aggregated"].keys())
    all_types = sorted(all_types)

    for htype in all_types:
        lines.append(f"### {htype}\n")

        for r in all_results:
            if htype not in r["type_aggregated"]:
                continue
            ta = r["type_aggregated"][htype]
            m = ta["metrics"]
            lines.append(f"**{r['model']}** ({ta['count']} heads):")
            lines.append(f"- QK: decay={m['qk_decay_rate']['mean']:.3f}+-{m['qk_decay_rate']['std']:.3f}, "
                         f"eff_rank={m['qk_eff_rank']['mean']:.1f}, "
                         f"id_cos={m['qk_id_cos']['mean']:.3f}, "
                         f"eig_pos={m['qk_eig_pos_frac']['mean']:.3f}")
            lines.append(f"- OV: decay={m['ov_decay_rate']['mean']:.3f}+-{m['ov_decay_rate']['std']:.3f}, "
                         f"eff_rank={m['ov_eff_rank']['mean']:.1f}, "
                         f"id_cos={m['ov_id_cos']['mean']:.3f}, "
                         f"eig_pos={m['ov_eig_pos_frac']['mean']:.3f}")
            lines.append(f"- Layer position: {m['layer_position_mean']:.2f}+-{m['layer_position_std']:.2f}")
            lines.append("")

    # NMF results
    lines.append("## NMF Factorization Results\n")
    for r in all_results:
        lines.append(f"### {r['model']}\n")
        for htype, nmf in r["nmf_summary"].items():
            lines.append(f"**{htype}**:")
            lines.append(f"- QK recon error: {nmf['qk_recon_error_mean']:.3f}")
            lines.append(f"- OV recon error: {nmf['ov_recon_error_mean']:.3f}")
            lines.append(f"- QK component importance: {nmf['qk_component_importance_mean']}")
            lines.append(f"- OV component importance: {nmf['ov_component_importance_mean']}")
            lines.append("")

    # Top correlations
    lines.append("## Weight-Behavior Correlations\n")
    for r in all_results:
        lines.append(f"### {r['model']}\n")
        # Sort by absolute correlation
        sorted_corrs = sorted(r["correlations"].items(), key=lambda x: -abs(x[1]["r"]))
        lines.append("| Correlation | r | p-value |")
        lines.append("|---|---|---|")
        for name, vals in sorted_corrs[:15]:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            lines.append(f"| {name} | {vals['r']:.3f}{sig} | {vals['p']:.1e} |")
        lines.append("")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Load head behavior data
    behavior_path = OUT_DIR / "head_behavior.json"
    if not behavior_path.exists():
        print(f"ERROR: {behavior_path} not found. Run analyze_head_behavior.py first.")
        return
    with open(behavior_path) as f:
        all_behavior = json.load(f)

    behavior_by_model = {b["model"]: b for b in all_behavior}

    all_results = []
    for spec in MODELS:
        if spec.name not in behavior_by_model:
            print(f"WARNING: No behavior data for {spec.name}, skipping.")
            continue
        result = analyze_model(spec, behavior_by_model[spec.name])
        all_results.append(result)

    # Save JSON
    json_path = OUT_DIR / "weight_signatures.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved data to {json_path}")

    # Save report
    report = generate_report(all_results)
    report_path = OUT_DIR / "weight_signatures_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
