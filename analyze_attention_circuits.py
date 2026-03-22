"""
Analyze attention circuit spectral structure across pre-trained models.

Extracts QK and OV circuits per head, computes SVD profiles, eigenvalue
distributions, and cross-layer similarity — looking for universal structure
that can be hardcoded as a compressed initialization.

Usage:
    pip install transformers torch numpy matplotlib
    python analyze_attention_circuits.py
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─── Model configs ───────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    hf_id: str
    arch: str  # "llama", "gpt2", "gptneox", "qwen2"

MODELS = [
    ModelSpec("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", "llama"),
    ModelSpec("Pythia-70M",   "EleutherAI/pythia-70m",       "gptneox"),
    ModelSpec("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",         "qwen2"),
    ModelSpec("GPT-2",        "gpt2",                        "gpt2"),
]

OUT_DIR = Path("analysis_results")

# ─── Weight extraction ───────────────────────────────────────────────────────

def extract_attention_weights(model, spec: ModelSpec):
    """
    Returns list of layers, each layer is a dict with:
        q: (num_heads, head_dim, d_model)
        k: (num_kv_heads, head_dim, d_model)
        v: (num_kv_heads, head_dim, d_model)
        o: (d_model, num_heads, head_dim)  -- reshaped from (d_model, d_model)
    """
    sd = model.state_dict()
    layers = []

    if spec.arch == "llama":
        # SmolLM2 uses LlamaForCausalLM
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        d = cfg.hidden_size
        hd = d // n_heads
        for i in range(n_layers):
            pfx = f"model.layers.{i}.self_attn"
            wq = sd[f"{pfx}.q_proj.weight"].float()  # (n_heads*hd, d)
            wk = sd[f"{pfx}.k_proj.weight"].float()  # (n_kv*hd, d)
            wv = sd[f"{pfx}.v_proj.weight"].float()
            wo = sd[f"{pfx}.o_proj.weight"].float()   # (d, n_heads*hd)
            layers.append(dict(
                q=wq.reshape(n_heads, hd, d),
                k=wk.reshape(n_kv, hd, d),
                v=wv.reshape(n_kv, hd, d),
                o=wo.reshape(d, n_heads, hd),
                n_heads=n_heads, n_kv=n_kv, hd=hd, d=d,
            ))

    elif spec.arch == "qwen2":
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        d = cfg.hidden_size
        hd = d // n_heads
        for i in range(n_layers):
            pfx = f"model.layers.{i}.self_attn"
            wq = sd[f"{pfx}.q_proj.weight"].float()
            wk = sd[f"{pfx}.k_proj.weight"].float()
            wv = sd[f"{pfx}.v_proj.weight"].float()
            wo = sd[f"{pfx}.o_proj.weight"].float()
            # Qwen2 may have bias on QKV
            layers.append(dict(
                q=wq.reshape(n_heads, hd, d),
                k=wk.reshape(n_kv, hd, d),
                v=wv.reshape(n_kv, hd, d),
                o=wo.reshape(d, n_heads, hd),
                n_heads=n_heads, n_kv=n_kv, hd=hd, d=d,
            ))

    elif spec.arch == "gptneox":
        # Pythia: fused QKV projection
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_heads = cfg.num_attention_heads
        d = cfg.hidden_size
        hd = d // n_heads
        for i in range(n_layers):
            pfx = f"gpt_neox.layers.{i}.attention"
            qkv = sd[f"{pfx}.query_key_value.weight"].float()  # (3*d, d)
            # GPT-NeoX interleaves: [q0,k0,v0, q1,k1,v1, ...] per head
            qkv = qkv.reshape(n_heads, 3, hd, d)
            wq = qkv[:, 0, :, :]  # (n_heads, hd, d)
            wk = qkv[:, 1, :, :]
            wv = qkv[:, 2, :, :]
            wo = sd[f"{pfx}.dense.weight"].float()  # (d, d)
            layers.append(dict(
                q=wq, k=wk, v=wv,
                o=wo.reshape(d, n_heads, hd),
                n_heads=n_heads, n_kv=n_heads, hd=hd, d=d,
            ))

    elif spec.arch == "gpt2":
        cfg = model.config
        n_layers = cfg.n_layer
        n_heads = cfg.n_head
        d = cfg.n_embd
        hd = d // n_heads
        for i in range(n_layers):
            pfx = f"transformer.h.{i}.attn"
            # GPT-2: c_attn is (d, 3*d) — note: Conv1D stores (in, out)
            c_attn = sd[f"{pfx}.c_attn.weight"].float()  # (d, 3*d)
            wq = c_attn[:, :d].T.reshape(n_heads, hd, d)          # (n_heads, hd, d)
            wk = c_attn[:, d:2*d].T.reshape(n_heads, hd, d)
            wv = c_attn[:, 2*d:].T.reshape(n_heads, hd, d)
            wo = sd[f"{pfx}.c_proj.weight"].float()  # (d, d) Conv1D
            wo = wo.T  # now (d, d) in standard layout
            layers.append(dict(
                q=wq, k=wk, v=wv,
                o=wo.reshape(d, n_heads, hd),
                n_heads=n_heads, n_kv=n_heads, hd=hd, d=d,
            ))

    return layers


# ─── Circuit analysis ────────────────────────────────────────────────────────

def analyze_head_circuits(layers):
    """
    For each layer and head, compute:
    - QK circuit: W_q[h]^T @ W_k[g]  (head_dim x head_dim in residual stream)
      For GQA, g = h // group_size
    - OV circuit: W_o[:, h, :] @ W_v[g]  (d_model x d_model, rank head_dim)
      We analyze the head_dim x head_dim core via SVD
    """
    results = []
    for li, layer in enumerate(layers):
        n_heads = layer["n_heads"]
        n_kv = layer["n_kv"]
        group = n_heads // n_kv
        hd = layer["hd"]
        d = layer["d"]

        layer_results = []
        for h in range(n_heads):
            g = h // group

            # QK circuit in head subspace: (hd, d) @ (d, hd) = (hd, hd)
            qk = layer["q"][h] @ layer["k"][g].T   # (hd, hd)
            # OV circuit in head subspace: (hd, d) @ (d, hd) = (hd, hd)
            # W_o for head h: layer["o"][:, h, :] is (d, hd), transposed = (hd, d)
            ov = layer["o"][:, h, :].T @ layer["v"][g].T  # wait...

            # Let me be careful:
            # v[g]: (hd, d) — maps d_model -> head_dim
            # o[:, h, :]: (d, hd) — maps head_dim -> d_model
            # OV circuit in residual stream: o[:, h, :] @ v[g] = (d, d), rank hd
            # In head subspace: v[g] @ o[:, h, :] would be (hd, hd) but that's VW_O
            # The standard is W_OV = W_O @ W_V = (d, hd) @ (hd, d) = (d, d)
            # We want the SVD of this (d,d) rank-hd matrix
            # But for efficiency, compute the hd x hd core:
            # If W_O @ W_V = U S V^T, the nonzero SVs come from the (hd, hd) core
            # Core = W_O^T_head @ W_O_other... actually let's just do the (hd, hd) product

            # W_V[g] = (hd, d), W_O[:, h] = (d, hd)
            # The OV circuit: W_O[:, h] @ W_V[g] has shape (d, d) rank hd
            # Its singular values = singular values of (hd, hd) matrix:
            #   R @ L where we SVD W_O[:,h] = U1 S1 V1^T and W_V[g] = U2 S2 V2^T
            # Simpler: just compute the hd x hd product V1^T @ U2 (rotated core)
            # Actually easiest: SVD of the full (d, d) is expensive. Instead:
            # The (hd, hd) matrix W_V[g] @ W_O[:,h]^T has same nonzero SVs as W_O @ W_V
            # Wait no. Let A = W_O[:,h] (d, hd), B = W_V[g] (hd, d)
            # AB is (d,d). BA is (hd, hd). They share nonzero singular values.
            ov_core = layer["v"][g] @ layer["o"][:, h, :]  # (hd, d) @ (d, hd) = (hd, hd)

            # SVD
            qk_svd = torch.linalg.svdvals(qk).numpy()
            ov_svd = torch.linalg.svdvals(ov_core).numpy()

            # Eigenvalues (these matrices aren't symmetric, so complex eigenvalues)
            qk_eig = torch.linalg.eigvals(qk).numpy()
            ov_eig = torch.linalg.eigvals(ov_core).numpy()

            # Frobenius norms
            qk_norm = float(torch.linalg.norm(qk, "fro"))
            ov_norm = float(torch.linalg.norm(ov_core, "fro"))

            # Identity similarity: how close is QK to scaled identity?
            qk_normed = qk / (qk_norm / (hd ** 0.5) + 1e-8)
            qk_identity_cos = float((qk_normed * torch.eye(hd)).sum() / hd)

            # OV identity similarity
            ov_normed = ov_core / (ov_norm / (hd ** 0.5) + 1e-8)
            ov_identity_cos = float((ov_normed * torch.eye(hd)).sum() / hd)

            # Effective rank (entropy-based)
            qk_p = qk_svd / (qk_svd.sum() + 1e-10)
            qk_eff_rank = float(np.exp(-np.sum(qk_p * np.log(qk_p + 1e-10))))
            ov_p = ov_svd / (ov_svd.sum() + 1e-10)
            ov_eff_rank = float(np.exp(-np.sum(ov_p * np.log(ov_p + 1e-10))))

            layer_results.append(dict(
                head=h, kv_group=g,
                qk_svs=qk_svd, ov_svs=ov_svd,
                qk_eig=qk_eig, ov_eig=ov_eig,
                qk_norm=qk_norm, ov_norm=ov_norm,
                qk_identity_cos=qk_identity_cos,
                ov_identity_cos=ov_identity_cos,
                qk_eff_rank=qk_eff_rank,
                ov_eff_rank=ov_eff_rank,
            ))
        results.append(layer_results)
    return results


# ─── Individual weight matrix analysis ───────────────────────────────────────

def analyze_individual_weights(layers):
    """Singular value profiles of raw Q, K, V, O matrices."""
    results = []
    for li, layer in enumerate(layers):
        n_heads = layer["n_heads"]
        n_kv = layer["n_kv"]
        q_svs = [torch.linalg.svdvals(layer["q"][h]).numpy() for h in range(n_heads)]
        k_svs = [torch.linalg.svdvals(layer["k"][g]).numpy() for g in range(n_kv)]
        v_svs = [torch.linalg.svdvals(layer["v"][g]).numpy() for g in range(n_kv)]
        o_svs = [torch.linalg.svdvals(layer["o"][:, h, :]).numpy() for h in range(n_heads)]
        results.append(dict(q=q_svs, k=k_svs, v=v_svs, o=o_svs))
    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_model_analysis(model_name, circuit_results, weight_results, pdf):
    n_layers = len(circuit_results)

    # 1. QK and OV identity cosine similarity heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_layers * 0.3)))
    n_heads = len(circuit_results[0])
    qk_id = np.array([[h["qk_identity_cos"] for h in layer] for layer in circuit_results])
    ov_id = np.array([[h["ov_identity_cos"] for h in layer] for layer in circuit_results])

    im0 = axes[0].imshow(qk_id, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title(f"{model_name}: QK Identity Similarity")
    axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ov_id, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title(f"{model_name}: OV Identity Similarity")
    axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # 2. Effective rank by layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    qk_ranks = np.array([[h["qk_eff_rank"] for h in layer] for layer in circuit_results])
    ov_ranks = np.array([[h["ov_eff_rank"] for h in layer] for layer in circuit_results])
    hd = circuit_results[0][0]["qk_svs"].shape[0]

    for h in range(n_heads):
        axes[0].plot(range(n_layers), qk_ranks[:, h], alpha=0.4, linewidth=0.8)
        axes[1].plot(range(n_layers), ov_ranks[:, h], alpha=0.4, linewidth=0.8)
    axes[0].plot(range(n_layers), qk_ranks.mean(axis=1), "k-", linewidth=2, label="mean")
    axes[1].plot(range(n_layers), ov_ranks.mean(axis=1), "k-", linewidth=2, label="mean")
    axes[0].axhline(y=hd, color="r", linestyle="--", alpha=0.5, label=f"max={hd}")
    axes[1].axhline(y=hd, color="r", linestyle="--", alpha=0.5, label=f"max={hd}")
    axes[0].set_title(f"{model_name}: QK Effective Rank by Layer")
    axes[1].set_title(f"{model_name}: OV Effective Rank by Layer")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Effective Rank")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Effective Rank")
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # 3. Singular value decay profiles (averaged across heads, select layers)
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    sample_layers = sorted(set(sample_layers))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for li in sample_layers:
        qk_mean = np.mean([h["qk_svs"] for h in circuit_results[li]], axis=0)
        ov_mean = np.mean([h["ov_svs"] for h in circuit_results[li]], axis=0)
        axes[0, 0].semilogy(qk_mean, label=f"L{li}")
        axes[0, 1].semilogy(ov_mean, label=f"L{li}")

        q_mean = np.mean(weight_results[li]["q"], axis=0)
        v_mean = np.mean(weight_results[li]["v"], axis=0)
        axes[1, 0].semilogy(q_mean, label=f"L{li}")
        axes[1, 1].semilogy(v_mean, label=f"L{li}")

    axes[0, 0].set_title("QK Circuit SVs"); axes[0, 0].legend()
    axes[0, 1].set_title("OV Circuit SVs"); axes[0, 1].legend()
    axes[1, 0].set_title("W_Q SVs"); axes[1, 0].legend()
    axes[1, 1].set_title("W_V SVs"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Singular Value Index"); ax.set_ylabel("Value")
    fig.suptitle(f"{model_name}: Singular Value Profiles", fontsize=14)
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # 4. Eigenvalue scatter (QK and OV) for first and last layer
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, li in enumerate([0, n_layers - 1]):
        for h in range(n_heads):
            qk_e = circuit_results[li][h]["qk_eig"]
            ov_e = circuit_results[li][h]["ov_eig"]
            axes[0, col].scatter(qk_e.real, qk_e.imag, alpha=0.3, s=10)
            axes[1, col].scatter(ov_e.real, ov_e.imag, alpha=0.3, s=10)
        axes[0, col].set_title(f"QK Eigenvalues — Layer {li}")
        axes[1, col].set_title(f"OV Eigenvalues — Layer {li}")
        axes[0, col].axhline(0, color="k", lw=0.5); axes[0, col].axvline(0, color="k", lw=0.5)
        axes[1, col].axhline(0, color="k", lw=0.5); axes[1, col].axvline(0, color="k", lw=0.5)
        axes[0, col].set_xlabel("Real"); axes[0, col].set_ylabel("Imag")
        axes[1, col].set_xlabel("Real"); axes[1, col].set_ylabel("Imag")
    fig.suptitle(f"{model_name}: Eigenvalue Distribution", fontsize=14)
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_cross_model_comparison(all_results, pdf):
    """Compare key metrics across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for model_name, (circuit_results, _) in all_results.items():
        n_layers = len(circuit_results)
        xs = np.linspace(0, 1, n_layers)  # normalize layer position

        qk_id_mean = [np.mean([h["qk_identity_cos"] for h in layer]) for layer in circuit_results]
        ov_id_mean = [np.mean([h["ov_identity_cos"] for h in layer]) for layer in circuit_results]
        qk_rank_mean = [np.mean([h["qk_eff_rank"] for h in layer]) for layer in circuit_results]
        ov_rank_mean = [np.mean([h["ov_eff_rank"] for h in layer]) for layer in circuit_results]

        axes[0, 0].plot(xs, qk_id_mean, label=model_name, linewidth=2)
        axes[0, 1].plot(xs, ov_id_mean, label=model_name, linewidth=2)
        axes[1, 0].plot(xs, qk_rank_mean, label=model_name, linewidth=2)
        axes[1, 1].plot(xs, ov_rank_mean, label=model_name, linewidth=2)

    axes[0, 0].set_title("QK Identity Similarity (mean across heads)")
    axes[0, 1].set_title("OV Identity Similarity (mean across heads)")
    axes[1, 0].set_title("QK Effective Rank (mean across heads)")
    axes[1, 1].set_title("OV Effective Rank (mean across heads)")
    for ax in axes.flat:
        ax.set_xlabel("Relative Layer Position (0=first, 1=last)")
        ax.legend()
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ─── Summary statistics ─────────────────────────────────────────────────────

def compute_summary(model_name, circuit_results, weight_results):
    """Compute summary stats that could parameterize an init scheme."""
    n_layers = len(circuit_results)
    hd = circuit_results[0][0]["qk_svs"].shape[0]

    # Fit exponential decay to OV SVs: sigma_i ~ a * exp(-rate * i)
    all_ov_svs = []
    for layer in circuit_results:
        for head in layer:
            normed = head["ov_svs"] / (head["ov_svs"][0] + 1e-10)
            all_ov_svs.append(normed)
    mean_ov_profile = np.mean(all_ov_svs, axis=0)

    # Fit log-linear: log(sv) = log(a) - rate * i
    indices = np.arange(len(mean_ov_profile))
    valid = mean_ov_profile > 1e-6
    if valid.sum() > 2:
        log_sv = np.log(mean_ov_profile[valid])
        coeffs = np.polyfit(indices[valid], log_sv, 1)
        ov_decay_rate = float(-coeffs[0])
    else:
        ov_decay_rate = 0.0

    # Same for QK
    all_qk_svs = []
    for layer in circuit_results:
        for head in layer:
            normed = head["qk_svs"] / (head["qk_svs"][0] + 1e-10)
            all_qk_svs.append(normed)
    mean_qk_profile = np.mean(all_qk_svs, axis=0)
    valid = mean_qk_profile > 1e-6
    if valid.sum() > 2:
        log_sv = np.log(mean_qk_profile[valid])
        coeffs = np.polyfit(indices[valid], log_sv, 1)
        qk_decay_rate = float(-coeffs[0])
    else:
        qk_decay_rate = 0.0

    # Average identity similarity by layer position
    qk_id_by_pos = [np.mean([h["qk_identity_cos"] for h in layer]) for layer in circuit_results]
    ov_id_by_pos = [np.mean([h["ov_identity_cos"] for h in layer]) for layer in circuit_results]

    return {
        "model": model_name,
        "n_layers": n_layers,
        "head_dim": int(hd),
        "n_heads": circuit_results[0][0]["qk_svs"].shape[0],  # actually head_dim
        "ov_sv_decay_rate": round(ov_decay_rate, 4),
        "qk_sv_decay_rate": round(qk_decay_rate, 4),
        "mean_ov_sv_profile": [round(float(x), 4) for x in mean_ov_profile],
        "mean_qk_sv_profile": [round(float(x), 4) for x in mean_qk_profile],
        "qk_identity_cos_first_layer": round(float(qk_id_by_pos[0]), 4),
        "qk_identity_cos_last_layer": round(float(qk_id_by_pos[-1]), 4),
        "ov_identity_cos_first_layer": round(float(ov_id_by_pos[0]), 4),
        "ov_identity_cos_last_layer": round(float(ov_id_by_pos[-1]), 4),
        "qk_identity_cos_mean": round(float(np.mean(qk_id_by_pos)), 4),
        "ov_identity_cos_mean": round(float(np.mean(ov_id_by_pos)), 4),
        "mean_qk_eff_rank": round(float(np.mean([[h["qk_eff_rank"] for h in l] for l in circuit_results])), 2),
        "mean_ov_eff_rank": round(float(np.mean([[h["ov_eff_rank"] for h in l] for l in circuit_results])), 2),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM

    OUT_DIR.mkdir(exist_ok=True)
    all_results = {}
    all_summaries = []

    for spec in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {spec.name} ({spec.hf_id})...")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
        )
        model.eval()

        print(f"  Extracting attention weights...")
        layers = extract_attention_weights(model, spec)
        print(f"  {len(layers)} layers, {layers[0]['n_heads']} heads, "
              f"{layers[0]['n_kv']} KV heads, head_dim={layers[0]['hd']}, d_model={layers[0]['d']}")

        print(f"  Analyzing circuits...")
        circuit_results = analyze_head_circuits(layers)
        weight_results = analyze_individual_weights(layers)

        summary = compute_summary(spec.name, circuit_results, weight_results)
        all_summaries.append(summary)
        all_results[spec.name] = (circuit_results, weight_results)

        # Free memory
        del model, layers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  QK identity cos (mean): {summary['qk_identity_cos_mean']:.4f}")
        print(f"  OV identity cos (mean): {summary['ov_identity_cos_mean']:.4f}")
        print(f"  QK SV decay rate: {summary['qk_sv_decay_rate']:.4f}")
        print(f"  OV SV decay rate: {summary['ov_sv_decay_rate']:.4f}")
        print(f"  QK eff rank: {summary['mean_qk_eff_rank']:.1f}/{summary['head_dim']}")
        print(f"  OV eff rank: {summary['mean_ov_eff_rank']:.1f}/{summary['head_dim']}")

    # Save summaries
    summary_path = OUT_DIR / "summaries.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummaries saved to {summary_path}")

    # Generate PDF report
    pdf_path = OUT_DIR / "attention_circuit_analysis.pdf"
    print(f"Generating plots to {pdf_path}...")
    with PdfPages(pdf_path) as pdf:
        for spec in MODELS:
            if spec.name in all_results:
                circuit_results, weight_results = all_results[spec.name]
                plot_model_analysis(spec.name, circuit_results, weight_results, pdf)
        plot_cross_model_comparison(all_results, pdf)

    print(f"\nDone! Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
