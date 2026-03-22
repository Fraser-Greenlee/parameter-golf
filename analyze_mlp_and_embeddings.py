"""
Script 3: MLP + Embedding Structure Analysis

Analyzes MLP weight structure (SVD profiles, effective rank, neuron importance,
dead neurons) and embedding structure (SVD, token norms, self-similarity)
across pre-trained models.

Usage:
    source .venv/bin/activate
    python analyze_mlp_and_embeddings.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_attention_circuits import MODELS, ModelSpec

OUT_DIR = Path("analysis_results")

# ─── MLP weight extraction ───────────────────────────────────────────────────

def extract_mlp_weights(model, spec):
    """
    Extract MLP weights per layer.
    Returns list of dicts with up/gate/down projection weights.
    Different architectures have different MLP structures:
    - GPT-2: fc (up) + proj (down), GELU activation
    - GPT-NeoX: dense_h_to_4h (up) + dense_4h_to_h (down), GELU
    - Llama/Qwen2: gate_proj + up_proj + down_proj, SiLU gated
    """
    sd = model.state_dict()
    layers = []

    if spec.arch == "llama":
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        for i in range(n_layers):
            pfx = f"model.layers.{i}.mlp"
            gate = sd[f"{pfx}.gate_proj.weight"].float()  # (hidden, d)
            up = sd[f"{pfx}.up_proj.weight"].float()      # (hidden, d)
            down = sd[f"{pfx}.down_proj.weight"].float()   # (d, hidden)
            layers.append(dict(gate=gate, up=up, down=down, gated=True))

    elif spec.arch == "qwen2":
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        for i in range(n_layers):
            pfx = f"model.layers.{i}.mlp"
            gate = sd[f"{pfx}.gate_proj.weight"].float()
            up = sd[f"{pfx}.up_proj.weight"].float()
            down = sd[f"{pfx}.down_proj.weight"].float()
            layers.append(dict(gate=gate, up=up, down=down, gated=True))

    elif spec.arch == "gptneox":
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        for i in range(n_layers):
            pfx = f"gpt_neox.layers.{i}.mlp"
            up = sd[f"{pfx}.dense_h_to_4h.weight"].float()    # (4d, d)
            down = sd[f"{pfx}.dense_4h_to_h.weight"].float()  # (d, 4d)
            layers.append(dict(up=up, down=down, gated=False))

    elif spec.arch == "gpt2":
        cfg = model.config
        n_layers = cfg.n_layer
        for i in range(n_layers):
            pfx = f"transformer.h.{i}.mlp"
            # GPT-2 Conv1D: stored as (in, out), need to transpose
            up = sd[f"{pfx}.c_fc.weight"].float().T      # (4d, d)
            down = sd[f"{pfx}.c_proj.weight"].float().T   # (d, 4d)
            layers.append(dict(up=up, down=down, gated=False))

    return layers


def extract_embeddings(model, spec):
    """Extract embedding weights."""
    sd = model.state_dict()

    if spec.arch == "llama":
        tok_emb = sd["model.embed_tokens.weight"].float()
        return dict(tok_emb=tok_emb, pos_emb=None)
    elif spec.arch == "qwen2":
        tok_emb = sd["model.embed_tokens.weight"].float()
        return dict(tok_emb=tok_emb, pos_emb=None)
    elif spec.arch == "gptneox":
        tok_emb = sd["gpt_neox.embed_in.weight"].float()
        return dict(tok_emb=tok_emb, pos_emb=None)
    elif spec.arch == "gpt2":
        tok_emb = sd["transformer.wte.weight"].float()
        pos_emb = sd["transformer.wpe.weight"].float()
        return dict(tok_emb=tok_emb, pos_emb=pos_emb)


# ─── MLP analysis ────────────────────────────────────────────────────────────

def effective_rank(svs):
    """Entropy-based effective rank from singular values."""
    p = svs / (svs.sum() + 1e-10)
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def analyze_mlp_layer(layer_weights, layer_idx):
    """Analyze a single MLP layer's weight structure."""
    result = {"layer": layer_idx}

    # SVD of up projection
    up = layer_weights["up"]
    up_svs = torch.linalg.svdvals(up).numpy()
    result["up_svs_top10"] = [round(float(x), 4) for x in up_svs[:10]]
    result["up_frob_norm"] = round(float(torch.linalg.norm(up, "fro")), 4)
    result["up_eff_rank"] = round(effective_rank(up_svs), 2)
    result["up_shape"] = list(up.shape)

    # SVD of down projection
    down = layer_weights["down"]
    down_svs = torch.linalg.svdvals(down).numpy()
    result["down_svs_top10"] = [round(float(x), 4) for x in down_svs[:10]]
    result["down_frob_norm"] = round(float(torch.linalg.norm(down, "fro")), 4)
    result["down_eff_rank"] = round(effective_rank(down_svs), 2)
    result["down_shape"] = list(down.shape)

    # Gate projection (if gated MLP)
    if layer_weights.get("gated"):
        gate = layer_weights["gate"]
        gate_svs = torch.linalg.svdvals(gate).numpy()
        result["gate_svs_top10"] = [round(float(x), 4) for x in gate_svs[:10]]
        result["gate_frob_norm"] = round(float(torch.linalg.norm(gate, "fro")), 4)
        result["gate_eff_rank"] = round(effective_rank(gate_svs), 2)

    # Down @ Up product: effective rank of MLP's residual contribution
    # down: (d, hidden), up: (hidden, d) -> product: (d, d)
    # For efficiency, compute SVD of the product via the smaller matrix
    # The nonzero SVs of down @ up equal those of up @ down (hidden x hidden is bigger)
    # But d < hidden typically, so down @ up is (d, d) - cheaper
    product = down @ up  # (d, d)
    prod_svs = torch.linalg.svdvals(product).numpy()
    result["product_svs_top10"] = [round(float(x), 4) for x in prod_svs[:10]]
    result["product_eff_rank"] = round(effective_rank(prod_svs), 2)
    result["product_frob_norm"] = round(float(torch.linalg.norm(product, "fro")), 4)

    # Neuron importance: L2 norm of each neuron's column in down (output contribution)
    # down: (d, hidden) -> each column is one neuron's output direction
    neuron_norms = torch.linalg.norm(down, dim=0).numpy()  # (hidden,)
    result["neuron_norm_mean"] = round(float(neuron_norms.mean()), 4)
    result["neuron_norm_std"] = round(float(neuron_norms.std()), 4)
    result["neuron_norm_max"] = round(float(neuron_norms.max()), 4)
    result["neuron_norm_min"] = round(float(neuron_norms.min()), 4)
    # Gini coefficient of neuron importance
    sorted_norms = np.sort(neuron_norms)
    n = len(sorted_norms)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_norms) / (n * np.sum(sorted_norms))) - (n + 1) / n)
    result["neuron_gini"] = round(gini, 4)

    return result


def estimate_dead_neurons(model, tokenizer, spec, mlp_layers):
    """Run short text and count neurons with zero activation."""
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of growing complexity, understanding the fundamental "
        "principles that govern language and thought becomes increasingly important."
    )
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)

    # Register hooks to capture activations
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    if spec.arch in ("llama", "qwen2"):
        cfg = model.config
        for i in range(cfg.num_hidden_layers):
            # Hook the up_proj output (before gating)
            layer = model.model.layers[i].mlp.up_proj
            hooks.append(layer.register_forward_hook(make_hook(f"mlp_{i}_up")))
    elif spec.arch == "gptneox":
        for i in range(model.config.num_hidden_layers):
            layer = model.gpt_neox.layers[i].mlp.dense_h_to_4h
            hooks.append(layer.register_forward_hook(make_hook(f"mlp_{i}_up")))
    elif spec.arch == "gpt2":
        for i in range(model.config.n_layer):
            layer = model.transformer.h[i].mlp.c_fc
            hooks.append(layer.register_forward_hook(make_hook(f"mlp_{i}_up")))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    dead_counts = []
    for i in range(len(mlp_layers)):
        key = f"mlp_{i}_up"
        if key in activations:
            act = activations[key]  # (batch, seq, hidden)
            # A neuron is "dead" if it never activates positively across all positions
            max_act = act.max(dim=0).values.max(dim=0).values  # (hidden,)
            dead = int((max_act <= 0).sum())
            total = int(max_act.shape[0])
            dead_counts.append({"layer": i, "dead_neurons": dead, "total_neurons": total,
                                "dead_frac": round(dead / total, 4)})
        else:
            dead_counts.append({"layer": i, "dead_neurons": -1, "total_neurons": -1,
                                "dead_frac": -1})

    return dead_counts


# ─── Embedding analysis ──────────────────────────────────────────────────────

def analyze_embeddings(emb_dict):
    """Analyze embedding matrix structure."""
    result = {}

    tok_emb = emb_dict["tok_emb"]  # (vocab_size, d_model)
    vocab_size, d_model = tok_emb.shape
    result["vocab_size"] = int(vocab_size)
    result["d_model"] = int(d_model)

    # SVD of token embeddings
    svs = torch.linalg.svdvals(tok_emb).numpy()
    result["tok_emb_svs_top20"] = [round(float(x), 4) for x in svs[:20]]
    result["tok_emb_eff_rank"] = round(effective_rank(svs), 2)
    result["tok_emb_frob_norm"] = round(float(torch.linalg.norm(tok_emb, "fro")), 4)

    # Effective dimensionality: how many SVs needed for 90% of total?
    cumulative = np.cumsum(svs ** 2) / np.sum(svs ** 2)
    dims_90 = int(np.searchsorted(cumulative, 0.9) + 1)
    dims_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    dims_99 = int(np.searchsorted(cumulative, 0.99) + 1)
    result["dims_for_90pct_var"] = dims_90
    result["dims_for_95pct_var"] = dims_95
    result["dims_for_99pct_var"] = dims_99

    # Token norm distribution
    token_norms = torch.linalg.norm(tok_emb, dim=1).numpy()
    result["token_norm_mean"] = round(float(token_norms.mean()), 4)
    result["token_norm_std"] = round(float(token_norms.std()), 4)
    result["token_norm_max"] = round(float(token_norms.max()), 4)
    result["token_norm_min"] = round(float(token_norms.min()), 4)

    # Embedding self-similarity: mean cosine similarity
    # For efficiency, sample if vocab is large
    if vocab_size > 1000:
        rng = np.random.RandomState(42)
        idx = rng.choice(vocab_size, 1000, replace=False)
        sample = tok_emb[idx]
    else:
        sample = tok_emb
    norms = torch.linalg.norm(sample, dim=1, keepdim=True).clamp(min=1e-8)
    normed = sample / norms
    sim_matrix = (normed @ normed.T).numpy()
    # Mean off-diagonal similarity
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    result["mean_cosine_similarity"] = round(float(sim_matrix[mask].mean()), 4)
    result["std_cosine_similarity"] = round(float(sim_matrix[mask].std()), 4)

    # Position embeddings (GPT-2 only)
    if emb_dict["pos_emb"] is not None:
        pos_emb = emb_dict["pos_emb"]
        pos_svs = torch.linalg.svdvals(pos_emb).numpy()
        result["pos_emb_svs_top20"] = [round(float(x), 4) for x in pos_svs[:20]]
        result["pos_emb_eff_rank"] = round(effective_rank(pos_svs), 2)
        result["pos_emb_shape"] = list(pos_emb.shape)

        # Position embedding distance structure
        pos_norms = torch.linalg.norm(pos_emb, dim=1).numpy()
        result["pos_norm_mean"] = round(float(pos_norms.mean()), 4)
        result["pos_norm_std"] = round(float(pos_norms.std()), 4)

    return result


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_model(spec):
    """Full MLP + embedding analysis for one model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}")
    print(f"Analyzing {spec.name} ({spec.hf_id})...")
    print(f"{'=' * 60}")

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)

    # MLP analysis
    print(f"  Extracting MLP weights...")
    mlp_layers = extract_mlp_weights(model, spec)
    print(f"  {len(mlp_layers)} MLP layers")

    print(f"  Analyzing MLP structure...")
    mlp_results = []
    for i, layer_w in enumerate(mlp_layers):
        mlp_results.append(analyze_mlp_layer(layer_w, i))
        if (i + 1) % 5 == 0:
            print(f"    Layer {i + 1}/{len(mlp_layers)} done")

    # Dead neuron estimation
    print(f"  Estimating dead neurons...")
    dead_neurons = estimate_dead_neurons(model, tokenizer, spec, mlp_layers)

    # Embedding analysis
    print(f"  Analyzing embeddings...")
    emb_dict = extract_embeddings(model, spec)
    emb_results = analyze_embeddings(emb_dict)

    del model, mlp_layers, emb_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary statistics
    n_layers = len(mlp_results)
    up_ranks = [r["up_eff_rank"] for r in mlp_results]
    down_ranks = [r["down_eff_rank"] for r in mlp_results]
    prod_ranks = [r["product_eff_rank"] for r in mlp_results]
    up_norms = [r["up_frob_norm"] for r in mlp_results]
    down_norms = [r["down_frob_norm"] for r in mlp_results]

    print(f"  Up eff rank: {np.mean(up_ranks):.1f} (std={np.std(up_ranks):.1f})")
    print(f"  Down eff rank: {np.mean(down_ranks):.1f} (std={np.std(down_ranks):.1f})")
    print(f"  Product eff rank: {np.mean(prod_ranks):.1f} (std={np.std(prod_ranks):.1f})")
    print(f"  Embedding eff rank: {emb_results['tok_emb_eff_rank']}")

    return {
        "model": spec.name,
        "mlp_layers": mlp_results,
        "dead_neurons": dead_neurons,
        "embeddings": emb_results,
    }


# ─── Report generation ───────────────────────────────────────────────────────

def generate_report(all_results):
    """Generate markdown report."""
    lines = ["# MLP and Embedding Structure Report\n"]

    # Cross-model MLP summary
    lines.append("## MLP Effective Rank by Model\n")
    lines.append("| Model | Up Rank (mean) | Down Rank (mean) | Product Rank (mean) | Neuron Gini (mean) |")
    lines.append("|---|---|---|---|---|")
    for r in all_results:
        mlp = r["mlp_layers"]
        up_r = np.mean([l["up_eff_rank"] for l in mlp])
        down_r = np.mean([l["down_eff_rank"] for l in mlp])
        prod_r = np.mean([l["product_eff_rank"] for l in mlp])
        gini = np.mean([l["neuron_gini"] for l in mlp])
        lines.append(f"| {r['model']} | {up_r:.1f} | {down_r:.1f} | {prod_r:.1f} | {gini:.3f} |")
    lines.append("")

    # Dead neurons summary
    lines.append("## Dead Neurons\n")
    lines.append("| Model | Total Dead | Total Neurons | Dead % |")
    lines.append("|---|---|---|---|")
    for r in all_results:
        dn = r["dead_neurons"]
        valid = [d for d in dn if d["dead_neurons"] >= 0]
        total_dead = sum(d["dead_neurons"] for d in valid)
        total_neurons = sum(d["total_neurons"] for d in valid)
        pct = 100 * total_dead / max(total_neurons, 1)
        lines.append(f"| {r['model']} | {total_dead} | {total_neurons} | {pct:.1f}% |")
    lines.append("")

    # Embedding summary
    lines.append("## Embedding Structure\n")
    lines.append("| Model | Vocab | d_model | Eff Rank | Dims 90% | Dims 99% | Mean Cos Sim |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        e = r["embeddings"]
        lines.append(
            f"| {r['model']} | {e['vocab_size']} | {e['d_model']} | "
            f"{e['tok_emb_eff_rank']} | {e['dims_for_90pct_var']} | "
            f"{e['dims_for_99pct_var']} | {e['mean_cosine_similarity']:.3f} |"
        )
    lines.append("")

    # Per-model details
    for r in all_results:
        lines.append(f"## {r['model']} Details\n")

        # MLP rank by layer
        lines.append("### MLP Effective Rank by Layer\n")
        lines.append("| Layer | Up Rank | Down Rank | Product Rank | Neuron Gini | Dead Neurons |")
        lines.append("|---|---|---|---|---|---|")
        for i, mlp in enumerate(r["mlp_layers"]):
            dn = r["dead_neurons"][i]
            dead_str = f"{dn['dead_neurons']}/{dn['total_neurons']}" if dn["dead_neurons"] >= 0 else "N/A"
            lines.append(
                f"| {i} | {mlp['up_eff_rank']:.1f} | {mlp['down_eff_rank']:.1f} | "
                f"{mlp['product_eff_rank']:.1f} | {mlp['neuron_gini']:.3f} | {dead_str} |"
            )
        lines.append("")

        # Frobenius norm growth
        lines.append("### Frobenius Norm by Layer\n")
        lines.append("| Layer | Up Norm | Down Norm | Product Norm |")
        lines.append("|---|---|---|---|")
        for mlp in r["mlp_layers"]:
            lines.append(
                f"| {mlp['layer']} | {mlp['up_frob_norm']:.2f} | "
                f"{mlp['down_frob_norm']:.2f} | {mlp['product_frob_norm']:.2f} |"
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
    json_path = OUT_DIR / "mlp_embedding_structure.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved data to {json_path}")

    # Save report
    report = generate_report(all_results)
    report_path = OUT_DIR / "mlp_embedding_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
