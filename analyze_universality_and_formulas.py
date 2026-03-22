"""
Script 4: Universality Analysis + Code Generation

Synthesizes results from Scripts 1-3 to:
1. Assess universality of each head type and MLP structure across models
2. Derive parametric formulas for universal structures
3. Validate formulas via forward-pass replacement in Pythia-70M
4. Rank structures by compressibility (universality_score)

Depends on: head_behavior.json, weight_signatures.json, mlp_embedding_structure.json

Usage:
    source .venv/bin/activate
    python analyze_universality_and_formulas.py
"""

import json
import textwrap
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from analyze_attention_circuits import MODELS, ModelSpec, extract_attention_weights

OUT_DIR = Path("analysis_results")


# ─── Load all prior results ──────────────────────────────────────────────────

def load_results():
    """Load outputs from Scripts 1-3."""
    results = {}

    p = OUT_DIR / "head_behavior.json"
    if p.exists():
        with open(p) as f:
            results["behavior"] = json.load(f)
    else:
        print(f"WARNING: {p} not found")
        results["behavior"] = []

    p = OUT_DIR / "weight_signatures.json"
    if p.exists():
        with open(p) as f:
            results["signatures"] = json.load(f)
    else:
        print(f"WARNING: {p} not found")
        results["signatures"] = []

    p = OUT_DIR / "mlp_embedding_structure.json"
    if p.exists():
        with open(p) as f:
            results["mlp_emb"] = json.load(f)
    else:
        print(f"WARNING: {p} not found")
        results["mlp_emb"] = []

    return results


# ─── Universality analysis ────────────────────────────────────────────────────

def analyze_universality(results):
    """
    For each head type: prevalence, consistency, layer position pattern.
    For MLP: spectral consistency, rank growth pattern.
    """
    behavior = results["behavior"]
    signatures = results["signatures"]
    mlp_emb = results["mlp_emb"]

    # Head type universality
    head_universality = {}
    all_types = set()
    for b in behavior:
        all_types.update(b["type_distribution"].keys())

    for htype in sorted(all_types):
        type_info = {"prevalence": {}, "consistency": {}, "layer_pattern": {}}

        # Prevalence per model
        for b in behavior:
            total = b["n_layers"] * b["n_heads"]
            count = b["type_distribution"].get(htype, 0)
            type_info["prevalence"][b["model"]] = round(count / total, 4)

        # Cross-model prevalence stats
        prevs = list(type_info["prevalence"].values())
        type_info["prevalence_mean"] = round(float(np.mean(prevs)), 4)
        type_info["prevalence_std"] = round(float(np.std(prevs)), 4)

        # Consistency from weight signatures
        for sig in signatures:
            ta = sig.get("type_aggregated", {})
            if htype in ta:
                metrics = ta[htype]["metrics"]
                # Key consistency metrics: how tight are the distributions?
                for key in ["qk_decay_rate", "ov_decay_rate", "qk_id_cos", "ov_id_cos"]:
                    if key in metrics:
                        m = metrics[key]
                        cv = abs(m["std"] / (m["mean"] + 1e-10))  # coefficient of variation
                        if sig["model"] not in type_info["consistency"]:
                            type_info["consistency"][sig["model"]] = {}
                        type_info["consistency"][sig["model"]][key] = round(cv, 4)

        # Layer position pattern
        for b in behavior:
            n_layers = b["n_layers"]
            layer_counts = defaultdict(int)
            for h in b["heads"]:
                if h["type"] == htype:
                    # Bin into early/middle/late thirds
                    pos = h["layer"] / max(n_layers - 1, 1)
                    if pos < 0.33:
                        layer_counts["early"] += 1
                    elif pos < 0.67:
                        layer_counts["middle"] += 1
                    else:
                        layer_counts["late"] += 1
            total = sum(layer_counts.values())
            if total > 0:
                type_info["layer_pattern"][b["model"]] = {
                    k: round(v / total, 3) for k, v in layer_counts.items()
                }

        head_universality[htype] = type_info

    # MLP universality
    mlp_universality = {}
    if mlp_emb:
        # Rank growth pattern across models
        for me in mlp_emb:
            model = me["model"]
            layers = me["mlp_layers"]
            up_ranks = [l["up_eff_rank"] for l in layers]
            down_ranks = [l["down_eff_rank"] for l in layers]
            prod_ranks = [l["product_eff_rank"] for l in layers]

            # Fit linear trend to rank vs layer
            n = len(up_ranks)
            x = np.arange(n) / max(n - 1, 1)
            up_slope = float(np.polyfit(x, up_ranks, 1)[0]) if n > 1 else 0
            down_slope = float(np.polyfit(x, down_ranks, 1)[0]) if n > 1 else 0
            prod_slope = float(np.polyfit(x, prod_ranks, 1)[0]) if n > 1 else 0

            mlp_universality[model] = {
                "up_rank_mean": round(float(np.mean(up_ranks)), 2),
                "up_rank_slope": round(up_slope, 2),
                "down_rank_mean": round(float(np.mean(down_ranks)), 2),
                "down_rank_slope": round(down_slope, 2),
                "product_rank_mean": round(float(np.mean(prod_ranks)), 2),
                "product_rank_slope": round(prod_slope, 2),
                "norm_growth": [round(l["up_frob_norm"], 2) for l in layers],
            }

    return head_universality, mlp_universality


# ─── Parametric formulas ─────────────────────────────────────────────────────

def derive_formulas(results, head_universality):
    """
    Derive parametric formulas for each universal structure.
    Returns dict of formula name -> {code, parameters, description}.
    """
    signatures = results["signatures"]
    formulas = {}

    # Collect per-type spectral templates across all models
    type_templates = defaultdict(lambda: {"qk": [], "ov": [], "params": {}})

    for sig in signatures:
        for htype, ta in sig.get("type_aggregated", {}).items():
            m = ta["metrics"]
            if "qk_spectral_template" in m:
                type_templates[htype]["qk"].append(m["qk_spectral_template"])
            if "ov_spectral_template" in m:
                type_templates[htype]["ov"].append(m["ov_spectral_template"])
            # Collect key parameters
            for key in ["qk_decay_rate", "ov_decay_rate", "qk_id_cos", "ov_id_cos",
                         "qk_norm", "ov_norm"]:
                if key in m:
                    if key not in type_templates[htype]["params"]:
                        type_templates[htype]["params"][key] = []
                    type_templates[htype]["params"][key].append(m[key]["mean"])

    # Formula 1: Previous-token head init
    if "previous_token" in type_templates:
        pt = type_templates["previous_token"]
        qk_id_cos = np.mean(pt["params"].get("qk_id_cos", [0]))
        qk_norm = np.mean(pt["params"].get("qk_norm", [1]))
        ov_id_cos = np.mean(pt["params"].get("ov_id_cos", [0]))
        ov_decay = np.mean(pt["params"].get("ov_decay_rate", [0.04]))

        formulas["previous_token_head"] = {
            "description": "Previous-token head: QK biased toward shifted-diagonal, OV near identity with decay",
            "parameters": {
                "qk_id_cos": round(float(qk_id_cos), 4),
                "qk_norm_scale": round(float(qk_norm), 4),
                "ov_id_cos": round(float(ov_id_cos), 4),
                "ov_decay_rate": round(float(ov_decay), 4),
            },
            "code": textwrap.dedent(f"""\
                def init_previous_token_head(hd, qk_scale={qk_norm:.4f}, ov_decay={ov_decay:.4f}):
                    # QK circuit: shifted diagonal (attend to i-1)
                    qk = torch.zeros(hd, hd)
                    for i in range(1, hd):
                        qk[i, i-1] = 1.0
                    qk = qk * qk_scale / (hd ** 0.5)
                    # OV circuit: identity with exponential SV decay
                    sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
                    ov = torch.diag(sigmas)
                    return qk, ov
            """),
        }

    # Formula 2: Induction head init
    if "induction" in type_templates:
        ind = type_templates["induction"]
        qk_id_cos = np.mean(ind["params"].get("qk_id_cos", [0]))
        qk_norm = np.mean(ind["params"].get("qk_norm", [1]))
        ov_id_cos = np.mean(ind["params"].get("ov_id_cos", [0]))
        ov_decay = np.mean(ind["params"].get("ov_decay_rate", [0.04]))

        formulas["induction_head"] = {
            "description": "Induction head: QK near identity (match tokens), OV near identity (copy)",
            "parameters": {
                "qk_id_cos": round(float(qk_id_cos), 4),
                "qk_norm_scale": round(float(qk_norm), 4),
                "ov_id_cos": round(float(ov_id_cos), 4),
                "ov_decay_rate": round(float(ov_decay), 4),
            },
            "code": textwrap.dedent(f"""\
                def init_induction_head(hd, beta={qk_norm:.4f}, gamma={ov_id_cos:.4f}, ov_decay={ov_decay:.4f}):
                    # QK circuit: identity-like (attend to matching tokens)
                    qk = beta * torch.eye(hd)
                    # OV circuit: identity with decay (copy content)
                    sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
                    ov = gamma * torch.diag(sigmas)
                    return qk, ov
            """),
        }

    # Formula 3: Copy head OV
    if "copy" in type_templates:
        cp = type_templates["copy"]
        ov_id_cos = np.mean(cp["params"].get("ov_id_cos", [0.3]))
        ov_decay = np.mean(cp["params"].get("ov_decay_rate", [0.04]))

        formulas["copy_head_ov"] = {
            "description": "Copy head: OV circuit is scaled identity with exponential SV decay",
            "parameters": {
                "ov_scale": round(float(ov_id_cos), 4),
                "ov_decay_rate": round(float(ov_decay), 4),
            },
            "code": textwrap.dedent(f"""\
                def init_copy_ov(hd, delta={ov_id_cos:.4f}, decay={ov_decay:.4f}):
                    sigmas = torch.exp(-decay * torch.arange(hd, dtype=torch.float32))
                    ov = delta * torch.diag(sigmas)
                    return ov
            """),
        }

    # Formula 4: Generic head spectral template
    # Fit exponential decay to the cross-model average spectral template
    all_qk_templates = []
    all_ov_templates = []
    for htype, tt in type_templates.items():
        all_qk_templates.extend(tt["qk"])
        all_ov_templates.extend(tt["ov"])

    if all_qk_templates and all_ov_templates:
        mean_qk = np.mean(all_qk_templates, axis=0)
        mean_ov = np.mean(all_ov_templates, axis=0)

        # Fit decay rates
        hd = len(mean_qk)
        indices = np.arange(hd)
        valid_qk = mean_qk > 1e-6
        valid_ov = mean_ov > 1e-6
        r_qk = 0.05
        r_ov = 0.04
        if valid_qk.sum() > 2:
            log_qk = np.log(mean_qk[valid_qk])
            c = np.polyfit(indices[valid_qk], log_qk, 1)
            r_qk = float(-c[0])
        if valid_ov.sum() > 2:
            log_ov = np.log(mean_ov[valid_ov])
            c = np.polyfit(indices[valid_ov], log_ov, 1)
            r_ov = float(-c[0])

        formulas["generic_spectral_template"] = {
            "description": "Generic head: exponential SV decay for both QK and OV circuits",
            "parameters": {
                "qk_decay_rate": round(r_qk, 4),
                "ov_decay_rate": round(r_ov, 4),
                "head_dim": hd,
            },
            "code": textwrap.dedent(f"""\
                def init_generic_head(hd, d_model, r_qk={r_qk:.4f}, r_ov={r_ov:.4f}):
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
            """),
        }

    # Formula 5: MLP init (if consistent structure found)
    mlp_emb = results.get("mlp_emb", [])
    if mlp_emb:
        # Average neuron gini and rank across models
        all_ginis = []
        all_up_ranks = []
        all_prod_ranks = []
        for me in mlp_emb:
            for l in me["mlp_layers"]:
                all_ginis.append(l["neuron_gini"])
                all_up_ranks.append(l["up_eff_rank"])
                all_prod_ranks.append(l["product_eff_rank"])

        mean_gini = float(np.mean(all_ginis))
        mean_up_rank = float(np.mean(all_up_ranks))

        formulas["mlp_spectral_init"] = {
            "description": "MLP initialization with matched spectral profile",
            "parameters": {
                "mean_neuron_gini": round(mean_gini, 4),
                "mean_up_eff_rank": round(mean_up_rank, 2),
            },
            "code": textwrap.dedent(f"""\
                def init_mlp_spectral(d_model, hidden_dim, eff_rank={mean_up_rank:.1f}):
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
            """),
        }

    return formulas


# ─── Validation via forward pass ──────────────────────────────────────────────

def validate_formulas(formulas):
    """
    For each formula, generate synthetic weights and replace in Pythia-70M.
    Compare attention patterns and output logits vs original.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = ModelSpec("Pythia-70M", "EleutherAI/pythia-70m", "gptneox")
    print(f"\n  Loading Pythia-70M for validation...")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.float32, trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)

    test_text = (
        "The cat sat on the mat. The cat sat on the mat. "
        "Alice met Bob at the store. Then Alice went home."
    )
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)

    # Get original outputs
    with torch.no_grad():
        orig_outputs = model(**inputs, output_attentions=True)
    orig_logits = orig_outputs.logits[0].clone()
    orig_attentions = [a[0].clone() for a in orig_outputs.attentions]

    # Extract original circuit structure for reference
    layers = extract_attention_weights(model, spec)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hd = cfg.hidden_size // n_heads

    validation_results = {}

    # Test each formula that produces head-level circuits
    for fname, formula in formulas.items():
        if fname == "mlp_spectral_init":
            continue  # Skip MLP formula for attention-based validation

        print(f"  Validating {fname}...")

        # Create modified model
        model_mod = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager",
        )
        model_mod.eval()
        sd = model_mod.state_dict()

        # Replace a few heads in layer 1 with formula-generated weights
        test_layer = min(1, n_layers - 1)
        test_heads = list(range(min(4, n_heads)))  # Replace first 4 heads

        for h in test_heads:
            # Generate synthetic QK and OV circuits
            if fname == "previous_token_head":
                params = formula["parameters"]
                qk_scale = params["qk_norm_scale"]
                ov_decay = params["ov_decay_rate"]
                qk = torch.zeros(hd, hd)
                for i in range(1, hd):
                    qk[i, i - 1] = 1.0
                qk = qk * qk_scale / (hd ** 0.5)
                sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
                ov = torch.diag(sigmas)

            elif fname == "induction_head":
                params = formula["parameters"]
                beta = params["qk_norm_scale"]
                gamma = params["ov_id_cos"]
                ov_decay = params["ov_decay_rate"]
                qk = beta * torch.eye(hd)
                sigmas = torch.exp(-ov_decay * torch.arange(hd, dtype=torch.float32))
                ov = gamma * torch.diag(sigmas)

            elif fname == "copy_head_ov":
                params = formula["parameters"]
                delta = params["ov_scale"]
                decay = params["ov_decay_rate"]
                sigmas = torch.exp(-decay * torch.arange(hd, dtype=torch.float32))
                ov = delta * torch.diag(sigmas)
                qk = torch.eye(hd)  # Default identity for QK

            elif fname == "generic_spectral_template":
                params = formula["parameters"]
                r_qk = params["qk_decay_rate"]
                r_ov = params["ov_decay_rate"]
                U_qk = torch.linalg.qr(torch.randn(hd, hd))[0]
                V_qk = torch.linalg.qr(torch.randn(hd, hd))[0]
                U_ov = torch.linalg.qr(torch.randn(hd, hd))[0]
                V_ov = torch.linalg.qr(torch.randn(hd, hd))[0]
                s_qk = torch.exp(-r_qk * torch.arange(hd, dtype=torch.float32))
                s_ov = torch.exp(-r_ov * torch.arange(hd, dtype=torch.float32))
                qk = U_qk @ torch.diag(s_qk) @ V_qk.T
                ov = U_ov @ torch.diag(s_ov) @ V_ov.T
            else:
                continue

            # Factor QK into Q and K: QK = Q @ K^T
            # Use SVD: QK = U S V^T, then Q = U * sqrt(S), K = V * sqrt(S)
            U, S, Vh = torch.linalg.svd(qk)
            sqrt_S = S.sqrt()
            W_Q_head = (U * sqrt_S).T  # (hd, hd) - but we need (hd, d)
            W_K_head = (Vh.T * sqrt_S).T  # (hd, hd)

            # Factor OV into V and O: OV = V_proj @ O_proj (in head subspace)
            U2, S2, Vh2 = torch.linalg.svd(ov)
            sqrt_S2 = S2.sqrt()
            W_V_head = (U2 * sqrt_S2).T  # (hd, hd)
            W_O_head = Vh2.T * sqrt_S2   # (hd, hd)

            # Inject into the fused QKV weight of Pythia
            d = cfg.hidden_size
            pfx = f"gpt_neox.layers.{test_layer}.attention"
            qkv = sd[f"{pfx}.query_key_value.weight"].float()
            qkv_reshaped = qkv.reshape(n_heads, 3, hd, d)

            # We can only set the head-subspace part (hd x hd) — embed via projection
            # For validation, just set the (hd, hd) part assuming identity projection
            # This is approximate but shows if the spectral structure is right
            qkv_reshaped[h, 0, :, :hd] = W_Q_head
            qkv_reshaped[h, 1, :, :hd] = W_K_head
            qkv_reshaped[h, 2, :, :hd] = W_V_head

            wo = sd[f"{pfx}.dense.weight"].float().reshape(d, n_heads, hd)
            wo[:hd, h, :] = W_O_head.T
            sd[f"{pfx}.dense.weight"] = wo.reshape(d, d)
            sd[f"{pfx}.query_key_value.weight"] = qkv_reshaped.reshape(n_heads * 3 * hd, d)

        model_mod.load_state_dict(sd)

        # Run modified model
        with torch.no_grad():
            mod_outputs = model_mod(**inputs, output_attentions=True)
        mod_logits = mod_outputs.logits[0]
        mod_attentions = [a[0] for a in mod_outputs.attentions]

        # Compare attention patterns at test layer
        orig_attn = orig_attentions[test_layer]  # (n_heads, seq, seq)
        mod_attn = mod_attentions[test_layer]

        # Cosine similarity of attention patterns for modified heads
        attn_cos_sims = []
        for h in test_heads:
            orig_flat = orig_attn[h].flatten()
            mod_flat = mod_attn[h].flatten()
            cos = float(torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0), mod_flat.unsqueeze(0)
            ))
            attn_cos_sims.append(cos)

        # KL divergence of output logits
        orig_probs = torch.softmax(orig_logits, dim=-1)
        mod_probs = torch.softmax(mod_logits, dim=-1)
        kl = float(torch.nn.functional.kl_div(
            mod_probs.log().clamp(min=-100), orig_probs,
            reduction="batchmean", log_target=False
        ))

        validation_results[fname] = {
            "attn_cosine_similarity_mean": round(float(np.mean(attn_cos_sims)), 4),
            "attn_cosine_similarity_per_head": [round(x, 4) for x in attn_cos_sims],
            "logit_kl_divergence": round(kl, 6),
            "test_layer": test_layer,
            "test_heads": test_heads,
        }

        del model_mod

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return validation_results


# ─── Scoring and ranking ─────────────────────────────────────────────────────

def rank_structures(formulas, head_universality, validation_results):
    """
    Rank structures by: universality_score = prevalence * consistency * pattern_similarity / code_bytes
    """
    rankings = []

    for fname, formula in formulas.items():
        code_bytes = len(formula["code"].encode("utf-8"))

        # Get head type from formula name
        type_map = {
            "previous_token_head": "previous_token",
            "induction_head": "induction",
            "copy_head_ov": "copy",
            "generic_spectral_template": None,  # applies to all
            "mlp_spectral_init": None,
        }
        htype = type_map.get(fname)

        # Prevalence
        if htype and htype in head_universality:
            prevalence = head_universality[htype].get("prevalence_mean", 0.1)
        else:
            prevalence = 0.5  # generic applies broadly

        # Consistency (inverse of coefficient of variation)
        consistency = 0.5  # default
        if htype and htype in head_universality:
            hu = head_universality[htype]
            cvs = []
            for model_cvs in hu.get("consistency", {}).values():
                cvs.extend(model_cvs.values())
            if cvs:
                mean_cv = np.mean(cvs)
                consistency = 1.0 / (1.0 + mean_cv)

        # Pattern similarity from validation
        pattern_sim = 0.5  # default
        if fname in validation_results:
            vr = validation_results[fname]
            pattern_sim = max(0, vr["attn_cosine_similarity_mean"])

        universality_score = prevalence * consistency * pattern_sim / max(code_bytes, 1) * 1000

        rankings.append({
            "name": fname,
            "description": formula["description"],
            "prevalence": round(prevalence, 4),
            "consistency": round(consistency, 4),
            "pattern_similarity": round(pattern_sim, 4),
            "code_bytes": code_bytes,
            "universality_score": round(universality_score, 4),
            "parameters": formula["parameters"],
            "validation": validation_results.get(fname),
        })

    rankings.sort(key=lambda x: -x["universality_score"])
    return rankings


# ─── Report generation ───────────────────────────────────────────────────────

def generate_report(head_universality, mlp_universality, formulas,
                    validation_results, rankings):
    """Generate the final synthesis report."""
    lines = ["# Universality and Formula Analysis Report\n"]

    # Executive summary
    lines.append("## Executive Summary\n")
    if rankings:
        lines.append("Structures ranked by universality score (higher = more compressible):\n")
        lines.append("| Rank | Structure | Score | Prevalence | Consistency | Pattern Sim | Code Bytes |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(rankings):
            lines.append(
                f"| {i+1} | {r['name']} | {r['universality_score']:.3f} | "
                f"{r['prevalence']:.3f} | {r['consistency']:.3f} | "
                f"{r['pattern_similarity']:.3f} | {r['code_bytes']} |"
            )
        lines.append("")

    # Head type universality
    lines.append("## Head Type Universality\n")
    for htype, info in sorted(head_universality.items()):
        lines.append(f"### {htype}\n")
        lines.append(f"- **Prevalence**: mean={info['prevalence_mean']:.3f}, std={info['prevalence_std']:.3f}")
        lines.append(f"  - Per model: {info['prevalence']}")
        if info.get("layer_pattern"):
            lines.append(f"- **Layer position**:")
            for model, pat in info["layer_pattern"].items():
                lines.append(f"  - {model}: {pat}")
        if info.get("consistency"):
            lines.append(f"- **Weight consistency (CV)**:")
            for model, cvs in info["consistency"].items():
                lines.append(f"  - {model}: {cvs}")
        lines.append("")

    # MLP universality
    lines.append("## MLP Structure Universality\n")
    if mlp_universality:
        lines.append("| Model | Up Rank | Up Slope | Prod Rank | Prod Slope |")
        lines.append("|---|---|---|---|---|")
        for model, info in mlp_universality.items():
            lines.append(
                f"| {model} | {info['up_rank_mean']:.1f} | {info['up_rank_slope']:.1f} | "
                f"{info['product_rank_mean']:.1f} | {info['product_rank_slope']:.1f} |"
            )
        lines.append("")

    # Formulas with code
    lines.append("## Parametric Formulas\n")
    for fname, formula in formulas.items():
        lines.append(f"### {fname}\n")
        lines.append(f"**{formula['description']}**\n")
        lines.append(f"Parameters: {json.dumps(formula['parameters'], indent=2)}\n")
        lines.append("```python")
        lines.append(formula["code"].rstrip())
        lines.append("```\n")

    # Validation results
    lines.append("## Validation Results (Pythia-70M)\n")
    if validation_results:
        lines.append("| Formula | Attn Cosine Sim | Logit KL Div |")
        lines.append("|---|---|---|")
        for fname, vr in validation_results.items():
            lines.append(
                f"| {fname} | {vr['attn_cosine_similarity_mean']:.4f} | "
                f"{vr['logit_kl_divergence']:.6f} |"
            )
        lines.append("")
        for fname, vr in validation_results.items():
            lines.append(f"### {fname}")
            lines.append(f"- Test layer: {vr['test_layer']}, heads: {vr['test_heads']}")
            lines.append(f"- Per-head attention cosine similarity: {vr['attn_cosine_similarity_per_head']}")
            lines.append("")

    # Recommended init scheme
    lines.append("## Recommended Initialization for Parameter-Golf\n")
    lines.append("Based on the analysis, the recommended init scheme combines:\n")
    if rankings:
        for i, r in enumerate(rankings[:3]):
            lines.append(f"{i+1}. **{r['name']}** (score={r['universality_score']:.3f}): {r['description']}")
    lines.append("")
    lines.append("Key parameters for `train_gpt.py` circuit-aware init:")
    lines.append("```python")
    lines.append("# Derived from cross-model analysis")
    for fname, formula in formulas.items():
        for pname, pval in formula["parameters"].items():
            lines.append(f"# {fname}.{pname} = {pval}")
    lines.append("```\n")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading prior analysis results...")
    results = load_results()

    print("Analyzing universality...")
    head_universality, mlp_universality = analyze_universality(results)

    print("Deriving parametric formulas...")
    formulas = derive_formulas(results, head_universality)
    print(f"  Derived {len(formulas)} formulas: {list(formulas.keys())}")

    print("Validating formulas on Pythia-70M...")
    validation_results = validate_formulas(formulas)

    print("Ranking structures...")
    rankings = rank_structures(formulas, head_universality, validation_results)

    # Save JSON
    output = {
        "head_universality": head_universality,
        "mlp_universality": mlp_universality,
        "formulas": {k: {kk: vv for kk, vv in v.items() if kk != "code"}
                     for k, v in formulas.items()},
        "formula_code": {k: v["code"] for k, v in formulas.items()},
        "validation": validation_results,
        "rankings": rankings,
    }
    json_path = OUT_DIR / "universality_report.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved data to {json_path}")

    # Save report
    report = generate_report(
        head_universality, mlp_universality, formulas,
        validation_results, rankings
    )
    report_path = OUT_DIR / "universality_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    # Print top rankings
    print("\n" + "=" * 60)
    print("STRUCTURE RANKINGS (by universality score)")
    print("=" * 60)
    for i, r in enumerate(rankings):
        print(f"  {i+1}. {r['name']}: score={r['universality_score']:.3f} "
              f"(prev={r['prevalence']:.3f}, cons={r['consistency']:.3f}, "
              f"sim={r['pattern_similarity']:.3f}, bytes={r['code_bytes']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
