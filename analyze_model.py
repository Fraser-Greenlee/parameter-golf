"""
Comprehensive model analysis for parameter-golf SOTA.

Trains the SOTA model (or loads a checkpoint), then collects extensive
diagnostics on how the model behaves on validation data.

Usage:
    # Train + analyze (requires GPU, ~12 min)
    srun ... python analyze_model.py

    # Analyze a pre-trained checkpoint
    srun ... python analyze_model.py --checkpoint final_model_XYZ.pt

    # Quick smoke test (fewer batches)
    python analyze_model.py --checkpoint final_model.pt --max-batches 2

Outputs:
    analysis_results/<run_id>/results.pkl   — full results dict
    analysis_results/<run_id>/summary.txt   — human-readable summary
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Import model definitions from the SOTA submission
# ---------------------------------------------------------------------------
SOTA_DIR = Path(__file__).parent / "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
sys.path.insert(0, str(SOTA_DIR))
from train_gpt import (
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    load_data_shard,
    quantize_int6_per_row,
    quantize_float_tensor,
    _classify_param,
    CONTROL_TENSOR_NAME_PATTERNS,
)
sys.path.pop(0)

# ---------------------------------------------------------------------------
# Data loading (standalone — no torchrun needed)
# ---------------------------------------------------------------------------

def load_val_tokens(args: Hyperparameters, seq_len: int) -> Tensor:
    files = sorted(glob.glob(args.val_files))
    if not files:
        raise FileNotFoundError(f"No val files: {args.val_files}")
    tokens = torch.cat([load_data_shard(Path(f)) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def build_model(args: Hyperparameters, device: torch.device) -> GPT:
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Analysis: instrumented forward pass
# ---------------------------------------------------------------------------

def analyze_forward(
    model: GPT,
    x: Tensor,       # [B, T]
    y: Tensor,       # [B, T]
    sp: spm.SentencePieceProcessor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> dict:
    """Run a single forward pass collecting all diagnostics."""
    device = x.device
    B, T = x.shape
    n = model.num_layers
    results = {}

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # ---- Embedding + pre-attention features ----
        emb = model.tok_emb(x)  # [B, T, D]
        bigram_out = model.bigram(x) if model.bigram is not None else torch.zeros_like(emb)
        x_normed = F.rms_norm(emb + bigram_out, (emb.size(-1),))

        # SmearGate analysis
        g = torch.sigmoid(model.smear.gate.to(dtype=x_normed.dtype))
        x_prev = torch.cat([torch.zeros_like(x_normed[:, :1]), x_normed[:, :-1]], dim=1)
        x_smeared = (1 - g[None, None, :]) * x_normed + g[None, None, :] * x_prev

        smear_gate_values = g.float().cpu()
        smear_delta_norm = (x_smeared - x_normed).float().norm(dim=-1).cpu()  # [B, T]
        results["smear_gate_values"] = smear_gate_values  # [D]
        results["smear_delta_norm_mean"] = smear_delta_norm.mean().item()
        results["smear_gate_mean"] = smear_gate_values.mean().item()
        results["smear_gate_std"] = smear_gate_values.std().item()

        # ---- Layer-by-layer analysis ----
        h = x_smeared
        x0 = h.clone()
        v0 = None
        skips = []
        ve_cache = {}

        layer_residual_norms = []
        layer_attn_contrib_norms = []
        layer_mlp_contrib_norms = []
        layer_cosine_with_final = []
        layer_ffn_sparsity = []
        layer_ffn_activation_stats = []
        layer_attn_scale_stats = []
        layer_mlp_scale_stats = []
        layer_resid_mix_stats = []
        per_layer_outputs = []

        for i in range(n):
            block = model.blocks[i]
            is_encoder = i < model.num_encoder_layers

            # Get weights from banks
            q_w = model.qo_bank[i]
            out_w = model.qo_bank[n + i]
            k_w = model.kv_bank[i]
            v_w = model.kv_bank[n + i]
            up_w = model.mlp_up_bank[i]
            down_w = model.mlp_down_bank[i]

            # Skip connections (decoder)
            if not is_encoder and skips:
                dec_idx = i - model.num_encoder_layers
                h = h + model.skip_weights[dec_idx].to(dtype=h.dtype)[None, None, :] * skips.pop()

            # Residual mix
            mix = block.resid_mix.to(dtype=h.dtype)
            x_in = mix[0][None, None, :] * h + mix[1][None, None, :] * x0

            # resid_mix analysis
            mix_float = block.resid_mix.float().cpu()
            layer_resid_mix_stats.append({
                "mix_x_mean": mix_float[0].mean().item(),
                "mix_x0_mean": mix_float[1].mean().item(),
            })

            # Attention
            ve = model._get_ve(i, x, ve_cache)
            h_normed = block.attn_norm(x_in) * block.ln_scale_factor
            attn_out, raw_v = block.attn(h_normed, q_w, k_w, v_w, out_w, v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v

            attn_scaled = block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_after_attn = x_in + attn_scaled

            # MLP
            mlp_normed = block.mlp_norm(x_after_attn) * block.ln_scale_factor
            mlp_pre = F.linear(mlp_normed, up_w.to(mlp_normed.dtype))
            mlp_activated = F.leaky_relu(mlp_pre, negative_slope=0.5)
            mlp_squared = mlp_activated.square()
            mlp_out_raw = F.linear(mlp_squared, down_w.to(mlp_squared.dtype))
            mlp_scaled = block.mlp_scale.to(dtype=x_after_attn.dtype)[None, None, :] * mlp_out_raw

            h_new = x_after_attn + mlp_scaled

            # FFN sparsity (fraction of neurons with relu output = 0)
            # With leaky_relu(0.5), nothing is truly zero, but near-zero counts
            relu_mask = (mlp_pre > 0).float()
            frac_positive = relu_mask.mean().item()
            layer_ffn_sparsity.append(frac_positive)

            # FFN activation magnitude stats
            act_magnitudes = mlp_squared.float().mean(dim=(0, 1))  # [hidden]
            layer_ffn_activation_stats.append({
                "mean": act_magnitudes.mean().item(),
                "std": act_magnitudes.std().item(),
                "max": act_magnitudes.max().item(),
                "p90": float(torch.quantile(act_magnitudes, 0.9).item()),
            })

            # Residual contribution norms
            resid_norm = h_new.float().norm(dim=-1).mean().item()
            attn_contrib_norm = attn_scaled.float().norm(dim=-1).mean().item()
            mlp_contrib_norm = mlp_scaled.float().norm(dim=-1).mean().item()

            layer_residual_norms.append(resid_norm)
            layer_attn_contrib_norms.append(attn_contrib_norm)
            layer_mlp_contrib_norms.append(mlp_contrib_norm)

            # Scale parameter stats
            layer_attn_scale_stats.append({
                "mean": block.attn_scale.float().mean().item(),
                "std": block.attn_scale.float().std().item(),
            })
            layer_mlp_scale_stats.append({
                "mean": block.mlp_scale.float().mean().item(),
                "std": block.mlp_scale.float().std().item(),
            })

            per_layer_outputs.append(h_new.detach())

            if is_encoder:
                skips.append(h_new)
            h = h_new

        # Final norm
        h_final = model.final_norm(h)

        # Cosine similarity of each layer's output with final
        for i, lo in enumerate(per_layer_outputs):
            lo_flat = lo.float().reshape(-1, lo.size(-1))
            hf_flat = h_final.float().reshape(-1, h_final.size(-1))
            cos = F.cosine_similarity(lo_flat, hf_flat, dim=-1).mean().item()
            layer_cosine_with_final.append(cos)

        results["layer_residual_norms"] = layer_residual_norms
        results["layer_attn_contrib_norms"] = layer_attn_contrib_norms
        results["layer_mlp_contrib_norms"] = layer_mlp_contrib_norms
        results["layer_cosine_with_final"] = layer_cosine_with_final
        results["layer_ffn_sparsity"] = layer_ffn_sparsity
        results["layer_ffn_activation_stats"] = layer_ffn_activation_stats
        results["layer_attn_scale_stats"] = layer_attn_scale_stats
        results["layer_mlp_scale_stats"] = layer_mlp_scale_stats
        results["layer_resid_mix_stats"] = layer_resid_mix_stats

        # ---- Logits and per-token loss ----
        if model.tie_embeddings:
            logits_proj = F.linear(h_final, model.tok_emb.weight)
        else:
            logits_proj = model.lm_head(h_final)
        logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
        logits_f32 = logits.float()

        # Per-token NLL
        nll = F.cross_entropy(
            logits_f32.reshape(-1, logits_f32.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape(B, T).cpu()  # [B, T]

        results["per_token_nll"] = nll  # [B, T]

        # ---- Logit statistics ----
        logit_mean = logits_f32.mean(dim=-1).cpu()  # [B, T]
        logit_std = logits_f32.std(dim=-1).cpu()
        logit_max = logits_f32.max(dim=-1).values.cpu()

        results["logit_mean"] = logit_mean.mean().item()
        results["logit_std"] = logit_std.mean().item()
        results["logit_max"] = logit_max.mean().item()
        results["logit_softcap"] = model.logit_softcap

        # How often does softcap bind? (pre-tanh magnitude > 0.8 * cap)
        pre_tanh = (logits_proj.float() / model.logit_softcap).abs()
        results["softcap_binding_frac"] = (pre_tanh > 0.8).float().mean().item()
        results["softcap_near_saturation_frac"] = (pre_tanh > 0.95).float().mean().item()

        # Top-1 / Top-5 probability spread
        probs = F.softmax(logits_f32, dim=-1)
        top5_probs, _ = probs.topk(5, dim=-1)
        results["top1_prob_mean"] = top5_probs[:, :, 0].mean().item()
        results["top5_prob_sum_mean"] = top5_probs.sum(dim=-1).mean().item()

        # Model entropy
        model_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [B, T] nats
        results["model_entropy_mean_nats"] = model_entropy.mean().item()
        results["model_entropy_mean_bits"] = (model_entropy / math.log(2)).mean().item()

        # KL(model || unigram): how much does the model deviate from unigram?
        unigram_counts = torch.zeros(logits_f32.size(-1), device="cpu")
        for token_id in y.reshape(-1).cpu():
            unigram_counts[token_id] += 1
        unigram_dist = (unigram_counts / unigram_counts.sum()).clamp(min=1e-10).to(device)
        # KL = sum p(x) * log(p(x)/q(x)), p=model, q=unigram
        log_model = (probs + 1e-10).log()
        log_unigram = unigram_dist.log().unsqueeze(0).unsqueeze(0).expand_as(probs)
        kl_from_unigram = (probs * (log_model - log_unigram)).sum(dim=-1)
        results["kl_from_unigram_mean"] = kl_from_unigram.mean().item()

    # ---- Token metadata (CPU, no autocast) ----
    x_cpu = x.cpu()
    y_cpu = y.cpu()

    # Word position: is this token word-initial?
    y_flat_dev = y.reshape(-1).long()
    has_space = has_leading_space_lut[y_flat_dev].reshape(B, T).cpu()
    is_boundary = is_boundary_token_lut[y_flat_dev].reshape(B, T).cpu()
    token_bytes = base_bytes_lut[y_flat_dev].reshape(B, T).cpu()

    results["target_has_leading_space"] = has_space  # [B, T]
    results["target_is_boundary"] = is_boundary
    results["target_token_bytes"] = token_bytes
    results["input_ids"] = x_cpu
    results["target_ids"] = y_cpu

    # Position within sequence
    results["seq_positions"] = torch.arange(T).unsqueeze(0).expand(B, T)

    return results


# ---------------------------------------------------------------------------
# Quantization error analysis
# ---------------------------------------------------------------------------

def analyze_quantization(model: GPT) -> dict:
    """Measure int6 quantization error per weight tensor."""
    results = {}
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    n = model.num_layers

    # Map bank indices to layer names
    weight_groups = {}

    # Banks
    for i in range(n):
        for bank_name, idx, label in [
            ("qo_bank", i, f"layer{i}_q"),
            ("qo_bank", n + i, f"layer{i}_out"),
            ("kv_bank", i, f"layer{i}_k"),
            ("kv_bank", n + i, f"layer{i}_v"),
            ("mlp_up_bank", i, f"layer{i}_mlp_up"),
            ("mlp_down_bank", i, f"layer{i}_mlp_down"),
        ]:
            key = f"{bank_name}[{idx}]"
            t = sd[bank_name][idx]
            weight_groups[label] = t

    # Embedding
    weight_groups["tok_emb"] = sd["tok_emb.weight"]
    if "bigram.embed.weight" in sd:
        weight_groups["bigram_embed"] = sd["bigram.embed.weight"]
    if "bigram.proj.weight" in sd:
        weight_groups["bigram_proj"] = sd["bigram.proj.weight"]
    if model.ve_shared is not None and "ve_shared.embed.weight" in sd:
        weight_groups["ve_embed"] = sd["ve_shared.embed.weight"]

    quant_errors = {}
    for name, t in weight_groups.items():
        if not t.is_floating_point() or t.ndim < 2 or t.numel() <= 65536:
            continue
        t32 = t.float()

        # Check if this would be int6 or int8
        # Banks are "attn" or "mlp" category -> int6
        is_int6 = any(tag in name for tag in ["_q", "_k", "_v", "_out", "_mlp_up", "_mlp_down"])
        if is_int6:
            q, s = quantize_int6_per_row(t)
            clip_range = 31
        else:
            # int8
            q, s = quantize_float_tensor(t)
            clip_range = 127

        # Reconstruct
        if s.ndim > 0:
            recon = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            recon = q.float() * float(s.item())

        mse = (t32 - recon).pow(2).mean().item()
        max_err = (t32 - recon).abs().max().item()
        rel_err = mse / (t32.pow(2).mean().item() + 1e-10)

        quant_errors[name] = {
            "mse": mse,
            "max_abs_error": max_err,
            "relative_mse": rel_err,
            "quant_type": "int6" if is_int6 else "int8",
            "shape": list(t.shape),
            "weight_std": t32.std().item(),
            "weight_max": t32.abs().max().item(),
        }

    return quant_errors


# ---------------------------------------------------------------------------
# Aggregate results across batches
# ---------------------------------------------------------------------------

def aggregate_results(batch_results: list[dict]) -> dict:
    """Merge per-batch results into aggregate statistics."""
    agg = {}
    n_layers = len(batch_results[0]["layer_residual_norms"])

    # ---- Per-layer stats: average across batches ----
    for key in [
        "layer_residual_norms", "layer_attn_contrib_norms",
        "layer_mlp_contrib_norms", "layer_cosine_with_final",
        "layer_ffn_sparsity",
    ]:
        vals = np.array([r[key] for r in batch_results])  # [n_batches, n_layers]
        agg[key] = vals.mean(axis=0).tolist()

    for key in ["layer_ffn_activation_stats", "layer_attn_scale_stats",
                "layer_mlp_scale_stats", "layer_resid_mix_stats"]:
        # Each is a list of dicts per layer — average the scalar values
        merged = []
        for li in range(n_layers):
            all_dicts = [r[key][li] for r in batch_results]
            avg_dict = {}
            for k in all_dicts[0]:
                avg_dict[k] = np.mean([d[k] for d in all_dicts])
            merged.append(avg_dict)
        agg[key] = merged

    # ---- Scalar stats: average across batches ----
    for key in [
        "smear_gate_mean", "smear_gate_std", "smear_delta_norm_mean",
        "logit_mean", "logit_std", "logit_max",
        "softcap_binding_frac", "softcap_near_saturation_frac",
        "top1_prob_mean", "top5_prob_sum_mean",
        "model_entropy_mean_nats", "model_entropy_mean_bits",
        "kl_from_unigram_mean",
    ]:
        agg[key] = np.mean([r[key] for r in batch_results])

    # SmearGate values (same across batches — just take first)
    agg["smear_gate_values"] = batch_results[0]["smear_gate_values"]

    # ---- Token-level loss decomposition ----
    all_nll = torch.cat([r["per_token_nll"] for r in batch_results], dim=0)  # [total_B, T]
    all_has_space = torch.cat([r["target_has_leading_space"] for r in batch_results], dim=0)
    all_is_boundary = torch.cat([r["target_is_boundary"] for r in batch_results], dim=0)
    all_token_bytes = torch.cat([r["target_token_bytes"] for r in batch_results], dim=0)
    all_target_ids = torch.cat([r["target_ids"] for r in batch_results], dim=0)
    all_positions = torch.cat([r["seq_positions"] for r in batch_results], dim=0)

    nll_flat = all_nll.reshape(-1).float()
    space_flat = all_has_space.reshape(-1).bool()
    boundary_flat = all_is_boundary.reshape(-1).bool()
    bytes_flat = all_token_bytes.reshape(-1).float()
    target_flat = all_target_ids.reshape(-1).long()
    pos_flat = all_positions.reshape(-1).long()

    # Overall loss
    agg["overall_mean_nll"] = nll_flat.mean().item()
    agg["overall_mean_bpb"] = (nll_flat.sum() / (bytes_flat.sum() * math.log(2))).item()
    agg["total_tokens_analyzed"] = int(nll_flat.numel())

    # Loss by word position
    word_initial = space_flat & ~boundary_flat
    word_continuation = ~space_flat & ~boundary_flat
    agg["loss_word_initial"] = nll_flat[word_initial].mean().item() if word_initial.any() else 0.0
    agg["loss_word_continuation"] = nll_flat[word_continuation].mean().item() if word_continuation.any() else 0.0
    agg["loss_boundary"] = nll_flat[boundary_flat].mean().item() if boundary_flat.any() else 0.0
    agg["frac_word_initial"] = word_initial.float().mean().item()
    agg["frac_word_continuation"] = word_continuation.float().mean().item()
    agg["frac_boundary"] = boundary_flat.float().mean().item()

    # What fraction of total loss comes from word-initial tokens?
    total_loss = nll_flat.sum().item()
    agg["loss_share_word_initial"] = nll_flat[word_initial].sum().item() / total_loss if total_loss > 0 else 0
    agg["loss_share_word_continuation"] = nll_flat[word_continuation].sum().item() / total_loss if total_loss > 0 else 0

    # Loss by token byte length
    for blen in [1, 2, 3, 4]:
        mask = bytes_flat == blen
        if mask.any():
            agg[f"loss_bytelen_{blen}"] = nll_flat[mask].mean().item()
            agg[f"frac_bytelen_{blen}"] = mask.float().mean().item()

    # Loss by sequence position (bucketed into 8 bins)
    T = all_nll.size(1)
    bin_size = T // 8
    pos_losses = []
    for b in range(8):
        start, end = b * bin_size, (b + 1) * bin_size
        mask = (pos_flat >= start) & (pos_flat < end)
        pos_losses.append(nll_flat[mask].mean().item() if mask.any() else 0.0)
    agg["loss_by_position_bin"] = pos_losses

    # Loss by position fine-grained (first 256 positions)
    fine_pos_losses = []
    for p in range(min(256, T)):
        mask = pos_flat == p
        if mask.any():
            fine_pos_losses.append(nll_flat[mask].mean().item())
        else:
            fine_pos_losses.append(0.0)
    agg["loss_by_position_fine"] = fine_pos_losses

    # ---- Top-k hardest/easiest sequences ----
    seq_mean_loss = all_nll.float().mean(dim=1)  # [total_B]
    k = min(50, seq_mean_loss.size(0))
    hardest_idx = seq_mean_loss.topk(k).indices.tolist()
    easiest_idx = seq_mean_loss.topk(k, largest=False).indices.tolist()

    def _pack_sequences(indices):
        return {
            "indices": indices,
            "mean_losses": seq_mean_loss[indices].tolist(),
            "token_ids": all_target_ids[indices].tolist(),        # [k, T]
            "per_token_nll": all_nll[indices].float().tolist(),   # [k, T]
        }

    agg["hardest_sequences"] = _pack_sequences(hardest_idx)
    agg["easiest_sequences"] = _pack_sequences(easiest_idx)

    # ---- Per-token frequency analysis ----
    # Compute loss bucketed by token frequency
    token_counts = torch.zeros(1024, dtype=torch.long)
    for tid in target_flat:
        token_counts[tid] += 1
    total_toks = target_flat.numel()
    # Bucket: rare (<0.1%), medium (0.1%-1%), common (>1%)
    freq = token_counts.float() / total_toks
    rare_tokens = set((freq < 0.001).nonzero(as_tuple=True)[0].tolist())
    common_tokens = set((freq > 0.01).nonzero(as_tuple=True)[0].tolist())
    rare_mask = torch.tensor([t.item() in rare_tokens for t in target_flat])
    common_mask = torch.tensor([t.item() in common_tokens for t in target_flat])
    medium_mask = ~rare_mask & ~common_mask

    agg["loss_rare_tokens"] = nll_flat[rare_mask].mean().item() if rare_mask.any() else 0
    agg["loss_common_tokens"] = nll_flat[common_mask].mean().item() if common_mask.any() else 0
    agg["loss_medium_tokens"] = nll_flat[medium_mask].mean().item() if medium_mask.any() else 0
    agg["n_rare_types"] = len(rare_tokens)
    agg["n_common_types"] = len(common_tokens)

    return agg


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(agg: dict, quant: dict, sp: spm.SentencePieceProcessor) -> str:
    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 80)
    p("MODEL ANALYSIS SUMMARY")
    p("=" * 80)

    p(f"\nTokens analyzed: {agg['total_tokens_analyzed']:,}")
    p(f"Overall mean NLL: {agg['overall_mean_nll']:.4f}")
    p(f"Overall mean BPB: {agg['overall_mean_bpb']:.4f}")

    p("\n--- TOKEN-LEVEL LOSS DECOMPOSITION ---")
    p(f"  Word-initial tokens:      loss={agg['loss_word_initial']:.4f}  "
      f"frac={agg['frac_word_initial']:.3f}  "
      f"loss_share={agg['loss_share_word_initial']:.1%}")
    p(f"  Word-continuation tokens: loss={agg['loss_word_continuation']:.4f}  "
      f"frac={agg['frac_word_continuation']:.3f}  "
      f"loss_share={agg['loss_share_word_continuation']:.1%}")
    p(f"  Boundary tokens:          loss={agg['loss_boundary']:.4f}  "
      f"frac={agg['frac_boundary']:.3f}")

    p("\n  Loss by token byte length:")
    for blen in [1, 2, 3, 4]:
        k_loss = f"loss_bytelen_{blen}"
        k_frac = f"frac_bytelen_{blen}"
        if k_loss in agg:
            p(f"    {blen}-byte tokens: loss={agg[k_loss]:.4f}  frac={agg[k_frac]:.3f}")

    p(f"\n  Loss by token frequency:")
    p(f"    Rare (<0.1%):   loss={agg['loss_rare_tokens']:.4f}  ({agg['n_rare_types']} types)")
    p(f"    Medium:         loss={agg['loss_medium_tokens']:.4f}")
    p(f"    Common (>1%):   loss={agg['loss_common_tokens']:.4f}  ({agg['n_common_types']} types)")

    p("\n  Loss by sequence position (8 bins):")
    for i, loss in enumerate(agg["loss_by_position_bin"]):
        T = len(agg["loss_by_position_fine"]) if agg["loss_by_position_fine"] else 2048
        bsz = T // 8
        p(f"    pos {i*bsz:4d}-{(i+1)*bsz:4d}: loss={loss:.4f}")

    p("\n--- PER-LAYER RESIDUAL ANALYSIS ---")
    p(f"  {'Layer':>5} {'ResidNorm':>10} {'AttnNorm':>10} {'MLPNorm':>10} "
      f"{'Attn/Resid':>10} {'MLP/Resid':>10} {'CosFinal':>10} {'FFNAct%':>8}")
    for i in range(len(agg["layer_residual_norms"])):
        rn = agg["layer_residual_norms"][i]
        an = agg["layer_attn_contrib_norms"][i]
        mn = agg["layer_mlp_contrib_norms"][i]
        cos = agg["layer_cosine_with_final"][i]
        sp_val = agg["layer_ffn_sparsity"][i]
        p(f"  {i:5d} {rn:10.2f} {an:10.2f} {mn:10.2f} "
          f"{an/rn:10.4f} {mn/rn:10.4f} {cos:10.4f} {sp_val:7.1%}")

    p("\n--- PER-LAYER SCALE PARAMETERS ---")
    p(f"  {'Layer':>5} {'AttnScale':>10} {'MLPScale':>10} {'MixX':>8} {'MixX0':>8}")
    for i in range(len(agg["layer_attn_scale_stats"])):
        asc = agg["layer_attn_scale_stats"][i]["mean"]
        msc = agg["layer_mlp_scale_stats"][i]["mean"]
        mx = agg["layer_resid_mix_stats"][i]["mix_x_mean"]
        mx0 = agg["layer_resid_mix_stats"][i]["mix_x0_mean"]
        p(f"  {i:5d} {asc:10.4f} {msc:10.4f} {mx:8.4f} {mx0:8.4f}")

    p("\n--- FFN ACTIVATION STATISTICS ---")
    p(f"  {'Layer':>5} {'MeanAct':>10} {'StdAct':>10} {'MaxAct':>10} {'P90Act':>10}")
    for i, stats in enumerate(agg["layer_ffn_activation_stats"]):
        p(f"  {i:5d} {stats['mean']:10.4f} {stats['std']:10.4f} "
          f"{stats['max']:10.2f} {stats['p90']:10.4f}")

    p("\n--- SMEARGATE ---")
    p(f"  Gate mean: {agg['smear_gate_mean']:.4f}")
    p(f"  Gate std:  {agg['smear_gate_std']:.4f}")
    p(f"  Delta norm (mean): {agg['smear_delta_norm_mean']:.4f}")
    g = agg["smear_gate_values"]
    p(f"  Gate range: [{g.min().item():.4f}, {g.max().item():.4f}]")
    p(f"  Dims with gate > 0.3: {(g > 0.3).sum().item()}/{g.numel()}")
    p(f"  Dims with gate < 0.1: {(g < 0.1).sum().item()}/{g.numel()}")

    p("\n--- LOGIT STATISTICS ---")
    p(f"  Mean logit:  {agg['logit_mean']:.4f}")
    p(f"  Std logit:   {agg['logit_std']:.4f}")
    p(f"  Max logit:   {agg['logit_max']:.4f}")
    p(f"  Softcap:     {agg.get('logit_softcap', 30.0)}")
    p(f"  Softcap binding (>80%):     {agg['softcap_binding_frac']:.4f}")
    p(f"  Softcap near-saturation (>95%): {agg['softcap_near_saturation_frac']:.4f}")
    p(f"  Top-1 prob (mean):  {agg['top1_prob_mean']:.4f}")
    p(f"  Top-5 prob sum:     {agg['top5_prob_sum_mean']:.4f}")
    p(f"  Model entropy:      {agg['model_entropy_mean_bits']:.4f} bits")
    p(f"  KL from unigram:    {agg['kl_from_unigram_mean']:.4f}")

    p("\n--- QUANTIZATION ERROR (int6/int8) ---")
    p(f"  {'Weight':>25} {'Type':>5} {'Shape':>16} {'MSE':>12} {'RelMSE':>12} {'MaxErr':>10}")
    sorted_quant = sorted(quant.items(), key=lambda kv: kv[1]["relative_mse"], reverse=True)
    for name, stats in sorted_quant:
        shape_str = "x".join(str(s) for s in stats["shape"])
        p(f"  {name:>25} {stats['quant_type']:>5} {shape_str:>16} "
          f"{stats['mse']:12.2e} {stats['relative_mse']:12.2e} {stats['max_abs_error']:10.4f}")

    p("\n--- HARDEST SEQUENCES (top 10, with per-token loss) ---")
    hardest = agg["hardest_sequences"]
    for rank_i in range(min(10, len(hardest["indices"]))):
        idx = hardest["indices"][rank_i]
        loss = hardest["mean_losses"][rank_i]
        token_ids = hardest["token_ids"][rank_i]
        token_nlls = hardest["per_token_nll"][rank_i]
        text = sp.decode(token_ids[:50])[:150].replace("\n", "\\n")
        p(f"  #{rank_i+1} (mean_loss={loss:.4f}, idx={idx}): {text}...")
        # Show the 10 hardest tokens in this sequence
        sorted_pos = sorted(range(len(token_nlls)), key=lambda i: token_nlls[i], reverse=True)[:10]
        for pos in sorted_pos:
            tok_id = token_ids[pos]
            piece = sp.id_to_piece(tok_id).replace("\n", "\\n")
            p(f"      pos={pos:4d}  nll={token_nlls[pos]:6.3f}  tok={tok_id:4d}  piece='{piece}'")

    p("\n--- EASIEST SEQUENCES (top 5, with per-token loss) ---")
    easiest = agg["easiest_sequences"]
    for rank_i in range(min(5, len(easiest["indices"]))):
        idx = easiest["indices"][rank_i]
        loss = easiest["mean_losses"][rank_i]
        token_ids = easiest["token_ids"][rank_i]
        token_nlls = easiest["per_token_nll"][rank_i]
        text = sp.decode(token_ids[:50])[:150].replace("\n", "\\n")
        p(f"  #{rank_i+1} (mean_loss={loss:.4f}, idx={idx}): {text}...")
        # Show the 5 hardest tokens even in easy sequences
        sorted_pos = sorted(range(len(token_nlls)), key=lambda i: token_nlls[i], reverse=True)[:5]
        for pos in sorted_pos:
            tok_id = token_ids[pos]
            piece = sp.id_to_piece(tok_id).replace("\n", "\\n")
            p(f"      pos={pos:4d}  nll={token_nlls[pos]:6.3f}  tok={tok_id:4d}  piece='{piece}'")

    p("\n" + "=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter-golf SOTA model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt model checkpoint. If not given, trains from scratch.")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Max batches to analyze (0 = all val data)")
    parser.add_argument("--batch-seqs", type=int, default=32,
                        help="Sequences per batch")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                        help="Output directory")
    cli_args = parser.parse_args()

    # Use SOTA defaults
    args = Hyperparameters()
    # Override with SOTA-specific settings from the submission
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", "1536"))
    args.xsa_last_n = int(os.environ.get("XSA_LAST_N", "4"))
    args.rope_dims = int(os.environ.get("ROPE_DIMS", "16"))
    args.ln_scale = True
    args.ve_enabled = True
    args.ve_dim = 128
    args.ve_layers = "9,10"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for analysis (flash_attn)")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    seq_len = args.train_seq_len
    print(f"Loading tokenizer from {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(args.tokenizer_path)

    print("Loading validation tokens...")
    val_tokens = load_val_tokens(args, seq_len)
    print(f"  {val_tokens.numel():,} tokens")

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    print("Building model...")
    model = build_model(args, device)
    model = model.bfloat16()

    if cli_args.checkpoint:
        print(f"Loading checkpoint: {cli_args.checkpoint}")
        sd = torch.load(cli_args.checkpoint, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        print("WARNING: No checkpoint provided — analyzing randomly initialized model.")
        print("         Results will not be meaningful. Pass --checkpoint to load a trained model.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} parameters")

    # ---- Quantization analysis (doesn't need forward pass) ----
    print("Analyzing quantization error...")
    quant_results = analyze_quantization(model)

    # ---- Forward pass analysis ----
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = cli_args.batch_seqs
    n_batches = total_seqs // batch_seqs
    if cli_args.max_batches > 0:
        n_batches = min(n_batches, cli_args.max_batches)

    print(f"Running analysis: {n_batches} batches x {batch_seqs} seqs x {seq_len} tokens")
    batch_results = []
    t0 = time.time()

    for bi in range(n_batches):
        raw_start = bi * batch_seqs * seq_len
        raw_end = raw_start + batch_seqs * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(batch_seqs, seq_len)
        y = local[1:].reshape(batch_seqs, seq_len)

        res = analyze_forward(
            model, x, y, sp,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        batch_results.append(res)

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  batch {bi+1}/{n_batches}  ({elapsed:.1f}s)")

    # ---- Aggregate ----
    print("Aggregating results...")
    agg = aggregate_results(batch_results)
    agg["quant_errors"] = quant_results

    # ---- Save ----
    run_id = args.run_id[:8]
    out_dir = Path(cli_args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(agg, f)
    print(f"Results saved to {pkl_path}")

    # ---- Print summary ----
    summary = print_summary(agg, quant_results, sp)
    print(summary)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
