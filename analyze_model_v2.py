"""
Full-stack model analysis v2 for parameter-golf SOTA.

Extends analyze_model.py with:
  Module 1: Data characterization (token metadata, document types, heuristic filters)
  Module 2: Loss × data cross-correlation (per-token-type loss, word position, doc position)
  Module 3: Model activation deep-dive (embeddings, skip connections, VE, bigram hash)

Usage:
    srun --partition=dev --nodes=1 --exclusive --gpus=8 \
        --cpus-per-task=208 --mem=0 --time=00:20:00 \
        --job-name=analysis-v2 \
        bash -c 'source .venv/bin/activate && \
        BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 \
        python -u analyze_model_v2.py \
            --checkpoint final_model.pt \
            --max-batches 100 \
            --output-dir analysis_results_v2 2>&1'
"""
from __future__ import annotations

import argparse
import csv
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
    CONTROL_TENSOR_NAME_PATTERNS,
)
sys.path.pop(0)


# ===================================================================
# Module 1: Data Characterization (CPU-only, no model needed)
# ===================================================================

def build_token_metadata(sp: spm.SentencePieceProcessor, vocab_size: int) -> dict:
    """Build per-token-type metadata LUTs."""
    meta = {
        "is_word_start": np.zeros(vocab_size, dtype=bool),
        "is_uppercase_start": np.zeros(vocab_size, dtype=bool),
        "is_punctuation": np.zeros(vocab_size, dtype=bool),
        "is_digit": np.zeros(vocab_size, dtype=bool),
        "is_single_char": np.zeros(vocab_size, dtype=bool),
        "is_byte_fallback": np.zeros(vocab_size, dtype=bool),
        "is_control": np.zeros(vocab_size, dtype=bool),
        "byte_length": np.zeros(vocab_size, dtype=np.int32),
        "piece": [""] * vocab_size,
    }
    for i in range(min(sp.vocab_size(), vocab_size)):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            meta["is_control"][i] = True
            meta["piece"][i] = sp.id_to_piece(i)
            continue
        if sp.is_byte(i):
            meta["is_byte_fallback"][i] = True
            meta["byte_length"][i] = 1
            meta["piece"][i] = sp.id_to_piece(i)
            continue
        piece = sp.id_to_piece(i)
        meta["piece"][i] = piece
        if piece.startswith("▁"):
            meta["is_word_start"][i] = True
            inner = piece[1:]
        else:
            inner = piece
        meta["byte_length"][i] = len(piece.encode("utf-8"))
        if inner and inner[0].isupper():
            meta["is_uppercase_start"][i] = True
        if inner and all(c in '.,;:!?-"\'()[]{}|/<>@#$%^&*~`' for c in inner):
            meta["is_punctuation"][i] = True
        if inner and inner.isdigit():
            meta["is_digit"][i] = True
        if len(inner) <= 1:
            meta["is_single_char"][i] = True
    return meta


def compute_word_positions(tokens: np.ndarray, token_meta: dict) -> np.ndarray:
    """Compute position-within-word for each token (0=word start)."""
    word_pos = np.zeros(len(tokens), dtype=np.int32)
    pos = 0
    for i in range(len(tokens)):
        t = int(tokens[i])
        if token_meta["is_word_start"][t] or token_meta["is_control"][t]:
            pos = 0
        word_pos[i] = pos
        pos += 1
    return word_pos


def find_document_boundaries(tokens: np.ndarray, bos_id: int = 1) -> np.ndarray:
    """Find document start positions (where <s> token appears)."""
    return np.where(tokens == bos_id)[0]


def compute_doc_position(tokens: np.ndarray, doc_starts: np.ndarray) -> np.ndarray:
    """Compute position-within-document for each token."""
    doc_pos = np.zeros(len(tokens), dtype=np.int32)
    doc_idx = 0
    for i in range(len(tokens)):
        if doc_idx + 1 < len(doc_starts) and i >= doc_starts[doc_idx + 1]:
            doc_idx += 1
        doc_pos[i] = i - doc_starts[doc_idx]
    return doc_pos


def classify_documents(
    tokens: np.ndarray,
    doc_starts: np.ndarray,
    token_meta: dict,
) -> list[dict]:
    """Classify each document by heuristic features."""
    docs = []
    for di in range(len(doc_starts)):
        start = doc_starts[di]
        end = doc_starts[di + 1] if di + 1 < len(doc_starts) else len(tokens)
        seg = tokens[start:end].astype(int)
        length = len(seg)
        if length < 10:
            continue

        n_word_start = token_meta["is_word_start"][seg].sum()
        n_upper = token_meta["is_uppercase_start"][seg].sum()
        n_punct = token_meta["is_punctuation"][seg].sum()
        n_digit = token_meta["is_digit"][seg].sum()
        n_byte = token_meta["is_byte_fallback"][seg].sum()

        # Mean word length: count words (word-starts) and divide total tokens
        n_words = max(n_word_start, 1)
        mean_word_len = length / n_words

        # Repetition: fraction of 4-grams that repeat
        if length >= 4:
            fourgrams = set()
            repeated = 0
            for i in range(length - 3):
                fg = tuple(seg[i:i+4])
                if fg in fourgrams:
                    repeated += 1
                fourgrams.add(fg)
            repetition = repeated / max(length - 3, 1)
        else:
            repetition = 0.0

        docs.append({
            "start": int(start),
            "end": int(end),
            "length": length,
            "uppercase_ratio": float(n_upper / length),
            "punct_density": float(n_punct / length),
            "digit_density": float(n_digit / length),
            "byte_fallback_ratio": float(n_byte / length),
            "mean_word_len": float(mean_word_len),
            "repetition_score": float(repetition),
        })
    return docs


def analyze_data(
    val_tokens: np.ndarray,
    token_meta: dict,
    sp: spm.SentencePieceProcessor,
) -> dict:
    """Module 1: Pure data characterization (no model needed)."""
    results = {}
    total = len(val_tokens)

    # Token category frequencies
    for cat in ["is_word_start", "is_uppercase_start", "is_punctuation",
                "is_digit", "is_single_char", "is_byte_fallback", "is_control"]:
        count = int(token_meta[cat][val_tokens.astype(int)].sum())
        results[f"data_{cat}_frac"] = count / total
        results[f"data_{cat}_count"] = count

    # Word position distribution
    word_pos = compute_word_positions(val_tokens, token_meta)
    for p in range(7):
        mask = word_pos == p
        results[f"data_word_pos_{p}_frac"] = float(mask.mean())
    results[f"data_word_pos_7plus_frac"] = float((word_pos >= 7).mean())
    results["word_positions"] = word_pos

    # Document boundaries
    doc_starts = find_document_boundaries(val_tokens)
    doc_lengths = np.diff(np.append(doc_starts, total))
    results["data_n_documents"] = len(doc_starts)
    results["data_doc_length_mean"] = float(doc_lengths.mean())
    results["data_doc_length_median"] = float(np.median(doc_lengths))
    results["data_doc_length_p10"] = float(np.percentile(doc_lengths, 10))
    results["data_doc_length_p90"] = float(np.percentile(doc_lengths, 90))

    # Document position
    doc_pos = compute_doc_position(val_tokens, doc_starts)
    results["doc_positions"] = doc_pos

    # Document classification
    docs = classify_documents(val_tokens, doc_starts, token_meta)
    results["documents"] = docs

    # Unigram frequency
    counts = np.bincount(val_tokens.astype(int), minlength=1024)
    results["token_unigram_counts"] = counts

    # Bigram stats
    bigrams = set()
    for i in range(len(val_tokens) - 1):
        bigrams.add((int(val_tokens[i]), int(val_tokens[i+1])))
    results["data_unique_bigrams"] = len(bigrams)
    results["data_bigram_coverage"] = len(bigrams) / (1024 * 1024)

    return results


# ===================================================================
# Module 2 + 3: Model forward pass with enriched instrumentation
# ===================================================================

def analyze_forward_v2(
    model: GPT,
    x: Tensor,
    y: Tensor,
    token_meta: dict,
    word_positions: np.ndarray,  # [B*T] or matching slice
    doc_positions: np.ndarray,   # [B*T] or matching slice
) -> dict:
    """Enriched forward pass collecting all diagnostics."""
    device = x.device
    B, T = x.shape
    n = model.num_layers
    results = {}

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # ---- Embedding analysis ----
        emb = model.tok_emb(x)
        emb_norms = emb.float().norm(dim=-1).cpu()  # [B, T]
        results["emb_norm_mean"] = emb_norms.mean().item()

        # Bigram embedding contribution
        if model.bigram is not None:
            bigram_out = model.bigram(x)
            bigram_norms = bigram_out.float().norm(dim=-1).cpu()
            results["bigram_contrib_norm_mean"] = bigram_norms.mean().item()
            results["bigram_to_emb_ratio"] = (bigram_norms / (emb_norms + 1e-8)).mean().item()
            results["bigram_scale"] = model.bigram.scale.item()
            x_pre = emb + bigram_out
        else:
            results["bigram_contrib_norm_mean"] = 0.0
            results["bigram_to_emb_ratio"] = 0.0
            x_pre = emb

        x_normed = F.rms_norm(x_pre, (x_pre.size(-1),))

        # SmearGate analysis — broken down by word position
        g = torch.sigmoid(model.smear.gate.to(dtype=x_normed.dtype))
        x_prev = torch.cat([torch.zeros_like(x_normed[:, :1]), x_normed[:, :-1]], dim=1)
        x_smeared = (1 - g[None, None, :]) * x_normed + g[None, None, :] * x_prev

        smear_delta = (x_smeared - x_normed).float()
        smear_delta_norm = smear_delta.norm(dim=-1).cpu()  # [B, T]
        results["smear_gate_values"] = g.float().cpu()
        results["smear_delta_norm_mean"] = smear_delta_norm.mean().item()

        # Smear delta by word position
        smear_flat = smear_delta_norm.reshape(-1).numpy()
        for p in range(5):
            mask = word_positions[:B*T] == p
            if mask.any():
                results[f"smear_delta_word_pos_{p}"] = float(smear_flat[mask].mean())

        # Cosine similarity smeared vs unsmeared, by word position
        cos_smear = F.cosine_similarity(
            x_smeared.float().reshape(-1, x_smeared.size(-1)),
            x_normed.float().reshape(-1, x_normed.size(-1)),
            dim=-1,
        ).cpu().numpy()
        x_ids_flat = x.reshape(-1).cpu().numpy()
        ws_mask = token_meta["is_word_start"][x_ids_flat]
        results["smear_cos_word_start"] = float(cos_smear[ws_mask].mean()) if ws_mask.any() else 0
        results["smear_cos_continuation"] = float(cos_smear[~ws_mask].mean()) if (~ws_mask).any() else 0

        # ---- Layer-by-layer analysis ----
        h = x_smeared
        x0 = h.clone()
        v0 = None
        skips_list = []
        ve_cache = {}

        layer_stats = []
        per_layer_residual_by_word_pos = []  # list of dicts
        skip_connection_stats = []

        for i in range(n):
            block = model.blocks[i]
            is_encoder = i < model.num_encoder_layers

            q_w = model.qo_bank[i]
            out_w = model.qo_bank[n + i]
            k_w = model.kv_bank[i]
            v_w = model.kv_bank[n + i]
            up_w = model.mlp_up_bank[i]
            down_w = model.mlp_down_bank[i]

            h_before_skip = h.clone()

            # Skip connection (decoder)
            skip_norm = 0.0
            if not is_encoder and skips_list:
                dec_idx = i - model.num_encoder_layers
                skip_val = model.skip_weights[dec_idx].to(dtype=h.dtype)[None, None, :] * skips_list.pop()
                skip_norm = skip_val.float().norm(dim=-1).mean().item()
                h = h + skip_val
                skip_connection_stats.append({
                    "layer": i,
                    "dec_idx": dec_idx,
                    "skip_norm": skip_norm,
                    "skip_weight_mean": model.skip_weights[dec_idx].float().mean().item(),
                    "residual_norm": h.float().norm(dim=-1).mean().item(),
                    "skip_to_residual_ratio": skip_norm / (h.float().norm(dim=-1).mean().item() + 1e-8),
                })

            # Residual mix
            mix = block.resid_mix.to(dtype=h.dtype)
            x_in = mix[0][None, None, :] * h + mix[1][None, None, :] * x0

            # Attention
            ve = model._get_ve(i, x, ve_cache)
            h_normed = block.attn_norm(x_in) * block.ln_scale_factor

            # Compute Q, K norms and q_gain for attention analysis
            q_raw = F.linear(h_normed, q_w.to(h_normed.dtype))
            q_heads = q_raw.reshape(B, T, block.attn.num_heads, block.attn.head_dim)
            q_norms = q_heads.float().norm(dim=-1).mean(dim=(0, 1))  # [num_heads]

            k_raw = F.linear(h_normed, k_w.to(h_normed.dtype))
            k_heads = k_raw.reshape(B, T, block.attn.num_kv_heads, block.attn.head_dim)
            k_norms = k_heads.float().norm(dim=-1).mean(dim=(0, 1))  # [num_kv_heads]

            # Full attention via block
            attn_out, raw_v = block.attn(h_normed, q_w, k_w, v_w, out_w, v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v

            attn_scaled = block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_after_attn = x_in + attn_scaled

            # VE contribution (layers 9, 10)
            ve_norm = 0.0
            if ve is not None:
                ve_norm = ve.float().norm(dim=-1).mean().item()

            # MLP — manually to capture activations
            mlp_normed = block.mlp_norm(x_after_attn) * block.ln_scale_factor
            mlp_pre = F.linear(mlp_normed, up_w.to(mlp_normed.dtype))
            mlp_activated = F.leaky_relu(mlp_pre, negative_slope=0.5)
            mlp_squared = mlp_activated.square()
            mlp_out_raw = F.linear(mlp_squared, down_w.to(mlp_squared.dtype))
            mlp_scaled = block.mlp_scale.to(dtype=x_after_attn.dtype)[None, None, :] * mlp_out_raw

            h_new = x_after_attn + mlp_scaled

            # Collect stats
            resid_norm = h_new.float().norm(dim=-1).mean().item()
            attn_norm = attn_scaled.float().norm(dim=-1).mean().item()
            mlp_norm = mlp_scaled.float().norm(dim=-1).mean().item()
            frac_positive = (mlp_pre > 0).float().mean().item()

            # Residual norm by word position
            h_norms_flat = h_new.float().norm(dim=-1).reshape(-1).cpu().numpy()
            wp_stats = {}
            for p in range(5):
                mask = word_positions[:B*T] == p
                if mask.any():
                    wp_stats[f"resid_norm_wp{p}"] = float(h_norms_flat[mask].mean())
            per_layer_residual_by_word_pos.append(wp_stats)

            layer_stats.append({
                "resid_norm": resid_norm,
                "attn_contrib_norm": attn_norm,
                "mlp_contrib_norm": mlp_norm,
                "attn_to_resid": attn_norm / (resid_norm + 1e-8),
                "mlp_to_resid": mlp_norm / (resid_norm + 1e-8),
                "ffn_frac_positive": frac_positive,
                "ffn_activation_mean": mlp_squared.float().mean().item(),
                "attn_scale_mean": block.attn_scale.float().mean().item(),
                "mlp_scale_mean": block.mlp_scale.float().mean().item(),
                "resid_mix_x": block.resid_mix[0].float().mean().item(),
                "resid_mix_x0": block.resid_mix[1].float().mean().item(),
                "q_gain": block.attn.q_gain.float().cpu().tolist(),
                "q_norm_per_head": q_norms.cpu().tolist(),
                "k_norm_per_head": k_norms.cpu().tolist(),
                "ve_contrib_norm": ve_norm,
                "use_xsa": block.attn.use_xsa,
            })

            if is_encoder:
                skips_list.append(h_new)
            h = h_new

        # Final norm + logits
        h_final = model.final_norm(h)

        # Cosine similarity of each layer with final
        layer_cos_final = []
        # We need to re-run to get per-layer outputs... but we didn't save them all.
        # Instead store the final h for the skip analysis we already have.

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
        ).reshape(B, T).cpu()

        results["per_token_nll"] = nll
        results["layer_stats"] = layer_stats
        results["per_layer_residual_by_word_pos"] = per_layer_residual_by_word_pos
        results["skip_connection_stats"] = skip_connection_stats

        # Logit stats
        probs = F.softmax(logits_f32, dim=-1)
        top5_probs, _ = probs.topk(5, dim=-1)
        model_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        results["top1_prob_mean"] = top5_probs[:, :, 0].mean().item()
        results["top5_prob_sum_mean"] = top5_probs.sum(dim=-1).mean().item()
        results["model_entropy_bits"] = (model_entropy / math.log(2)).mean().item()

        pre_tanh = (logits_proj.float() / model.logit_softcap).abs()
        results["softcap_binding_frac"] = (pre_tanh > 0.8).float().mean().item()

    results["input_ids"] = x.cpu()
    results["target_ids"] = y.cpu()
    return results


# ===================================================================
# Aggregation: combine batch results + data analysis
# ===================================================================

def aggregate_all(
    batch_results: list[dict],
    data_info: dict,
    token_meta: dict,
    sp: spm.SentencePieceProcessor,
) -> dict:
    """Aggregate everything into final results."""
    agg = {}
    n_layers = len(batch_results[0]["layer_stats"])

    # ---- Data stats (from Module 1) ----
    for k, v in data_info.items():
        if isinstance(v, (int, float, str, bool)):
            agg[k] = v

    # ---- Per-layer stats ----
    for key in ["resid_norm", "attn_contrib_norm", "mlp_contrib_norm",
                "attn_to_resid", "mlp_to_resid", "ffn_frac_positive",
                "ffn_activation_mean", "attn_scale_mean", "mlp_scale_mean",
                "resid_mix_x", "resid_mix_x0", "ve_contrib_norm"]:
        agg[f"layer_{key}"] = [
            np.mean([r["layer_stats"][li][key] for r in batch_results])
            for li in range(n_layers)
        ]

    # q_gain (static, just take from first batch)
    agg["layer_q_gain"] = [batch_results[0]["layer_stats"][li]["q_gain"] for li in range(n_layers)]
    agg["layer_use_xsa"] = [batch_results[0]["layer_stats"][li]["use_xsa"] for li in range(n_layers)]

    # Q/K norms per head
    agg["layer_q_norm_per_head"] = [
        np.mean([r["layer_stats"][li]["q_norm_per_head"] for r in batch_results], axis=0).tolist()
        for li in range(n_layers)
    ]
    agg["layer_k_norm_per_head"] = [
        np.mean([r["layer_stats"][li]["k_norm_per_head"] for r in batch_results], axis=0).tolist()
        for li in range(n_layers)
    ]

    # Skip connection stats
    all_skips = [s for r in batch_results for s in r["skip_connection_stats"]]
    skip_by_layer = defaultdict(list)
    for s in all_skips:
        skip_by_layer[s["layer"]].append(s)
    agg["skip_connection_stats"] = {
        layer: {
            "skip_norm": np.mean([s["skip_norm"] for s in ss]),
            "skip_weight_mean": ss[0]["skip_weight_mean"],
            "skip_to_residual_ratio": np.mean([s["skip_to_residual_ratio"] for s in ss]),
        }
        for layer, ss in skip_by_layer.items()
    }

    # Residual norm by word position per layer
    agg["per_layer_residual_by_word_pos"] = []
    for li in range(n_layers):
        merged = {}
        for k in batch_results[0]["per_layer_residual_by_word_pos"][li]:
            vals = [r["per_layer_residual_by_word_pos"][li].get(k, 0) for r in batch_results]
            merged[k] = float(np.mean(vals))
        agg["per_layer_residual_by_word_pos"].append(merged)

    # ---- Scalar stats ----
    for key in ["emb_norm_mean", "bigram_contrib_norm_mean", "bigram_to_emb_ratio",
                "bigram_scale", "smear_delta_norm_mean",
                "smear_cos_word_start", "smear_cos_continuation",
                "top1_prob_mean", "top5_prob_sum_mean", "model_entropy_bits",
                "softcap_binding_frac"]:
        vals = [r.get(key, 0) for r in batch_results]
        agg[key] = float(np.mean(vals))

    agg["smear_gate_values"] = batch_results[0]["smear_gate_values"]

    # Smear delta by word position
    for p in range(5):
        key = f"smear_delta_word_pos_{p}"
        vals = [r.get(key, 0) for r in batch_results]
        agg[key] = float(np.mean(vals))

    # ---- Token-level loss analysis ----
    all_nll = torch.cat([r["per_token_nll"] for r in batch_results], dim=0)
    all_targets = torch.cat([r["target_ids"] for r in batch_results], dim=0)
    all_inputs = torch.cat([r["input_ids"] for r in batch_results], dim=0)

    nll_flat = all_nll.reshape(-1).float().numpy()
    target_flat = all_targets.reshape(-1).numpy().astype(int)
    input_flat = all_inputs.reshape(-1).numpy().astype(int)
    BT = len(nll_flat)

    agg["overall_mean_nll"] = float(nll_flat.mean())
    agg["total_tokens_analyzed"] = BT

    # Per-token-type loss (1024 entries)
    token_loss_sum = np.zeros(1024, dtype=np.float64)
    token_loss_count = np.zeros(1024, dtype=np.int64)
    for i in range(BT):
        tid = target_flat[i]
        token_loss_sum[tid] += nll_flat[i]
        token_loss_count[tid] += 1
    token_mean_loss = np.divide(token_loss_sum, token_loss_count,
                                 out=np.zeros(1024), where=token_loss_count > 0)
    agg["token_type_loss"] = token_mean_loss
    agg["token_type_count"] = token_loss_count

    # Word-initial vs continuation (sanity check)
    ws = token_meta["is_word_start"][target_flat]
    agg["loss_word_initial"] = float(nll_flat[ws].mean()) if ws.any() else 0
    agg["loss_word_continuation"] = float(nll_flat[~ws].mean()) if (~ws).any() else 0
    agg["frac_word_initial"] = float(ws.mean())

    # Loss by word position within word
    # Reconstruct word positions for the analyzed tokens
    T = all_nll.size(1)
    all_wp = []
    for bi in range(all_targets.size(0)):
        seq_targets = all_targets[bi].numpy().astype(int)
        wp = compute_word_positions(seq_targets, token_meta)
        all_wp.append(wp)
    all_wp = np.concatenate(all_wp)

    for p in range(7):
        mask = all_wp == p
        if mask.any():
            agg[f"loss_word_pos_{p}"] = float(nll_flat[mask].mean())
            agg[f"frac_word_pos_{p}"] = float(mask.mean())
    mask7 = all_wp >= 7
    if mask7.any():
        agg["loss_word_pos_7plus"] = float(nll_flat[mask7].mean())
        agg["frac_word_pos_7plus"] = float(mask7.mean())

    # Loss by document position
    # Compute doc positions for analyzed tokens (approximate: use first seq_len tokens of val)
    doc_pos_data = data_info.get("doc_positions")
    if doc_pos_data is not None and len(doc_pos_data) >= BT:
        # Map analyzed token positions to doc positions
        # Tokens are sequential from start of val set
        dp = doc_pos_data[:BT]
        for label, lo, hi in [("first_20", 0, 20), ("20_100", 20, 100),
                               ("100_500", 100, 500), ("500_plus", 500, 999999)]:
            mask = (dp >= lo) & (dp < hi)
            if mask.any():
                agg[f"loss_doc_pos_{label}"] = float(nll_flat[mask].mean())
                agg[f"frac_doc_pos_{label}"] = float(mask.mean())

    # Loss by preceding-token category
    prev_cats = {
        "after_word_start": token_meta["is_word_start"][input_flat],
        "after_punctuation": token_meta["is_punctuation"][input_flat],
        "after_digit": token_meta["is_digit"][input_flat],
        "after_uppercase": token_meta["is_uppercase_start"][input_flat],
    }
    for cat_name, mask in prev_cats.items():
        if mask.any():
            agg[f"loss_{cat_name}"] = float(nll_flat[mask].mean())
            agg[f"frac_{cat_name}"] = float(mask.mean())

    # Loss by text-type (cross-reference with document classification)
    docs = data_info.get("documents", [])
    if docs and doc_pos_data is not None:
        # Bucket documents by each heuristic
        for heuristic in ["uppercase_ratio", "punct_density", "digit_density",
                          "byte_fallback_ratio", "repetition_score"]:
            vals = np.array([d[heuristic] for d in docs])
            # Split into terciles
            p33 = np.percentile(vals, 33)
            p66 = np.percentile(vals, 66)
            for label, lo, hi in [("low", -1, p33), ("mid", p33, p66), ("high", p66, 999)]:
                doc_indices = [i for i, d in enumerate(docs) if lo < d[heuristic] <= hi]
                if not doc_indices:
                    continue
                # Gather token losses for these documents
                doc_nll_sum = 0.0
                doc_nll_count = 0
                doc_starts = data_info.get("documents", [])
                for di in doc_indices:
                    d = docs[di]
                    s, e = d["start"], min(d["end"], BT)
                    if s >= BT:
                        continue
                    e = min(e, BT)
                    if s < e:
                        doc_nll_sum += nll_flat[s:e].sum()
                        doc_nll_count += e - s
                if doc_nll_count > 0:
                    agg[f"loss_{heuristic}_{label}"] = float(doc_nll_sum / doc_nll_count)
                    agg[f"count_{heuristic}_{label}"] = doc_nll_count

    # ---- Embedding space analysis ----
    # Per-token-type embedding norms (from model weights, not forward pass)
    emb_w = batch_results[0].get("_emb_weight")  # we'll add this

    # ---- Hardest/easiest sequences ----
    seq_mean_loss = all_nll.float().mean(dim=1)
    k = min(50, seq_mean_loss.size(0))
    hardest_idx = seq_mean_loss.topk(k).indices.tolist()
    easiest_idx = seq_mean_loss.topk(k, largest=False).indices.tolist()

    def _pack(indices):
        return {
            "indices": indices,
            "mean_losses": seq_mean_loss[indices].tolist(),
            "token_ids": all_targets[indices].tolist(),
            "per_token_nll": all_nll[indices].float().tolist(),
        }
    agg["hardest_sequences"] = _pack(hardest_idx)
    agg["easiest_sequences"] = _pack(easiest_idx)

    return agg


# ===================================================================
# Quantization analysis (reused from v1)
# ===================================================================

def analyze_quantization(model: GPT) -> dict:
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    n = model.num_layers
    weight_groups = {}
    for i in range(n):
        for bank_name, idx, label in [
            ("qo_bank", i, f"layer{i}_q"), ("qo_bank", n + i, f"layer{i}_out"),
            ("kv_bank", i, f"layer{i}_k"), ("kv_bank", n + i, f"layer{i}_v"),
            ("mlp_up_bank", i, f"layer{i}_mlp_up"),
            ("mlp_down_bank", i, f"layer{i}_mlp_down"),
        ]:
            weight_groups[label] = sd[bank_name][idx]
    weight_groups["tok_emb"] = sd["tok_emb.weight"]
    if "bigram.embed.weight" in sd:
        weight_groups["bigram_embed"] = sd["bigram.embed.weight"]
    if model.ve_shared is not None and "ve_shared.embed.weight" in sd:
        weight_groups["ve_embed"] = sd["ve_shared.embed.weight"]

    quant_errors = {}
    for name, t in weight_groups.items():
        if not t.is_floating_point() or t.ndim < 2 or t.numel() <= 65536:
            continue
        t32 = t.float()
        is_int6 = any(tag in name for tag in ["_q", "_k", "_v", "_out", "_mlp_up", "_mlp_down"])
        if is_int6:
            q, s = quantize_int6_per_row(t)
        else:
            q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            recon = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            recon = q.float() * float(s.item())
        mse = (t32 - recon).pow(2).mean().item()
        rel_err = mse / (t32.pow(2).mean().item() + 1e-10)
        quant_errors[name] = {
            "mse": mse, "relative_mse": rel_err,
            "quant_type": "int6" if is_int6 else "int8",
            "shape": list(t.shape),
        }
    return quant_errors


# ===================================================================
# Summary printer
# ===================================================================

def print_summary(agg: dict, quant: dict, sp: spm.SentencePieceProcessor, token_meta: dict) -> str:
    lines = []
    def p(s=""):
        lines.append(s)

    p("=" * 90)
    p("FULL-STACK MODEL ANALYSIS v2")
    p("=" * 90)

    p(f"\nTokens analyzed: {agg['total_tokens_analyzed']:,}")
    p(f"Overall mean NLL: {agg['overall_mean_nll']:.4f}")

    # ---- Data characterization ----
    p("\n" + "=" * 90)
    p("MODULE 1: DATA CHARACTERIZATION")
    p("=" * 90)

    p("\n--- Token Category Frequencies (in validation data) ---")
    for cat in ["is_word_start", "is_uppercase_start", "is_punctuation",
                "is_digit", "is_single_char", "is_byte_fallback", "is_control"]:
        frac = agg.get(f"data_{cat}_frac", 0)
        p(f"  {cat:25s}: {frac:.1%}")

    p("\n--- Word Position Distribution ---")
    for p_idx in range(7):
        frac = agg.get(f"data_word_pos_{p_idx}_frac", 0)
        p(f"  Position {p_idx}: {frac:.1%}")
    p(f"  Position 7+: {agg.get('data_word_pos_7plus_frac', 0):.1%}")

    p("\n--- Document Statistics ---")
    p(f"  Documents: {agg.get('data_n_documents', 0):,}")
    p(f"  Length mean/median: {agg.get('data_doc_length_mean', 0):.0f} / {agg.get('data_doc_length_median', 0):.0f}")
    p(f"  Length p10/p90: {agg.get('data_doc_length_p10', 0):.0f} / {agg.get('data_doc_length_p90', 0):.0f}")
    p(f"  Unique bigrams: {agg.get('data_unique_bigrams', 0):,} ({agg.get('data_bigram_coverage', 0):.2%} of possible)")

    # ---- Loss × Data ----
    p("\n" + "=" * 90)
    p("MODULE 2: LOSS × DATA CROSS-CORRELATION")
    p("=" * 90)

    p("\n--- Loss by Word Position (within word) ---")
    p(f"  {'Pos':>4} {'Mean NLL':>10} {'Fraction':>10} {'Difficulty':>12}")
    for p_idx in range(7):
        loss = agg.get(f"loss_word_pos_{p_idx}", 0)
        frac = agg.get(f"frac_word_pos_{p_idx}", 0)
        difficulty = "HARD" if loss > 2.5 else "medium" if loss > 1.5 else "easy"
        p(f"  {p_idx:4d} {loss:10.4f} {frac:10.1%} {difficulty:>12}")
    loss_7p = agg.get("loss_word_pos_7plus", 0)
    frac_7p = agg.get("frac_word_pos_7plus", 0)
    p(f"  {'7+':>4} {loss_7p:10.4f} {frac_7p:10.1%}")

    p("\n--- Loss by Document Position ---")
    for label in ["first_20", "20_100", "100_500", "500_plus"]:
        loss = agg.get(f"loss_doc_pos_{label}", 0)
        frac = agg.get(f"frac_doc_pos_{label}", 0)
        p(f"  {label:>12}: loss={loss:.4f}  frac={frac:.1%}")

    p("\n--- Loss by Preceding Token Category ---")
    for cat in ["after_word_start", "after_punctuation", "after_digit", "after_uppercase"]:
        loss = agg.get(f"loss_{cat}", 0)
        frac = agg.get(f"frac_{cat}", 0)
        p(f"  {cat:>25}: loss={loss:.4f}  frac={frac:.1%}")

    p("\n--- Loss by Text Type (document heuristic terciles) ---")
    for heuristic in ["uppercase_ratio", "punct_density", "digit_density",
                      "byte_fallback_ratio", "repetition_score"]:
        vals = []
        for label in ["low", "mid", "high"]:
            loss = agg.get(f"loss_{heuristic}_{label}", 0)
            count = agg.get(f"count_{heuristic}_{label}", 0)
            vals.append(f"{label}={loss:.4f} ({count:,}tok)")
        p(f"  {heuristic:>25}: {' | '.join(vals)}")

    p("\n--- Top 20 Hardest Token Types (by mean NLL) ---")
    token_loss = agg.get("token_type_loss", np.zeros(1024))
    token_count = agg.get("token_type_count", np.zeros(1024))
    # Filter to tokens with >= 100 occurrences
    valid = token_count >= 100
    sorted_ids = np.argsort(-token_loss)
    p(f"  {'Rank':>4} {'ID':>5} {'Piece':>15} {'Mean NLL':>10} {'Count':>8} {'WordStart':>10}")
    rank = 0
    for tid in sorted_ids:
        if not valid[tid]:
            continue
        rank += 1
        if rank > 20:
            break
        piece = token_meta["piece"][tid]
        ws = "yes" if token_meta["is_word_start"][tid] else "no"
        p(f"  {rank:4d} {tid:5d} {repr(piece):>15} {token_loss[tid]:10.4f} {int(token_count[tid]):8d} {ws:>10}")

    p("\n--- Top 20 Easiest Token Types (by mean NLL, >=100 occurrences) ---")
    sorted_ids_easy = np.argsort(token_loss)
    p(f"  {'Rank':>4} {'ID':>5} {'Piece':>15} {'Mean NLL':>10} {'Count':>8} {'WordStart':>10}")
    rank = 0
    for tid in sorted_ids_easy:
        if not valid[tid]:
            continue
        rank += 1
        if rank > 20:
            break
        piece = token_meta["piece"][tid]
        ws = "yes" if token_meta["is_word_start"][tid] else "no"
        p(f"  {rank:4d} {tid:5d} {repr(piece):>15} {token_loss[tid]:10.4f} {int(token_count[tid]):8d} {ws:>10}")

    # ---- Model activations ----
    p("\n" + "=" * 90)
    p("MODULE 3: MODEL ACTIVATION DEEP-DIVE")
    p("=" * 90)

    p("\n--- Embedding Space ---")
    p(f"  Token embedding norm (mean): {agg.get('emb_norm_mean', 0):.4f}")
    p(f"  Bigram contribution norm:    {agg.get('bigram_contrib_norm_mean', 0):.4f}")
    p(f"  Bigram/Embedding ratio:      {agg.get('bigram_to_emb_ratio', 0):.4f}")
    p(f"  Bigram learned scale:        {agg.get('bigram_scale', 0):.4f}")

    p("\n--- SmearGate by Word Position ---")
    g = agg.get("smear_gate_values")
    if g is not None:
        p(f"  Gate mean: {g.mean().item():.4f}, std: {g.std().item():.4f}, range: [{g.min().item():.4f}, {g.max().item():.4f}]")
        p(f"  Dims gate > 0.3: {(g > 0.3).sum().item()}/512, Dims gate < 0.1: {(g < 0.1).sum().item()}/512")
    for pp in range(5):
        delta = agg.get(f"smear_delta_word_pos_{pp}", 0)
        p(f"  Word pos {pp}: smear delta norm = {delta:.4f}")
    p(f"  Cosine(smeared, unsmeared) — word-start: {agg.get('smear_cos_word_start', 0):.4f}")
    p(f"  Cosine(smeared, unsmeared) — continuation: {agg.get('smear_cos_continuation', 0):.4f}")

    p("\n--- Per-Layer Summary ---")
    p(f"  {'Lyr':>3} {'ResNorm':>8} {'Attn%':>6} {'MLP%':>6} {'FFN+%':>6} "
      f"{'AttnSc':>7} {'MLPSc':>7} {'MixX':>6} {'MixX0':>6} {'VE':>6} {'XSA':>4}")
    for li in range(len(agg.get("layer_resid_norm", []))):
        rn = agg["layer_resid_norm"][li]
        at = agg["layer_attn_to_resid"][li]
        mt = agg["layer_mlp_to_resid"][li]
        fp = agg["layer_ffn_frac_positive"][li]
        asc = agg["layer_attn_scale_mean"][li]
        msc = agg["layer_mlp_scale_mean"][li]
        mx = agg["layer_resid_mix_x"][li]
        mx0 = agg["layer_resid_mix_x0"][li]
        ve = agg["layer_ve_contrib_norm"][li]
        xsa = "yes" if agg["layer_use_xsa"][li] else ""
        p(f"  {li:3d} {rn:8.1f} {at:6.1%} {mt:6.1%} {fp:6.1%} "
          f"{asc:7.3f} {msc:7.3f} {mx:6.3f} {mx0:6.3f} {ve:6.2f} {xsa:>4}")

    p("\n--- Per-Head Q-Gain (learned attention temperature) ---")
    for li in range(len(agg.get("layer_q_gain", []))):
        gains = agg["layer_q_gain"][li]
        gains_str = " ".join(f"{g:.3f}" for g in gains)
        p(f"  Layer {li:2d}: [{gains_str}]")

    p("\n--- Skip Connections (U-Net decoder) ---")
    skip_stats = agg.get("skip_connection_stats", {})
    if skip_stats:
        p(f"  {'Layer':>5} {'SkipNorm':>10} {'SkipWeight':>11} {'Skip/Resid':>11}")
        for layer in sorted(skip_stats.keys()):
            s = skip_stats[layer]
            p(f"  {layer:5d} {s['skip_norm']:10.2f} {s['skip_weight_mean']:11.4f} {s['skip_to_residual_ratio']:11.4f}")

    p("\n--- Residual Norm by Word Position (per layer) ---")
    wp_data = agg.get("per_layer_residual_by_word_pos", [])
    if wp_data:
        header = f"  {'Lyr':>3}"
        for pp in range(5):
            header += f" {'WP'+str(pp):>8}"
        p(header)
        for li, d in enumerate(wp_data):
            row = f"  {li:3d}"
            for pp in range(5):
                val = d.get(f"resid_norm_wp{pp}", 0)
                row += f" {val:8.1f}"
            p(row)

    # ---- Quantization ----
    p("\n--- Quantization Error (top 15 by relative MSE) ---")
    sorted_q = sorted(quant.items(), key=lambda kv: kv[1]["relative_mse"], reverse=True)[:15]
    p(f"  {'Weight':>25} {'Type':>5} {'RelMSE':>10}")
    for name, stats in sorted_q:
        p(f"  {name:>25} {stats['quant_type']:>5} {stats['relative_mse']:10.2e}")

    # ---- Hardest/easiest ----
    p("\n--- Hardest Sequences (top 5) ---")
    hardest = agg.get("hardest_sequences", {})
    for ri in range(min(5, len(hardest.get("indices", [])))):
        idx = hardest["indices"][ri]
        loss = hardest["mean_losses"][ri]
        tids = hardest["token_ids"][ri][:50]
        text = sp.decode(tids)[:150].replace("\n", "\\n")
        p(f"  #{ri+1} (loss={loss:.4f}): {text}...")
        nlls = hardest["per_token_nll"][ri]
        worst = sorted(range(len(nlls)), key=lambda i: nlls[i], reverse=True)[:5]
        for pos in worst:
            tid = hardest["token_ids"][ri][pos]
            piece = sp.id_to_piece(tid)
            p(f"      pos={pos:4d} nll={nlls[pos]:6.3f} tok={tid:4d} piece={repr(piece)}")

    p("\n" + "=" * 90)
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Full-stack model analysis v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="analysis_results_v2")
    cli_args = parser.parse_args()

    args = Hyperparameters()
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", "1536"))
    args.xsa_last_n = int(os.environ.get("XSA_LAST_N", "4"))
    args.rope_dims = int(os.environ.get("ROPE_DIMS", "16"))
    args.ln_scale = True
    args.ve_enabled = True
    args.ve_dim = 128
    args.ve_layers = "9,10"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    seq_len = args.train_seq_len
    print(f"Loading tokenizer from {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(args.tokenizer_path)

    print("Building token metadata LUTs...")
    token_meta = build_token_metadata(sp, args.vocab_size)

    print("Loading validation tokens...")
    val_files = sorted(glob.glob(args.val_files))
    val_tokens_torch = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    usable = ((val_tokens_torch.numel() - 1) // seq_len) * seq_len
    val_tokens_torch = val_tokens_torch[:usable + 1]
    val_tokens_np = val_tokens_torch.numpy().astype(int)
    print(f"  {val_tokens_torch.numel():,} tokens")

    # ---- Module 1: Data characterization (CPU) ----
    print("Running Module 1: Data characterization...")
    t0 = time.time()
    data_info = analyze_data(val_tokens_np, token_meta, sp)
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ---- Build model ----
    print("Building model...")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

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
    ).to(device).bfloat16()

    if cli_args.checkpoint:
        print(f"Loading checkpoint: {cli_args.checkpoint}")
        sd = torch.load(cli_args.checkpoint, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    else:
        print("WARNING: No checkpoint — analyzing random weights.")
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---- Quantization analysis ----
    print("Analyzing quantization error...")
    quant_results = analyze_quantization(model)

    # ---- Module 2+3: Forward pass analysis ----
    total_seqs = (val_tokens_torch.numel() - 1) // seq_len
    batch_seqs = cli_args.batch_seqs
    n_batches = total_seqs // batch_seqs
    if cli_args.max_batches > 0:
        n_batches = min(n_batches, cli_args.max_batches)

    print(f"Running Module 2+3: {n_batches} batches × {batch_seqs} seqs × {seq_len} tokens")
    batch_results = []
    t0 = time.time()

    # Precompute word positions and doc positions for the full val set
    word_positions_full = data_info.get("word_positions", compute_word_positions(val_tokens_np, token_meta))
    doc_positions_full = data_info.get("doc_positions", np.zeros(len(val_tokens_np), dtype=np.int32))

    for bi in range(n_batches):
        raw_start = bi * batch_seqs * seq_len
        raw_end = raw_start + batch_seqs * seq_len + 1
        local = val_tokens_torch[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(batch_seqs, seq_len)
        y = local[1:].reshape(batch_seqs, seq_len)

        # Word positions for targets (y = tokens at positions raw_start+1 .. raw_end-1)
        wp_slice = word_positions_full[raw_start + 1: raw_end]
        dp_slice = doc_positions_full[raw_start + 1: raw_end]

        res = analyze_forward_v2(model, x, y, token_meta, wp_slice, dp_slice)
        batch_results.append(res)

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f"  batch {bi+1}/{n_batches}  ({time.time() - t0:.1f}s)")

    # ---- Aggregate ----
    print("Aggregating results...")
    agg = aggregate_all(batch_results, data_info, token_meta, sp)
    agg["quant_errors"] = quant_results

    # ---- Save ----
    run_id = args.run_id[:8]
    out_dir = Path(cli_args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle
    # Remove large non-serializable items before saving
    save_agg = {k: v for k, v in agg.items()
                if k not in ("word_positions", "doc_positions", "documents")}
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(save_agg, f)

    # Save per-token-type CSV
    csv_path = out_dir / "token_type_loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "piece", "is_word_start", "mean_nll", "count",
                          "byte_length", "is_punct", "is_digit", "is_uppercase"])
        for tid in range(1024):
            writer.writerow([
                tid, token_meta["piece"][tid], int(token_meta["is_word_start"][tid]),
                f"{agg['token_type_loss'][tid]:.6f}", int(agg['token_type_count'][tid]),
                int(token_meta["byte_length"][tid]),
                int(token_meta["is_punctuation"][tid]),
                int(token_meta["is_digit"][tid]),
                int(token_meta["is_uppercase_start"][tid]),
            ])

    # Save doc segments CSV
    docs = data_info.get("documents", [])
    if docs:
        doc_csv = out_dir / "doc_segments.csv"
        with open(doc_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_idx", "start", "length", "uppercase_ratio",
                              "punct_density", "digit_density", "byte_fallback_ratio",
                              "mean_word_len", "repetition_score"])
            for di, d in enumerate(docs):
                writer.writerow([
                    di, d["start"], d["length"],
                    f"{d['uppercase_ratio']:.4f}", f"{d['punct_density']:.4f}",
                    f"{d['digit_density']:.4f}", f"{d['byte_fallback_ratio']:.4f}",
                    f"{d['mean_word_len']:.2f}", f"{d['repetition_score']:.4f}",
                ])

    # Print summary
    summary = print_summary(agg, quant_results, sp, token_meta)
    print(summary)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
