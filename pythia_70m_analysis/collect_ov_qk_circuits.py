"""Compute QK and OV circuits in token space for each Pythia-70M attention head.

For each head:
- OV circuit: W_U @ W_O @ W_V @ W_E  (when attending to token X, what logits change?)
- QK circuit: W_E @ W_Q^T @ W_K @ W_E^T  (which tokens prefer to attend to which?)

Uses a ~500 token vocabulary subset for tractable analysis.
"""
import sys, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).parent / "circuits"
N_LAYERS = 6
N_HEADS = 8
HEAD_DIM = 64
N_VOCAB_SUBSET = 500
TOP_K = 5


def load_model():
    print("Loading Pythia-70M...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    return model, tokenizer


def select_vocab_subset(tokenizer, n=500):
    """Select n most interpretable tokens (shortest decoded strings first)."""
    all_tokens = []
    for i in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([i])
            if decoded and not decoded.startswith("<") and len(decoded.strip()) > 0:
                all_tokens.append((i, decoded, len(decoded)))
        except Exception:
            pass
    # Shortest tokens first (most common in BPE), then by index
    all_tokens.sort(key=lambda x: (x[2], x[0]))
    selected = all_tokens[:n]
    return [t[0] for t in selected], [t[1] for t in selected]


def extract_head_weights(model, layer, head):
    """Extract W_Q, W_K, W_V, W_O for a given head.

    QKV layout: interleaved [Q_h0(64x512), K_h0(64x512), V_h0(64x512), Q_h1, ...]
    Dense layout: [512, 512], head h uses columns [h*64:(h+1)*64]
    """
    qkv = model.gpt_neox.layers[layer].attention.query_key_value.weight.data.float()
    dense = model.gpt_neox.layers[layer].attention.dense.weight.data.float()
    base = head * 3 * HEAD_DIM
    W_Q = qkv[base:base + HEAD_DIM, :]               # [64, 512]
    W_K = qkv[base + HEAD_DIM:base + 2 * HEAD_DIM, :]  # [64, 512]
    W_V = qkv[base + 2 * HEAD_DIM:base + 3 * HEAD_DIM, :]  # [64, 512]
    W_O = dense[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]  # [512, 64]
    return W_Q, W_K, W_V, W_O


def analyze_ov_circuit(model, layer, head, token_ids, token_strs):
    """Compute OV circuit in token space and extract key patterns."""
    W_Q, W_K, W_V, W_O = extract_head_weights(model, layer, head)
    W_E = model.gpt_neox.embed_in.weight.data[token_ids].float()   # [n, 512]
    W_U = model.embed_out.weight.data[token_ids].float()            # [n, 512]

    # OV circuit: W_O @ W_V maps input -> residual contribution
    # In token space: result[out_tok, src_tok] = logit change for out_tok when attending to src_tok
    W_OV = W_O @ W_V  # [512, 512]
    ov_token = (W_U @ W_OV @ W_E.T).numpy()  # [n_tokens, n_tokens]

    n = len(token_strs)
    diag = np.diag(ov_token)

    # Copy fraction: how often is self the top-boosted output?
    top1_output = np.argmax(ov_token, axis=0)  # for each source, which output is top?
    copy_frac = np.mean(top1_output == np.arange(n))

    # Suppression: how many sources have negative self-boost?
    suppress_frac = np.mean(diag < 0)
    mean_self_boost = diag.mean()

    # SVD of OV circuit for rank analysis
    U, S, Vh = np.linalg.svd(W_OV.numpy(), full_matrices=False)
    eff_rank = np.exp(-(S / S.sum() * np.log(S / S.sum() + 1e-10)).sum())

    lines = [
        f"=== Layer {layer}, Head {head} - OV Circuit ===",
        f"When attending to source token X, which output logits change?",
        f"",
        f"Copy fraction (self = top-1 boosted): {copy_frac:.1%}",
        f"Suppress fraction (negative self-boost): {suppress_frac:.1%}",
        f"Mean self-boost: {mean_self_boost:.4f}",
        f"OV effective rank: {eff_rank:.1f} / {HEAD_DIM}",
        f"Top 5 singular values: {', '.join(f'{s:.3f}' for s in S[:5])}",
        f"",
    ]

    # Top copying sources (highest self-boost)
    lines.append("--- Top 20 Copying Sources (self-boost, source -> top outputs) ---")
    for s_idx in np.argsort(-diag)[:20]:
        row = ov_token[:, s_idx]
        top_out = np.argsort(-row)[:TOP_K]
        pairs = ", ".join(f"'{token_strs[o]}' ({row[o]:+.4f})" for o in top_out)
        lines.append(f"  '{token_strs[s_idx]}' (self={diag[s_idx]:+.4f}) -> {pairs}")

    # Top suppression sources (most negative self-boost)
    lines.append("")
    lines.append("--- Top 20 Suppression Sources (negative self-boost) ---")
    for s_idx in np.argsort(diag)[:20]:
        if diag[s_idx] >= 0:
            break
        row = ov_token[:, s_idx]
        top_out = np.argsort(-row)[:TOP_K]
        bot_out = np.argsort(row)[:3]
        boost_str = ", ".join(f"'{token_strs[o]}' ({row[o]:+.4f})" for o in top_out)
        suppress_str = ", ".join(f"'{token_strs[o]}' ({row[o]:+.4f})" for o in bot_out)
        lines.append(f"  '{token_strs[s_idx]}' (self={diag[s_idx]:+.4f})")
        lines.append(f"    Boosts: {boost_str}")
        lines.append(f"    Suppresses: {suppress_str}")

    # Strongest off-diagonal pairs
    lines.append("")
    lines.append("--- Strongest Off-Diagonal Pairs (source -> different output) ---")
    np.fill_diagonal(ov_token, -np.inf)
    flat = ov_token.flatten()
    top_flat = np.argsort(-flat)[:20]
    for idx in top_flat:
        out_idx, src_idx = divmod(idx, n)
        lines.append(f"  '{token_strs[src_idx]}' -> '{token_strs[out_idx]}' ({ov_token[out_idx, src_idx]:+.4f})")

    return "\n".join(lines)


def analyze_qk_circuit(model, layer, head, token_ids, token_strs):
    """Compute QK circuit in token space (content component only)."""
    W_Q, W_K, W_V, W_O = extract_head_weights(model, layer, head)
    W_E = model.gpt_neox.embed_in.weight.data[token_ids].float()  # [n, 512]

    # QK circuit: W_Q^T @ W_K in model space
    # In token space: score[q, k] = embedding[q] @ W_Q^T @ W_K @ embedding[k]
    W_QK = W_Q.T @ W_K  # [512, 512]
    qk_token = (W_E @ W_QK @ W_E.T).numpy()  # [n, n]

    n = len(token_strs)
    diag = np.diag(qk_token)

    # Self-preference: how often is self in top-3?
    self_in_top3 = 0
    for q in range(n):
        top3 = np.argsort(-qk_token[q])[:3]
        if q in top3:
            self_in_top3 += 1
    self_pref = self_in_top3 / n

    # Mean self-score relative to max
    row_max = qk_token.max(axis=1)
    rel_self = np.mean(diag / (np.abs(row_max) + 1e-8))

    # SVD for rank analysis
    U, S, Vh = np.linalg.svd(W_QK.numpy(), full_matrices=False)
    eff_rank = np.exp(-(S / S.sum() * np.log(S / S.sum() + 1e-10)).sum())

    lines = [
        f"=== Layer {layer}, Head {head} - QK Circuit (content only) ===",
        f"NOTE: 75% of head dims are content (no RoPE). This shows content preference only.",
        f"",
        f"Self-preference (self in top-3): {self_pref:.1%}",
        f"Relative self-score (self / row_max): {rel_self:.3f}",
        f"QK effective rank: {eff_rank:.1f} / {HEAD_DIM}",
        f"Top 5 singular values: {', '.join(f'{s:.3f}' for s in S[:5])}",
        f"",
    ]

    # Top 30 queries with their preferred sources
    lines.append("--- Query -> Preferred Sources (top 30 queries by self-score) ---")
    for q_idx in np.argsort(-diag)[:30]:
        row = qk_token[q_idx]
        top_k = np.argsort(-row)[:TOP_K]
        pairs = ", ".join(f"'{token_strs[k]}' ({row[k]:.3f})" for k in top_k)
        lines.append(f"  '{token_strs[q_idx]}' prefers: {pairs}")

    # Strongest overall QK pairs (excluding self)
    lines.append("")
    lines.append("--- Strongest QK Pairs (global, excluding self) ---")
    qk_noselfdiag = qk_token.copy()
    np.fill_diagonal(qk_noselfdiag, -np.inf)
    flat = qk_noselfdiag.flatten()
    top_flat = np.argsort(-flat)[:30]
    for idx in top_flat:
        q, k = divmod(idx, n)
        lines.append(f"  '{token_strs[q]}' -> '{token_strs[k]}' ({qk_token[q, k]:.3f})")

    # Weakest pairs (strongest anti-preference)
    lines.append("")
    lines.append("--- Weakest QK Pairs (strongest anti-preference) ---")
    bot_flat = np.argsort(flat)[:15]
    for idx in bot_flat:
        q, k = divmod(idx, n)
        lines.append(f"  '{token_strs[q]}' avoids '{token_strs[k]}' ({qk_token[q, k]:.3f})")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    token_ids, token_strs = select_vocab_subset(tokenizer, N_VOCAB_SUBSET)
    print(f"Selected {len(token_ids)} tokens for analysis")
    print(f"Sample tokens: {token_strs[:20]}")

    t0 = time.time()
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            ov_text = analyze_ov_circuit(model, layer, head, token_ids, token_strs)
            qk_text = analyze_qk_circuit(model, layer, head, token_ids, token_strs)
            (OUT_DIR / f"L{layer}_H{head}_ov.txt").write_text(ov_text)
            (OUT_DIR / f"L{layer}_H{head}_qk.txt").write_text(qk_text)
        print(f"  Layer {layer} done ({time.time() - t0:.1f}s)")

    print(f"\nDone. {N_LAYERS * N_HEADS * 2} files in {OUT_DIR}/ ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
