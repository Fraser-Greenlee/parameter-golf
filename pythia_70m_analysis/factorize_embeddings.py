"""Factorize Pythia-70M embeddings to discover natural components.

Following the Circuits approach: use NMF and SVD to decompose the embedding
matrix into interpretable factors, then characterize each factor by examining
which tokens score highest/lowest on it.

The embedding matrix E [50254, 512] maps token IDs to 512-dim vectors.
Factoring E ≈ W @ H where W [50254, k] and H [k, 512] gives us:
- k factors (columns of W / rows of H)
- Each factor is a direction in embedding space
- Each token's embedding is a weighted combination of these factors
"""
import sys
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from sklearn.decomposition import NMF, PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).parent / "summary"


def decode_token(tokenizer, tok_id):
    """Clean token decoding for display."""
    decoded = tokenizer.decode([tok_id])
    # Show whitespace explicitly
    if decoded.startswith(' '):
        return '▁' + decoded[1:]
    return repr(decoded)[1:-1] if not decoded.isprintable() else decoded


def characterize_factor(factor_scores, tokenizer, top_k=20):
    """Given per-token scores on a factor, find top/bottom tokens and patterns."""
    sorted_idx = np.argsort(-factor_scores)
    top_tokens = [(decode_token(tokenizer, i), float(factor_scores[i])) for i in sorted_idx[:top_k]]
    bot_tokens = [(decode_token(tokenizer, i), float(factor_scores[i])) for i in sorted_idx[-top_k:]]
    return top_tokens, bot_tokens


def main():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', torch_dtype=torch.float32)

    E = model.gpt_neox.embed_in.weight.data.float().numpy()[:tokenizer.vocab_size]  # [50254, 512]
    U = model.embed_out.weight.data.float().numpy()[:tokenizer.vocab_size]

    V = tokenizer.vocab_size
    print(f"Embedding matrix: {E.shape}")

    lines = ["# Pythia-70M Embedding Factorization", ""]

    # =============================================
    # 1. SVD -- the natural linear factorization
    # =============================================
    print("Computing SVD...")
    E_mean = E.mean(axis=0)
    E_centered = E - E_mean
    U_svd, S_svd, Vt_svd = np.linalg.svd(E_centered, full_matrices=False)

    lines.append("## SVD Factors")
    lines.append("")
    lines.append("SVD decomposes E into orthogonal directions ordered by variance explained.")
    lines.append("Each factor is a direction in 512-dim space; tokens project onto it positively or negatively.")
    lines.append("")

    for k in range(20):
        # Each token's score on factor k is U_svd[:, k] * S_svd[k]
        scores = U_svd[:, k] * S_svd[k]
        var_explained = (S_svd[k] ** 2) / (S_svd ** 2).sum()
        top, bot = characterize_factor(scores, tokenizer, top_k=15)

        lines.append(f"### SVD Factor {k} ({var_explained:.1%} variance)")
        lines.append("")
        lines.append(f"**Positive (+):** {', '.join(f'`{t}` ({s:.2f})' for t, s in top[:10])}")
        lines.append("")
        lines.append(f"**Negative (-):** {', '.join(f'`{t}` ({s:.2f})' for t, s in bot[:10])}")
        lines.append("")

        # Print to console too
        print(f"\nSVD Factor {k} ({var_explained:.1%}):")
        print(f"  (+): {', '.join(f'{t}' for t, _ in top[:8])}")
        print(f"  (-): {', '.join(f'{t}' for t, _ in bot[:8])}")

    # =============================================
    # 2. NMF -- non-negative factorization
    # =============================================
    # NMF requires non-negative input. Shift embeddings to be non-negative.
    # Two approaches: (a) use E + offset, (b) split into positive/negative parts

    # Approach: split each dim into pos and neg channels, then NMF on that
    print("\n\nComputing NMF...")
    E_pos = np.maximum(E, 0)   # [V, 512]
    E_neg = np.maximum(-E, 0)  # [V, 512]
    E_split = np.hstack([E_pos, E_neg])  # [V, 1024] -- all non-negative

    n_components = 30
    nmf = NMF(n_components=n_components, init='nndsvd', max_iter=500, random_state=42)
    W_nmf = nmf.fit_transform(E_split)  # [V, 30] -- per-token factor weights
    H_nmf = nmf.components_             # [30, 1024] -- factor directions

    reconstruction_error = nmf.reconstruction_err_
    E_split_norm = np.linalg.norm(E_split, 'fro')
    r2_nmf = 1 - (reconstruction_error / E_split_norm) ** 2
    print(f"NMF R² (split embedding, {n_components} factors): {r2_nmf:.4f}")

    lines.append("---")
    lines.append("")
    lines.append("## NMF Factors")
    lines.append("")
    lines.append(f"NMF with {n_components} factors on split-polarity embedding (pos/neg channels).")
    lines.append(f"R² = {r2_nmf:.4f}")
    lines.append("")
    lines.append("Each factor is non-negative: tokens either load on it or not. This reveals")
    lines.append("clusters/groups of tokens that share embedding structure.")
    lines.append("")

    for k in range(n_components):
        scores = W_nmf[:, k]  # per-token weight on this factor
        top, _ = characterize_factor(scores, tokenizer, top_k=15)

        # How many tokens have significant weight?
        threshold = scores.max() * 0.1
        n_active = (scores > threshold).sum()
        sparsity = 1 - n_active / V

        lines.append(f"### NMF Factor {k} ({n_active} active tokens, {sparsity:.0%} sparse)")
        lines.append("")
        lines.append(f"**Top tokens:** {', '.join(f'`{t}` ({s:.2f})' for t, s in top[:12])}")
        lines.append("")

        print(f"\nNMF Factor {k} ({n_active} tokens, {sparsity:.0%} sparse):")
        print(f"  Top: {', '.join(f'{t}' for t, _ in top[:10])}")

    # =============================================
    # 3. Embed_out factorization (the output/unembedding matrix)
    # =============================================
    print("\n\nComputing SVD of embed_out...")
    U_centered = U - U.mean(axis=0)
    Uu, Su, Vtu = np.linalg.svd(U_centered, full_matrices=False)

    lines.append("---")
    lines.append("")
    lines.append("## Embed_out (Unembedding) SVD Factors")
    lines.append("")
    lines.append("The output embedding determines what the model predicts.")
    lines.append("Its factors show what output patterns the model uses.")
    lines.append("")

    for k in range(10):
        scores = Uu[:, k] * Su[k]
        var_explained = (Su[k] ** 2) / (Su ** 2).sum()
        top, bot = characterize_factor(scores, tokenizer, top_k=12)

        lines.append(f"### Unembed Factor {k} ({var_explained:.1%} variance)")
        lines.append("")
        lines.append(f"**Positive:** {', '.join(f'`{t}`' for t, _ in top[:8])}")
        lines.append(f"**Negative:** {', '.join(f'`{t}`' for t, _ in bot[:8])}")
        lines.append("")

        print(f"\nUnembed Factor {k} ({var_explained:.1%}):")
        print(f"  (+): {', '.join(f'{t}' for t, _ in top[:8])}")
        print(f"  (-): {', '.join(f'{t}' for t, _ in bot[:8])}")

    # =============================================
    # 4. Cross-analysis: embed_in vs embed_out shared structure
    # =============================================
    # CCA or simple: SVD of E.T @ U to find shared directions
    print("\n\nCross-analysis: shared structure between embed_in and embed_out...")
    cross = E_centered.T @ U_centered  # [512, 512]
    Uc, Sc, Vtc = np.linalg.svd(cross, full_matrices=False)

    lines.append("---")
    lines.append("")
    lines.append("## Shared Structure: Embed_in × Embed_out")
    lines.append("")
    lines.append("SVD of E.T @ U reveals directions that are correlated between input and output embeddings.")
    lines.append("These are the directions the model uses for 'what I read' → 'what I predict'.")
    lines.append("")
    lines.append(f"Top 10 cross singular values: {', '.join(f'{s:.2f}' for s in Sc[:10])}")
    lines.append(f"Total cross energy in top 10: {(Sc[:10]**2).sum() / (Sc**2).sum():.1%}")
    lines.append(f"Total cross energy in top 50: {(Sc[:50]**2).sum() / (Sc**2).sum():.1%}")
    lines.append("")

    for k in range(5):
        # Project tokens onto the shared direction
        # In embed_in space: direction = Uc[:, k]
        # In embed_out space: direction = Vtc[k]
        in_scores = E_centered @ Uc[:, k]
        out_scores = U_centered @ Vtc[k]
        top_in, bot_in = characterize_factor(in_scores, tokenizer, top_k=8)
        top_out, bot_out = characterize_factor(out_scores, tokenizer, top_k=8)

        lines.append(f"### Cross Factor {k} (sv={Sc[k]:.2f})")
        lines.append(f"- **Input (+):** {', '.join(f'`{t}`' for t, _ in top_in[:6])}")
        lines.append(f"- **Input (-):** {', '.join(f'`{t}`' for t, _ in bot_in[:6])}")
        lines.append(f"- **Output (+):** {', '.join(f'`{t}`' for t, _ in top_out[:6])}")
        lines.append(f"- **Output (-):** {', '.join(f'`{t}`' for t, _ in bot_out[:6])}")
        lines.append("")

    # Write report
    out_path = OUT_DIR / "embedding_factorization.md"
    out_path.write_text("\n".join(lines))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
