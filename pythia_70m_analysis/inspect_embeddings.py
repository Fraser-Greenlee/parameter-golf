"""Analyze the structure of Pythia-70M's trained embeddings.

Goal: understand what features the 50304x512 embedding matrix encodes
so we can recreate it from code (NLTK POS tags, frequency stats, etc.)

Key questions:
1. What do the principal components correspond to?
2. How do tokens cluster by POS category?
3. How does frequency/norm relate to embedding structure?
4. Can we predict embedding directions from known features?
"""
import sys
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import nltk

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('universal_tagset', quiet=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).parent / "summary"


def classify_vocab(tokenizer):
    """Classify all tokens using NLTK POS tags and heuristics."""
    categories = {}  # token_id -> category string
    pos_tags = {}    # token_id -> POS tag

    # Collect full words (tokens that decode to clean alpha strings)
    full_words = []
    for i in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([i])
        stripped = decoded.strip()

        if not stripped or decoded.startswith('<'):
            categories[i] = 'special'
        elif stripped and not stripped[0].isalpha() and not stripped[0].isdigit():
            categories[i] = 'punctuation'
        elif stripped.isdigit() or (len(stripped) > 1 and stripped.replace('.', '').replace(',', '').isdigit()):
            categories[i] = 'number'
        elif decoded.startswith(' ') and stripped.isalpha() and len(stripped) >= 2:
            full_words.append((i, stripped))
            categories[i] = 'word'  # will be refined by POS
        elif not decoded.startswith(' ') and stripped.isalpha():
            categories[i] = 'subword'
        else:
            categories[i] = 'other'

    # POS-tag full words
    words_only = [w for _, w in full_words]
    tagged = nltk.pos_tag(words_only, tagset='universal')
    function_pos = {'DET', 'ADP', 'CONJ', 'PRON', 'PRT'}

    for (tok_id, word), (_, pos) in zip(full_words, tagged):
        pos_tags[tok_id] = pos
        if pos in function_pos:
            categories[tok_id] = 'function'
        elif pos == 'VERB':
            categories[tok_id] = 'verb'
        elif pos == 'NOUN':
            categories[tok_id] = 'noun'
        elif pos == 'ADJ':
            categories[tok_id] = 'adjective'
        elif pos == 'ADV':
            categories[tok_id] = 'adverb'
        else:
            categories[tok_id] = 'content_other'

    return categories, pos_tags


def main():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', torch_dtype=torch.float32)

    embed_in = model.gpt_neox.embed_in.weight.data.float().numpy()   # [50304, 512]
    embed_out = model.embed_out.weight.data.float().numpy()           # [50304, 512]
    vocab_size = tokenizer.vocab_size  # 50254 actual tokens (50304 with padding)

    # Only use actual vocab tokens
    E = embed_in[:vocab_size]  # [50254, 512]
    U = embed_out[:vocab_size]

    print(f"Embed_in shape: {E.shape}, Embed_out shape: {U.shape}")

    # --- Basic statistics ---
    norms = np.linalg.norm(E, axis=1)
    print(f"\nEmbedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")

    # --- Classify vocab ---
    print("\nClassifying vocabulary...")
    categories, pos_tags = classify_vocab(tokenizer)

    cat_counts = defaultdict(int)
    for i in range(vocab_size):
        cat_counts[categories.get(i, 'unknown')] += 1
    print("\nVocab categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:16s}: {count:6d}")

    # --- Norms by category ---
    print("\nEmbedding norms by category:")
    for cat in ['function', 'verb', 'noun', 'adjective', 'adverb', 'subword', 'punctuation', 'number']:
        ids = [i for i in range(vocab_size) if categories.get(i) == cat]
        if ids:
            cat_norms = norms[ids]
            print(f"  {cat:16s}: mean={cat_norms.mean():.4f}, std={cat_norms.std():.4f}, n={len(ids)}")

    # --- PCA of embeddings ---
    print("\nComputing PCA...")
    E_centered = E - E.mean(axis=0)
    U_svd, S_svd, Vt_svd = np.linalg.svd(E_centered, full_matrices=False)

    print(f"Top 20 singular values: {S_svd[:20].tolist()}")
    explained_var = (S_svd ** 2) / (S_svd ** 2).sum()
    cum_var = np.cumsum(explained_var)
    print(f"Cumulative variance: PC1={cum_var[0]:.3f}, PC5={cum_var[4]:.3f}, "
          f"PC10={cum_var[9]:.3f}, PC50={cum_var[49]:.3f}, PC100={cum_var[99]:.3f}")

    # --- What do the top PCs correspond to? ---
    print("\n\n=== Top Principal Components ===")
    for pc_idx in range(10):
        direction = Vt_svd[pc_idx]  # [512]
        projections = E_centered @ direction  # [50254]

        # Top and bottom tokens on this PC
        top_idx = np.argsort(-projections)[:15]
        bot_idx = np.argsort(projections)[:15]

        print(f"\n--- PC{pc_idx} (explains {explained_var[pc_idx]:.1%} variance) ---")
        print(f"  Top (+): ", end="")
        print(", ".join(f"'{tokenizer.decode([i]).strip()}'" for i in top_idx))
        print(f"  Bottom (-): ", end="")
        print(", ".join(f"'{tokenizer.decode([i]).strip()}'" for i in bot_idx))

        # Category breakdown: which categories are high vs low on this PC?
        cat_means = {}
        for cat in ['function', 'verb', 'noun', 'subword', 'punctuation']:
            ids = [i for i in range(vocab_size) if categories.get(i) == cat]
            if ids:
                cat_means[cat] = projections[ids].mean()
        sorted_cats = sorted(cat_means.items(), key=lambda x: -x[1])
        print(f"  Category means: ", end="")
        print(", ".join(f"{cat}={mean:+.3f}" for cat, mean in sorted_cats))

    # --- Cluster analysis: mean embedding by category ---
    print("\n\n=== Category Centroids ===")
    centroids = {}
    for cat in ['function', 'verb', 'noun', 'adjective', 'subword', 'punctuation', 'number']:
        ids = [i for i in range(vocab_size) if categories.get(i) == cat]
        if ids:
            centroids[cat] = E[ids].mean(axis=0)

    # Cosine similarity between category centroids
    print("\nCosine similarity between category centroids:")
    cat_list = list(centroids.keys())
    print(f"{'':16s}", end="")
    for c in cat_list:
        print(f" {c[:8]:>8s}", end="")
    print()
    for c1 in cat_list:
        print(f"{c1:16s}", end="")
        for c2 in cat_list:
            cos = np.dot(centroids[c1], centroids[c2]) / (np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]))
            print(f" {cos:8.3f}", end="")
        print()

    # --- Embed_in vs Embed_out relationship ---
    print("\n\n=== Embed_in vs Embed_out ===")
    # Per-token cosine similarity between embed_in and embed_out
    cos_sims = (E * U).sum(axis=1) / (np.linalg.norm(E, axis=1) * np.linalg.norm(U, axis=1) + 1e-10)
    print(f"Cosine similarity (embed_in, embed_out): mean={cos_sims.mean():.4f}, "
          f"std={cos_sims.std():.4f}")

    for cat in ['function', 'verb', 'noun', 'subword', 'punctuation']:
        ids = [i for i in range(vocab_size) if categories.get(i) == cat]
        if ids:
            print(f"  {cat:16s}: mean cos={cos_sims[ids].mean():.4f}")

    # --- Can we predict embeddings from simple features? ---
    print("\n\n=== Feature Predictability ===")
    # How much variance can we explain with:
    # 1. Just the category (one-hot) -> 7-dim
    # 2. Token length
    # 3. First character
    # Build a simple feature matrix and check R^2

    # Category one-hot
    all_cats = list(set(categories.values()))
    cat_to_idx = {c: i for i, c in enumerate(all_cats)}
    X_cat = np.zeros((vocab_size, len(all_cats)))
    for i in range(vocab_size):
        X_cat[i, cat_to_idx[categories.get(i, 'unknown')]] = 1

    # Token length
    lengths = np.array([len(tokenizer.decode([i]).strip()) for i in range(vocab_size)]).reshape(-1, 1)

    # Predict embedding from category one-hot via least squares
    X = X_cat
    # E_centered = E - E.mean(0)
    # Solve: X @ W = E_centered
    W, residuals, rank, sv = np.linalg.lstsq(X, E_centered, rcond=None)
    E_pred = X @ W
    ss_total = (E_centered ** 2).sum()
    ss_residual = ((E_centered - E_pred) ** 2).sum()
    r2_cat = 1 - ss_residual / ss_total
    print(f"R² from POS category alone: {r2_cat:.4f}")
    print(f"  (category centroids explain {r2_cat:.1%} of embedding variance)")

    # How many dimensions would we need to capture 90% variance?
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_dims = np.searchsorted(cum_var, threshold) + 1
        print(f"  Dims for {threshold:.0%} variance: {n_dims}")


if __name__ == "__main__":
    main()
