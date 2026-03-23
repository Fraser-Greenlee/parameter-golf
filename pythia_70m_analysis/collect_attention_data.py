"""Collect empirical attention patterns from Pythia-70M on diverse text samples.

For each head, writes a human-readable file showing what tokens attend to what,
including summary statistics. Captures full attention including RoPE positional effects.
"""
import sys, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from text_samples import TEXT_SAMPLES, SAMPLE_LABELS

OUT_DIR = Path(__file__).parent / "heads"
N_LAYERS = 6
N_HEADS = 8
TOP_K = 3  # top attended positions per query
MAX_DETAIL_POS = 15  # show detailed attention for first N positions per sample


def load_model():
    print("Loading Pythia-70M (eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    return model, tokenizer


def run_all_samples(model, tokenizer):
    """Run all text samples, return list of (tokens, attn_per_layer)."""
    results = []
    for i, text in enumerate(TEXT_SAMPLES):
        inputs = tokenizer(text, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        # out.attentions is tuple of [batch, heads, seq, seq] per layer
        attn_layers = [a[0].cpu().numpy() for a in out.attentions]  # each [heads, seq, seq]
        results.append((tokens, attn_layers))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(TEXT_SAMPLES)} samples")
    return results


def compute_head_stats(layer, head, all_results):
    """Compute aggregate statistics for one head across all samples."""
    stats = {
        "self_attn_top1": 0,
        "prev_attn_top1": 0,
        "bos_attn_top1": 0,
        "total_positions": 0,
        "entropies": [],
        "top1_offsets": [],  # (query_pos - top1_pos) for all positions
    }
    for tokens, attn_layers in all_results:
        attn = attn_layers[layer][head]  # [seq, seq]
        seq_len = len(tokens)
        for q in range(1, seq_len):  # skip position 0
            row = attn[q, :q + 1]  # causal: attend to 0..q
            top1 = np.argmax(row)
            stats["total_positions"] += 1
            if top1 == q:
                stats["self_attn_top1"] += 1
            if top1 == q - 1:
                stats["prev_attn_top1"] += 1
            if top1 == 0:
                stats["bos_attn_top1"] += 1
            stats["top1_offsets"].append(q - top1)
            entropy = -(row * np.log(row + 1e-10)).sum()
            stats["entropies"].append(entropy)
    return stats


def write_head_file(layer, head, all_results):
    """Write detailed attention patterns for one head."""
    stats = compute_head_stats(layer, head, all_results)
    total = max(stats["total_positions"], 1)
    mean_ent = np.mean(stats["entropies"])
    offsets = np.array(stats["top1_offsets"])

    lines = [
        f"=== Layer {layer}, Head {head} ===",
        f"Samples: {len(all_results)} | Positions: {total}",
        f"",
        f"--- Aggregate Stats ---",
        f"Mean entropy: {mean_ent:.3f}",
        f"Self-attention top-1 rate: {stats['self_attn_top1']/total:.1%}",
        f"Previous-token top-1 rate: {stats['prev_attn_top1']/total:.1%}",
        f"BOS/first-token top-1 rate: {stats['bos_attn_top1']/total:.1%}",
        f"Mean top-1 offset (positions back): {offsets.mean():.1f}",
        f"Median top-1 offset: {np.median(offsets):.0f}",
        f"Top-1 offset distribution: 0={np.mean(offsets==0):.0%}, 1={np.mean(offsets==1):.0%}, "
        f"2-5={np.mean((offsets>=2)&(offsets<=5)):.0%}, 6+={np.mean(offsets>=6):.0%}",
        f"",
    ]

    # Detailed per-sample attention patterns
    for s_idx, (tokens, attn_layers) in enumerate(all_results):
        attn = attn_layers[layer][head]  # [seq, seq]
        seq_len = len(tokens)
        label = SAMPLE_LABELS[s_idx]

        # Truncate token display for header
        tok_preview = " ".join(f"'{t}'" for t in tokens[:8])
        if seq_len > 8:
            tok_preview += " ..."
        lines.append(f"--- Sample {s_idx} [{label}] ({seq_len} tokens): {tok_preview} ---")

        # Show detailed attention for first MAX_DETAIL_POS positions
        n_detail = min(MAX_DETAIL_POS, seq_len)
        for q in range(1, n_detail):
            row = attn[q, :q + 1]
            k = min(TOP_K, len(row))
            top_idx = np.argsort(-row)[:k]
            attn_str = "  ".join(
                f"'{tokens[i]}'@{i} ({row[i]:.2f})"
                for i in top_idx
            )
            lines.append(f"  '{tokens[q]}'@{q} -> {attn_str}")

        # Per-sample aggregate for remaining positions
        if seq_len > n_detail:
            self_ct = 0
            prev_ct = 0
            for q in range(n_detail, seq_len):
                row = attn[q, :q + 1]
                top1 = np.argmax(row)
                if top1 == q:
                    self_ct += 1
                if top1 == q - 1:
                    prev_ct += 1
            remaining = seq_len - n_detail
            lines.append(f"  [remaining {remaining} positions: self={self_ct/remaining:.0%}, prev={prev_ct/remaining:.0%}]")

        lines.append("")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()
    print("Running samples...")
    all_results = run_all_samples(model, tokenizer)

    t0 = time.time()
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            text = write_head_file(layer, head, all_results)
            (OUT_DIR / f"L{layer}_H{head}.txt").write_text(text)
        print(f"  Layer {layer} written ({time.time() - t0:.1f}s)")

    print(f"\nDone. {N_LAYERS * N_HEADS} files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
