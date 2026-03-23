"""Deep inspection of L2_H1 (previous-token head) on real FineWeb data.

The trained L2_H1 and our engineered version both achieve ~99-100% prev-token
top-1 attention, but replacing it costs +2.22 nats. This script investigates
what content-dependent features the trained head uses that our pure positional
bias misses.

Key questions:
1. Where does the trained head NOT attend to prev-token? What does it attend to instead?
2. How does the attention DISTRIBUTION shape vary (not just argmax)?
3. What's different about the QK content dims in the trained head?
"""
import sys, math, copy
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from engineer_weights import (
    N_LAYERS, N_HEADS, HEAD_DIM, ROTARY_DIM,
    extract_trained_subspaces, XSA_HEADS, CIRCUIT_PLAN,
    make_prev_token_head, get_subspace, get_trained_vo,
    set_qkv_for_head, set_output_for_head,
    USE_TRAINED_SUBSPACES,
)

OUT_DIR = Path(__file__).parent / "summary"
DATA_PATH = Path(__file__).parent.parent / "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
SP_MODEL_PATH = Path(__file__).parent.parent / "data/tokenizers/fineweb_1024_bpe.model"


def load_fineweb_texts(n_docs=40):
    """Load real FineWeb validation texts."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SP_MODEL_PATH))
    raw_tokens = np.fromfile(str(DATA_PATH), dtype="<u2", offset=1024, count=400000)
    bos_positions = np.where(raw_tokens == 1)[0]
    texts = []
    for i in range(min(n_docs, len(bos_positions) - 1)):
        start = bos_positions[i]
        end = bos_positions[i + 1]
        doc_tokens = raw_tokens[start:end].tolist()
        text = sp.Decode(doc_tokens)
        if len(text) > 1500:
            text = text[:1500]
        if len(text) > 100:
            texts.append(text)
    return texts


def analyze_head(model, tokenizer, texts, layer=2, head=1):
    """Deep analysis of a single head's behavior on real text."""
    model.eval()

    all_prev_weights = []      # attention weight on prev token at each position
    all_self_weights = []      # attention weight on self
    all_entropies = []
    all_top1_is_prev = []
    non_prev_examples = []     # positions where top-1 is NOT prev-token
    prev_weight_by_token = {}  # token -> list of prev-token attention weights

    for doc_idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
        seq_len = len(tokens)
        if seq_len < 3:
            continue

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        attn = out.attentions[layer][0, head].numpy()  # [seq, seq]

        for q in range(1, seq_len):
            row = attn[q, :q + 1]
            prev_w = float(row[q - 1])
            self_w = float(row[q])
            entropy = -float((row * np.log(row + 1e-10)).sum())
            top1 = int(np.argmax(row))

            all_prev_weights.append(prev_w)
            all_self_weights.append(self_w)
            all_entropies.append(entropy)
            all_top1_is_prev.append(top1 == q - 1)

            tok = tokens[q]
            if tok not in prev_weight_by_token:
                prev_weight_by_token[tok] = []
            prev_weight_by_token[tok].append(prev_w)

            # Record non-prev-token examples
            if top1 != q - 1:
                # Get context window
                ctx_start = max(0, q - 5)
                ctx_end = min(seq_len, q + 3)
                context = tokens[ctx_start:ctx_end]
                ctx_str = " ".join(f"[{tokens[i]}]" if i == q else tokens[i]
                                   for i in range(ctx_start, ctx_end))

                # Top 3 attended positions
                top3_idx = np.argsort(-row)[:3]
                top3 = [(int(idx), tokens[idx], float(row[idx])) for idx in top3_idx]

                non_prev_examples.append({
                    "doc": doc_idx,
                    "pos": q,
                    "token": tok,
                    "prev_token": tokens[q - 1],
                    "context": ctx_str,
                    "top3": top3,
                    "prev_weight": prev_w,
                    "self_weight": self_w,
                    "entropy": entropy,
                })

    return {
        "prev_weights": np.array(all_prev_weights),
        "self_weights": np.array(all_self_weights),
        "entropies": np.array(all_entropies),
        "top1_is_prev": np.array(all_top1_is_prev),
        "non_prev_examples": non_prev_examples,
        "prev_weight_by_token": prev_weight_by_token,
    }


def write_report(trained_stats, engineered_stats):
    """Write detailed comparison report."""
    out_path = OUT_DIR / "prev_token_head_deep_dive.md"
    lines = []

    lines.append("# L2_H1 Previous-Token Head: Deep Dive")
    lines.append("")
    lines.append("Comparing trained vs engineered (bias-based + trained V/O) on 40 FineWeb docs.")
    lines.append("")

    # --- Distribution statistics ---
    lines.append("## Attention Distribution Statistics")
    lines.append("")
    for label, stats in [("Trained", trained_stats), ("Engineered", engineered_stats)]:
        pw = stats["prev_weights"]
        sw = stats["self_weights"]
        ent = stats["entropies"]
        top1 = stats["top1_is_prev"]
        lines.append(f"### {label}")
        lines.append(f"- Total positions: {len(pw)}")
        lines.append(f"- Top-1 is prev-token: {top1.mean():.1%}")
        lines.append(f"- Prev-token weight: mean={pw.mean():.4f}, median={np.median(pw):.4f}, "
                      f"min={pw.min():.4f}, p5={np.percentile(pw, 5):.4f}, p95={np.percentile(pw, 95):.4f}")
        lines.append(f"- Self weight: mean={sw.mean():.4f}, median={np.median(sw):.4f}")
        lines.append(f"- Entropy: mean={ent.mean():.4f}, median={np.median(ent):.4f}, "
                      f"p5={np.percentile(ent, 5):.4f}, p95={np.percentile(ent, 95):.4f}")
        lines.append("")

    # --- Prev-token weight distribution comparison ---
    lines.append("## Prev-Token Weight Distribution")
    lines.append("")
    lines.append("| Bucket | Trained | Engineered |")
    lines.append("|--------|---------|------------|")
    buckets = [(0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.001)]
    for lo, hi in buckets:
        t_frac = ((trained_stats["prev_weights"] >= lo) & (trained_stats["prev_weights"] < hi)).mean()
        e_frac = ((engineered_stats["prev_weights"] >= lo) & (engineered_stats["prev_weights"] < hi)).mean()
        lines.append(f"| {lo:.2f}-{hi:.2f} | {t_frac:.1%} | {e_frac:.1%} |")
    lines.append("")

    # --- Non-prev-token examples from trained model ---
    lines.append("## Where Trained L2_H1 Does NOT Attend to Previous Token")
    lines.append("")
    lines.append(f"Found {len(trained_stats['non_prev_examples'])} positions "
                 f"({len(trained_stats['non_prev_examples'])/len(trained_stats['prev_weights']):.1%} of total)")
    lines.append("")

    # Group by pattern
    examples = trained_stats["non_prev_examples"]
    if examples:
        # Sort by entropy (most interesting = highest entropy)
        examples_sorted = sorted(examples, key=lambda x: -x["entropy"])

        lines.append("### Top 30 Non-Prev-Token Positions (by entropy)")
        lines.append("")
        lines.append("| Token | Prev Token | Context | Top-1 Attended | Prev Weight | Entropy |")
        lines.append("|-------|-----------|---------|---------------|-------------|---------|")
        for ex in examples_sorted[:30]:
            top1_pos, top1_tok, top1_w = ex["top3"][0]
            lines.append(
                f"| `{ex['token']}` | `{ex['prev_token']}` "
                f"| {ex['context'][:60]} "
                f"| `{top1_tok}`@{top1_pos} ({top1_w:.2f}) "
                f"| {ex['prev_weight']:.3f} | {ex['entropy']:.3f} |"
            )
        lines.append("")

        # Categorize the non-prev positions
        self_attend = [ex for ex in examples if ex["top3"][0][0] == ex["pos"]]
        bos_attend = [ex for ex in examples if ex["top3"][0][0] == 0]
        other_attend = [ex for ex in examples if ex["top3"][0][0] not in (0, ex["pos"], ex["pos"] - 1)]

        lines.append(f"### Breakdown of non-prev attention targets")
        lines.append(f"- Attends to **self**: {len(self_attend)} ({len(self_attend)/max(len(examples),1):.0%})")
        lines.append(f"- Attends to **BOS/position 0**: {len(bos_attend)} ({len(bos_attend)/max(len(examples),1):.0%})")
        lines.append(f"- Attends to **other position**: {len(other_attend)} ({len(other_attend)/max(len(examples),1):.0%})")
        lines.append("")

    # --- Token-specific prev-token weight analysis ---
    lines.append("## Prev-Token Weight by Current Token (trained model)")
    lines.append("")
    lines.append("Which tokens cause the head to attend LESS to the previous token?")
    lines.append("")

    # Find tokens with lowest mean prev-token weight
    token_means = []
    for tok, weights in trained_stats["prev_weight_by_token"].items():
        if len(weights) >= 5:  # need enough samples
            token_means.append((tok, np.mean(weights), len(weights)))
    token_means.sort(key=lambda x: x[1])

    lines.append("### Tokens with LOWEST prev-token attention (weakest prev-token signal)")
    lines.append("")
    lines.append("| Token | Mean Prev Weight | Count |")
    lines.append("|-------|-----------------|-------|")
    for tok, mean_w, count in token_means[:20]:
        lines.append(f"| `{tok}` | {mean_w:.4f} | {count} |")
    lines.append("")

    lines.append("### Tokens with HIGHEST prev-token attention (strongest prev-token signal)")
    lines.append("")
    lines.append("| Token | Mean Prev Weight | Count |")
    lines.append("|-------|-----------------|-------|")
    for tok, mean_w, count in token_means[-20:]:
        lines.append(f"| `{tok}` | {mean_w:.4f} | {count} |")

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    print("Loading FineWeb texts...")
    texts = load_fineweb_texts(40)
    print(f"  {len(texts)} docs loaded")

    print("Loading trained Pythia-70M...")
    model_full = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model_full.eval()

    # --- Analyze trained head ---
    print("Analyzing trained L2_H1...")
    trained_stats = analyze_head(model_full, tokenizer, texts, layer=2, head=1)
    print(f"  {len(trained_stats['prev_weights'])} positions, "
          f"{trained_stats['top1_is_prev'].mean():.1%} prev-token, "
          f"{len(trained_stats['non_prev_examples'])} non-prev positions")

    # --- Build engineered version ---
    print("Building engineered L2_H1...")
    extract_trained_subspaces(model_full)
    model_eng = copy.deepcopy(model_full)

    subspace = get_subspace(1, layer_idx=2)
    tvo = get_trained_vo(2, 1)
    W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_prev_token_head(subspace, alpha=10.0, trained_vo=tvo)

    orig_dense_bias = model_full.gpt_neox.layers[2].attention.dense.bias.data.clone()
    layer = model_eng.gpt_neox.layers[2]
    set_qkv_for_head(layer.attention.query_key_value.weight,
                     layer.attention.query_key_value.bias,
                     2, 1, W_Q, W_K, W_V, b_Q, b_K, b_V)
    set_output_for_head(layer.attention.dense.weight,
                        layer.attention.dense.bias,
                        1, W_O)
    layer.attention.dense.bias.data.copy_(orig_dense_bias)
    model_eng.eval()

    print("Analyzing engineered L2_H1...")
    engineered_stats = analyze_head(model_eng, tokenizer, texts, layer=2, head=1)
    print(f"  {len(engineered_stats['prev_weights'])} positions, "
          f"{engineered_stats['top1_is_prev'].mean():.1%} prev-token")

    # --- Write report ---
    write_report(trained_stats, engineered_stats)

    # --- Quick loss comparison ---
    print("\nLoss comparison on these docs:")
    loss_full = 0
    loss_eng = 0
    total_tokens = 0
    for text in texts[:20]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        n = inputs.input_ids.shape[1] - 1
        if n < 1:
            continue
        with torch.no_grad():
            logits_f = model_full(**inputs).logits[:, :-1]
            logits_e = model_eng(**inputs).logits[:, :-1]
        labels = inputs.input_ids[:, 1:]
        loss_full += F.cross_entropy(logits_f.reshape(-1, logits_f.size(-1)),
                                     labels.reshape(-1), reduction='sum').item()
        loss_eng += F.cross_entropy(logits_e.reshape(-1, logits_e.size(-1)),
                                    labels.reshape(-1), reduction='sum').item()
        total_tokens += n
    print(f"  Trained:    {loss_full/total_tokens:.4f} nats")
    print(f"  Engineered: {loss_eng/total_tokens:.4f} nats")
    print(f"  Delta:      {(loss_eng-loss_full)/total_tokens:+.4f} nats")


if __name__ == "__main__":
    main()
