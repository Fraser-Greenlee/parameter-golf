"""Deep inspection of L0_H7 on real FineWeb data.

L0_H7 is the 2nd worst engineered head (+0.24 nats, KL=2.3, activity=0.61).
From earlier analysis: entropy 1.75, 43% prev-token, 22% self, 10% distant.
It's a "soft previous-token" head that also does content-dependent routing.

Goal: understand exactly what content features drive its attention decisions
so we can engineer a better QK circuit than generic identity.
"""
import sys, math, copy
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path(__file__).parent / "summary"
DATA_PATH = Path(__file__).parent.parent / "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
SP_MODEL_PATH = Path(__file__).parent.parent / "data/tokenizers/fineweb_1024_bpe.model"

LAYER = 0
HEAD = 7


def load_fineweb_texts(n_docs=50):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(SP_MODEL_PATH))
    raw_tokens = np.fromfile(str(DATA_PATH), dtype="<u2", offset=1024, count=500000)
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


def analyze_head(model, tokenizer, texts):
    """Collect detailed per-position attention data."""
    model.eval()

    positions = []  # list of dicts with full context

    for doc_idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
        seq_len = len(tokens)
        if seq_len < 5:
            continue

        with torch.no_grad():
            out = model(**inputs, output_attentions=True)

        attn = out.attentions[LAYER][0, HEAD].numpy()

        for q in range(1, seq_len):
            row = attn[q, :q + 1]
            top1 = int(np.argmax(row))
            top3_idx = np.argsort(-row)[:3]
            delta = q - top1  # how far back top-1 is

            row_safe = np.clip(row, 1e-10, 1.0)
            entropy = -float((row_safe * np.log(row_safe)).sum())

            # Classify attention target
            if top1 == q:
                target_type = "self"
            elif top1 == q - 1:
                target_type = "prev"
            elif delta <= 5:
                target_type = "near"
            else:
                target_type = "distant"

            # Context window
            ctx_start = max(0, q - 4)
            ctx_end = min(seq_len, q + 2)

            positions.append({
                "doc": doc_idx,
                "pos": q,
                "token": tokens[q],
                "prev_token": tokens[q - 1],
                "target_type": target_type,
                "top1_pos": top1,
                "top1_token": tokens[top1],
                "top1_weight": float(row[top1]),
                "prev_weight": float(row[q - 1]),
                "self_weight": float(row[q]),
                "delta": delta,
                "entropy": entropy,
                "top3": [(int(i), tokens[i], float(row[i])) for i in top3_idx],
                "context": " ".join(tokens[ctx_start:ctx_end]),
            })

    return positions


def write_report(positions):
    out_path = OUT_DIR / "L0_H7_deep_dive.md"
    lines = []

    lines.append(f"# L0_H7 Deep Dive")
    lines.append(f"")
    lines.append(f"Analyzed {len(positions)} positions across FineWeb validation docs.")
    lines.append(f"")

    # --- Overall distribution ---
    types = defaultdict(int)
    for p in positions:
        types[p["target_type"]] += 1
    total = len(positions)

    lines.append("## Attention Target Distribution")
    lines.append("")
    for t in ["prev", "self", "near", "distant"]:
        lines.append(f"- **{t}**: {types[t]} ({types[t]/total:.1%})")

    pw = np.array([p["prev_weight"] for p in positions])
    sw = np.array([p["self_weight"] for p in positions])
    ent = np.array([p["entropy"] for p in positions])

    lines.append(f"")
    lines.append(f"Prev weight: mean={pw.mean():.3f}, median={np.median(pw):.3f}")
    lines.append(f"Self weight: mean={sw.mean():.3f}, median={np.median(sw):.3f}")
    lines.append(f"Entropy: mean={ent.mean():.3f}, median={np.median(ent):.3f}")
    lines.append("")

    # --- What determines whether it attends to prev vs self vs other? ---
    # Group by current token
    lines.append("## Behavior by Current Token")
    lines.append("")
    lines.append("Which tokens cause which attention pattern?")
    lines.append("")

    token_stats = defaultdict(lambda: {"prev": 0, "self": 0, "near": 0, "distant": 0,
                                        "count": 0, "prev_weights": []})
    for p in positions:
        ts = token_stats[p["token"]]
        ts[p["target_type"]] += 1
        ts["count"] += 1
        ts["prev_weights"].append(p["prev_weight"])

    # Sort by count, show tokens with enough data
    frequent_tokens = [(tok, s) for tok, s in token_stats.items() if s["count"] >= 10]
    frequent_tokens.sort(key=lambda x: -x[1]["count"])

    lines.append("### Most frequent tokens and their attention patterns")
    lines.append("")
    lines.append("| Token | Count | Prev% | Self% | Near% | Dist% | Mean Prev Weight |")
    lines.append("|-------|-------|-------|-------|-------|-------|-----------------|")
    for tok, s in frequent_tokens[:40]:
        c = s["count"]
        mpw = np.mean(s["prev_weights"])
        lines.append(
            f"| `{tok}` | {c} | {s['prev']/c:.0%} | {s['self']/c:.0%} "
            f"| {s['near']/c:.0%} | {s['distant']/c:.0%} | {mpw:.3f} |"
        )

    # Tokens with highest self-attention
    lines.append("")
    lines.append("### Tokens with highest SELF-attention rate (>30%, min 5 occurrences)")
    lines.append("")
    lines.append("| Token | Count | Self% | Prev% | Mean Prev Wt | Example Context |")
    lines.append("|-------|-------|-------|-------|-------------|-----------------|")
    high_self = [(tok, s) for tok, s in token_stats.items()
                 if s["count"] >= 5 and s["self"] / s["count"] > 0.30]
    high_self.sort(key=lambda x: -x[1]["self"] / x[1]["count"])
    for tok, s in high_self[:25]:
        c = s["count"]
        # Find an example
        ex = next(p for p in positions if p["token"] == tok and p["target_type"] == "self")
        lines.append(
            f"| `{tok}` | {c} | {s['self']/c:.0%} | {s['prev']/c:.0%} "
            f"| {np.mean(s['prev_weights']):.3f} | {ex['context'][:50]} |"
        )

    # Tokens with highest prev-token rate
    lines.append("")
    lines.append("### Tokens with highest PREV-TOKEN rate (>70%, min 5 occurrences)")
    lines.append("")
    lines.append("| Token | Count | Prev% | Self% | Mean Prev Wt |")
    lines.append("|-------|-------|-------|-------|-------------|")
    high_prev = [(tok, s) for tok, s in token_stats.items()
                 if s["count"] >= 5 and s["prev"] / s["count"] > 0.70]
    high_prev.sort(key=lambda x: -x[1]["prev"] / x[1]["count"])
    for tok, s in high_prev[:25]:
        c = s["count"]
        lines.append(
            f"| `{tok}` | {c} | {s['prev']/c:.0%} | {s['self']/c:.0%} "
            f"| {np.mean(s['prev_weights']):.3f} |"
        )

    # --- What does the PREVIOUS token predict about the attention? ---
    lines.append("")
    lines.append("## Behavior by Previous Token")
    lines.append("")
    lines.append("Does the prev token determine the attention pattern?")
    lines.append("")

    prev_token_stats = defaultdict(lambda: {"prev": 0, "self": 0, "near": 0, "distant": 0,
                                             "count": 0, "prev_weights": []})
    for p in positions:
        ps = prev_token_stats[p["prev_token"]]
        ps[p["target_type"]] += 1
        ps["count"] += 1
        ps["prev_weights"].append(p["prev_weight"])

    # Prev tokens that cause high self-attention (current token attends to self)
    lines.append("### Previous tokens that cause HIGH self-attention in the NEXT position")
    lines.append("")
    lines.append("| Prev Token | Count | Next Self% | Next Prev% | Mean Prev Wt |")
    lines.append("|-----------|-------|-----------|-----------|-------------|")
    prev_high_self = [(tok, s) for tok, s in prev_token_stats.items()
                      if s["count"] >= 10 and s["self"] / s["count"] > 0.25]
    prev_high_self.sort(key=lambda x: -x[1]["self"] / x[1]["count"])
    for tok, s in prev_high_self[:20]:
        c = s["count"]
        lines.append(
            f"| `{tok}` | {c} | {s['self']/c:.0%} | {s['prev']/c:.0%} "
            f"| {np.mean(s['prev_weights']):.3f} |"
        )

    # --- Distant attention examples ---
    lines.append("")
    lines.append("## Distant Attention Examples (delta > 5)")
    lines.append("")
    lines.append("What makes the head look far back?")
    lines.append("")

    distant = [p for p in positions if p["target_type"] == "distant"]
    distant.sort(key=lambda x: -x["delta"])

    lines.append("### Top 30 longest-range attention (sorted by distance)")
    lines.append("")
    lines.append("| Token | Attended Token | Delta | Weight | Context |")
    lines.append("|-------|---------------|-------|--------|---------|")
    for p in distant[:30]:
        lines.append(
            f"| `{p['token']}` | `{p['top1_token']}`@{p['top1_pos']} "
            f"| {p['delta']} | {p['top1_weight']:.2f} | {p['context'][:55]} |"
        )

    # --- Token pair analysis: (prev_token, current_token) -> attention ---
    lines.append("")
    lines.append("## Bigram Analysis: (prev, current) -> attention pattern")
    lines.append("")
    lines.append("Do specific token PAIRS predict the attention pattern?")
    lines.append("")

    pair_stats = defaultdict(lambda: {"prev": 0, "self": 0, "other": 0, "count": 0})
    for p in positions:
        pair = (p["prev_token"], p["token"])
        ps = pair_stats[pair]
        ps["count"] += 1
        if p["target_type"] == "prev":
            ps["prev"] += 1
        elif p["target_type"] == "self":
            ps["self"] += 1
        else:
            ps["other"] += 1

    # Pairs with highest self-attention
    lines.append("### Bigrams where current token attends to SELF (not prev)")
    lines.append("")
    lines.append("| Prev -> Current | Count | Self% | Prev% |")
    lines.append("|----------------|-------|-------|-------|")
    pair_self = [(pair, s) for pair, s in pair_stats.items()
                 if s["count"] >= 3 and s["self"] / s["count"] > 0.50]
    pair_self.sort(key=lambda x: (-x[1]["self"] / x[1]["count"], -x[1]["count"]))
    for (prev, cur), s in pair_self[:30]:
        c = s["count"]
        lines.append(f"| `{prev}` -> `{cur}` | {c} | {s['self']/c:.0%} | {s['prev']/c:.0%} |")

    out_path.write_text("\n".join(lines))
    print(f"Report written to {out_path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    print("Loading FineWeb texts...")
    texts = load_fineweb_texts(50)
    print(f"  {len(texts)} docs")

    print("Loading Pythia-70M...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model.eval()

    print(f"Analyzing L{LAYER}_H{HEAD}...")
    positions = analyze_head(model, tokenizer, texts)
    print(f"  {len(positions)} positions collected")

    write_report(positions)


if __name__ == "__main__":
    main()
