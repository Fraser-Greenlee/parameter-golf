"""Test each engineered head in isolation against trained Pythia-70M.

For each of the 48 heads:
1. Start from full trained Pythia-70M
2. Replace ONLY that one head's attention weights with engineered version
3. Measure loss delta and attention pattern agreement

Uses real FineWeb validation text decoded from the sp1024 tokenized shards.
"""
import sys, time, math, copy
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from engineer_weights import (
    CIRCUIT_PLAN, XSA_HEADS, apply_xsa_to_attention,
    N_LAYERS, N_HEADS, HEAD_DIM,
    get_subspace, make_prev_token_head, make_induction_head, make_copy_head,
    make_suppression_head, make_content_head,
    set_qkv_for_head, set_output_for_head,
)

OUT_DIR = Path(__file__).parent / "summary"
DATA_PATH = Path(__file__).parent.parent / "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
SP_MODEL_PATH = Path(__file__).parent.parent / "data/tokenizers/fineweb_1024_bpe.model"

N_EVAL_DOCS = 20  # number of FineWeb documents to evaluate on


def load_fineweb_texts(n_docs=20):
    """Load real FineWeb validation texts by decoding sp1024 tokenized data."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(SP_MODEL_PATH))

    # Load enough tokens to get n_docs documents
    raw_tokens = np.fromfile(str(DATA_PATH), dtype="<u2", offset=1024, count=200000)

    # Find document boundaries (BOS token = 1)
    bos_positions = np.where(raw_tokens == 1)[0]
    texts = []
    for i in range(min(n_docs, len(bos_positions) - 1)):
        start = bos_positions[i]
        end = bos_positions[i + 1]
        doc_tokens = raw_tokens[start:end].tolist()
        text = sp.Decode(doc_tokens)
        # Truncate very long docs to ~300 Pythia tokens worth (~1500 chars)
        if len(text) > 1500:
            text = text[:1500]
        if len(text) > 50:  # skip tiny docs
            texts.append(text)

    return texts


def compute_loss(model, tokenizer, texts):
    """Mean cross-entropy loss across texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(**inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += input_ids.shape[1] - 1
    return total_loss / max(total_tokens, 1)


def compute_attention_stats(model_full, model_patched, tokenizer, texts,
                            layer, head, is_xsa=False):
    """Compute attention agreement and stats for a specific head."""
    agree = 0
    total = 0
    stats_full = {"prev": 0, "self": 0, "bos": 0}
    stats_eng = {"prev": 0, "self": 0, "bos": 0}

    for text in texts[:8]:  # use subset for attention comparison (speed)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        seq_len = inputs.input_ids.shape[1]
        if seq_len < 3:
            continue

        with torch.no_grad():
            out_full = model_full(**inputs, output_attentions=True)
            out_patched = model_patched(**inputs, output_attentions=True)

        attn_full = out_full.attentions[layer][0, head].numpy()
        attn_eng_all = out_patched.attentions[layer][0].numpy()
        if is_xsa:
            attn_eng_all = apply_xsa_to_attention(attn_eng_all, layer)
        attn_eng = attn_eng_all[head]

        for q in range(1, seq_len):
            row_f = attn_full[q, :q + 1]
            row_e = attn_eng[q, :q + 1]
            top1_f = int(np.argmax(row_f))
            top1_e = int(np.argmax(row_e))
            if top1_f == top1_e:
                agree += 1
            total += 1

            if top1_f == q: stats_full["self"] += 1
            if top1_f == q - 1: stats_full["prev"] += 1
            if top1_f == 0: stats_full["bos"] += 1
            if top1_e == q: stats_eng["self"] += 1
            if top1_e == q - 1: stats_eng["prev"] += 1
            if top1_e == 0: stats_eng["bos"] += 1

    t = max(total, 1)
    return {
        "agreement": agree / t,
        "full_prev": stats_full["prev"] / t,
        "full_self": stats_full["self"] / t,
        "full_bos": stats_full["bos"] / t,
        "eng_prev": stats_eng["prev"] / t,
        "eng_self": stats_eng["self"] / t,
        "eng_bos": stats_eng["bos"] / t,
    }


def make_engineered_head(layer_idx, head_idx):
    """Create engineered weights for a specific head."""
    circuit = CIRCUIT_PLAN[(layer_idx, head_idx)]
    subspace = get_subspace(head_idx)
    prev_token_subspace = get_subspace(1)

    if circuit == "prev_token":
        return make_prev_token_head(subspace, alpha=10.0)
    elif circuit == "induction":
        return make_induction_head(subspace, prev_token_subspace, alpha=4.0, content_alpha=2.0)
    elif circuit == "copy":
        return make_copy_head(subspace, ov_strength=0.8, qk_content_strength=1.5)
    elif circuit == "suppress":
        return make_suppression_head(subspace, ov_strength=-0.4)
    else:
        return make_content_head(subspace, scale=0.5)


def test_single_head(model_full, tokenizer, eval_texts, layer_idx, head_idx, baseline_loss):
    """Replace one head in trained model with engineered version, measure impact."""
    model_patched = copy.deepcopy(model_full)

    W_Q, W_K, W_V, W_O, b_Q, b_K, b_V = make_engineered_head(layer_idx, head_idx)

    # Save original dense bias before patching (set_output_for_head zeros it)
    orig_dense_bias = model_full.gpt_neox.layers[layer_idx].attention.dense.bias.data.clone()

    layer = model_patched.gpt_neox.layers[layer_idx]
    set_qkv_for_head(layer.attention.query_key_value.weight,
                     layer.attention.query_key_value.bias,
                     layer_idx, head_idx, W_Q, W_K, W_V, b_Q, b_K, b_V)
    set_output_for_head(layer.attention.dense.weight,
                        layer.attention.dense.bias,
                        head_idx, W_O)

    # Restore dense bias (shared across heads)
    layer.attention.dense.bias.data.copy_(orig_dense_bias)

    model_patched.eval()

    patched_loss = compute_loss(model_patched, tokenizer, eval_texts)

    is_xsa = (layer_idx, head_idx) in XSA_HEADS
    attn_stats = compute_attention_stats(
        model_full, model_patched, tokenizer, eval_texts, layer_idx, head_idx, is_xsa
    )

    return {
        "layer": layer_idx,
        "head": head_idx,
        "circuit": CIRCUIT_PLAN[(layer_idx, head_idx)],
        "xsa": is_xsa,
        "loss_delta": patched_loss - baseline_loss,
        **attn_stats,
    }


def write_results(results, baseline_loss):
    """Write results to markdown file."""
    out_path = OUT_DIR / "per_head_comparison.md"

    lines = [
        "# Per-Head Engineered vs Trained Comparison",
        "",
        f"Baseline full Pythia-70M loss: **{baseline_loss:.4f} nats** "
        f"({baseline_loss / math.log(2):.4f} bpb)",
        f"",
        f"Evaluated on {N_EVAL_DOCS} real FineWeb validation documents.",
        "",
        "Each row: replace ONE head in trained Pythia-70M with our engineered version,",
        "measure the loss increase and attention pattern agreement.",
        "",
        "| Layer | Head | Circuit | XSA | Loss Delta | Agree | Trained prev/self/bos | Eng prev/self/bos |",
        "|-------|------|---------|-----|------------|-------|----------------------|-------------------|",
    ]

    for r in results:
        xsa = "yes" if r["xsa"] else ""
        lines.append(
            f"| {r['layer']} | {r['head']} | {r['circuit']:<10} | {xsa:<3} "
            f"| {r['loss_delta']:+.4f} "
            f"| {r['agreement']:.0%} "
            f"| {r['full_prev']:.0%}/{r['full_self']:.0%}/{r['full_bos']:.0%} "
            f"| {r['eng_prev']:.0%}/{r['eng_self']:.0%}/{r['eng_bos']:.0%} |"
        )

    # Summary by circuit type
    lines.extend(["", "## Summary by Circuit Type", ""])
    lines.append("| Circuit | Count | Mean Loss Delta | Mean Agreement | Best Head (agree) |")
    lines.append("|---------|-------|----------------|----------------|-------------------|")
    for ct in ["prev_token", "induction", "copy", "suppress", "content"]:
        entries = [r for r in results if r["circuit"] == ct]
        if not entries:
            continue
        mean_delta = np.mean([r["loss_delta"] for r in entries])
        mean_agree = np.mean([r["agreement"] for r in entries])
        best = max(entries, key=lambda r: r["agreement"])
        lines.append(
            f"| {ct:<9} | {len(entries):>5} | {mean_delta:>+14.4f} "
            f"| {mean_agree:>14.0%} "
            f"| L{best['layer']}_H{best['head']} ({best['agreement']:.0%}) |"
        )

    # Summary by layer
    lines.extend(["", "## Summary by Layer", ""])
    lines.append("| Layer | Mean Loss Delta | Mean Agreement |")
    lines.append("|-------|----------------|----------------|")
    for layer in range(N_LAYERS):
        lr = [r for r in results if r["layer"] == layer]
        lines.append(
            f"| {layer} | {np.mean([r['loss_delta'] for r in lr]):>+14.4f} "
            f"| {np.mean([r['agreement'] for r in lr]):>14.0%} |"
        )

    # Worst and best heads
    for label, sorted_results in [
        ("Worst 10 Heads (largest loss increase)", sorted(results, key=lambda r: -r["loss_delta"])[:10]),
        ("Best 10 Heads (smallest loss increase)", sorted(results, key=lambda r: r["loss_delta"])[:10]),
    ]:
        lines.extend(["", f"## {label}", ""])
        lines.append("| Head | Circuit | XSA | Loss Delta | Agreement |")
        lines.append("|------|---------|-----|------------|-----------|")
        for r in sorted_results:
            xsa = "yes" if r["xsa"] else ""
            lines.append(
                f"| L{r['layer']}_H{r['head']} | {r['circuit']} | {xsa} "
                f"| {r['loss_delta']:+.4f} | {r['agreement']:.0%} |"
            )

    out_path.write_text("\n".join(lines))
    print(f"\nResults written to {out_path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    print(f"Loading {N_EVAL_DOCS} FineWeb validation documents...")
    eval_texts = load_fineweb_texts(N_EVAL_DOCS)
    total_chars = sum(len(t) for t in eval_texts)
    print(f"  Loaded {len(eval_texts)} docs, {total_chars:,} chars total")

    print("Loading full Pythia-70M...")
    model_full = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model_full.eval()

    print("Computing baseline loss...")
    baseline_loss = compute_loss(model_full, tokenizer, eval_texts)
    print(f"  Baseline: {baseline_loss:.4f} nats ({baseline_loss / math.log(2):.4f} bpb)")

    # Populate XSA_HEADS set
    XSA_HEADS.clear()
    for (l, h), circuit in CIRCUIT_PLAN.items():
        if circuit in ("prev_token", "induction"):
            XSA_HEADS.add((l, h))

    print(f"\nTesting all 48 heads individually...")
    results = []
    t0 = time.time()

    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            r = test_single_head(model_full, tokenizer, eval_texts,
                                 layer_idx, head_idx, baseline_loss)
            results.append(r)
            circuit = r["circuit"]
            xsa_tag = " [XSA]" if r["xsa"] else ""
            print(f"  L{layer_idx}_H{head_idx} ({circuit}{xsa_tag}): "
                  f"delta={r['loss_delta']:+.4f}, agree={r['agreement']:.0%}")

        elapsed = time.time() - t0
        rate = elapsed / ((layer_idx + 1) * N_HEADS)
        remaining = rate * (N_LAYERS - layer_idx - 1) * N_HEADS
        print(f"  --- Layer {layer_idx} done ({elapsed:.0f}s, ~{remaining:.0f}s left) ---")

    write_results(results, baseline_loss)
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
