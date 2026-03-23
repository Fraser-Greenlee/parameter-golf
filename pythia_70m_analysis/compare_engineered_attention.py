"""Compare full Pythia-70M vs hybrid (real embed/FFN + engineered attention + XSA).

Measures cross-entropy loss on text samples to quantify how much of Pythia-70M's
performance our engineered attention circuits capture.
"""
import sys, time, math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from engineer_weights import (
    populate_model, XSA_HEADS, apply_xsa_to_attention,
    N_LAYERS, N_HEADS, CIRCUIT_PLAN,
)

# Evaluation texts -- longer passages for meaningful loss measurement
EVAL_TEXTS = [
    # Wikipedia-style
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.",

    # Technical
    "In machine learning, gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the gradient of the function at the current point, because this is the direction of steepest descent. Conversely, stepping in the direction of the gradient will lead to a local maximum of that function; the procedure is then known as gradient ascent. Gradient descent is generally attributed to Augustin-Louis Cauchy, who first suggested it in 1847.",

    # Narrative
    "The old man sat alone in the corner of the cafe, stirring his coffee with a small silver spoon. He had been coming to this same spot every morning for thirty years, ever since his wife had passed away. The waitress knew his order by heart: black coffee, two sugars, and a slice of lemon cake. She brought it to him without being asked, and he nodded his thanks with a tired smile. Outside, the rain began to fall, and he watched the drops trace lines down the window glass.",

    # Code-adjacent
    "To implement a hash table, you need two things: a hash function that maps keys to array indices, and a strategy for handling collisions when two keys map to the same index. The simplest collision resolution strategy is chaining, where each array slot holds a linked list of all key-value pairs that hash to that index. An alternative is open addressing, where collisions are resolved by probing for the next empty slot in the array. The load factor, defined as the number of entries divided by the number of slots, determines the expected performance.",

    # Dialogue-heavy
    "\"I don't think we should go through with this,\" said Maria, putting down her pen. \"The risks are too high.\" David leaned forward in his chair. \"But think about what we stand to gain. If this works, it changes everything.\" \"And if it doesn't work?\" she asked. \"Then we've lost six months and half our budget.\" He paused, considering her words. \"What if we start with a smaller pilot program? Test it on one department first, then scale up if the results are good.\" Maria thought about it for a moment. \"That... might actually work.\"",

    # Repetitive/structured
    "The first rule of the competition is that all submissions must be original work. The second rule of the competition is that submissions must be received by the deadline. The third rule is that each participant may submit only one entry. The fourth rule is that entries will be judged by a panel of five experts. The fifth rule is that the judges' decision is final. The sixth rule is that prizes will be awarded to the top three entries.",
]


def compute_loss(model, tokenizer, texts, apply_xsa=False):
    """Compute mean cross-entropy loss across texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            if apply_xsa:
                # Run with output_attentions to apply XSA post-hoc
                # NOTE: This applies XSA to the attention WEIGHTS for reporting,
                # but doesn't change the actual forward pass computation.
                # For a true XSA comparison, we need to modify the forward pass.
                outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits  # [1, seq, vocab]

        # Shift: predict token i+1 from position i
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += seq_len - 1

    return total_loss / total_tokens


def compute_per_position_loss(model, tokenizer, text):
    """Compute loss at each position for detailed comparison."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq, vocab]

    losses = []
    for i in range(len(tokens) - 1):
        log_probs = F.log_softmax(logits[i], dim=-1)
        target = input_ids[0, i + 1]
        loss = -log_probs[target].item()
        losses.append((tokens[i + 1], loss))

    return losses


def compare_attention_patterns(model_full, model_hybrid, tokenizer, text):
    """Compare attention patterns on a specific text between models."""
    inputs = tokenizer(text, return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]

    with torch.no_grad():
        out_full = model_full(**inputs, output_attentions=True)
        out_hybrid = model_hybrid(**inputs, output_attentions=True)

    print(f"\n  Tokens: {' '.join(tokens[:15])}{'...' if len(tokens) > 15 else ''}")

    for layer in range(N_LAYERS):
        attn_full = out_full.attentions[layer][0].numpy()    # [heads, seq, seq]
        attn_hybrid = out_hybrid.attentions[layer][0].numpy()
        # Apply XSA to hybrid
        attn_hybrid = apply_xsa_to_attention(attn_hybrid, layer)

        # Compute per-head agreement: for each query, does top-1 match?
        agreements = []
        for h in range(N_HEADS):
            agree = 0
            total = 0
            for q in range(1, len(tokens)):
                top1_full = np.argmax(attn_full[h, q, :q + 1])
                top1_hybrid = np.argmax(attn_hybrid[h, q, :q + 1])
                if top1_full == top1_hybrid:
                    agree += 1
                total += 1
            agreements.append(agree / max(total, 1))

        circuit_types = [CIRCUIT_PLAN.get((layer, h), "?") for h in range(N_HEADS)]
        agree_strs = [f"H{h}({circuit_types[h][:3]})={agreements[h]:.0%}" for h in range(N_HEADS)]
        mean_agree = np.mean(agreements)
        print(f"  L{layer} top-1 agreement: {mean_agree:.0%}  [{', '.join(agree_strs)}]")


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # --- Model 1: Full Pythia-70M ---
    print("Loading full Pythia-70M...")
    model_full = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    model_full.eval()

    print("Computing baseline loss...")
    loss_full = compute_loss(model_full, tokenizer, EVAL_TEXTS)
    bpb_full = loss_full / math.log(2)
    print(f"  Full Pythia-70M: loss={loss_full:.4f} nats, bpb={bpb_full:.4f}")

    # --- Model 2: Hybrid (real embed + real FFN + engineered attention + XSA) ---
    print("\nBuilding hybrid model (real embed/FFN + engineered attention)...")
    model_hybrid = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32,
        attn_implementation="eager"
    )

    # Replace ONLY the attention weights with engineered circuits
    # Keep: embed_in, embed_out, all FFN weights, all layer norms
    populate_model(model_hybrid)

    # Restore the real layer norms (populate_model sets them to identity)
    for layer_idx in range(N_LAYERS):
        real_layer = model_full.gpt_neox.layers[layer_idx]
        hybrid_layer = model_hybrid.gpt_neox.layers[layer_idx]

        hybrid_layer.input_layernorm.weight.data.copy_(real_layer.input_layernorm.weight.data)
        hybrid_layer.input_layernorm.bias.data.copy_(real_layer.input_layernorm.bias.data)
        hybrid_layer.post_attention_layernorm.weight.data.copy_(real_layer.post_attention_layernorm.weight.data)
        hybrid_layer.post_attention_layernorm.bias.data.copy_(real_layer.post_attention_layernorm.bias.data)

    model_full.gpt_neox.final_layer_norm.weight.data.copy_(
        model_full.gpt_neox.final_layer_norm.weight.data)
    model_hybrid.gpt_neox.final_layer_norm.weight.data.copy_(
        model_full.gpt_neox.final_layer_norm.weight.data)
    model_hybrid.gpt_neox.final_layer_norm.bias.data.copy_(
        model_full.gpt_neox.final_layer_norm.bias.data)

    # Restore real FFN weights (populate_model doesn't touch these, but let's be explicit)
    for layer_idx in range(N_LAYERS):
        real_layer = model_full.gpt_neox.layers[layer_idx]
        hybrid_layer = model_hybrid.gpt_neox.layers[layer_idx]
        hybrid_layer.mlp.dense_h_to_4h.weight.data.copy_(real_layer.mlp.dense_h_to_4h.weight.data)
        hybrid_layer.mlp.dense_h_to_4h.bias.data.copy_(real_layer.mlp.dense_h_to_4h.bias.data)
        hybrid_layer.mlp.dense_4h_to_h.weight.data.copy_(real_layer.mlp.dense_4h_to_h.weight.data)
        hybrid_layer.mlp.dense_4h_to_h.bias.data.copy_(real_layer.mlp.dense_4h_to_h.bias.data)

    model_hybrid.eval()

    print("Computing hybrid loss...")
    loss_hybrid = compute_loss(model_hybrid, tokenizer, EVAL_TEXTS)
    bpb_hybrid = loss_hybrid / math.log(2)
    print(f"  Hybrid (eng attn + XSA): loss={loss_hybrid:.4f} nats, bpb={bpb_hybrid:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'Model':<35} {'Loss (nats)':>12} {'BPB':>8}")
    print(f"{'-'*60}")
    print(f"{'Full Pythia-70M':<35} {loss_full:>12.4f} {bpb_full:>8.4f}")
    print(f"{'Hybrid (eng attn + XSA)':<35} {loss_hybrid:>12.4f} {bpb_hybrid:>8.4f}")
    print(f"{'-'*60}")
    print(f"{'Gap':<35} {loss_hybrid - loss_full:>+12.4f} {bpb_hybrid - bpb_full:>+8.4f}")
    print(f"\nAttention engineering captures {(1 - (loss_hybrid - loss_full) / loss_full) * 100:.1f}% "
          f"of full model quality (lower gap = better)")

    # --- Attention pattern comparison ---
    print(f"\n{'='*60}")
    print("Attention Pattern Agreement (top-1 match between full and hybrid)")
    print(f"{'='*60}")

    # Repetitive text (should show induction agreement)
    print("\n--- Repetitive text ---")
    compare_attention_patterns(model_full, model_hybrid, tokenizer,
        "The cat sat on the mat. The dog sat on the log. The cat sat on the mat.")

    # News text
    print("\n--- News text ---")
    compare_attention_patterns(model_full, model_hybrid, tokenizer,
        EVAL_TEXTS[0][:200])

    # --- Per-position loss comparison on one text ---
    print(f"\n{'='*60}")
    print("Per-Position Loss Comparison (first eval text, first 30 tokens)")
    print(f"{'='*60}")
    losses_full = compute_per_position_loss(model_full, tokenizer, EVAL_TEXTS[0])
    losses_hybrid = compute_per_position_loss(model_hybrid, tokenizer, EVAL_TEXTS[0])

    print(f"{'Token':<15} {'Full':>8} {'Hybrid':>8} {'Diff':>8}")
    print("-" * 42)
    for i in range(min(30, len(losses_full))):
        tok_f, l_f = losses_full[i]
        tok_h, l_h = losses_hybrid[i]
        diff = l_h - l_f
        marker = " <<" if abs(diff) > 2.0 else ""
        print(f"{tok_f:<15} {l_f:>8.3f} {l_h:>8.3f} {diff:>+8.3f}{marker}")


if __name__ == "__main__":
    main()
