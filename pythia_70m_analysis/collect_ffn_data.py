"""Collect FFN neuron activation profiles from Pythia-70M on diverse text samples.

For each layer, captures which neurons fire and on what tokens, producing
human-readable per-layer dumps. Hooks into the post-GELU activation.
"""
import sys, time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from text_samples import TEXT_SAMPLES, SAMPLE_LABELS

OUT_DIR = Path(__file__).parent / "ffn"
N_LAYERS = 6
N_NEURONS = 2048
FIRE_THRESHOLD = 0.1
TOP_K_PER_NEURON = 30  # keep top activations per neuron


def load_model():
    print("Loading Pythia-70M...")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m", torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    return model, tokenizer


def collect_activations(model, tokenizer):
    """Run all samples and collect per-neuron activation data."""
    # Storage: per-neuron top activations
    # neuron_tops[layer][neuron] = list of (activation_value, token_str, sample_idx, position, context_str)
    neuron_tops = {L: {n: [] for n in range(N_NEURONS)} for L in range(N_LAYERS)}
    fire_counts = np.zeros((N_LAYERS, N_NEURONS), dtype=np.int64)
    sum_acts = np.zeros((N_LAYERS, N_NEURONS), dtype=np.float64)
    max_acts = np.full((N_LAYERS, N_NEURONS), -np.inf, dtype=np.float64)
    total_positions = 0

    # Register hooks
    activations = {}
    handles = []
    for L in range(N_LAYERS):
        def make_hook(layer_idx):
            def hook_fn(mod, inp, out):
                activations[layer_idx] = out.detach().cpu()
            return hook_fn
        h = model.gpt_neox.layers[L].mlp.act.register_forward_hook(make_hook(L))
        handles.append(h)

    for s_idx, text in enumerate(TEXT_SAMPLES):
        inputs = tokenizer(text, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0].tolist()]
        seq_len = len(tokens)
        total_positions += seq_len

        with torch.no_grad():
            model(**inputs)

        for L in range(N_LAYERS):
            act = activations[L][0].numpy()  # [seq_len, 2048]
            abs_act = np.abs(act)

            # Update counts
            firing = abs_act > FIRE_THRESHOLD
            fire_counts[L] += firing.sum(axis=0)
            sum_acts[L] += abs_act.sum(axis=0)
            layer_max = abs_act.max(axis=0)
            max_acts[L] = np.maximum(max_acts[L], layer_max)

            # Update top activations for neurons that fired
            for pos in range(seq_len):
                fired_neurons = np.where(firing[pos])[0]
                # Get context window (3 tokens before + current)
                ctx_start = max(0, pos - 3)
                ctx = " ".join(f"'{tokens[j]}'" for j in range(ctx_start, min(pos + 1, seq_len)))

                for n_idx in fired_neurons:
                    val = float(act[pos, n_idx])
                    entry = (abs(val), val, tokens[pos], s_idx, pos, ctx)
                    top_list = neuron_tops[L][n_idx]
                    if len(top_list) < TOP_K_PER_NEURON:
                        top_list.append(entry)
                    elif abs(val) > top_list[-1][0]:
                        top_list[-1] = entry
                        top_list.sort(key=lambda x: -x[0])

        if (s_idx + 1) % 10 == 0:
            print(f"  Processed {s_idx+1}/{len(TEXT_SAMPLES)} samples")

    for h in handles:
        h.remove()

    return neuron_tops, fire_counts, sum_acts, max_acts, total_positions


def analyze_neuron_output_weights(model, layer, neuron_idx, tokenizer, top_k=10):
    """What tokens does this neuron boost/suppress in the output?

    The FFN output for neuron n is: activation_n * W_out[:, n]
    This vector gets added to the residual stream, then projected to logits via W_U.
    Effect on logits: activation_n * W_U @ W_out[:, n]
    """
    W_out = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.data.float()  # [512, 2048]
    W_U = model.embed_out.weight.data.float()  # [50304, 512]

    neuron_dir = W_out[:, neuron_idx]  # [512]
    logit_effect = (W_U @ neuron_dir).numpy()  # [50304]

    # Top boosted tokens
    top_boost = np.argsort(-logit_effect)[:top_k]
    boost_strs = []
    for t_idx in top_boost:
        tok = tokenizer.decode([t_idx])
        boost_strs.append(f"'{tok}' ({logit_effect[t_idx]:+.3f})")

    # Top suppressed tokens
    top_suppress = np.argsort(logit_effect)[:top_k]
    suppress_strs = []
    for t_idx in top_suppress:
        tok = tokenizer.decode([t_idx])
        suppress_strs.append(f"'{tok}' ({logit_effect[t_idx]:+.3f})")

    return boost_strs, suppress_strs


def write_layer_file(layer, neuron_tops, fire_counts, sum_acts, max_acts, total_pos, model, tokenizer):
    """Write per-layer FFN neuron analysis."""
    fire_rates = fire_counts[layer] / max(total_pos, 1)
    mean_acts_arr = sum_acts[layer] / max(total_pos, 1)

    active = (fire_rates > 0.10).sum()
    dead = (fire_rates < 0.01).sum()
    sparse = ((fire_rates >= 0.01) & (fire_rates < 0.05)).sum()

    lines = [
        f"=== Layer {layer} FFN Neurons ({N_NEURONS} neurons) ===",
        f"Total token positions analyzed: {total_pos}",
        f"",
        f"Active neurons (>10% fire rate): {active}/{N_NEURONS}",
        f"Sparse neurons (1-5% fire rate): {sparse}/{N_NEURONS}",
        f"Dead neurons (<1% fire rate): {dead}/{N_NEURONS}",
        f"Mean firing rate: {fire_rates.mean():.1%}",
        f"",
    ]

    # Top 50 by firing rate
    lines.append("--- Top 50 Most Active Neurons (by firing rate) ---")
    lines.append("")
    top_by_fire = np.argsort(-fire_rates)[:50]
    for rank, n_idx in enumerate(top_by_fire):
        top_entries = sorted(neuron_tops[layer][n_idx], key=lambda e: -e[0])[:8]
        trigger_tokens = [e[2] for e in top_entries]
        trigger_str = ", ".join(f"'{t}' ({e[1]:.2f})" for t, e in zip(trigger_tokens, top_entries))
        # Context for top activation
        if top_entries:
            top_ctx = top_entries[0][5]
        else:
            top_ctx = "N/A"

        # What does this neuron boost in output?
        boost_strs, suppress_strs = analyze_neuron_output_weights(model, layer, int(n_idx), tokenizer, top_k=5)

        lines.append(f"Neuron {n_idx:4d} | fire={fire_rates[n_idx]:.0%} | mean={mean_acts_arr[n_idx]:.3f} | max={max_acts[layer, n_idx]:.2f}")
        lines.append(f"  Triggers: {trigger_str}")
        lines.append(f"  Top context: {top_ctx}")
        lines.append(f"  Output boosts: {', '.join(boost_strs[:5])}")
        lines.append(f"  Output suppresses: {', '.join(suppress_strs[:5])}")
        lines.append("")

    # Sparse feature detectors
    lines.append("--- Top 30 Sparse Feature Detectors (1-5% fire rate, by max activation) ---")
    lines.append("")
    sparse_mask = (fire_rates >= 0.01) & (fire_rates < 0.05)
    sparse_neurons = np.where(sparse_mask)[0]
    sparse_by_max = sorted(sparse_neurons, key=lambda n: -max_acts[layer, n])[:30]
    for n_idx in sparse_by_max:
        top_entries = sorted(neuron_tops[layer][n_idx], key=lambda e: -e[0])[:8]
        trigger_str = ", ".join(f"'{e[2]}' ({e[1]:.2f})" for e in top_entries)
        if top_entries:
            top_ctx = top_entries[0][5]
        else:
            top_ctx = "N/A"

        boost_strs, suppress_strs = analyze_neuron_output_weights(model, layer, int(n_idx), tokenizer, top_k=5)

        lines.append(f"Neuron {n_idx:4d} | fire={fire_rates[n_idx]:.1%} | max={max_acts[layer, n_idx]:.2f}")
        lines.append(f"  Triggers: {trigger_str}")
        lines.append(f"  Top context: {top_ctx}")
        lines.append(f"  Output boosts: {', '.join(boost_strs[:5])}")
        lines.append(f"  Output suppresses: {', '.join(suppress_strs[:5])}")
        lines.append("")

    # Very sparse / rare feature detectors
    lines.append("--- Ultra-Sparse Neurons (<1% fire rate but fired at least once) ---")
    lines.append("")
    ultra_sparse = np.where((fire_rates > 0) & (fire_rates < 0.01))[0]
    ultra_by_max = sorted(ultra_sparse, key=lambda n: -max_acts[layer, n])[:15]
    for n_idx in ultra_by_max:
        top_entries = sorted(neuron_tops[layer][n_idx], key=lambda e: -e[0])[:5]
        trigger_str = ", ".join(f"'{e[2]}' ({e[1]:.2f})" for e in top_entries)
        lines.append(f"  Neuron {n_idx:4d} | fire={fire_rates[n_idx]:.2%} | max={max_acts[layer, n_idx]:.2f} | triggers: {trigger_str}")

    # Dead neurons
    truly_dead = np.where(fire_counts[layer] == 0)[0]
    lines.append(f"\n--- Truly Dead Neurons (never fired): {len(truly_dead)} ---")
    if len(truly_dead) > 0:
        lines.append(f"  Indices: {truly_dead[:80].tolist()}{'...' if len(truly_dead) > 80 else ''}")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()

    print("Collecting FFN activations...")
    t0 = time.time()
    neuron_tops, fire_counts, sum_acts, max_acts, total_pos = collect_activations(model, tokenizer)
    print(f"  Collection done ({time.time() - t0:.1f}s)")

    print("Writing layer files...")
    for layer in range(N_LAYERS):
        text = write_layer_file(layer, neuron_tops, fire_counts, sum_acts, max_acts, total_pos, model, tokenizer)
        (OUT_DIR / f"L{layer}_neurons.txt").write_text(text)
        print(f"  Layer {layer} written")

    print(f"\nDone. {N_LAYERS} files in {OUT_DIR}/ ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
