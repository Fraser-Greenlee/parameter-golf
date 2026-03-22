"""
Sweep weight engineering strategies on MLX.

Generates a modified train_gpt_mlx.py with engineered weight init, runs
short training experiments, and compares val_bpb across strategies.

Strategies are defined as init functions that modify the model after
construction. Each receives the full model and can modify any weights.

Usage:
    source .venv/bin/activate
    # Run all strategies:
    python sweep_attn_init.py
    # Run specific strategies:
    STRATEGIES=baseline,bigram_emb_plus_circuit python sweep_attn_init.py
"""

import os
import sys
import subprocess
import json
import re
import time
from pathlib import Path

SWEEP_DIR = Path("analysis_results/init_sweep")

ALL_STRATEGIES = [
    "baseline",
    "bigram_emb",
    "full_circuit",
    "bigram_emb_plus_circuit",
]

_strat_override = os.environ.get("STRATEGIES", "")
STRATEGIES = _strat_override.split(",") if _strat_override else ALL_STRATEGIES

BASE_ENV = {
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "524288",
    "MAX_VAL_BATCHES": "20",
    "TRAIN_LOG_EVERY": "25",
}


# ─── Init code injected into the generated training script ────────────────────

INIT_FUNCTIONS = r'''
# ─── Engineered weight initialization ─────────────────────────────────────────

def apply_weight_init(model, strategy):
    """Apply engineered weight initialization based on strategy name."""
    import numpy as np
    from pathlib import Path

    if strategy == "baseline":
        return

    # ── Bigram SVD embedding init ──
    if "bigram_emb" in strategy:
        emb_path = Path("analysis_results/bigram_emb_init.npy")
        if emb_path.exists():
            emb_init = np.load(str(emb_path))  # (1024, 512) unit-norm rows
            tied_std = 0.005
            emb_init = emb_init * tied_std * (512 ** 0.5)
            model.tok_emb.weight = mx.array(emb_init).astype(COMPUTE_DTYPE)

    # ── Full circuit attention init ──
    if "circuit" in strategy:
        _init_full_circuit(model)


def _init_full_circuit(model):
    """Identity QK + copy OV + orthogonal subspaces + spectral decay."""
    import numpy as np

    dim = model.blocks[0].attn.c_q.weight.shape[0]
    num_heads = model.blocks[0].attn.num_heads
    num_kv_heads = model.blocks[0].attn.num_kv_heads
    head_dim = dim // num_heads
    n_layers = len(model.blocks)
    kv_dim = num_kv_heads * head_dim
    group_size = num_heads // num_kv_heads

    r_qk = 0.06
    r_ov = 0.04
    beta_qk = 0.3
    scale = 1.0 / dim**0.5
    depth_scale = 0.1 / n_layers**0.5
    noise_scale = 0.01

    for li, block in enumerate(model.blocks):
        attn = block.attn
        basis = np.linalg.qr(np.random.randn(dim, dim).astype(np.float32))[0]

        W_Q = np.zeros((dim, dim), dtype=np.float32)
        W_K = np.zeros((kv_dim, dim), dtype=np.float32)
        W_V = np.zeros((kv_dim, dim), dtype=np.float32)
        W_O = np.zeros((dim, dim), dtype=np.float32)

        for g in range(num_kv_heads):
            heads_in_group = range(g * group_size, (g + 1) * group_size)
            g_start = g * head_dim
            P_g = basis[:, g_start:g_start+head_dim]

            s_k = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
            K_struct = P_g * s_k * scale
            K_id = np.zeros((dim, head_dim), dtype=np.float32)
            K_id[g_start:g_start+head_dim, :] = np.eye(head_dim) * beta_qk * scale
            W_K[g_start:g_start+head_dim, :] = (K_struct + K_id).T

            s_v = np.exp(-r_ov * np.arange(head_dim)).astype(np.float32)
            W_V[g_start:g_start+head_dim, :] = (P_g * np.sqrt(s_v) * scale).T

            for h in heads_in_group:
                h_start = h * head_dim
                P_h = basis[:, h_start:h_start+head_dim]
                s_q = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
                Q_struct = P_h * s_q * scale
                Q_id = np.zeros((dim, head_dim), dtype=np.float32)
                Q_id[h_start:h_start+head_dim, :] = np.eye(head_dim) * beta_qk * scale
                W_Q[h_start:h_start+head_dim, :] = (Q_struct + Q_id).T
                W_O[:, h_start:h_start+head_dim] = P_g * np.sqrt(s_v) * scale

        attn.c_q.weight = mx.array(W_Q) + mx.random.normal(W_Q.shape) * noise_scale
        attn.c_k.weight = mx.array(W_K) + mx.random.normal(W_K.shape) * noise_scale
        attn.c_v.weight = mx.array(W_V) + mx.random.normal(W_V.shape) * noise_scale
        attn.proj.weight = mx.array(W_O) * depth_scale + mx.random.normal(W_O.shape) * noise_scale * depth_scale

# ─── End engineered weight init ───────────────────────────────────────────────
'''

# Code injected into main() after model construction
INIT_CALL = '''
    # Apply engineered weight init
    _init_strategy = os.environ.get("ATTN_INIT", "baseline")
    apply_weight_init(model, _init_strategy)
    log(f"weight_init:{_init_strategy}")
'''

# ─── Result parsing ──────────────────────────────────────────────────────────

def parse_results(log_text):
    """Extract val_loss, val_bpb, and train_loss from training log output."""
    results = {"val": [], "train": []}
    for line in log_text.split("\n"):
        m = re.search(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            results["val"].append({
                "step": int(m.group(1)),
                "val_loss": float(m.group(2)),
                "val_bpb": float(m.group(3)),
            })
        m = re.search(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
        if m:
            results["train"].append({
                "step": int(m.group(1)),
                "train_loss": float(m.group(2)),
            })
    m = re.search(r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", log_text)
    if m:
        results["val"].append({
            "step": -1,
            "val_loss": float(m.group(1)),
            "val_bpb": float(m.group(2)),
            "tag": "int8_roundtrip",
        })
    return results


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(strategy, train_script_path):
    """Run one training experiment, streaming output live."""
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["RUN_ID"] = f"sweep_{strategy}"
    env["ATTN_INIT"] = strategy

    print(f"\n{'=' * 60}")
    print(f"Running: {strategy}")
    print(f"{'=' * 60}", flush=True)

    t0 = time.time()
    log_path = SWEEP_DIR / f"{strategy}.log"
    output_lines = []

    proc = subprocess.Popen(
        [sys.executable, "-u", str(train_script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(log_path, "w") as logf:
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            logf.write(line + "\n")
            if any(k in line for k in ["step:", "val_loss:", "final_",
                                         "train_loss:", "weight_init:",
                                         "Error", "Traceback"]):
                print(f"  [{strategy}] {line}", flush=True)
    proc.wait()
    elapsed = time.time() - t0

    output = "\n".join(output_lines)
    parsed = parse_results(output)
    if parsed["val"]:
        final = parsed["val"][-1]
        print(f"  => {strategy} DONE: val_bpb={final['val_bpb']:.4f} ({elapsed:.0f}s)", flush=True)
    else:
        print(f"  => {strategy} WARNING: No val results ({elapsed:.0f}s)")
        if proc.returncode != 0:
            print(f"  Exit code: {proc.returncode}")
            print(f"  Last lines: {output_lines[-5:]}")

    return {
        "strategy": strategy,
        "val_results": parsed["val"],
        "train_results": parsed["train"],
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
    }


# ─── Script generation ────────────────────────────────────────────────────────

def create_modified_train_script():
    """Create train_gpt_mlx.py copy with engineered init and val batch cap."""
    src = Path("train_gpt_mlx.py").read_text()

    # 1. Inject init functions before model classes
    class_marker = "class CastedLinear(nn.Module):"
    src = src.replace(class_marker, INIT_FUNCTIONS + "\n" + class_marker, 1)

    # 2. Inject init call after model construction, before optimizer
    call_marker = "    opt = SplitOptimizers(model, args)"
    if call_marker not in src:
        for candidate in ["    opt = SplitOptimizer", "    compiled_loss = mx.compile"]:
            if candidate in src:
                call_marker = candidate
                break
    src = src.replace(call_marker, INIT_CALL + "\n    " + call_marker.lstrip(), 1)

    # 3. Cap validation batches for fast sweeps
    val_loop = "    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):"
    if val_loop in src:
        val_cap = (
            '    max_val_batches = int(os.environ.get("MAX_VAL_BATCHES", "0"))\n'
            '    if max_val_batches > 0:\n'
            '        total_batches = min(total_batches, max_val_batches)\n'
            '    ' + val_loop.lstrip()
        )
        src = src.replace(val_loop, val_cap, 1)
        # Early break after processing enough val batches
        bytes_line = "        total_bytes += float(bytes_np.astype(np.float64).sum())"
        if bytes_line in src:
            src = src.replace(
                bytes_line,
                bytes_line + "\n        if max_val_batches > 0 and batch_idx >= max_val_batches:\n            break",
                1,
            )

    out_path = Path("train_gpt_mlx_sweep.py")
    out_path.write_text(src)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    train_script = create_modified_train_script()
    print(f"Created: {train_script}")

    all_results = []
    for strategy in STRATEGIES:
        result = run_experiment(strategy, train_script)
        all_results.append(result)

    json_path = SWEEP_DIR / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    iters = int(BASE_ENV["ITERATIONS"])
    print(f"\n{'=' * 80}")
    print("SWEEP RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<30} {'Train@25':<11} {'Train@50':<11} {'Train@{}':<11} {'Val BPB':<11} {'Time':<8}".format(iters))
    print("-" * 85)
    for r in all_results:
        trains = {v["step"]: v["train_loss"] for v in r.get("train_results", [])}
        vals = {v["step"]: v["val_bpb"] for v in r["val_results"] if "tag" not in v}
        final_val = max(vals.items(), key=lambda x: x[0])[1] if vals else None
        row = f"{r['strategy']:<30}"
        for step in [25, 50, iters]:
            tl = trains.get(step)
            row += f" {tl:<11.4f}" if tl else f" {'N/A':<11}"
        row += f" {final_val:<11.4f}" if final_val else f" {'N/A':<11}"
        row += f" {r['elapsed_seconds']:.0f}s"
        print(row)

    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
