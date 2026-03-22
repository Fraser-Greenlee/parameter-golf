"""
Sweep attention initialization strategies on MLX.

Tests circuit-inspired init schemes derived from our cross-model analysis:
1. baseline     — default (Kaiming uniform Q/K/V, zero proj)
2. mimetic      — mimetic init: QK ~ alpha*noise + beta*I, OV ~ alpha*noise + beta*I
3. spectral     — exponential SV decay matching trained model profiles
4. identity_qk  — QK circuit biased toward identity (attend-to-self)
5. ortho_head   — orthogonal subspace allocation per head
6. copy_ov      — OV circuit biased toward identity (copy)
7. full_circuit — combine identity_qk + copy_ov + ortho_head + spectral decay

Each strategy only changes initialization of c_q, c_k, c_v, proj weights.
The rest of training is identical.

Usage:
    source .venv/bin/activate
    python sweep_attn_init.py
"""

import os
import sys
import subprocess
import json
import re
import time
from pathlib import Path

SWEEP_DIR = Path("analysis_results/init_sweep")

STRATEGIES = [
    "baseline",
    "mimetic",
    "spectral",
    "identity_qk",
    "ortho_head",
    "copy_ov",
    "full_circuit",
]

# Short run config for local Mac testing
# VAL_BATCH_SIZE must be large enough to avoid 60k+ tiny val batches
BASE_ENV = {
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "524288",
    "MAX_VAL_BATCHES": "20",
    "TRAIN_LOG_EVERY": "25",
}


def parse_results(log_text):
    """Extract val_loss, val_bpb, and train_loss from training log output."""
    results = {"val": [], "train": []}
    for line in log_text.split("\n"):
        # Val results
        m = re.search(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            results["val"].append({
                "step": int(m.group(1)),
                "val_loss": float(m.group(2)),
                "val_bpb": float(m.group(3)),
            })
        # Train loss
        m = re.search(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
        if m:
            results["train"].append({
                "step": int(m.group(1)),
                "train_loss": float(m.group(2)),
            })
    # Final int8 result
    m = re.search(r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", log_text)
    if m:
        results["val"].append({
            "step": -1,
            "val_loss": float(m.group(1)),
            "val_bpb": float(m.group(2)),
            "tag": "int8_roundtrip",
        })
    return results


def run_experiment(strategy, train_script_path):
    """Run one training experiment with given init strategy, streaming output live."""
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
            # Print key lines live, skip warmup/val_progress spam
            if any(k in line for k in ["step:", "val_loss:", "val_bpb:", "final_",
                                         "train_loss:", "Applied", "Using baseline",
                                         "Error", "Traceback"]):
                print(f"  [{strategy}] {line}", flush=True)
    proc.wait()
    elapsed = time.time() - t0

    output = "\n".join(output_lines)
    parsed = parse_results(output)
    if parsed["val"]:
        final = parsed["val"][-1]
        print(f"  => {strategy} DONE: val_loss={final['val_loss']:.4f} val_bpb={final['val_bpb']:.4f} ({elapsed:.0f}s)", flush=True)
    elif parsed["train"]:
        final = parsed["train"][-1]
        print(f"  => {strategy} DONE: train_loss={final['train_loss']:.4f} ({elapsed:.0f}s)", flush=True)
    else:
        print(f"  => {strategy} WARNING: No results found ({elapsed:.0f}s)")
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


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Create the modified training script
    train_script = create_modified_train_script()
    print(f"Created modified training script: {train_script}")

    all_results = []
    for strategy in STRATEGIES:
        result = run_experiment(strategy, train_script)
        all_results.append(result)

    # Save summary
    json_path = SWEEP_DIR / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("INIT STRATEGY SWEEP RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<20} {'Train@25':<11} {'Train@50':<11} {'Train@100':<11} {'Train@200':<11} {'Final Val':<11} {'Time':<8}")
    print("-" * 90)
    for r in all_results:
        trains = {v["step"]: v["train_loss"] for v in r.get("train_results", [])}
        vals = {v["step"]: v["val_bpb"] for v in r["val_results"] if "tag" not in v}
        final_val = vals.get(200, vals.get(max(vals.keys())) if vals else None)
        row = f"{r['strategy']:<20}"
        for step in [25, 50, 100, 200]:
            tl = trains.get(step, None)
            row += f" {tl:<11.4f}" if tl else f" {'N/A':<11}"
        row += f" {final_val:<11.4f}" if final_val else f" {'N/A':<11}"
        row += f" {r['elapsed_seconds']:.0f}s"
        print(row)

    print(f"\nResults saved to {json_path}")


def create_modified_train_script():
    """
    Create a copy of train_gpt_mlx.py with circuit-aware init injected.
    """
    src = Path("train_gpt_mlx.py").read_text()

    # We inject the init code right after model construction, before training.
    # Find where proj weights are zeroed and add our init after.
    init_code = '''

# ─── Circuit-aware attention initialization ──────────────────────────────────
def apply_attn_init(model, strategy):
    """Apply circuit-aware attention initialization."""
    import numpy as np

    if strategy == "baseline":
        return  # Use default init

    dim = model.blocks[0].attn.c_q.weight.shape[0]
    num_heads = model.blocks[0].attn.num_heads
    num_kv_heads = model.blocks[0].attn.num_kv_heads
    head_dim = dim // num_heads
    n_layers = len(model.blocks)
    kv_dim = num_kv_heads * head_dim
    group_size = num_heads // num_kv_heads

    for li, block in enumerate(model.blocks):
        attn = block.attn
        layer_pos = li / max(n_layers - 1, 1)  # 0 to 1

        if strategy == "mimetic":
            # Mimetic init: QK ~ alpha*noise + beta*I, factor via SVD
            alpha_qk, beta_qk = 0.7, 0.7
            alpha_ov, beta_ov = 0.4, 0.4

            # Build QK target per KV group, factor into Q and K
            W_Q_all = np.zeros((dim, dim), dtype=np.float32)
            W_K_all = np.zeros((kv_dim, dim), dtype=np.float32)
            W_V_all = np.zeros((kv_dim, dim), dtype=np.float32)
            W_O_all = np.zeros((dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                for h in heads_in_group:
                    h_start = h * head_dim
                    # QK circuit in residual stream: target is alpha*noise + beta*I
                    Z = np.random.randn(dim, dim).astype(np.float32) / dim**0.5
                    M_qk = alpha_qk * Z + beta_qk * np.eye(dim, dtype=np.float32)
                    U, S, Vh = np.linalg.svd(M_qk, full_matrices=False)
                    sqrt_S = np.sqrt(S[:head_dim])
                    W_Q_all[h_start:h_start+head_dim, :] = (Vh[:head_dim].T * sqrt_S).T

                # K is shared for the group
                g_start = g * head_dim
                Z = np.random.randn(dim, dim).astype(np.float32) / dim**0.5
                M_qk = alpha_qk * Z + beta_qk * np.eye(dim, dtype=np.float32)
                U, S, Vh = np.linalg.svd(M_qk, full_matrices=False)
                sqrt_S = np.sqrt(S[:head_dim])
                W_K_all[g_start:g_start+head_dim, :] = (Vh[:head_dim].T * sqrt_S).T

                # OV circuit
                Z2 = np.random.randn(dim, dim).astype(np.float32) / dim**0.5
                M_ov = alpha_ov * Z2 + beta_ov * np.eye(dim, dtype=np.float32)
                U2, S2, Vh2 = np.linalg.svd(M_ov, full_matrices=False)
                sqrt_S2 = np.sqrt(S2[:head_dim])
                W_V_all[g_start:g_start+head_dim, :] = (Vh2[:head_dim].T * sqrt_S2).T
                for h in heads_in_group:
                    h_start = h * head_dim
                    W_O_all[:, h_start:h_start+head_dim] = U2[:, :head_dim] * sqrt_S2

            attn.c_q.weight = mx.array(W_Q_all)
            attn.c_k.weight = mx.array(W_K_all)
            attn.c_v.weight = mx.array(W_V_all)
            attn.proj.weight = mx.array(W_O_all) * (0.1 / n_layers**0.5)

        elif strategy == "spectral":
            # Match empirical spectral profiles: exp decay with fitted rates
            # From our analysis: qk_decay ~ 0.06, ov_decay ~ 0.04
            r_qk = 0.06
            r_ov = 0.04
            scale = 1.0 / dim**0.5

            W_Q = np.zeros((dim, dim), dtype=np.float32)
            W_K = np.zeros((kv_dim, dim), dtype=np.float32)
            W_V = np.zeros((kv_dim, dim), dtype=np.float32)
            W_O = np.zeros((dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                g_start = g * head_dim

                # K: random orthogonal with spectral shaping
                K_base = np.linalg.qr(np.random.randn(dim, head_dim).astype(np.float32))[0]
                s_k = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
                W_K[g_start:g_start+head_dim, :] = (K_base * s_k * scale).T

                # V
                V_base = np.linalg.qr(np.random.randn(dim, head_dim).astype(np.float32))[0]
                s_v = np.exp(-r_ov * np.arange(head_dim)).astype(np.float32)
                W_V[g_start:g_start+head_dim, :] = (V_base * s_v * scale).T

                for h in heads_in_group:
                    h_start = h * head_dim
                    # Q
                    Q_base = np.linalg.qr(np.random.randn(dim, head_dim).astype(np.float32))[0]
                    s_q = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
                    W_Q[h_start:h_start+head_dim, :] = (Q_base * s_q * scale).T
                    # O
                    O_base = np.linalg.qr(np.random.randn(dim, head_dim).astype(np.float32))[0]
                    s_o = np.exp(-r_ov * np.arange(head_dim)).astype(np.float32)
                    W_O[:, h_start:h_start+head_dim] = O_base * s_o * scale

            attn.c_q.weight = mx.array(W_Q)
            attn.c_k.weight = mx.array(W_K)
            attn.c_v.weight = mx.array(W_V)
            attn.proj.weight = mx.array(W_O) * (0.1 / n_layers**0.5)

        elif strategy == "identity_qk":
            # QK circuit biased toward identity: heads attend to similar content
            beta = 0.5  # identity strength
            scale = 1.0 / dim**0.5

            W_Q = np.zeros((dim, dim), dtype=np.float32)
            W_K = np.zeros((kv_dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                g_start = g * head_dim

                # Shared K direction: mix of identity subspace + random
                K_id = np.zeros((dim, head_dim), dtype=np.float32)
                K_id[g_start:g_start+head_dim, :] = np.eye(head_dim) * beta
                K_rand = np.random.randn(dim, head_dim).astype(np.float32) * scale * (1 - beta)
                W_K[g_start:g_start+head_dim, :] = (K_id + K_rand).T

                for h in heads_in_group:
                    h_start = h * head_dim
                    Q_id = np.zeros((dim, head_dim), dtype=np.float32)
                    Q_id[h_start:h_start+head_dim, :] = np.eye(head_dim) * beta
                    Q_rand = np.random.randn(dim, head_dim).astype(np.float32) * scale * (1 - beta)
                    W_Q[h_start:h_start+head_dim, :] = (Q_id + Q_rand).T

            attn.c_q.weight = mx.array(W_Q)
            attn.c_k.weight = mx.array(W_K)
            # V and O stay default (V = kaiming, O = zero)

        elif strategy == "ortho_head":
            # Assign each head an orthogonal subspace of residual stream
            basis = np.linalg.qr(np.random.randn(dim, dim).astype(np.float32))[0]
            scale = 1.0 / dim**0.5

            W_Q = np.zeros((dim, dim), dtype=np.float32)
            W_K = np.zeros((kv_dim, dim), dtype=np.float32)
            W_V = np.zeros((kv_dim, dim), dtype=np.float32)
            W_O = np.zeros((dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                g_start = g * head_dim
                P_g = basis[:, g_start:g_start+head_dim]  # (dim, head_dim)

                W_K[g_start:g_start+head_dim, :] = (P_g * scale).T
                W_V[g_start:g_start+head_dim, :] = (P_g * scale).T

                for h in heads_in_group:
                    h_start = h * head_dim
                    P_h = basis[:, h_start:h_start+head_dim]
                    W_Q[h_start:h_start+head_dim, :] = (P_h * scale).T
                    W_O[:, h_start:h_start+head_dim] = P_h * scale

            attn.c_q.weight = mx.array(W_Q)
            attn.c_k.weight = mx.array(W_K)
            attn.c_v.weight = mx.array(W_V)
            attn.proj.weight = mx.array(W_O) * (0.1 / n_layers**0.5)

        elif strategy == "copy_ov":
            # OV circuit near identity with exponential SV decay
            r_ov = 0.04
            scale = 1.0 / dim**0.5

            W_V = np.zeros((kv_dim, dim), dtype=np.float32)
            W_O = np.zeros((dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                g_start = g * head_dim

                # V: project into subspace with SV decay
                V_base = np.linalg.qr(np.random.randn(dim, head_dim).astype(np.float32))[0]
                sigmas = np.exp(-r_ov * np.arange(head_dim)).astype(np.float32)
                W_V[g_start:g_start+head_dim, :] = (V_base * np.sqrt(sigmas) * scale).T

                for h in heads_in_group:
                    h_start = h * head_dim
                    # O: matching subspace so OV ~ identity with decay
                    W_O[:, h_start:h_start+head_dim] = V_base * np.sqrt(sigmas) * scale

            attn.c_v.weight = mx.array(W_V)
            attn.proj.weight = mx.array(W_O) * (0.1 / n_layers**0.5)
            # Q, K stay default

        elif strategy == "full_circuit":
            # Combine: ortho subspaces + identity QK + copy OV + spectral decay + depth scaling
            basis = np.linalg.qr(np.random.randn(dim, dim).astype(np.float32))[0]
            r_qk = 0.06
            r_ov = 0.04
            beta_qk = 0.3  # identity bias in QK
            scale = 1.0 / dim**0.5
            depth_scale = 0.1 / n_layers**0.5

            W_Q = np.zeros((dim, dim), dtype=np.float32)
            W_K = np.zeros((kv_dim, dim), dtype=np.float32)
            W_V = np.zeros((kv_dim, dim), dtype=np.float32)
            W_O = np.zeros((dim, dim), dtype=np.float32)

            for g in range(num_kv_heads):
                heads_in_group = range(g * group_size, (g + 1) * group_size)
                g_start = g * head_dim
                P_g = basis[:, g_start:g_start+head_dim]

                # K: orthogonal subspace + identity bias + spectral decay
                s_k = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
                K_struct = P_g * s_k * scale
                K_id = np.zeros((dim, head_dim), dtype=np.float32)
                K_id[g_start:g_start+head_dim, :] = np.eye(head_dim) * beta_qk * scale
                W_K[g_start:g_start+head_dim, :] = (K_struct + K_id).T

                # V: orthogonal subspace + spectral decay (copy-like OV)
                s_v = np.exp(-r_ov * np.arange(head_dim)).astype(np.float32)
                W_V[g_start:g_start+head_dim, :] = (P_g * np.sqrt(s_v) * scale).T

                for h in heads_in_group:
                    h_start = h * head_dim
                    P_h = basis[:, h_start:h_start+head_dim]

                    # Q: orthogonal subspace + identity bias + spectral decay
                    s_q = np.exp(-r_qk * np.arange(head_dim)).astype(np.float32)
                    Q_struct = P_h * s_q * scale
                    Q_id = np.zeros((dim, head_dim), dtype=np.float32)
                    Q_id[h_start:h_start+head_dim, :] = np.eye(head_dim) * beta_qk * scale
                    W_Q[h_start:h_start+head_dim, :] = (Q_struct + Q_id).T

                    # O: matching V subspace for copy-like OV
                    W_O[:, h_start:h_start+head_dim] = P_g * np.sqrt(s_v) * scale

            attn.c_q.weight = mx.array(W_Q)
            attn.c_k.weight = mx.array(W_K)
            attn.c_v.weight = mx.array(W_V)
            attn.proj.weight = mx.array(W_O) * depth_scale

            # Add small noise for symmetry breaking
            noise_scale = 0.01
            attn.c_q.weight = attn.c_q.weight + mx.random.normal(attn.c_q.weight.shape) * noise_scale
            attn.c_k.weight = attn.c_k.weight + mx.random.normal(attn.c_k.weight.shape) * noise_scale
            attn.c_v.weight = attn.c_v.weight + mx.random.normal(attn.c_v.weight.shape) * noise_scale
            attn.proj.weight = attn.proj.weight + mx.random.normal(attn.proj.weight.shape) * noise_scale * depth_scale

# ─── End circuit-aware init ──────────────────────────────────────────────────
'''

    # Find the injection point: after "self.tok_emb.weight = ..." line in GPT.__init__
    # We need to inject a call site in the training loop, after model creation
    # Look for where the model is created in main()
    inject_after = "    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)"

    # Actually, better to inject right after the model is constructed and weights are zeroed.
    # Find the zero-init block in GPT.__init__ equivalent
    # In MLX script, this is around line 407-412
    marker = "        self.tok_emb.weight = (\n            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std\n        ).astype(COMPUTE_DTYPE)"

    if marker not in src:
        print("ERROR: Could not find injection point in train_gpt_mlx.py")
        print("Looking for marker text...")
        # Try a simpler marker
        marker = ".astype(COMPUTE_DTYPE)"
        # Fall back to injecting after model creation in main
        pass

    # Inject the init functions at module level, and the call after model creation
    # Find "def main" or the training function
    # Better approach: inject init code as a function, then call it after model is built

    # Find where model is used for the first time after creation
    # In the MLX script, model is created around line ~920, then used
    call_marker = "    q_eval_ms = 1000.0"

    # Inject right after model creation, before optimizer
    call_marker = "    opt = SplitOptimizers(model, args)"

    if call_marker not in src:
        print(f"WARNING: Could not find call_marker, trying alternatives...")
        for candidate in ["    opt = SplitOptimizer", "    compiled_loss = mx.compile"]:
            if candidate in src:
                call_marker = candidate
                print(f"  Using alternative: {candidate[:50]}")
                break

    # Inject function definition before the model class
    class_marker = "class CastedLinear(nn.Module):"
    src = src.replace(class_marker, init_code + "\n" + class_marker, 1)

    # Inject the call after model creation, before training starts
    init_call = '''
    # Apply circuit-aware attention init
    attn_init_strategy = os.environ.get("ATTN_INIT", "baseline")
    if attn_init_strategy != "baseline":
        apply_attn_init(model, attn_init_strategy)
        log(f"Applied attention init strategy: {attn_init_strategy}")
    else:
        log("Using baseline attention init")

'''
    src = src.replace(call_marker, init_call + "    " + call_marker.lstrip(), 1)

    # Cap validation batches to keep sweep fast on Mac
    # Inject MAX_VAL_BATCHES limit into the eval_val loop
    val_loop_marker = "    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):"
    if val_loop_marker in src:
        max_val_patch = """    max_val_batches = int(os.environ.get("MAX_VAL_BATCHES", "0"))
    if max_val_batches > 0:
        total_batches = min(total_batches, max_val_batches)
""" + "    " + val_loop_marker.lstrip()
        src = src.replace(val_loop_marker, max_val_patch, 1)
        # Add early break at the END of the loop body (after total_bytes +=)
        bytes_marker = "        total_bytes += float(bytes_np.astype(np.float64).sum())"
        if bytes_marker in src:
            src = src.replace(
                bytes_marker,
                bytes_marker + "\n        if max_val_batches > 0 and batch_idx >= max_val_batches:\n            break",
                1,
            )

    out_path = Path("train_gpt_mlx_sweep.py")
    out_path.write_text(src)
    return out_path


if __name__ == "__main__":
    main()
