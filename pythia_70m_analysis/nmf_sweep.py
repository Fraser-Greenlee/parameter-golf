"""Sweep NMF factor count vs R² for Pythia-70M embeddings.

Run: python3 pythia_70m_analysis/nmf_sweep.py

Results so far (k ≤ 75):
    k |   NMF R² |   SVD R²
    2 |   0.328  |   0.024
    5 |   0.343  |   0.052
   10 |   0.359  |   0.088
   20 |   0.377  |   0.140
   30 |   0.390  |   0.183
   50 |   0.407  |   0.256
   75 |   0.425  |   0.328
"""
import numpy as np, torch, time, json
from pathlib import Path
from sklearn.decomposition import NMF
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

OUT_PATH = Path(__file__).parent / "summary" / "nmf_sweep_results.json"

print("Loading Pythia-70M...", flush=True)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', torch_dtype=torch.float32)
E = model.gpt_neox.embed_in.weight.data.float().numpy()[:tokenizer.vocab_size]
del model  # free memory

# Prepare matrices
E_pos = np.maximum(E, 0)
E_neg = np.maximum(-E, 0)
E_split = np.hstack([E_pos, E_neg])  # [50254, 1024]
E_split_norm = np.linalg.norm(E_split, 'fro')

E_centered = E - E.mean(axis=0)
_, S_svd, _ = np.linalg.svd(E_centered, full_matrices=False)
total_var = (S_svd ** 2).sum()

print(f"Matrix: {E_split.shape}, Frobenius norm: {E_split_norm:.2f}", flush=True)

results = []
ks = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 512]

print(f"\n{'k':>5} | {'NMF R²':>8} | {'SVD R²':>8} | {'Iters':>6} | {'Time':>8} | Progress", flush=True)
print("-" * 65, flush=True)

for ki, k in enumerate(ks):
    svd_r2 = float((S_svd[:k] ** 2).sum() / total_var)

    t0 = time.time()
    nmf = NMF(n_components=k, init='nndsvd', max_iter=1000, random_state=42, tol=1e-4)
    W = nmf.fit_transform(E_split)
    elapsed = time.time() - t0

    nmf_r2 = float(1 - (nmf.reconstruction_err_ / E_split_norm) ** 2)
    n_iter = nmf.n_iter_

    bar = f"[{'█' * (ki+1)}{'░' * (len(ks)-ki-1)}]"
    print(f"{k:>5} | {nmf_r2:>8.4f} | {svd_r2:>8.4f} | {n_iter:>6} | {elapsed:>7.1f}s | {bar} {ki+1}/{len(ks)}",
          flush=True)

    results.append({"k": k, "nmf_r2": nmf_r2, "svd_r2": svd_r2,
                    "n_iter": n_iter, "time_s": elapsed})

    # Save incrementally
    OUT_PATH.write_text(json.dumps(results, indent=2))

print(f"\nResults saved to {OUT_PATH}", flush=True)
