# Initializing attention weights to skip early training dynamics

**Trained transformer attention heads converge to remarkably consistent low-rank structures with identity-like QK circuits and copying OV circuits — structures that can be directly encoded at initialization to bypass the costly phase transition that dominates early training.** The most critical finding across seven research streams is that the induction head phase transition at ~1–2% of training represents the model discovering just three effective parameters (match, route, copy) in a 19-dimensional subspace. This transition can potentially be shortcut by initializing W_Q·W_K^T ≈ I and W_O·W_V ≈ ±I with appropriate spectral profiles, using SVD factorization to decompose these target products into the rectangular weight matrices the architecture requires.

---

## The anatomy of attention circuits reveals simple target structures

Anthropic's *Mathematical Framework for Transformer Circuits* (Elhage et al., 2021) decomposes each attention head into two independent low-rank circuits. The **QK circuit** (W_Q^T·W_K, rank ≤ d_head) determines where information flows — which tokens attend to which. The **OV circuit** (W_O·W_V, rank ≤ d_head) determines what information moves when attention fires. Both are inherently rank-bottlenecked: with d_head = 64 and d_model = 768, each head operates in a **64-dimensional subspace** of the 768-dimensional residual stream.

The residual stream acts as a shared communication bus. Every head reads from it via linear projection and writes back additively. Different heads avoid interference by operating in approximately orthogonal subspaces. This additive, low-rank structure makes the target geometry for initialization surprisingly tractable.

The most important circuit discovered is the **induction head** — a two-layer composition that implements the pattern [A][B]...[A] → predict B. It requires coordination between a **previous-token head** in layer 0 (QK circuit implements a shifted diagonal over positional embeddings; OV circuit copies token identity) and an **induction head** in layer 1 (QK circuit matches the current token against "what preceded me" information written by the layer-0 head via K-composition; OV circuit is approximately identity — a pure copying circuit). The eigenvalue signature is distinctive: both the K-composed QK circuit and the OV circuit of trained induction heads show **strongly positive real eigenvalues**, placing them in the extreme upper-right corner of a (QK-positivity, OV-positivity) scatter plot. Random initialization produces the Ginibre distribution — eigenvalues uniformly scattered across the complex plane, about as far from the target as possible.

---

## Spectral signatures of trained weights follow heavy-tailed exponential decay

Empirical analysis of trained transformers reveals consistent spectral structure across models ranging from GPT-2 to LLaMA-3-70B. The OV circuit (W_O·W_V) of GPT-2 medium shows **approximate exponential singular value decay** — linear on a log scale — with a super-exponential dropoff in the top 5–10 directions, then a more gradual exponential tail until the hard cutoff at rank d_head. Crucially, the network utilizes most of the available rank: the spectrum doesn't collapse to rank 1 or 2 but fills the d_head-dimensional subspace, with the top singular vectors being highly interpretable when projected to token space.

The broader weight matrix landscape follows what Martin & Mahoney call **heavy-tailed self-regularization**: empirical spectral densities exhibit power-law behavior with exponents α typically between **1.5 and 3.5**. Millidge's analysis of Pythia models (up to 1.3B) during training found that trained attention Q/K/V weight matrices develop a distinctive distribution with excess near-zero singular values combined with an extremely heavy tail — heavier than logistic or power-law. This distribution **forms rapidly at the beginning of training** and remains consistent throughout, suggesting it reflects architectural inductive bias rather than dataset properties.

Different matrix types show different profiles. Query matrices deviate most from random matrix theory predictions (the Marchenko-Pastur law), indicating they undergo the most feature learning. **Attention output matrices remain closest to random initialization** — a striking finding suggesting W_O undergoes milder updates and could potentially be initialized more carefully. The Loki paper (NeurIPS 2024) measured effective key vector rank across 11 LLMs: **~80 out of 128 dimensions** capture 90% of variance, with initial layers having much lower rank than later layers. Value vectors are near full rank, while keys and queries are significantly lower-dimensional.

| Matrix type | Spectral profile | Deviation from random | Effective rank |
|---|---|---|---|
| Query (W_Q) | Heavy-tailed, strong outliers | Strongest | ~62% of d_head |
| Key (W_K) | Heavy-tailed | Strong | ~62% of d_head |
| Value (W_V) | Near full-rank | Moderate | ~90%+ of d_head |
| Output (W_O) | Near-Gaussian | Mildest | Variable |
| OV circuit | Exponential decay | Strong (positive eigenvalues) | Uses most of d_head |

---

## The phase transition proceeds through condensation then rank collapse

The induction head phase transition occurs at approximately **2.5–5 billion tokens** of training (roughly 1–2% of a typical run). It produces the only non-convex region of the training loss curve — a visible "bump" where loss drops faster than the surrounding trajectory. Before the transition, in-context learning score is below 0.15 nats; after, it jumps to **~0.4 nats** and remains constant across model sizes from 13M to 13B parameters.

Recent theoretical work by Musat et al. (2025) proved that induction head formation in a minimal two-layer transformer is constrained to a **19-dimensional subspace** of parameter space, with only **3 principal pseudo-parameters** driving the transition: α₃ (previous-token attending), β₂ (query-key matching), and γ₃ (label copying). These correspond precisely to the three subcircuits identified by Singh et al. (2024): each subcircuit learns smoothly, but their **combination** produces the sharp phase transition. The emergence time scales quadratically with context length: T_ICL = Θ(N²).

The weight geometry evolves in two distinct stages. During **Stage 1 (condensation)**, the outer parameters (W_V, W_O, FFN weights) develop block structure in their cosine similarity matrices while their effective rank steadily decreases. The attention parameters W_Q and W_K remain **largely static and unstructured**. During **Stage 2 (rank collapse)**, the outer parameters stabilize and attention parameters become the optimization focus. A **sharp rank collapse** occurs in the key-query matrices — structure suddenly appears, coinciding with the loss drop. This two-stage pattern holds on both synthetic data and WikiText.

Before the transition, attention patterns appear noisy and uninterpretable. A hidden "attention progress measure" gradually increases during the plateau, but the attention maps only become interpretable after the sudden drop. The model starts as essentially an n-gram model, then abruptly transitions to a qualitatively different regime with long-range context utilization. Interestingly, many heads that later become task-encoding "function vector" heads begin their life as induction heads, suggesting induction heads serve as **inductive scaffolding** for richer circuits.

---

## Mimetic initialization is the closest existing method to circuit-aware init

The most relevant prior work is **mimetic initialization** (Trockman & Kolter, ICML 2023), which examined pre-trained ViT weights and discovered that W_Q·W_K^T ≈ positive identity and W_V·W_O ≈ negative identity (sign flipped for language models). Their method constructs target matrices M_QK = α·Z + β·I (random noise + identity) and M_VO = α·Z − β·I, then factors each via truncated SVD:

```python
def mimetic_init_head(d_model, d_head, alpha_qk=0.7, beta_qk=0.7, alpha_vo=0.4, beta_vo=0.4):
    # QK circuit: target is alpha*noise + beta*I, truncated to rank d_head
    Z = torch.randn(d_model, d_model) / d_model**0.5
    M_qk = alpha_qk * Z + beta_qk * torch.eye(d_model)
    U, S, Vh = torch.linalg.svd(M_qk, full_matrices=False)
    sqrt_S = S[:d_head].sqrt()
    W_Q = U[:, :d_head] * sqrt_S          # (d_model, d_head)
    W_K = Vh[:d_head].T * sqrt_S           # (d_model, d_head)
    
    # OV circuit: target is alpha*noise - beta*I (vision) or +beta*I (language)
    Z2 = torch.randn(d_model, d_model) / d_model**0.5
    M_ov = alpha_vo * Z2 + beta_vo * torch.eye(d_model)  # +beta for language/copy
    U2, S2, Vh2 = torch.linalg.svd(M_ov, full_matrices=False)
    sqrt_S2 = S2[:d_head].sqrt()
    W_V = U2[:, :d_head] * sqrt_S2         # (d_model, d_head)
    W_O = (Vh2[:d_head].T * sqrt_S2).T     # (d_head, d_model)
    return W_Q, W_K, W_V, W_O
```

This yields +5% accuracy on CIFAR-10 and +4% on ImageNet for ViTs trained from scratch. The hyperparameters α=β=0.7 for QK and α=β=0.4 for VO were tuned via grid search. Sinusoidal positional embeddings are required for full benefit (they create structured local mixing via the P·P^T term in attention). The method performs slightly *better* than using actual pretrained weights as initialization.

Beyond mimetic init, several other approaches shape early attention dynamics. **ReZero** initializes a per-layer scalar to zero, making attention layers invisible at initialization (output ≈ identity) and converging **56% faster**. **T-Fixup** scales V and O projection weights by (9N)^{−1/4} and FFN weights by (12N)^{−1/4}, enabling 200-layer training without warmup or LayerNorm. **μP** (maximal update parameterization) changes attention logit scaling from 1/√d_head to **1/d_head**, accounting for Q-K correlation during training, and enables hyperparameter transfer across model widths. **σReparam** reparameterizes weight matrices by their spectral norm to prevent attention entropy collapse — the primary source of training instability. No existing work directly initializes heads to implement specific known circuits (previous-token, induction), making this a clear open opportunity.

---

## Weight sharing across layers reveals attention's inherent redundancy

Evidence from looped and weight-tied architectures strongly supports the feasibility of structured shared initialization. **ALBERT** shares all parameters across layers with minimal attention degradation (−0.7 points vs. unshared). The **Reuse Transformer** (Bhojanapalli et al., 2021) measured attention score similarity between layers using total variation distance and found adjacent layers show **0.8–0.9 similarity** for best-matched heads — a pattern that emerges from training, not architecture. **MASA** decomposes Q/K/V/O matrices across layers into shared dictionary atoms and achieves on-par performance with **66.7% fewer attention parameters**.

**Huginn** (Geiping et al., 2025) uses 4 weight-tied recurrent blocks cycled 16–128 times, relying solely on evolving hidden states (not iteration-specific encodings) for attention diversification. The **Relaxed Recursive Transformer** found that SVD-based initialization from averaged original weights achieves a **37% improvement** over random initialization, confirming that a single well-chosen set of attention weights provides a strong multi-layer starting point. The diversity needed for effective processing arises naturally from the residual stream's evolving state.

The practical implication is direct: a single structured initialization can serve all layers, with small per-layer perturbations (LoRA-scale) recovering nearly full expressivity if needed. The most important structural requirement is that shared weights implement a **contractive, convergent transformation** — one that refines representations toward a useful fixed point.

---

## A practical blueprint for circuit-aware attention initialization

Synthesizing all findings, the optimal initialization strategy combines five elements. First, **target QK identity structure**: set W_Q·W_K^T ≈ αZ + βI with positive β to bias attention toward self/similar tokens, factored via truncated SVD. For RoPE models, the identity structure naturally interacts with rotary embeddings to produce local-context attention. Second, **target OV copy structure**: set W_O·W_V ≈ scaled identity with exponential singular value decay (matching the empirical σ_i ≈ e^{−0.3i} profile), factored similarly. Third, **orthogonal subspace allocation**: assign each head a different d_head-dimensional subspace of the residual stream by drawing from a shared orthogonal basis, preventing inter-head interference. Fourth, **depth-aware scaling**: scale residual-path weights by 1/√N (GPT-2 style) or use ReZero-style learnable scalars initialized to small values. Fifth, **symmetry breaking**: add small Gaussian noise (σ ≈ 0.01) to the structured initialization, ensuring heads can specialize during training.

```python
def circuit_aware_init(d_model, n_heads, n_layers, alpha=0.5, beta=0.7, ov_decay=0.3):
    """Initialize all attention weights with circuit-aware structure."""
    d_head = d_model // n_heads
    basis = torch.linalg.qr(torch.randn(d_model, d_model))[0]  # shared orthogonal basis
    
    for layer in range(n_layers):
        for h in range(n_heads):
            # Orthogonal subspace for this head
            P = basis[:, h*d_head:(h+1)*d_head]  # (d_model, d_head)
            
            # QK: identity-like in subspace (attend to similar tokens)
            noise = torch.randn(d_head, d_head) / d_head**0.5
            M_qk = alpha * noise + beta * torch.eye(d_head)
            U, S, Vh = torch.linalg.svd(M_qk)
            W_Q = P @ (U * S.sqrt())                          # (d_model, d_head)
            W_K = P @ (Vh.T * S.sqrt())                       # (d_model, d_head)
            
            # OV: identity with exponential SV decay (copy circuit)
            sigmas = torch.exp(-ov_decay * torch.arange(d_head).float())
            sigmas *= (0.5 / n_layers**0.5)  # depth scaling
            W_V = P * sigmas.sqrt()                            # (d_model, d_head)
            W_O = (P * sigmas.sqrt()).T                        # (d_head, d_model)
            
            # Small noise for symmetry breaking
            W_Q += 0.01 * torch.randn_like(W_Q)
            W_K += 0.01 * torch.randn_like(W_K)
            W_V += 0.01 * torch.randn_like(W_V)
            W_O += 0.01 * torch.randn_like(W_O)
            # Assign to model parameters...
```

The key insight from phase transition research is that the induction head circuit requires only 3 effective parameters to form — but the standard training process takes billions of tokens to discover them because it must coordinate learning across two layers simultaneously. By initializing QK circuits near identity (enabling same-token matching) and OV circuits near identity (enabling copying), the gradient signal for the remaining coordination — the K-composition path — becomes immediately available rather than requiring a long condensation period.

---

## Conclusion

The research converges on a clear picture: trained attention weights occupy a small, structured region of parameter space characterized by identity-like QK circuits, copying OV circuits, exponential singular value decay, and orthogonal inter-head subspaces. The standard random initialization places weights maximally far from this target, requiring a costly phase transition at ~1–2% of training to discover structure that can be directly specified. **Mimetic initialization is the strongest validated method**, improving ViT training by 4–5% accuracy; extending it with exponential spectral profiles, orthogonal subspace allocation, and depth scaling represents the natural next step. The most promising unexplored direction is **differentiating heads by role at initialization** — assigning some heads shifted-diagonal QK circuits (for previous-token attention) and others identity QK circuits (for same-token matching), directly encoding the two-layer induction head circuit structure. Weight-sharing research confirms that this shared structural template generalizes across layers, with attention diversification arising naturally from evolving hidden states rather than requiring distinct per-layer initialization.