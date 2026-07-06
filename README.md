# Representation Geometry Fields (RGF) — Python Implementation

An implementation of the *Representation Geometry Field* framework: the pullback
metric induced on transformer hidden-state space by the model's own downstream
computation, `G_l(h) = J_{T̃}(h)ᵀ J_{T̃}(h)`, where `T̃ = Π ∘ T_{l:L}` projects
onto the zero-mean logit space `R^|V| / R·1`.

Built on GPT-2 via HuggingFace `transformers`. Every theorem in the paper is
implemented as a numerical check, plus tools to visualize how the RGF's
spectrum, rank, and total sensitivity evolve across layers on real text.

## ⚠️ Network note for this sandbox

This code was developed and tested inside a sandbox that cannot reach
`huggingface.co`, so it defaults to a **randomly-initialized model with GPT-2's
exact architecture** (`RGFModel(pretrained=False)`). All math has been
verified against this model to machine precision (see "Verification results"
below). To run against **real GPT-2 weights**, use `RGFModel(pretrained=True)`
or `--pretrained` on a machine with normal internet access — the code path is
identical, only the weights differ.

## Install

```bash
pip install torch transformers matplotlib numpy
```

## Structure

```
rgf/
  model_wrapper.py   RGFModel: loads GPT-2, extracts hidden states, and builds
                      the "downstream map" T_{l:L}^{(x,i)}: R^d -> R^|V|
                      (Definition 2.5) as a differentiable function of a
                      single hidden state, holding the rest of the context fixed.
  metric.py           compute_rgf() and one function per theorem:
                      pullback_check, psd_check, rank_check, trace_check,
                      spectral_check, tensor_transform_check,
                      final_layer_flatness_check.
  trajectory.py       Section 7: layer trajectories, metric-normalized
                      velocity ||v_l||_G, and the s(l) heuristic.
  visualize.py        Plotting helpers (eigenvalue spectra, effective rank
                      per layer, metric heatmaps, trajectory speed).

examples/
  verify_theorems.py  Numerically verifies all theorems at a chosen layer.
  demo_gpt2.py         Full empirical walkthrough on a real sentence:
                       RGF spectrum/rank/trace across all layers + plots.
```

## Usage

Verify every theorem (defaults to the tiny random model, no internet needed):

```bash
python examples/verify_theorems.py --layer 3
```

Run the empirical demo, producing plots in `outputs/`:

```bash
python examples/demo_gpt2.py --outdir outputs
```

With real GPT-2 and your own sentence:

```bash
python examples/verify_theorems.py --pretrained
python examples/demo_gpt2.py --pretrained --text "The theorem states that" --position -1
```

## Design notes

**Downstream map.** `RGFModel.downstream_map(hidden_states_l, layer_idx, position_idx)`
returns a pure function `f: R^d -> R^|V|` built with functional (out-of-place)
tensor operations, so it plugs directly into `torch.func.jacrev`.

**Jacobian.** The paper notes `d << |V|` for transformers, which makes
forward-mode AD (`jacfwd`, cost `O(d)`) the asymptotically better choice over
reverse-mode (`jacrev`, cost `O(|V|)`). In practice, HuggingFace's attention
implementations don't reliably support forward-mode AD under `vmap` as of
this writing, so `metric.jacobian()` uses `jacrev`. For very large
vocabularies where a full `O(|V|)`-pass Jacobian is too memory-hungry, use
`metric.jacobian_looped()`, which computes it in chunks.

**The `ln_f` caveat.** Real GPT-2 inserts a final LayerNorm between the last
transformer block's output and the unembedding, i.e. the actual final map is
`U ∘ LN`, not the strictly affine `U` the paper's Theorem 6.1 assumes.
LayerNorm renormalizes by `‖h‖`, so it is *not* affine, and the final-layer
RGF is therefore **not exactly constant** on real GPT-2 — an interesting,
correct empirical departure from the idealized setting, not a bug. The
verification script demonstrates this directly: it checks flatness against
GPT-2's real final map (fails, ~1e-2 deviation) and against the raw affine
`U(h) = W_U h + b_U` via `RGFModel.unembed()` (passes to machine precision),
isolating `ln_f` as the cause.

## Verification results (tiny random model, float64)

All seven theorems were checked at every layer (0, 1, 2, and final layer 3)
of a 3-layer random GPT-2-architecture model:

| Theorem | Result |
|---|---|
| 3.1 Pullback identity | PASS (error ~1e-13) |
| 4.1 Positive semidefiniteness | PASS (min eigenvalue ~1e-15, i.e. zero up to float noise) |
| 4.2 Rank / kernel structure | PASS (rank(G) = rank(ΠJ) exactly; rank-drop matches whether **1** ∈ im(J)) |
| 4.4 Total sensitivity (trace) | PASS (error ~1e-13) |
| 4.5 Spectral interpretation | PASS (error ~1e-13) |
| 4.7 Tensor transformation law | PASS (matrix error ~1e-14, rank and quadratic form exactly preserved) |
| 6.1/6.2 Final-layer flatness | PASS for the strictly affine `U(h)` (error = 0.0); **fails** for GPT-2's real `U∘ln_f` pipeline, as expected |

## Extending to empirical curvature analysis

The paper explicitly scopes out full curvature computation (Section 6,
Proposition 6.1) as future work — nonconstant `G_l` is necessary but not
sufficient for curvature. `metric.py` and `trajectory.py` give you `G_l(h)`
at arbitrary points, so computing Christoffel symbols / Riemann curvature via
finite differences of `G_l` on the visible subbundle is a natural next step
if you want to go beyond what's implemented here.
