"""
Empirical demo: compute the RGF at every layer for a chosen token position in
a real sentence, and visualize how its spectrum / rank / trace / trajectory
speed evolve across the network.

By default this uses a randomly-initialized GPT-2-architecture model (no
internet access required in this sandbox). Pass --pretrained to use real
GPT-2 weights (downloads from huggingface.co -- run this on a machine with
internet access, not inside a network-restricted sandbox).
"""
import argparse
import os

import torch

from rgf import RGFModel
from rgf.metric import compute_rgf, effective_rank
from rgf.trajectory import layer_trajectory, metric_normalized_velocities, normalized_velocity_change
from rgf.visualize import (
    plot_eigenvalue_spectra, plot_scalar_across_layers,
    plot_metric_heatmap, plot_trajectory_speeds,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", help="Use real GPT-2 weights (requires internet access to huggingface.co)")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--position", type=int, default=-1, help="Token position to analyze (-1 = last token)")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.set_default_dtype(torch.float32)

    model = RGFModel(pretrained=args.pretrained, model_name="gpt2")
    print(f"Model: L={model.L} layers, d={model.d} hidden dim, |V|={model.V} vocab size")

    if args.pretrained:
        input_ids = model.tokenizer(args.text, return_tensors="pt").input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])
    else:
        torch.manual_seed(0)
        n = 8
        input_ids = torch.randint(0, model.V, (1, n))
        tokens = [str(t.item()) for t in input_ids[0]]

    n_tokens = input_ids.shape[1]
    position_idx = args.position if args.position >= 0 else n_tokens - 1
    print(f"Sequence ({n_tokens} tokens): {tokens}")
    print(f"Analyzing token position {position_idx} ('{tokens[position_idx]}') across all {model.L + 1} representation layers\n")

    hidden_states = model.hidden_states(input_ids)

    # --- Compute RGF at every layer 0..L for this (fixed context, position) ---
    layer_eigvals = {}
    layer_trace = {}
    layer_rank = {}
    layer_effrank = {}
    G_by_layer = {}

    for l in range(model.L + 1):
        h = hidden_states[l][0, position_idx, :].detach().clone()
        f = model.downstream_map(hidden_states[l], layer_idx=l, position_idx=position_idx)
        out = compute_rgf(f, h)
        layer_eigvals[l] = out["eigvals"]
        layer_trace[l] = torch.trace(out["G"]).item()
        layer_rank[l] = torch.linalg.matrix_rank(out["G"]).item()
        layer_effrank[l] = effective_rank(out["eigvals"])
        G_by_layer[l] = out["G"]
        print(f"  layer {l:2d}: trace={layer_trace[l]:.4g}  rank={layer_rank[l]}/{model.d}  eff_rank={layer_effrank[l]:.2f}")

    # --- Plots ---
    p1 = plot_eigenvalue_spectra(layer_eigvals, os.path.join(args.outdir, "eigenvalue_spectra.png"))
    p2 = plot_scalar_across_layers(layer_trace, os.path.join(args.outdir, "trace_across_layers.png"),
                                    ylabel="tr(G_l) = total local sensitivity", title="Total output sensitivity per layer")
    p3 = plot_scalar_across_layers(layer_effrank, os.path.join(args.outdir, "effective_rank_across_layers.png"),
                                    ylabel="effective rank", title="Effective rank of RGF per layer")
    mid = model.L // 2
    p4 = plot_metric_heatmap(G_by_layer[mid], os.path.join(args.outdir, f"metric_heatmap_layer{mid}.png"),
                              title=f"RGF G_{mid}(h)")

    # --- Layer trajectory / metric-normalized velocity ---
    velocities, speeds, unit_vels = metric_normalized_velocities(model, hidden_states, position_idx)
    s = normalized_velocity_change(unit_vels)
    p5 = plot_trajectory_speeds(speeds, os.path.join(args.outdir, "trajectory_speeds.png"))
    print("\nMetric-normalized velocities ||v_l||_G per layer transition:")
    for l, sp in enumerate(speeds):
        print(f"  layer {l} -> {l+1}: {sp:.4g}")
    print("\nNormalized velocity change s(l) (empirical heuristic, Section 7):")
    for l, val in enumerate(s):
        print(f"  s({l+1}) = {val if val is None else f'{val:.4g}'}")

    print("\nSaved plots:")
    for p in [p1, p2, p3, p4, p5]:
        print(" ", p)


if __name__ == "__main__":
    main()
