"""
Plotting helpers for empirically exploring RGF structure across layers.
All functions save a PNG and return the file path.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_eigenvalue_spectra(layer_eigvals: dict[int, "torch.Tensor"], out_path: str, title="RGF eigenvalue spectrum per layer", log=True):
    """layer_eigvals: {layer_idx: eigenvalues tensor (descending)}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for l, ev in sorted(layer_eigvals.items()):
        ev = ev.detach().cpu().numpy()
        ev = np.clip(ev, 1e-12, None)
        ax.plot(np.arange(1, len(ev) + 1), ev, marker="o", markersize=3, label=f"layer {l}")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (log scale)" if log else "eigenvalue")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_scalar_across_layers(layer_to_value: dict[int, float], out_path: str, ylabel: str, title: str):
    layers = sorted(layer_to_value.keys())
    vals = [layer_to_value[l] for l in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, vals, marker="o")
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_heatmap(G, out_path: str, title="RGF G_l(h)", max_dim=100):
    """Heatmap of the (possibly truncated) d x d metric matrix."""
    G = G.detach().cpu().numpy()
    d = G.shape[0]
    if d > max_dim:
        G = G[:max_dim, :max_dim]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max())
    ax.set_title(title + (f" (top-left {G.shape[0]}x{G.shape[0]})" if d > max_dim else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_trajectory_speeds(speeds: list[float], out_path: str, title="Metric-normalized velocity ||v_l||_G per layer"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(speeds)), speeds, marker="o", color="darkorange")
    ax.set_xlabel("layer l -> l+1")
    ax.set_ylabel("||v_l||_{G_l}")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
