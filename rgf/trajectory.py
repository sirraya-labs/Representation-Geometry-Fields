"""
Section 7: Layer trajectories.

gamma = (h^(0)_i, h^(1)_i, ..., h^(L)_i), velocities v_l = h^(l+1) - h^(l),
metric-normalized speed ||v_l||_{G_l} = sqrt(v_l^T G_l(h^(l)) v_l), and the
empirical heuristic s(l) = || v~_{l+1} - v~_l ||_2 for unit-normalized
velocities v~_l = v_l / ||v_l||_{G_l}.
"""
from __future__ import annotations

import torch

from .metric import compute_rgf


def layer_trajectory(hidden_states: tuple[torch.Tensor, ...], position_idx: int) -> list[torch.Tensor]:
    """Extract h^(l)_i for l = 0..L from the hidden_states tuple returned by RGFModel.hidden_states."""
    return [hs[0, position_idx, :].detach().clone() for hs in hidden_states]


def metric_normalized_velocities(model, hidden_states, position_idx: int):
    """
    Returns:
      velocities: list of v_l = h^{(l+1)} - h^{(l)}, l = 0..L-1
      speeds:     list of ||v_l||_{G_l} = sqrt(v_l^T G_l(h^{(l)}) v_l), l = 0..L-1
      unit_vels:  list of v_l / ||v_l||_{G_l} (None where speed ~ 0)
    """
    gamma = layer_trajectory(hidden_states, position_idx)
    L = model.L
    velocities, speeds, unit_vels = [], [], []
    for l in range(L):
        h_l = gamma[l]
        v_l = gamma[l + 1] - gamma[l]
        f_l = model.downstream_map(hidden_states[l], layer_idx=l, position_idx=position_idx)
        out = compute_rgf(f_l, h_l)
        speed_sq = (v_l @ out["G"] @ v_l).item()
        speed = speed_sq ** 0.5 if speed_sq > 0 else 0.0
        velocities.append(v_l)
        speeds.append(speed)
        unit_vels.append(v_l / speed if speed > 1e-8 else None)
    return velocities, speeds, unit_vels


def normalized_velocity_change(unit_vels: list) -> list:
    """s(l) = || v~_{l+1} - v~_l ||_2 for consecutive layers with defined unit velocity."""
    s = []
    for l in range(len(unit_vels) - 1):
        a, b = unit_vels[l], unit_vels[l + 1]
        if a is None or b is None:
            s.append(None)
        else:
            s.append((b - a).norm().item())
    return s
