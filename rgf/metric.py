"""
Core Representation Geometry Field (RGF) computation, and numerical checks
of every theorem in the specification:

  Thm 3.1  Pullback identity          -> pullback_check
  Thm 4.1  Positive semidefiniteness  -> psd_check
  Thm 4.2  Rank / kernel structure    -> rank_check
  Thm 4.4  Total sensitivity (trace)  -> trace_check
  Thm 4.5  Spectral interpretation    -> spectral_check
  Thm 4.7  Tensor transformation law  -> tensor_transform_check
  Thm 6.1/6.2 Final-layer flatness    -> final_layer_flatness_check
"""
from __future__ import annotations

import torch
from torch.func import jacrev


def project_pi(J: torch.Tensor) -> torch.Tensor:
    """
    Pi J, where Pi = I_{|V|} - (1/|V|) 1 1^T projects onto the zero-mean
    subspace of R^{|V|}. J has shape (|V|, d); Pi acts on the |V| (row) axis.
    """
    return J - J.mean(dim=0, keepdim=True)


def jacobian(f, h: torch.Tensor) -> torch.Tensor:
    """
    Jacobian of f: R^d -> R^{|V|} at h, shape (|V|, d).

    Uses reverse-mode AD (jacrev). In principle forward-mode (jacfwd) is the
    better asymptotic choice here since d << |V| for transformers, but as of
    this writing HuggingFace's attention modules don't reliably support
    forward-mode AD under vmap. jacrev costs O(|V|) vectorized backward
    passes; for very large vocabularies, use `jacobian_looped` with a
    restricted set of output coordinates instead.
    """
    return jacrev(f)(h)


def jacobian_looped(f, h: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """
    Fallback Jacobian computation via chunked reverse-mode passes, useful
    if vmap-based jacrev runs out of memory for very large vocabularies.
    Mathematically identical to `jacobian`, just more memory-frugal.
    """
    y = f(h)
    V = y.shape[0]
    rows = []
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        basis = torch.eye(end - start, dtype=h.dtype)

        def f_chunk(hh):
            return f(hh)[start:end]

        rows.append(jacrev(f_chunk)(h))
    return torch.cat(rows, dim=0)


def compute_rgf(f, h: torch.Tensor) -> dict:
    """
    Computes G_l(h) = J^T Pi J together with the intermediate quantities
    needed by the various theorem checks, so callers don't recompute the
    Jacobian repeatedly.

    Returns a dict with keys: J, PiJ, G, eigvals, eigvecs (eigvals ascending
    is NOT assumed; we sort descending to match the paper's lambda_1 >= ...).
    """
    J = jacobian(f, h)                 # (|V|, d)
    PiJ = project_pi(J)                # (|V|, d)
    G = PiJ.T @ PiJ                    # (d, d)
    G = 0.5 * (G + G.T)                # symmetrize away floating point noise
    eigvals, eigvecs = torch.linalg.eigh(G)
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return dict(J=J, PiJ=PiJ, G=G, eigvals=eigvals, eigvecs=eigvecs)


def effective_rank(eigvals: torch.Tensor) -> float:
    ev = eigvals.clamp(min=0)
    num = ev.sum() ** 2
    den = (ev ** 2).sum()
    if den <= 0:
        return 0.0
    return (num / den).item()


# ----------------------------------------------------------------------
# Theorem 3.1: Pullback identity  v^T G v = || Pi (J v) ||^2
# ----------------------------------------------------------------------
def pullback_check(f, h: torch.Tensor, n_trials: int = 8, atol: float = 1e-4) -> dict:
    out = compute_rgf(f, h)
    G, J = out["G"], out["J"]
    d = h.shape[0]
    errs = []
    for _ in range(n_trials):
        v = torch.randn(d, dtype=h.dtype)
        lhs = (v @ G @ v).item()
        Jv = J @ v
        rhs = (Jv - Jv.mean()).pow(2).sum().item()
        errs.append(abs(lhs - rhs))
    max_err = max(errs)
    return dict(max_abs_error=max_err, passed=max_err < atol, details=out)


# ----------------------------------------------------------------------
# Theorem 4.1: G(h) is PSD
# ----------------------------------------------------------------------
def psd_check(out: dict, tol: float = -1e-6) -> dict:
    min_eig = out["eigvals"].min().item()
    return dict(min_eigenvalue=min_eig, passed=min_eig >= tol)


# ----------------------------------------------------------------------
# Theorem 4.2: rank(G) = rank(Pi J) = rank(J) - dim(im(J) cap R*1)
# ----------------------------------------------------------------------
def rank_check(out: dict, tol: float = 1e-4) -> dict:
    J, PiJ, G = out["J"], out["PiJ"], out["G"]
    rank_J = torch.linalg.matrix_rank(J, tol=tol).item()
    rank_PiJ = torch.linalg.matrix_rank(PiJ, tol=tol).item()
    rank_G = torch.linalg.matrix_rank(G, tol=tol).item()

    # Does the all-ones vector lie (numerically) in the column space of J?
    Vsize = J.shape[0]
    ones = torch.ones(Vsize, dtype=J.dtype) / (Vsize ** 0.5)
    Q, _ = torch.linalg.qr(J)               # orthonormal basis for im(J)
    proj = Q @ (Q.T @ ones)
    residual = (ones - proj).norm().item()
    ones_in_image = residual < 1e-3
    correction_predicted = 1 if ones_in_image else 0

    return dict(
        rank_J=rank_J,
        rank_PiJ=rank_PiJ,
        rank_G=rank_G,
        consistent_G_eq_PiJ=(rank_G == rank_PiJ),
        ones_residual_norm=residual,
        ones_in_image_of_J=ones_in_image,
        rank_drop_observed=rank_J - rank_PiJ,
        rank_drop_predicted=correction_predicted,
        passed=(rank_G == rank_PiJ) and (rank_J - rank_PiJ == correction_predicted),
    )


# ----------------------------------------------------------------------
# Theorem 4.4: tr(G) = || Pi J ||_F^2
# ----------------------------------------------------------------------
def trace_check(out: dict, atol: float = 1e-3) -> dict:
    tr_G = torch.trace(out["G"]).item()
    frob_PiJ_sq = (out["PiJ"] ** 2).sum().item()
    err = abs(tr_G - frob_PiJ_sq)
    return dict(trace_G=tr_G, frob_PiJ_sq=frob_PiJ_sq, abs_error=err, passed=err < atol)


# ----------------------------------------------------------------------
# Theorem 4.5: lambda_i = || Pi J u_i ||^2
# ----------------------------------------------------------------------
def spectral_check(out: dict, atol: float = 1e-3) -> dict:
    PiJ, eigvals, eigvecs = out["PiJ"], out["eigvals"], out["eigvecs"]
    d = eigvecs.shape[1]
    errs = []
    for i in range(d):
        u_i = eigvecs[:, i]
        predicted = (PiJ @ u_i).pow(2).sum().item()
        errs.append(abs(predicted - eigvals[i].item()))
    max_err = max(errs)
    return dict(max_abs_error=max_err, passed=max_err < atol)


# ----------------------------------------------------------------------
# Theorem 4.7: G'(h') = A^{-T} G(h) A^{-1} under h' = A h
# ----------------------------------------------------------------------
def tensor_transform_check(f, h: torch.Tensor, atol: float = 1e-3) -> dict:
    d = h.shape[0]
    A = torch.eye(d, dtype=h.dtype) + 0.1 * torch.randn(d, d, dtype=h.dtype)
    A_inv = torch.linalg.inv(A)

    out_h = compute_rgf(f, h)
    G_h = out_h["G"]

    h_prime = A @ h

    def f_prime(hp):
        return f(A_inv @ hp)

    out_hp = compute_rgf(f_prime, h_prime)
    G_hp_actual = out_hp["G"]

    G_hp_predicted = A_inv.T @ G_h @ A_inv

    err = (G_hp_actual - G_hp_predicted).abs().max().item()

    # quadratic-form invariance check
    v = torch.randn(d, dtype=h.dtype)
    v_prime = A @ v
    qf_h = (v @ G_h @ v).item()
    qf_hp = (v_prime @ G_hp_actual @ v_prime).item()
    qf_err = abs(qf_h - qf_hp)

    rank_h = torch.linalg.matrix_rank(G_h).item()
    rank_hp = torch.linalg.matrix_rank(G_hp_actual).item()

    return dict(
        max_abs_error_matrix=err,
        quadratic_form_error=qf_err,
        rank_h=rank_h,
        rank_h_prime=rank_hp,
        rank_preserved=(rank_h == rank_hp),
        passed=(err < atol) and (qf_err < atol) and (rank_h == rank_hp),
    )


# ----------------------------------------------------------------------
# Theorems 6.1/6.2: final-layer RGF is constant (isometric to Euclidean)
# ----------------------------------------------------------------------
def final_layer_flatness_check(f_L, d: int, n_trials: int = 5, atol: float = 1e-3, dtype=torch.float32) -> dict:
    """
    f_L should be the layer-L downstream map (i.e. built with layer_idx == L,
    so no transformer blocks remain -- only the final projection to logits).
    Checks that G_L(h) is the same matrix regardless of h, as Theorem 6.1
    predicts for a strictly affine unembedding U(h) = W_U h + b_U.

    NOTE: real GPT-2 inserts a final LayerNorm between the last hidden state
    and the unembedding. LayerNorm is NOT affine (it renormalizes by ||h||),
    so if f_L includes ln_f, this check is expected to reveal small-to-large
    non-constancy -- an interesting empirical departure from the idealized
    paper setting, not a bug.
    """
    Gs = []
    for _ in range(n_trials):
        h = torch.randn(d, dtype=dtype)
        out = compute_rgf(f_L, h)
        Gs.append(out["G"])
    ref = Gs[0]
    max_devs = [(G - ref).abs().max().item() for G in Gs[1:]]
    max_dev = max(max_devs) if max_devs else 0.0
    return dict(max_deviation_across_h=max_dev, passed=max_dev < atol, G_sample=ref)
