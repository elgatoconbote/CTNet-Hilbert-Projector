#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact small-Hilbert baselines on the CTNet u/p basis.

Dense exact routines for audit at small n. The CTNet claim remains projective;
this file only provides reference dynamics when the full 2^n table is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .hilbert_projector import UPBranch, enumerate_up_branches


@dataclass(frozen=True)
class IsingConfig:
    """Transverse-field Ising parameters in the u/p chart.

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i.
    """

    n_qubits: int
    J: float = 1.0
    h: float = 0.5
    periodic: bool = False


def branch_z_value(symbol: str) -> float:
    if symbol == "u":
        return 1.0
    if symbol == "p":
        return -1.0
    raise ValueError(f"invalid u/p symbol: {symbol!r}")


def flip_branch(branch: Sequence[str], site: int) -> UPBranch:
    out = list(branch)
    out[site] = "p" if out[site] == "u" else "u"
    return tuple(out)


def branch_index(branches: Sequence[UPBranch]) -> Dict[UPBranch, int]:
    return {tuple(branch): i for i, branch in enumerate(branches)}


def transverse_field_ising_matrix(
    config: IsingConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, List[UPBranch]]:
    """Build the dense exact transverse-field Ising Hamiltonian."""
    if config.n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    device = device or torch.device("cpu")
    branches = enumerate_up_branches(config.n_qubits)
    idx = branch_index(branches)
    dim = len(branches)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    H = torch.zeros(dim, dim, device=device, dtype=real_dtype)
    n = config.n_qubits
    edge_count = n if config.periodic and n > 1 else max(0, n - 1)

    for col, branch in enumerate(branches):
        diag = 0.0
        for i in range(edge_count):
            j = (i + 1) % n
            diag += -config.J * branch_z_value(branch[i]) * branch_z_value(branch[j])
        H[col, col] += diag
        for site in range(n):
            row = idx[flip_branch(branch, site)]
            H[row, col] += -config.h
    return H.to(dtype=dtype), branches


def normalize_state(psi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if psi.ndim == 1:
        return psi / psi.norm().clamp_min(eps)
    if psi.ndim == 2:
        return psi / psi.norm(dim=-1, keepdim=True).clamp_min(eps)
    raise ValueError("psi must have shape [S] or [B,S]")


def basis_state(
    branches: Sequence[UPBranch],
    branch: Sequence[str],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    idx = branch_index(branches)
    key = tuple(branch)
    if key not in idx:
        raise ValueError(f"branch {key!r} is not present in basis")
    psi = torch.zeros(len(branches), device=device, dtype=dtype)
    psi[idx[key]] = 1.0 + 0.0j
    return psi


def evolve_exact(psi: torch.Tensor, H: torch.Tensor, *, dt: float = 0.1, steps: int = 1) -> torch.Tensor:
    """Evolve psi under exp(-i H dt steps)."""
    total_dt = float(dt) * int(steps)
    U = torch.matrix_exp((-1j * total_dt) * H)
    if psi.ndim == 1:
        return U @ psi
    if psi.ndim == 2:
        return torch.einsum("ij,bj->bi", U, psi)
    raise ValueError("psi must have shape [S] or [B,S]")


def global_phase_align(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Align candidate to reference by best global phase."""
    if reference.ndim == 1:
        overlap = torch.vdot(candidate, reference)
        return candidate * (overlap / overlap.abs().clamp_min(eps))
    if reference.ndim == 2:
        overlap = (candidate.conj() * reference).sum(dim=-1, keepdim=True)
        return candidate * (overlap / overlap.abs().clamp_min(eps))
    raise ValueError("states must have shape [S] or [B,S]")


def amplitude_l2_error(reference: torch.Tensor, candidate: torch.Tensor, *, phase_invariant: bool = True) -> torch.Tensor:
    if phase_invariant:
        candidate = global_phase_align(reference, candidate)
    diff = reference - candidate
    return diff.norm() if diff.ndim == 1 else diff.norm(dim=-1)


def probability_l1_error(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    diff = (reference.abs().pow(2) - candidate.abs().pow(2)).abs()
    return diff.sum() if diff.ndim == 1 else diff.sum(dim=-1)


def expectation(psi: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    if psi.ndim == 1:
        return torch.vdot(psi, operator @ psi)
    op_psi = torch.einsum("ij,bj->bi", operator, psi)
    return (psi.conj() * op_psi).sum(dim=-1)
