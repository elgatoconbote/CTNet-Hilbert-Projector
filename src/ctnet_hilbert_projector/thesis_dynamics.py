#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet quantum thesis dynamics.

This module implements the paper thesis as executable mechanics:

1. H_t deforms the persistent state Xi_t before projection.
2. Branch mass is generated from coherence, residue, relation and memory.
3. Phase is updated from reversible branch action plus Hamiltonian energy.
4. Non-separability is represented by an explicit relational cocycle chi.
5. A complete step returns Xi_{t+1} and A_{t+1}(sigma).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ctnet_omega_core import FoldedCTNetOmegaCubo26, FoldedOmegaCuboState
from .hilbert_projector import HilbertProjection, UPBranch, branch_to_tensor, enumerate_up_branches


@dataclass(frozen=True)
class ThesisDynamicsConfig:
    n_qubits: int
    beta_coherence: float = 1.0
    gamma_residue: float = 1.0
    lambda_relation: float = 0.5
    eta_memory: float = 0.5
    cocycle_strength: float = 0.25
    hamiltonian_state_strength: float = 0.05
    hamiltonian_phase_strength: float = 1.0
    coherence_clamp: float = 8.0
    eps: float = 1e-9


@dataclass
class ThesisProjection:
    branches: List[UPBranch]
    amplitudes: torch.Tensor
    probabilities: torch.Tensor
    masses: torch.Tensor
    phases: torch.Tensor
    coherence: torch.Tensor
    residue: torch.Tensor
    relation: torch.Tensor
    memory: torch.Tensor
    cocycle: torch.Tensor
    normalization_error: torch.Tensor

    def as_hilbert_projection(self) -> HilbertProjection:
        return HilbertProjection(
            branches=self.branches,
            amplitudes=self.amplitudes,
            probabilities=self.probabilities,
            masses=self.masses,
            phases=self.phases,
            normalization_error=self.normalization_error,
        )


def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] < dim:
        return F.pad(x, (0, dim - x.shape[-1]))
    return x[..., :dim]


def branch_matrix(branches: Sequence[UPBranch], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([branch_to_tensor(branch, device=device, dtype=dtype) for branch in branches], dim=0)


def hamiltonian_branch_descriptors(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return diagonal energy, coupling strength and signed phase drive per branch."""
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    diag = H.diagonal().real
    off = H - torch.diag_embed(H.diagonal())
    coupling = off.abs().sum(dim=-1).real
    phase_drive = diag + off.real.sum(dim=-1)
    return diag, coupling, phase_drive


def condition_state_with_hamiltonian(
    state: FoldedOmegaCuboState,
    H: torch.Tensor,
    branches: Sequence[UPBranch],
    *,
    strength: float = 0.05,
) -> FoldedOmegaCuboState:
    """Inject H into Z, M, R and C6 as a structural deformation of Xi.

    This is the concrete implementation of H_t -> \tilde Xi_t = T_rho(Xi_t,H_t)
    for the current finite technical chart. It does not append memory and does not
    materialize a growing relation table; it folds Hamiltonian energy, coupling
    and phase pressure back into the fixed state support.
    """
    device, dtype = state.z.device, state.z.dtype
    diag, coupling, phase_drive = hamiltonian_branch_descriptors(H.to(device=device))
    diag = diag.to(dtype=dtype)
    coupling = coupling.to(dtype=dtype)
    phase_drive = phase_drive.to(dtype=dtype)
    bmat = branch_matrix(branches, device=device, dtype=dtype)

    diag_n = (diag - diag.mean()) / diag.std().clamp_min(1e-6)
    coup_n = (coupling - coupling.mean()) / coupling.std().clamp_min(1e-6)
    phase_n = (phase_drive - phase_drive.mean()) / phase_drive.std().clamp_min(1e-6)

    modal_drive = torch.einsum("s,sn->n", diag_n + 0.5 * coup_n, bmat) / float(len(branches))
    rel_drive = torch.einsum("s,sn->n", phase_n, bmat) / float(len(branches))

    z_delta = torch.zeros_like(state.z)
    m_delta = torch.zeros_like(state.memory)
    r_delta = torch.zeros_like(state.relations)
    c_delta = torch.zeros_like(state.cubo)

    z_pattern = _match_dim(modal_drive, state.z.shape[-1]).view(1, 1, -1)
    r_pattern = _match_dim(rel_drive, state.relations.shape[-1]).view(1, 1, -1)
    z_delta = z_delta + z_pattern
    r_delta = r_delta + r_pattern
    m_delta = m_delta + 0.5 * _match_dim(modal_drive, state.memory.shape[-1]).view(1, 1, -1)

    c_stats = torch.stack([diag_n.mean(), diag_n.std(), coup_n.mean(), coup_n.std(), phase_n.mean(), phase_n.std()])
    c_delta[:, : min(c_delta.shape[-1], c_stats.numel())] = c_stats[: c_delta.shape[-1]]

    alpha = float(strength)
    return FoldedOmegaCuboState(
        z=state.z + alpha * z_delta,
        memory=state.memory + alpha * m_delta,
        relations=state.relations + alpha * r_delta,
        cubo=state.cubo + alpha * c_delta,
        pad=state.pad,
    )
