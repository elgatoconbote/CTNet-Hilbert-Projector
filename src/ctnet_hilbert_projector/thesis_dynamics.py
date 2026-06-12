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


@dataclass
class ThesisStepResult:
    preconditioned_state: FoldedOmegaCuboState
    next_state: FoldedOmegaCuboState
    projection: ThesisProjection


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
    """Inject H into Z, M, R and C6 as a structural deformation of Xi."""
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

    z_delta = _match_dim(modal_drive, state.z.shape[-1]).view(1, 1, -1).expand_as(state.z)
    r_delta = _match_dim(rel_drive, state.relations.shape[-1]).view(1, 1, -1).expand_as(state.relations)
    m_delta = 0.5 * _match_dim(modal_drive, state.memory.shape[-1]).view(1, 1, -1).expand_as(state.memory)
    c_delta = torch.zeros_like(state.cubo)
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


def _branch_features(state: FoldedOmegaCuboState, branches: Sequence[UPBranch]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = state.z.device, state.z.dtype
    bmat = branch_matrix(branches, device=device, dtype=dtype)
    z_mean = state.z.mean(dim=1)
    m_mean = _match_dim(state.memory.mean(dim=1), z_mean.shape[-1])
    r_mean = _match_dim(state.relations.mean(dim=1), z_mean.shape[-1])
    z_proj = torch.einsum("bd,sd->bs", z_mean, _match_dim(bmat, z_mean.shape[-1]))
    m_proj = torch.einsum("bd,sd->bs", m_mean, _match_dim(bmat, m_mean.shape[-1]))
    r_proj = torch.einsum("bd,sd->bs", r_mean, _match_dim(bmat, r_mean.shape[-1]))
    return z_proj, m_proj, r_proj


def relational_cocycle(state: FoldedOmegaCuboState, branches: Sequence[UPBranch], strength: float) -> torch.Tensor:
    device, dtype = state.z.device, state.z.dtype
    bmat = branch_matrix(branches, device=device, dtype=dtype)
    rel = state.relations.mean(dim=1)
    rel_vec = _match_dim(rel, bmat.shape[-1])
    gram = torch.einsum("bi,bj->bij", rel_vec, rel_vec)
    raw = torch.einsum("si,bij,sj->bs", bmat, gram, bmat)
    local = raw.mean(dim=-1, keepdim=True)
    return float(strength) * torch.tanh(raw - local)


def thesis_project(
    omega_core: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    H: torch.Tensor | None,
    config: ThesisDynamicsConfig,
    *,
    dt: float = 1.0,
) -> ThesisProjection:
    branches = enumerate_up_branches(config.n_qubits)
    z_proj, m_proj, r_proj = _branch_features(state, branches)
    obs = omega_core.cubo_observation(state)
    omega_global = obs["omega"].unsqueeze(-1).to(dtype=state.z.dtype)
    closure = obs["closure_score"].unsqueeze(-1).to(dtype=state.z.dtype)

    coherence = torch.tanh(z_proj + 0.5 * m_proj + 0.5 * r_proj + closure)
    memory = torch.sigmoid(m_proj)
    relation = torch.sigmoid(r_proj)
    residue = torch.relu(omega_global + 0.1 * torch.tanh(-z_proj))
    cocycle = relational_cocycle(state, branches, config.cocycle_strength)

    if H is not None:
        diag, coupling, phase_drive = hamiltonian_branch_descriptors(H.to(device=state.z.device))
        diag = diag.to(dtype=state.z.dtype).unsqueeze(0)
        coupling = coupling.to(dtype=state.z.dtype).unsqueeze(0)
        phase_drive = phase_drive.to(dtype=state.z.dtype).unsqueeze(0)
        coherence = coherence + 0.1 * torch.tanh(coupling)
    else:
        phase_drive = torch.zeros_like(coherence)

    mass_log = (
        config.beta_coherence * coherence.clamp(-config.coherence_clamp, config.coherence_clamp)
        - config.gamma_residue * residue
        + config.lambda_relation * relation
        + config.eta_memory * memory
        + cocycle
    )
    masses = torch.exp(mass_log.clamp(-config.coherence_clamp, config.coherence_clamp)) + config.eps
    probabilities = masses / masses.sum(dim=-1, keepdim=True).clamp_min(config.eps)
    phases = torch.tanh(z_proj + r_proj + cocycle) - float(dt) * config.hamiltonian_phase_strength * phase_drive
    amplitudes = torch.sqrt(probabilities).to(torch.complex64) * torch.exp(1j * phases.to(torch.complex64))
    normalization_error = (probabilities.sum(dim=-1) - 1.0).abs()
    return ThesisProjection(
        branches=branches,
        amplitudes=amplitudes,
        probabilities=probabilities,
        masses=masses,
        phases=phases,
        coherence=coherence,
        residue=residue,
        relation=relation,
        memory=memory,
        cocycle=cocycle,
        normalization_error=normalization_error,
    )


def thesis_quantum_step(
    omega_core: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    H: torch.Tensor,
    config: ThesisDynamicsConfig,
    *,
    dt: float = 1.0,
) -> ThesisStepResult:
    branches = enumerate_up_branches(config.n_qubits)
    pre = condition_state_with_hamiltonian(state, H, branches, strength=config.hamiltonian_state_strength)
    nxt = omega_core.forward_state(pre)
    proj = thesis_project(omega_core, nxt, H, config, dt=dt)
    return ThesisStepResult(preconditioned_state=pre, next_state=nxt, projection=proj)


def cocycle_nonseparability_score(projection: ThesisProjection) -> torch.Tensor:
    """Mean absolute non-local cocycle residue after subtracting branch mean."""
    centered = projection.cocycle - projection.cocycle.mean(dim=-1, keepdim=True)
    return centered.abs().mean(dim=-1)
