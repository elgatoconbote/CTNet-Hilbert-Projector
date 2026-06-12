#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CTNet quantum thesis dynamics.

Faithful executable version of the CTNet/u-p quantum thesis:

- H_t enters the active regime and deforms the persistent state Xi_t.
- Branch features phi_t(sigma) are read from Z, M, R, C6, H and Omega.
- K_t = D_t + U_t V_t^T is explicit and generates branch coherence.
- Mass is causal: mu = exp(beta c - gamma omega + lambda r + eta m + chi).
- Phase is reversible branch action driven by state, H, coherence and cocycle.
- Mass, phase, coherence and residue feed back into Xi before the CTNet step.
- The output vector is a projective family of amplitudes A_t(sigma)=Q_sigma(Xi_t).
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
    hamiltonian_mass_strength: float = 0.0
    cardinal_mass_strength: float = 0.0
    mass_feedback_strength: float = 0.05
    coherence_feedback_strength: float = 0.02
    atlas_strength: float = 0.20
    relation_branch_strength: float = 0.25
    memory_branch_strength: float = 0.15
    coherence_rank: int = 2
    feature_dim: int = 16
    coherence_clamp: float = 8.0
    eps: float = 1e-9


@dataclass
class CoherenceTensorState:
    """Explicit K_t = D_t + U_t V_t^T representation."""

    diagonal: torch.Tensor
    low_rank_u: torch.Tensor
    low_rank_v: torch.Tensor


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
    branch_features: torch.Tensor | None = None
    coherence_tensor: CoherenceTensorState | None = None
    atlas_gauge: torch.Tensor | None = None
    mass_contrast: torch.Tensor | None = None

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
    feedback_state: FoldedOmegaCuboState
    next_state: FoldedOmegaCuboState
    projection: ThesisProjection
    pre_feedback_projection: ThesisProjection


@dataclass
class ProjectiveCostReport:
    Cgen: float
    Cread: float
    Cobs: float
    Clist: float
    Dproj: float
    Ceff_amp: float


def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] < dim:
        return F.pad(x, (0, dim - x.shape[-1]))
    return x[..., :dim]


def _center_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x - x.mean()
    s = y.std()
    if bool(s < eps):
        return torch.zeros_like(x)
    return y / s.clamp_min(eps)


def _center_by_branch(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def branch_matrix(branches: Sequence[UPBranch], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([branch_to_tensor(branch, device=device, dtype=dtype) for branch in branches], dim=0)


def branch_index(branches: Sequence[UPBranch]) -> Dict[UPBranch, int]:
    return {tuple(branch): i for i, branch in enumerate(branches)}


def hamiltonian_branch_descriptors(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return diagonal energy, coupling strength and signed phase drive per branch."""
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    diag = H.diagonal().real
    off = H - torch.diag_embed(H.diagonal())
    coupling = off.abs().sum(dim=-1).real
    phase_drive = diag + off.real.sum(dim=-1)
    return diag, coupling, phase_drive


def _pair_drive(weights: torch.Tensor, bmat: torch.Tensor) -> torch.Tensor:
    """Fold pair structure into site coordinates."""
    n = bmat.shape[-1]
    out = torch.zeros(n, device=bmat.device, dtype=bmat.dtype)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = (weights * bmat[:, i] * bmat[:, j]).mean()
            out[i] = out[i] + val
            out[j] = out[j] + val
            count += 1
    if count:
        out = out / float(count)
    return out


def _branch_cardinal_signatures(bmat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return intrinsic u/p branch features without collapsing u/p to binary semantics."""
    n = bmat.shape[-1]
    activity = bmat.mean(dim=-1)
    if n > 1:
        adjacent = bmat[:, :-1] * bmat[:, 1:]
        pair_signature = adjacent.mean(dim=-1)
        transition_density = ((1.0 - adjacent) * 0.5).mean(dim=-1)
        boundary_signature = bmat[:, 0] * bmat[:, -1]
    else:
        pair_signature = bmat[:, 0]
        transition_density = torch.zeros_like(pair_signature)
        boundary_signature = bmat[:, 0]
    pos = torch.arange(n, device=bmat.device, dtype=bmat.dtype) + 1.0
    sine_weights = torch.sin(pos * 1.61803398875)
    cosine_weights = torch.cos(pos * 2.41421356237)
    ramp_weights = (pos - pos.mean()) / pos.std().clamp_min(1e-6)
    position_sine = torch.einsum("sn,n->s", bmat, sine_weights) / float(n)
    position_cosine = torch.einsum("sn,n->s", bmat, cosine_weights) / float(n)
    oriented_ramp = torch.einsum("sn,n->s", bmat, ramp_weights) / float(n)
    parity_signature = bmat.prod(dim=-1)
    return activity, pair_signature, transition_density, boundary_signature, position_sine, position_cosine, oriented_ramp, parity_signature


def branch_atlas_gauge(
    state: FoldedOmegaCuboState,
    bmat: torch.Tensor,
    H: torch.Tensor | None,
    config: ThesisDynamicsConfig,
) -> torch.Tensor:
    """Atlas gauge: branch-specific chart pressure from C6, M, R and H."""
    dtype, device = state.z.dtype, state.z.device
    n = bmat.shape[-1]
    mem_vec = _match_dim(state.memory.mean(dim=1), n)
    rel_vec = _match_dim(state.relations.mean(dim=1), n)
    cubo_a = _match_dim(state.cubo, n)
    cubo_b = _match_dim(torch.flip(state.cubo, dims=[-1]), n)
    mem_proj = torch.einsum("bd,sd->bs", mem_vec, bmat)
    rel_proj = torch.einsum("bd,sd->bs", rel_vec, bmat)
    c6_proj = torch.einsum("bd,sd->bs", cubo_a, bmat)
    c6_pair = torch.einsum("bd,sd->bs", cubo_b, bmat.flip(dims=[-1]))

    activity, pair_signature, transition_density, boundary_signature, position_sine, position_cosine, oriented_ramp, parity_signature = _branch_cardinal_signatures(bmat)
    sig = torch.stack([activity, pair_signature, transition_density, boundary_signature, position_sine, position_cosine, oriented_ramp, parity_signature], dim=-1)
    sig_weights = _match_dim(state.cubo, sig.shape[-1])
    sig_proj = torch.einsum("bf,sf->bs", sig_weights, sig)

    if H is not None:
        diag, coupling, phase_drive = hamiltonian_branch_descriptors(H.to(device=device))
        h_branch = (_center_scale(diag.to(dtype=dtype)) + 0.5 * _center_scale(coupling.to(dtype=dtype)) + 0.25 * _center_scale(phase_drive.to(dtype=dtype))).unsqueeze(0)
    else:
        h_branch = torch.zeros_like(mem_proj)

    raw = 0.25 * mem_proj + 0.25 * rel_proj + 0.20 * c6_proj + 0.15 * c6_pair + 0.10 * sig_proj + 0.05 * h_branch
    raw = raw - raw.mean(dim=-1, keepdim=True)
    return float(config.atlas_strength) * torch.tanh(raw)


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

    diag_n = _center_scale(diag)
    coup_n = _center_scale(coupling)
    phase_n = _center_scale(phase_drive)

    linear_drive = torch.einsum("s,sn->n", diag_n + 0.5 * coup_n + 0.25 * phase_n, bmat) / float(len(branches))
    pair_diag = _pair_drive(diag_n, bmat)
    pair_phase = _pair_drive(phase_n + 0.5 * coup_n, bmat)

    n = bmat.shape[-1]
    global_scale = H.real.abs().mean().to(dtype=dtype).clamp_min(1e-6)
    global_pattern_n = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype) * global_scale

    modal_drive = linear_drive + pair_diag + 0.1 * global_pattern_n
    rel_drive = pair_phase + 0.5 * linear_drive + 0.05 * torch.flip(global_pattern_n, dims=[0])
    mem_drive = 0.5 * modal_drive + 0.25 * rel_drive

    z_delta = _match_dim(modal_drive, state.z.shape[-1]).view(1, 1, -1).expand_as(state.z)
    r_delta = _match_dim(rel_drive, state.relations.shape[-1]).view(1, 1, -1).expand_as(state.relations)
    m_delta = _match_dim(mem_drive, state.memory.shape[-1]).view(1, 1, -1).expand_as(state.memory)
    c_delta = torch.zeros_like(state.cubo)
    c_stats = torch.stack([
        diag_n.mean(), diag_n.std(), coup_n.mean(), coup_n.std(), phase_n.mean(), phase_n.std(),
        pair_diag.abs().mean(), pair_phase.abs().mean(), global_scale,
    ])
    width = min(c_delta.shape[-1], c_stats.numel())
    c_delta[:, :width] = c_stats[:width]

    alpha = float(strength)
    return FoldedOmegaCuboState(
        z=state.z + alpha * z_delta,
        memory=state.memory + alpha * m_delta,
        relations=state.relations + alpha * r_delta,
        cubo=state.cubo + alpha * c_delta,
        pad=state.pad,
    )


def _branch_projections(state: FoldedOmegaCuboState, branches: Sequence[UPBranch]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = state.z.device, state.z.dtype
    bmat = branch_matrix(branches, device=device, dtype=dtype)
    z_mean = state.z.mean(dim=1)
    m_mean = _match_dim(state.memory.mean(dim=1), z_mean.shape[-1])
    r_mean = _match_dim(state.relations.mean(dim=1), z_mean.shape[-1])
    z_proj = torch.einsum("bd,sd->bs", z_mean, _match_dim(bmat, z_mean.shape[-1]))
    m_proj = torch.einsum("bd,sd->bs", m_mean, _match_dim(bmat, m_mean.shape[-1]))
    r_proj = torch.einsum("bd,sd->bs", r_mean, _match_dim(bmat, r_mean.shape[-1]))
    return z_proj, m_proj, r_proj, bmat


def relational_cocycle(state: FoldedOmegaCuboState, branches: Sequence[UPBranch], strength: float) -> torch.Tensor:
    device, dtype = state.z.device, state.z.dtype
    bmat = branch_matrix(branches, device=device, dtype=dtype)
    rel = state.relations.mean(dim=1)
    rel_vec = _match_dim(rel, bmat.shape[-1])
    gram = torch.einsum("bi,bj->bij", rel_vec, rel_vec)
    raw = torch.einsum("si,bij,sj->bs", bmat, gram, bmat)
    local = raw.mean(dim=-1, keepdim=True)
    return float(strength) * torch.tanh(raw - local)


def branch_feature_tensor(
    omega_core: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    branches: Sequence[UPBranch],
    H: torch.Tensor | None,
    config: ThesisDynamicsConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build phi_t(sigma; Z,M,R,C6,rho,Omega,H) with explicit cardinal signatures."""
    z_proj, m_proj, r_proj, bmat = _branch_projections(state, branches)
    obs = omega_core.cubo_observation(state)
    omega_global = obs["omega"].unsqueeze(-1).to(dtype=state.z.dtype)
    closure = obs["closure_score"].unsqueeze(-1).to(dtype=state.z.dtype)

    if H is not None:
        diag, coupling, phase_drive = hamiltonian_branch_descriptors(H.to(device=state.z.device))
        diag = _center_scale(diag.to(dtype=state.z.dtype)).unsqueeze(0)
        coupling = _center_scale(coupling.to(dtype=state.z.dtype)).unsqueeze(0)
        phase_drive = _center_scale(phase_drive.to(dtype=state.z.dtype)).unsqueeze(0)
    else:
        diag = torch.zeros_like(z_proj)
        coupling = torch.zeros_like(z_proj)
        phase_drive = torch.zeros_like(z_proj)

    activity, pair_signature, transition_density, boundary_signature, position_sine, position_cosine, oriented_ramp, parity_signature = _branch_cardinal_signatures(bmat)
    activity = activity.unsqueeze(0).expand_as(z_proj)
    pair_signature = pair_signature.unsqueeze(0).expand_as(z_proj)
    transition_density = transition_density.unsqueeze(0).expand_as(z_proj)
    boundary_signature = boundary_signature.unsqueeze(0).expand_as(z_proj)
    position_sine = position_sine.unsqueeze(0).expand_as(z_proj)
    position_cosine = position_cosine.unsqueeze(0).expand_as(z_proj)
    oriented_ramp = oriented_ramp.unsqueeze(0).expand_as(z_proj)
    parity_signature = parity_signature.unsqueeze(0).expand_as(z_proj)

    cocycle = relational_cocycle(state, branches, config.cocycle_strength)
    atlas = branch_atlas_gauge(state, bmat, H, config)
    omega = omega_global.expand_as(z_proj)
    clos = closure.expand_as(z_proj)
    phi = torch.stack([
        z_proj, m_proj, r_proj, diag, coupling, phase_drive, clos, omega,
        activity, pair_signature, transition_density, boundary_signature,
        position_sine, position_cosine, oriented_ramp, parity_signature,
    ], dim=-1)
    if phi.shape[-1] != config.feature_dim:
        phi = _match_dim(phi, config.feature_dim)
    return phi, z_proj, m_proj, r_proj, phase_drive, cocycle, atlas


def build_coherence_tensor(
    state: FoldedOmegaCuboState,
    H: torch.Tensor | None,
    config: ThesisDynamicsConfig,
) -> CoherenceTensorState:
    """Construct bounded K_t = D_t + U_t V_t^T from C6 and H statistics."""
    B = state.cubo.shape[0]
    Fdim = config.feature_dim
    rank = max(1, int(config.coherence_rank))
    dtype, device = state.cubo.dtype, state.cubo.device
    cubo = _match_dim(state.cubo, Fdim * (1 + 2 * rank))
    diagonal_raw = cubo[:, :Fdim]
    low_raw = cubo[:, Fdim:]
    diagonal = 0.02 + 0.18 * torch.sigmoid(diagonal_raw)
    if low_raw.shape[-1] < 2 * Fdim * rank:
        low_raw = F.pad(low_raw, (0, 2 * Fdim * rank - low_raw.shape[-1]))
    low_raw = low_raw[:, : 2 * Fdim * rank]
    u_raw, v_raw = low_raw.split(Fdim * rank, dim=-1)
    U = 0.05 * torch.tanh(u_raw.reshape(B, Fdim, rank))
    V = 0.05 * torch.tanh(v_raw.reshape(B, Fdim, rank))

    if H is not None:
        h_scale = H.real.abs().mean().to(device=device, dtype=dtype).view(1, 1)
        diagonal = diagonal + 0.005 * torch.tanh(h_scale)
        U = U + 0.002 * torch.tanh(h_scale).view(1, 1, 1)
        V = V - 0.002 * torch.tanh(h_scale).view(1, 1, 1)
    return CoherenceTensorState(diagonal=diagonal, low_rank_u=U, low_rank_v=V)


def coherence_from_tensor(phi: torch.Tensor, K: CoherenceTensorState) -> torch.Tensor:
    """c_t(sigma)=<phi,D phi> + sum_r <phi,U_r><phi,V_r>, scale-controlled."""
    phi_scale = phi.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    phi_n = phi / phi_scale
    diag_part = (phi_n.pow(2) * K.diagonal.unsqueeze(1)).sum(dim=-1)
    u_proj = torch.einsum("bsf,bfr->bsr", phi_n, K.low_rank_u)
    v_proj = torch.einsum("bsf,bfr->bsr", phi_n, K.low_rank_v)
    low_part = (u_proj * v_proj).sum(dim=-1)
    return (diag_part + low_part) / (float(phi.shape[-1]) ** 0.5)


def hamiltonian_mass_contrast(phi: torch.Tensor, atlas: torch.Tensor, config: ThesisDynamicsConfig) -> torch.Tensor:
    """Branch mass contrast forced by Hamiltonian and cardinal signatures.

    This is not a target-vector fit. It uses only branch-local H descriptors and
    u/p atlas signatures already contained in phi. It exists because phase-only
    H coupling can keep Born masses nearly flat even when exact projected dynamics
    develops a non-flat probability profile.
    """
    h_energy = _center_by_branch(phi[..., 3])
    h_coupling = _center_by_branch(phi[..., 4])
    h_phase = _center_by_branch(phi[..., 5])
    cardinal = _center_by_branch(
        0.45 * phi[..., 8]
        + 0.85 * phi[..., 9]
        - 0.55 * phi[..., 10]
        + 0.25 * phi[..., 11]
        + 0.45 * phi[..., 12]
        + 0.45 * phi[..., 13]
        + 0.25 * phi[..., 14]
        + 0.20 * phi[..., 15]
    )
    h_drive = torch.tanh(h_energy + 0.75 * h_coupling + 0.50 * h_phase)
    c_drive = torch.tanh(cardinal + 0.5 * _center_by_branch(atlas))
    return config.hamiltonian_mass_strength * h_drive + config.cardinal_mass_strength * c_drive


def thesis_project(
    omega_core: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    H: torch.Tensor | None,
    config: ThesisDynamicsConfig,
    *,
    dt: float = 1.0,
) -> ThesisProjection:
    branches = enumerate_up_branches(config.n_qubits)
    phi, z_proj, m_proj, r_proj, phase_drive, cocycle, atlas = branch_feature_tensor(omega_core, state, branches, H, config)
    K = build_coherence_tensor(state, H, config)

    raw_coherence = coherence_from_tensor(phi, K)
    coherence = torch.tanh((raw_coherence + 0.5 * atlas).clamp(-config.coherence_clamp, config.coherence_clamp))
    memory = torch.sigmoid(m_proj + config.memory_branch_strength * atlas)
    relation = torch.sigmoid(r_proj + config.relation_branch_strength * atlas)
    residue = torch.relu(phi[..., 7] + 0.1 * torch.tanh(-z_proj) - 0.05 * atlas)
    mass_contrast = hamiltonian_mass_contrast(phi, atlas, config)

    mass_log = (
        config.beta_coherence * coherence
        - config.gamma_residue * residue
        + config.lambda_relation * relation
        + config.eta_memory * memory
        + cocycle
        + atlas
        + mass_contrast
    )
    masses = torch.exp(mass_log.clamp(-config.coherence_clamp, config.coherence_clamp)) + config.eps
    probabilities = masses / masses.sum(dim=-1, keepdim=True).clamp_min(config.eps)

    reversible_action = z_proj + r_proj + 0.5 * m_proj + 0.25 * coherence + cocycle + atlas + 0.20 * mass_contrast
    phases = torch.tanh(reversible_action) - float(dt) * config.hamiltonian_phase_strength * phase_drive
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
        branch_features=phi,
        coherence_tensor=K,
        atlas_gauge=atlas,
        mass_contrast=mass_contrast,
    )


def feedback_state_with_mass(
    state: FoldedOmegaCuboState,
    projection: ThesisProjection,
    config: ThesisDynamicsConfig,
) -> FoldedOmegaCuboState:
    """Implement Xi_{t+1}=U_rho(Xi~,mu,Theta,K,Omega)."""
    bmat = branch_matrix(projection.branches, device=state.z.device, dtype=state.z.dtype)
    prob = projection.probabilities.to(dtype=state.z.dtype)
    mass_center = prob - prob.mean(dim=-1, keepdim=True)
    phase_sin = torch.sin(projection.phases).to(dtype=state.z.dtype)
    phase_cos = torch.cos(projection.phases).to(dtype=state.z.dtype)
    coh = projection.coherence.to(dtype=state.z.dtype)
    res = projection.residue.to(dtype=state.z.dtype)
    atlas = projection.atlas_gauge.to(dtype=state.z.dtype) if projection.atlas_gauge is not None else torch.zeros_like(prob)
    mass_contrast = projection.mass_contrast.to(dtype=state.z.dtype) if projection.mass_contrast is not None else torch.zeros_like(prob)

    mass_moment = torch.einsum("bs,sn->bn", mass_center, bmat)
    phase_moment = torch.einsum("bs,sn->bn", prob * phase_sin, bmat)
    coh_moment = torch.einsum("bs,sn->bn", prob * coh, bmat)
    res_moment = torch.einsum("bs,sn->bn", prob * res, bmat)
    atlas_moment = torch.einsum("bs,sn->bn", prob * atlas, bmat)
    contrast_moment = torch.einsum("bs,sn->bn", prob * mass_contrast, bmat)
    rel_moment = torch.einsum("bs,sn->bn", prob * projection.cocycle.to(dtype=state.z.dtype), bmat)

    z_drive = mass_moment + 0.5 * phase_moment + config.coherence_feedback_strength * coh_moment + 0.25 * atlas_moment + 0.30 * contrast_moment
    m_drive = 0.5 * mass_moment + 0.5 * coh_moment - 0.25 * res_moment + 0.25 * atlas_moment + 0.25 * contrast_moment
    r_drive = rel_moment + 0.25 * phase_moment + 0.5 * atlas_moment + 0.35 * contrast_moment

    z_delta = _match_dim(z_drive, state.z.shape[-1]).unsqueeze(1).expand_as(state.z)
    m_delta = _match_dim(m_drive, state.memory.shape[-1]).unsqueeze(1).expand_as(state.memory)
    r_delta = _match_dim(r_drive, state.relations.shape[-1]).unsqueeze(1).expand_as(state.relations)

    c_delta = torch.zeros_like(state.cubo)
    stats = torch.stack([
        prob.mean(dim=-1), prob.std(dim=-1), projection.masses.mean(dim=-1), projection.masses.std(dim=-1),
        projection.phases.mean(dim=-1), projection.phases.std(dim=-1), projection.coherence.mean(dim=-1),
        projection.residue.mean(dim=-1), projection.cocycle.abs().mean(dim=-1), phase_cos.mean(dim=-1),
        atlas.abs().mean(dim=-1), atlas.std(dim=-1), mass_contrast.mean(dim=-1), mass_contrast.std(dim=-1),
    ], dim=-1).to(dtype=state.cubo.dtype)
    width = min(c_delta.shape[-1], stats.shape[-1])
    c_delta[:, :width] = stats[:, :width]

    alpha = float(config.mass_feedback_strength)
    return FoldedOmegaCuboState(
        z=state.z + alpha * z_delta,
        memory=state.memory + alpha * m_delta,
        relations=state.relations + alpha * r_delta,
        cubo=state.cubo + alpha * c_delta,
        pad=state.pad,
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
    pre_proj = thesis_project(omega_core, pre, H, config, dt=dt)
    feedback = feedback_state_with_mass(pre, pre_proj, config)
    nxt = omega_core.forward_state(feedback)
    proj = thesis_project(omega_core, nxt, H, config, dt=dt)
    return ThesisStepResult(preconditioned_state=pre, feedback_state=feedback, next_state=nxt, projection=proj, pre_feedback_projection=pre_proj)


def cocycle_nonseparability_score(projection: ThesisProjection) -> torch.Tensor:
    centered = projection.cocycle - projection.cocycle.mean(dim=-1, keepdim=True)
    return centered.abs().mean(dim=-1)


def read_branch_amplitude(projection: ThesisProjection, branch: Sequence[str], batch: int = 0) -> torch.Tensor:
    idx = branch_index(projection.branches)[tuple(branch)]
    return projection.amplitudes[batch, idx]


def branch_active_count(branch: UPBranch) -> int:
    return sum(1 for x in branch if x == "u")


def sector_amplitude_by_active_count(projection: ThesisProjection, active_count: int) -> torch.Tensor:
    idx = [i for i, branch in enumerate(projection.branches) if branch_active_count(branch) == active_count]
    if not idx:
        return torch.zeros(projection.amplitudes.shape[0], device=projection.amplitudes.device, dtype=projection.amplitudes.dtype)
    return projection.amplitudes[:, idx].sum(dim=-1)


def observable_expectation(projection: ThesisProjection, operator: torch.Tensor) -> torch.Tensor:
    psi = projection.amplitudes.to(dtype=operator.dtype, device=operator.device)
    op_psi = torch.einsum("ij,bj->bi", operator, psi)
    return (psi.conj() * op_psi).sum(dim=-1)


def up_x_gate(n_qubits: int, site: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    branches = enumerate_up_branches(n_qubits)
    idx = branch_index(branches)
    G = torch.zeros(len(branches), len(branches), device=device, dtype=dtype)
    for col, branch in enumerate(branches):
        out = list(branch)
        out[site] = "p" if out[site] == "u" else "u"
        G[idx[tuple(out)], col] = 1.0 + 0.0j
    return G


def up_phase_gate(n_qubits: int, site: int, phi_u: float, phi_p: float = 0.0, *, device: torch.device | None = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    branches = enumerate_up_branches(n_qubits)
    diag = []
    for branch in branches:
        phi = phi_u if branch[site] == "u" else phi_p
        diag.append(torch.exp(torch.tensor(1j * phi, device=device, dtype=dtype)))
    return torch.diag(torch.stack(diag))


def up_hadamard_gate(n_qubits: int, site: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    branches = enumerate_up_branches(n_qubits)
    idx = branch_index(branches)
    G = torch.zeros(len(branches), len(branches), device=device, dtype=dtype)
    inv = 1.0 / (2.0 ** 0.5)
    for col, branch in enumerate(branches):
        same = list(branch)
        flip = list(branch)
        flip[site] = "p" if branch[site] == "u" else "u"
        sign = 1.0 if branch[site] == "u" else -1.0
        G[idx[tuple(same)], col] += inv
        G[idx[tuple(flip)], col] += sign * inv
    return G


def apply_projected_gate(projection: ThesisProjection, gate: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bj->bi", gate.to(device=projection.amplitudes.device, dtype=projection.amplitudes.dtype), projection.amplitudes)


def projective_commutation_error(before: ThesisProjection, after: ThesisProjection, unitary: torch.Tensor, *, phase_invariant: bool = True) -> torch.Tensor:
    target = torch.einsum("ij,bj->bi", unitary.to(device=before.amplitudes.device, dtype=before.amplitudes.dtype), before.amplitudes)
    cand = after.amplitudes
    if phase_invariant:
        overlap = (cand.conj() * target).sum(dim=-1, keepdim=True)
        cand = cand * (overlap / overlap.abs().clamp_min(1e-12))
    return (target - cand).norm(dim=-1)


def estimate_projective_costs(state: FoldedOmegaCuboState, projection: ThesisProjection) -> ProjectiveCostReport:
    Cgen = float(state.z.numel() + state.memory.numel() + state.relations.numel() + state.cubo.numel() + state.pad.numel())
    Cread = float((projection.branch_features.shape[-1] if projection.branch_features is not None else 1) + 2)
    Cobs = float(projection.amplitudes.shape[-1] ** 2)
    Clist = float(projection.amplitudes.shape[-1])
    Dproj = float(projection.amplitudes.shape[-1]) / max(Cgen, 1.0)
    Ceff_amp = Cgen / max(float(projection.amplitudes.shape[-1]), 1.0)
    return ProjectiveCostReport(Cgen=Cgen, Cread=Cread, Cobs=Cobs, Clist=Clist, Dproj=Dproj, Ceff_amp=Ceff_amp)
