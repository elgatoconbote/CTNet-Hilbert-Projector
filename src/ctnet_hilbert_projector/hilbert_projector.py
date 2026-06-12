#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hilbert projector on the CTNet u/p cardinal basis.

The projector treats the Hilbert vector as a projective family:

    A_t(sigma) = Q_sigma(Xi_t), sigma in {u,p}^n

It does not make the flat 2^n table the internal primitive. The table is a
materialized projection for auditing small n.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctnet_omega_core import FoldedCTNetOmegaCubo26, FoldedOmegaCuboState

UPBranch = Tuple[str, ...]


def enumerate_up_branches(n: int) -> List[UPBranch]:
    """Return all cardinal branches in lexicographic u/p order."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [tuple(bits) for bits in itertools.product(("u", "p"), repeat=n)]


def branch_to_tensor(branch: Sequence[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode a branch as +/- modal coordinates: u=+1, p=-1.

    This is not a reduction of u/p to 1/0. It is only a numerical chart for the
    branch readout. The structural meaning remains actuation/inertia.
    """
    vals = []
    for item in branch:
        if item == "u":
            vals.append(1.0)
        elif item == "p":
            vals.append(-1.0)
        else:
            raise ValueError(f"invalid branch symbol {item!r}; expected 'u' or 'p'")
    return torch.tensor(vals, device=device, dtype=dtype)


@dataclass(frozen=True)
class HilbertProjectorConfig:
    n_qubits: int = 3
    branch_hidden: int = 64
    readout_rank: int = 8
    beta_mass: float = 1.0
    omega_penalty: float = 1.0
    phase_scale: float = math.pi
    eps: float = 1e-9


@dataclass
class BranchReadout:
    branch: UPBranch
    amplitude: complex
    probability: float
    mass: float
    phase: float


@dataclass
class HilbertProjection:
    branches: List[UPBranch]
    amplitudes: torch.Tensor  # [B, 2^n], complex
    probabilities: torch.Tensor  # [B, 2^n]
    masses: torch.Tensor  # [B, 2^n]
    phases: torch.Tensor  # [B, 2^n]
    normalization_error: torch.Tensor  # [B]

    def as_branch_table(self, batch_index: int = 0) -> List[BranchReadout]:
        rows: List[BranchReadout] = []
        amps = self.amplitudes[batch_index].detach().cpu()
        probs = self.probabilities[batch_index].detach().cpu()
        masses = self.masses[batch_index].detach().cpu()
        phases = self.phases[batch_index].detach().cpu()
        for branch, amp, prob, mass, phase in zip(self.branches, amps, probs, masses, phases):
            rows.append(
                BranchReadout(
                    branch=branch,
                    amplitude=complex(float(amp.real), float(amp.imag)),
                    probability=float(prob),
                    mass=float(mass),
                    phase=float(phase),
                )
            )
        return rows


class UPPauli:
    """Small utilities for u/p basis observables."""

    @staticmethod
    def z_expectation(projection: HilbertProjection, site: int, *, u_value: float = 1.0, p_value: float = -1.0) -> torch.Tensor:
        vals = []
        for branch in projection.branches:
            vals.append(u_value if branch[site] == "u" else p_value)
        coeff = torch.tensor(vals, device=projection.probabilities.device, dtype=projection.probabilities.dtype)
        return (projection.probabilities * coeff.unsqueeze(0)).sum(dim=-1)

    @staticmethod
    def zz_correlation(projection: HilbertProjection, i: int, j: int) -> torch.Tensor:
        zi = UPPauli.z_expectation(projection, i)
        zj = UPPauli.z_expectation(projection, j)
        vals = []
        for branch in projection.branches:
            vi = 1.0 if branch[i] == "u" else -1.0
            vj = 1.0 if branch[j] == "u" else -1.0
            vals.append(vi * vj)
        coeff = torch.tensor(vals, device=projection.probabilities.device, dtype=projection.probabilities.dtype)
        zij = (projection.probabilities * coeff.unsqueeze(0)).sum(dim=-1)
        return zij - zi * zj


class UPHilbertProjector(nn.Module):
    """Read amplitudes on {u,p}^n from a folded CTNet state.

    The module reads a compact global descriptor from Xi=(Z,M,R,C6,pad), combines
    it with a branch chart sigma, then produces:

        mass  mu_t(sigma) > 0
        phase Theta_t(sigma)
        amplitude sqrt(mu/sum_mu) exp(i Theta)
    """

    def __init__(self, config: HilbertProjectorConfig | None = None):
        super().__init__()
        self.config = config or HilbertProjectorConfig()
        n = self.config.n_qubits
        h = self.config.branch_hidden
        self.branch_encoder = nn.Sequential(
            nn.Linear(n, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
        )
        # State descriptor: z mean, memory mean, relations mean, cubo, and scalar closures.
        # The descriptor dimension is inferred lazily at the first forward call.
        self.state_proj: nn.Linear | None = None
        self.mass_head: nn.Linear | None = None
        self.phase_head: nn.Linear | None = None
        self.interaction_head: nn.Linear | None = None
        self._branches = enumerate_up_branches(n)

    @property
    def branches(self) -> List[UPBranch]:
        return list(self._branches)

    def _state_descriptor(self, state: FoldedOmegaCuboState, omega_core: FoldedCTNetOmegaCubo26) -> torch.Tensor:
        obs = omega_core.cubo_observation(state)
        z_mean = state.z.mean(dim=1)
        mem_mean = state.memory.mean(dim=1)
        rel_mean = state.relations.mean(dim=1)
        scalars = torch.stack(
            [
                obs["residual"],
                obs["absorption"],
                obs["omega"],
                obs["closure_score"],
            ],
            dim=-1,
        )
        return torch.cat([z_mean, mem_mean, rel_mean, state.cubo, scalars], dim=-1)

    def _ensure_heads(self, descriptor_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        h = self.config.branch_hidden
        if self.state_proj is None:
            self.state_proj = nn.Linear(descriptor_dim, h)
            self.mass_head = nn.Linear(h, 1)
            self.phase_head = nn.Linear(h, 1)
            self.interaction_head = nn.Linear(h, 1)
            self.add_module("state_proj", self.state_proj)
            self.add_module("mass_head", self.mass_head)
            self.add_module("phase_head", self.phase_head)
            self.add_module("interaction_head", self.interaction_head)
        self.state_proj.to(device=device, dtype=dtype)
        self.mass_head.to(device=device, dtype=dtype)
        self.phase_head.to(device=device, dtype=dtype)
        self.interaction_head.to(device=device, dtype=dtype)

    def branch_matrix(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([branch_to_tensor(branch, device=device, dtype=dtype) for branch in self._branches], dim=0)

    def project(self, omega_core: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState) -> HilbertProjection:
        descriptor = self._state_descriptor(state, omega_core)
        batch, descriptor_dim = descriptor.shape
        device, dtype = descriptor.device, descriptor.dtype
        self._ensure_heads(descriptor_dim, device, dtype)
        assert self.state_proj is not None and self.mass_head is not None and self.phase_head is not None
        assert self.interaction_head is not None

        state_latent = torch.tanh(self.state_proj(descriptor))  # [B,H]
        branches = self.branch_matrix(device=device, dtype=dtype)  # [S,n]
        branch_latent = self.branch_encoder(branches)  # [S,H]

        joint = torch.tanh(state_latent.unsqueeze(1) + branch_latent.unsqueeze(0))  # [B,S,H]
        raw_mass = self.mass_head(joint).squeeze(-1)
        raw_phase = self.phase_head(joint).squeeze(-1)
        interaction = self.interaction_head(joint).squeeze(-1)

        # Cubo omega penalizes unabsorbed defect globally; interaction is branch specific.
        obs = omega_core.cubo_observation(state)
        omega = obs["omega"].unsqueeze(-1).to(dtype=dtype)
        mass_logits = self.config.beta_mass * raw_mass + interaction - self.config.omega_penalty * omega
        masses = F.softplus(mass_logits) + self.config.eps
        probabilities = masses / masses.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        phases = self.config.phase_scale * torch.tanh(raw_phase + interaction)
        amplitudes = torch.sqrt(probabilities) * torch.exp(1j * phases.to(torch.complex64))
        normalization_error = (probabilities.sum(dim=-1) - 1.0).abs()
        return HilbertProjection(
            branches=self.branches,
            amplitudes=amplitudes,
            probabilities=probabilities,
            masses=masses,
            phases=phases,
            normalization_error=normalization_error,
        )

    def forward(self, omega_core: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState) -> HilbertProjection:
        return self.project(omega_core, state)

    @torch.no_grad()
    def audit(self, omega_core: FoldedCTNetOmegaCubo26, *, batch: int = 1, steps: int = 1, seed: int = 0) -> Dict[str, float]:
        state = omega_core.random_state(batch=batch, seed=seed)
        for _ in range(max(0, steps)):
            state = omega_core.forward_state(state)
        projection = self.project(omega_core, state)
        probs = projection.probabilities
        return {
            "n_qubits": float(self.config.n_qubits),
            "num_branches": float(len(projection.branches)),
            "normalization_error_max": float(projection.normalization_error.max().detach().cpu()),
            "probability_min": float(probs.min().detach().cpu()),
            "probability_max": float(probs.max().detach().cpu()),
            "mass_min": float(projection.masses.min().detach().cpu()),
            "mass_max": float(projection.masses.max().detach().cpu()),
            "phase_min": float(projection.phases.min().detach().cpu()),
            "phase_max": float(projection.phases.max().detach().cpu()),
        }
