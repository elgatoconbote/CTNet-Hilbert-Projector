#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State preparation for CTNet Hilbert projection.

The default random CTNet state keeps memory and relation supports very small so
that reversibility audits start close to neutral. For quantum simulation this is
not enough: the thesis requires memory topological support and relational support
as active ingredients of the branch law A_t(sigma)=Q_sigma(Xi_t).

This module prepares a non-degenerate initial Xi without breaking fixed support.
It folds Z, H and C6 into M and R as structured memory/relations, not as an
external amplitude table.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .ctnet_omega_core import FoldedOmegaCuboState
from .hilbert_projector import UPBranch, branch_to_tensor


def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] < dim:
        return torch.nn.functional.pad(x, (0, dim - x.shape[-1]))
    return x[..., :dim]


def _branch_matrix(branches: Sequence[UPBranch], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([branch_to_tensor(branch, device=device, dtype=dtype) for branch in branches], dim=0)


def prepare_quantum_atlas_state(
    state: FoldedOmegaCuboState,
    H: torch.Tensor,
    branches: Sequence[UPBranch],
    *,
    memory_strength: float = 0.20,
    relation_strength: float = 0.25,
    cubo_strength: float = 0.05,
) -> FoldedOmegaCuboState:
    """Prepare M and R as active topological/relational supports.

    The construction is deterministic and uses only internal CTNet state plus the
    Hamiltonian chart. It does not store amplitudes. It creates a richer atlas so
    that branch reads can differ by memory, relation, position and H-coupling.
    """
    device, dtype = state.z.device, state.z.dtype
    H = H.to(device=device)
    bmat = _branch_matrix(branches, device=device, dtype=dtype)
    diag = H.diagonal().real.to(dtype=dtype)
    off = H - torch.diag_embed(H.diagonal())
    coupling = off.abs().sum(dim=-1).real.to(dtype=dtype)
    phase = (diag + off.real.sum(dim=-1).to(dtype=dtype))

    def norm(x: torch.Tensor) -> torch.Tensor:
        y = x - x.mean()
        return y / y.std().clamp_min(1e-6)

    diag_n = norm(diag)
    coup_n = norm(coupling)
    phase_n = norm(phase)
    n = bmat.shape[-1]
    pos = torch.arange(n, device=device, dtype=dtype) + 1.0
    wave1 = torch.sin(pos * 1.61803398875)
    wave2 = torch.cos(pos * 2.41421356237)
    wave3 = (pos - pos.mean()) / pos.std().clamp_min(1e-6)

    branch_drive = diag_n + 0.5 * coup_n + 0.25 * phase_n
    site_drive = torch.einsum("s,sn->n", branch_drive, bmat) / float(len(branches))
    pair_drive = torch.zeros_like(site_drive)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = (branch_drive * bmat[:, i] * bmat[:, j]).mean()
            pair_drive[i] = pair_drive[i] + val
            pair_drive[j] = pair_drive[j] + val
            count += 1
    if count:
        pair_drive = pair_drive / float(count)

    z_summary = _match_dim(state.z.mean(dim=1), n)
    z_drive = torch.tanh(z_summary + site_drive.view(1, -1) + 0.5 * pair_drive.view(1, -1))
    mem_base = _match_dim(z_drive, state.memory.shape[-1]).unsqueeze(1).expand_as(state.memory)
    rel_pattern = torch.tanh(z_drive * _match_dim(wave1 + 0.5 * wave2 + 0.25 * wave3, z_drive.shape[-1]).view(1, -1))
    rel_base = _match_dim(rel_pattern, state.relations.shape[-1]).unsqueeze(1).expand_as(state.relations)

    slot_phase = torch.linspace(0.0, torch.pi, state.memory.shape[1], device=device, dtype=dtype).view(1, -1, 1)
    edge_phase = torch.linspace(0.0, 2.0 * torch.pi, state.relations.shape[1], device=device, dtype=dtype).view(1, -1, 1)
    memory_shape = torch.sin(slot_phase + mem_base)
    relation_shape = torch.cos(edge_phase + rel_base)

    c_delta = torch.zeros_like(state.cubo)
    stats = torch.stack([
        diag_n.mean(), diag_n.std(), coup_n.mean(), coup_n.std(), phase_n.mean(), phase_n.std(),
        site_drive.abs().mean(), pair_drive.abs().mean(), z_drive.abs().mean(), rel_pattern.abs().mean(),
    ])
    c_delta[:, : min(c_delta.shape[-1], stats.numel())] = stats[: c_delta.shape[-1]]

    return FoldedOmegaCuboState(
        z=state.z,
        memory=state.memory + float(memory_strength) * memory_shape,
        relations=state.relations + float(relation_strength) * relation_shape,
        cubo=state.cubo + float(cubo_strength) * c_delta,
        pad=state.pad,
    )
