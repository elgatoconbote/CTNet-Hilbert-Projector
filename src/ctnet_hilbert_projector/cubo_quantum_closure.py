from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .ctnet_omega_core import FoldedOmegaCuboState
from .thesis_dynamics import ThesisProjection


@dataclass
class CuboQuantumClosureState:
    state: FoldedOmegaCuboState
    projection: ThesisProjection
    omega_q: torch.Tensor
    absorption_q: torch.Tensor
    closure_score: torch.Tensor
    loss: torch.Tensor
    metrics: Dict[str, float]


def align_global_phase(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Align one complex state to another by global phase only."""
    c = candidate.to(device=reference.device, dtype=torch.complex128)
    r = reference.to(device=reference.device, dtype=torch.complex128)
    if r.ndim == 1:
        r = r.unsqueeze(0)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    overlap = (c.conj() * r).sum(dim=-1, keepdim=True)
    return c * (overlap / overlap.abs().clamp_min(1e-15))


CUBO_QUANTUM_CLOSURE_ROUTE = "residual -> cubo6d -> omega_q -> gates -> Q_sigma(Xi)"
