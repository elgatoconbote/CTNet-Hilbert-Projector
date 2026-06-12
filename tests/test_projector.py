from __future__ import annotations

import torch

from ctnet_hilbert_projector import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    HilbertProjectorConfig,
    UPHilbertProjector,
    enumerate_up_branches,
)


def test_enumerate_up_branches() -> None:
    branches = enumerate_up_branches(3)
    assert len(branches) == 8
    assert branches[0] == ("u", "u", "u")
    assert branches[-1] == ("p", "p", "p")


def test_projection_normalizes_small_state() -> None:
    torch.manual_seed(0)
    layout = FoldLayout(N=64, d=16)
    omega = FoldedCTNetOmegaCubo26(layout=layout)
    projector = UPHilbertProjector(HilbertProjectorConfig(n_qubits=3))
    state = omega.random_state(batch=2, seed=0)
    projection = projector.project(omega, state)
    assert projection.amplitudes.shape == (2, 8)
    assert projection.probabilities.shape == (2, 8)
    assert torch.allclose(projection.probabilities.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert float(projection.normalization_error.max().detach()) < 1e-6


def test_ctnet_forward_inverse_audit_runs() -> None:
    torch.manual_seed(0)
    omega = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    audit = omega.audit(batch=1, steps=1, seed=0)
    assert audit["capacity"] == 1024.0
    assert audit["semantic_size"] == 797.0
    assert audit["pad_size"] == 227.0
    assert audit["packed_mae"] >= 0.0
