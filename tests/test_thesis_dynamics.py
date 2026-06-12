from __future__ import annotations

import torch

from ctnet_hilbert_projector import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    IsingConfig,
    transverse_field_ising_matrix,
)
from ctnet_hilbert_projector.thesis_dynamics import (
    ThesisDynamicsConfig,
    cocycle_nonseparability_score,
    condition_state_with_hamiltonian,
    thesis_project,
    thesis_quantum_step,
)


def test_hamiltonian_conditioning_changes_state() -> None:
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    H, branches = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    state = core.random_state(batch=1, seed=0)
    conditioned = condition_state_with_hamiltonian(state, H, branches, strength=0.05)

    delta_z = (conditioned.z - state.z).abs().mean()
    delta_m = (conditioned.memory - state.memory).abs().mean()
    delta_r = (conditioned.relations - state.relations).abs().mean()
    delta_c = (conditioned.cubo - state.cubo).abs().mean()

    assert float(delta_z.detach()) > 0.0
    assert float(delta_m.detach()) > 0.0
    assert float(delta_r.detach()) > 0.0
    assert float(delta_c.detach()) > 0.0


def test_thesis_project_generates_mass_phase_and_cocycle() -> None:
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    H, _ = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    state = core.random_state(batch=1, seed=0)
    proj = thesis_project(core, state, H, ThesisDynamicsConfig(n_qubits=3), dt=0.1)

    assert proj.amplitudes.shape == (1, 8)
    assert proj.coherence.shape == (1, 8)
    assert proj.residue.shape == (1, 8)
    assert proj.memory.shape == (1, 8)
    assert proj.relation.shape == (1, 8)
    assert proj.cocycle.shape == (1, 8)
    assert torch.allclose(proj.probabilities.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert float(proj.masses.min().detach()) > 0.0
    assert float(proj.normalization_error.max().detach()) < 1e-6


def test_thesis_quantum_step_runs_and_normalizes() -> None:
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    H, _ = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    state = core.random_state(batch=1, seed=0)

    result = thesis_quantum_step(
        core,
        state,
        H,
        ThesisDynamicsConfig(n_qubits=3),
        dt=0.1,
    )

    proj = result.projection
    assert proj.amplitudes.shape == (1, 8)
    assert torch.allclose(proj.probabilities.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert float(proj.normalization_error.max().detach()) < 1e-6
    assert float(cocycle_nonseparability_score(proj).max().detach()) >= 0.0
