from __future__ import annotations

import torch

from ctnet_hilbert_projector import FoldLayout, FoldedCTNetOmegaCubo26, IsingConfig, transverse_field_ising_matrix
from ctnet_hilbert_projector.presets import get_thesis_preset
from ctnet_hilbert_projector.thesis_dynamics import ThesisDynamicsConfig, thesis_project


def test_best_preset_points_to_ising_v4() -> None:
    preset = get_thesis_preset("best", 6)
    assert preset.name == "ising_v4"
    assert preset.config.hamiltonian_mass_strength > 0.0
    assert preset.config.cardinal_mass_strength > 0.0
    assert preset.config.hamiltonian_phase_strength > 2.0


def test_hamiltonian_contrast_produces_nonflat_probabilities() -> None:
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    H, _ = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    state = core.random_state(batch=1, seed=0)
    cfg = ThesisDynamicsConfig(
        n_qubits=3,
        hamiltonian_mass_strength=1.0,
        cardinal_mass_strength=1.0,
        atlas_strength=0.8,
    )
    proj = thesis_project(core, state, H, cfg, dt=0.1)

    assert proj.mass_contrast is not None
    assert float(proj.mass_contrast.std().detach()) > 0.0
    assert float(proj.probabilities.std().detach()) > 0.0
    assert torch.allclose(proj.probabilities.sum(dim=-1), torch.ones(1), atol=1e-6)
