from __future__ import annotations

from argparse import Namespace

import torch

from ctnet_hilbert_projector import FoldLayout, FoldedCTNetOmegaCubo26, IsingConfig, evolve_exact, probability_l1_error, transverse_field_ising_matrix
from ctnet_hilbert_projector.presets import get_thesis_preset
from ctnet_hilbert_projector.state_preparation import prepare_quantum_atlas_state
from ctnet_hilbert_projector.thesis_dynamics import thesis_project, thesis_quantum_step
from ctnet_hilbert_projector.hamiltonians import amplitude_l2_error


def test_raw_structural_transition_is_not_marked_exact_by_admissibility() -> None:
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16))
    H, branches = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    preset = get_thesis_preset("best", 3)
    state = core.random_state(batch=1, seed=0)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=preset.prep_memory,
        relation_strength=preset.prep_relation,
    )
    p0 = thesis_project(core, state, H, preset.config, dt=0.0)
    exact = evolve_exact(p0.amplitudes[0], H, dt=0.05, steps=2)

    current = state
    result = None
    for _ in range(2):
        result = thesis_quantum_step(core, current, H, preset.config, dt=0.05)
        current = result.next_state

    amp = float(amplitude_l2_error(exact, result.projection.amplitudes[0], phase_invariant=True).detach())
    prob = float(probability_l1_error(exact, result.projection.amplitudes[0]).detach())

    assert amp > 1e-6
    assert prob > 1e-6
    assert amp < 0.50
    assert prob < 0.50
