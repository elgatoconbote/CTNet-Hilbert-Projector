from __future__ import annotations

import torch

from ctnet_hilbert_projector import (
    IsingConfig,
    certify_exact_projective_evolution,
    mass_phase_lift,
    probability_l1_error,
    transverse_field_ising_matrix,
)


def test_mass_phase_lift_reconstructs_normalized_state() -> None:
    psi = torch.tensor([0.5 + 0.0j, 0.0 + 0.5j, -0.5 + 0.0j, 0.0 - 0.5j], dtype=torch.complex64)
    reconstructed, mass, phase = mass_phase_lift(psi)

    assert torch.allclose(reconstructed, psi, atol=1e-6)
    assert torch.allclose(mass.sum(), torch.tensor(1.0), atol=1e-6)
    assert phase.shape == psi.shape


def test_exact_projective_certificate_passes_for_ising_evolution() -> None:
    H, _ = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    psi0 = torch.zeros(8, dtype=torch.complex64)
    psi0[2] = 1.0 + 0.0j

    cert = certify_exact_projective_evolution(psi0, H, dt=0.05, steps=3)

    assert cert.certified
    assert cert.amp_l2 <= 1e-6
    assert cert.prob_l1 <= 1e-6
    assert cert.observable_abs <= 1e-6
    assert cert.normalization_error <= 1e-6
    assert cert.projective_commutation_error <= 1e-6


def test_mass_phase_lift_preserves_probabilities() -> None:
    psi = torch.tensor([0.1 + 0.2j, -0.3 + 0.4j, 0.5 - 0.1j, 0.2 - 0.6j], dtype=torch.complex64)
    psi = psi / psi.norm()
    reconstructed, _, _ = mass_phase_lift(psi)

    assert float(probability_l1_error(psi, reconstructed).detach()) <= 1e-6
