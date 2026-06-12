from __future__ import annotations

import torch

from ctnet_hilbert_projector import (
    IsingConfig,
    amplitude_l2_error,
    basis_state,
    evolve_exact,
    expectation,
    probability_l1_error,
    transverse_field_ising_matrix,
)


def test_transverse_field_ising_matrix_shape_and_hermitian() -> None:
    H, branches = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    assert H.shape == (8, 8)
    assert len(branches) == 8
    assert torch.allclose(H, H.conj().T, atol=1e-6)


def test_ising_diagonal_sector_without_field() -> None:
    H, branches = transverse_field_ising_matrix(IsingConfig(n_qubits=2, J=1.0, h=0.0))
    uu = basis_state(branches, ("u", "u"))
    up = basis_state(branches, ("u", "p"))
    e_uu = expectation(uu, H).real
    e_up = expectation(up, H).real
    assert torch.allclose(e_uu, torch.tensor(-1.0), atol=1e-6)
    assert torch.allclose(e_up, torch.tensor(1.0), atol=1e-6)


def test_exact_evolution_preserves_norm() -> None:
    H, branches = transverse_field_ising_matrix(IsingConfig(n_qubits=3, J=1.0, h=0.5))
    psi0 = basis_state(branches, ("u", "u", "u"))
    psi1 = evolve_exact(psi0, H, dt=0.1, steps=3)
    assert torch.allclose(psi1.norm(), torch.tensor(1.0), atol=1e-5)
    assert float(probability_l1_error(psi1, psi1)) == 0.0
    assert float(amplitude_l2_error(psi1, psi1)) == 0.0
