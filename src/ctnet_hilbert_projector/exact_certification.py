from __future__ import annotations

from dataclasses import dataclass

import torch

from .hamiltonians import amplitude_l2_error, evolve_exact, expectation, probability_l1_error


@dataclass(frozen=True)
class ExactProjectiveCertificate:
    """Numerical certificate for the exact projective layer.

    This certificate checks the formal CTNet layer used in the theorem:
    a quantum evolution U is first represented as projected amplitudes and then
    lifted exactly into mass-phase branch data. It is intentionally distinct
    from the calibrated structural dynamics, whose role is to approximate a
    concrete CTNet transition regime.
    """

    amp_l2: float
    prob_l1: float
    observable_abs: float
    normalization_error: float
    projective_commutation_error: float
    max_amp_threshold: float
    max_prob_threshold: float
    max_observable_threshold: float
    max_norm_threshold: float
    max_commutation_threshold: float

    @property
    def certified(self) -> bool:
        return (
            self.amp_l2 <= self.max_amp_threshold
            and self.prob_l1 <= self.max_prob_threshold
            and self.observable_abs <= self.max_observable_threshold
            and self.normalization_error <= self.max_norm_threshold
            and self.projective_commutation_error <= self.max_commutation_threshold
        )


def mass_phase_lift(amplitudes: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift amplitudes into CTNet mass-phase form and reconstruct them.

    For a branch family A(sigma), CTNet mass-phase data are

        mu(sigma) = |A(sigma)|^2
        P(sigma) = mu(sigma) / sum_tau mu(tau)
        theta(sigma) = arg A(sigma)
        A_rec(sigma) = sqrt(P(sigma)) exp(i theta(sigma))

    The reconstruction is exact up to floating-point precision when amplitudes
    are normalized.
    """

    if amplitudes.ndim != 1:
        raise ValueError("amplitudes must have shape [branches]")
    mass = amplitudes.abs().pow(2)
    prob = mass / mass.sum().clamp_min(eps)
    phase = torch.angle(amplitudes)
    reconstructed = torch.sqrt(prob).to(amplitudes.dtype) * torch.exp(1j * phase).to(amplitudes.dtype)
    return reconstructed, mass, phase


def certify_exact_projective_evolution(
    initial_amplitudes: torch.Tensor,
    hamiltonian: torch.Tensor,
    *,
    dt: float,
    steps: int,
    max_amp: float = 1e-6,
    max_prob: float = 1e-6,
    max_observable: float = 1e-6,
    max_norm: float = 1e-6,
    max_commutation: float = 1e-6,
) -> ExactProjectiveCertificate:
    """Certify exact CTNet projective representation of a quantum evolution.

    The check has two parts:

    1. Projective commutation in finite basis:
       exp(-i H t) applied to the projected initial amplitudes gives the exact
       target branch family.
    2. Mass-phase exactness:
       the target branch family is lifted to CTNet mass and phase and then
       reconstructed amplitude by amplitude.

    This certifies the exact layer required by the theorem. It does not claim
    that a calibrated structural transition has already reached zero residual.
    """

    if initial_amplitudes.ndim != 1:
        raise ValueError("initial_amplitudes must have shape [branches]")

    total_dt = float(dt) * int(steps)
    U = torch.matrix_exp((-1j * total_dt) * hamiltonian)
    target = evolve_exact(initial_amplitudes, hamiltonian, dt=dt, steps=steps)
    commuting_target = U @ initial_amplitudes
    reconstructed, _, _ = mass_phase_lift(target)

    obs_target = expectation(target, hamiltonian)
    obs_reconstructed = expectation(reconstructed, hamiltonian)

    return ExactProjectiveCertificate(
        amp_l2=float(amplitude_l2_error(target, reconstructed, phase_invariant=False).detach().cpu()),
        prob_l1=float(probability_l1_error(target, reconstructed).detach().cpu()),
        observable_abs=float((obs_target - obs_reconstructed).abs().detach().cpu()),
        normalization_error=float((reconstructed.abs().pow(2).sum() - 1.0).abs().detach().cpu()),
        projective_commutation_error=float(amplitude_l2_error(target, commuting_target, phase_invariant=False).detach().cpu()),
        max_amp_threshold=max_amp,
        max_prob_threshold=max_prob,
        max_observable_threshold=max_observable,
        max_norm_threshold=max_norm,
        max_commutation_threshold=max_commutation,
    )
