#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare CTNet u/p projection against an exact Ising baseline.

This demo does not claim the untrained CTNet core already implements Ising time
evolution. It is the first audit harness: project Xi -> amplitudes, evolve those
amplitudes exactly under Ising, run CTNet internal steps, project again, then
measure the discrepancy.
"""

from __future__ import annotations

import argparse

import torch

from ctnet_hilbert_projector import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    HilbertProjectorConfig,
    IsingConfig,
    UPHilbertProjector,
    amplitude_l2_error,
    evolve_exact,
    expectation,
    probability_l1_error,
    transverse_field_ising_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet u/p Ising exact baseline comparison")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    real_dtype = torch.float64 if args.fp64 else torch.float32
    complex_dtype = torch.complex128 if args.fp64 else torch.complex64

    layout = FoldLayout(N=64, d=16, z_tokens=32, z_dim=16, mem_slots=8, mem_dim=16, rel_edges=8, rel_dim=16)
    omega_core = FoldedCTNetOmegaCubo26(layout=layout).to(device=device, dtype=real_dtype)
    projector = UPHilbertProjector(HilbertProjectorConfig(n_qubits=args.n)).to(device=device, dtype=real_dtype)

    H, branches = transverse_field_ising_matrix(
        IsingConfig(n_qubits=args.n, J=args.J, h=args.h, periodic=args.periodic),
        device=device,
        dtype=complex_dtype,
    )

    state0 = omega_core.random_state(batch=1, seed=args.seed, device=device, dtype=real_dtype)
    proj0 = projector.project(omega_core, state0)
    psi0 = proj0.amplitudes[0].to(dtype=complex_dtype)
    psi_exact = evolve_exact(psi0, H, dt=args.dt, steps=args.steps)

    state_ct = state0
    for _ in range(max(0, args.steps)):
        state_ct = omega_core.forward_state(state_ct)
    proj_ct = projector.project(omega_core, state_ct)
    psi_ct = proj_ct.amplitudes[0].to(dtype=complex_dtype)

    amp_err = amplitude_l2_error(psi_exact, psi_ct, phase_invariant=True)
    prob_err = probability_l1_error(psi_exact, psi_ct)
    e0 = expectation(psi0, H).real
    e_exact = expectation(psi_exact, H).real
    e_ct = expectation(psi_ct, H).real

    print("=== CTNet Hilbert Projector / Ising exact audit ===")
    print(f"device={device} dtype={real_dtype}")
    print(f"n={args.n} branches={len(branches)} J={args.J} h={args.h} dt={args.dt} steps={args.steps} periodic={args.periodic}")
    print(f"norm0_error={float(proj0.normalization_error.max().detach().cpu()):.12g}")
    print(f"norm_ct_error={float(proj_ct.normalization_error.max().detach().cpu()):.12g}")
    print(f"amp_l2_error_phase_aligned={float(amp_err.detach().cpu()):.12g}")
    print(f"prob_l1_error={float(prob_err.detach().cpu()):.12g}")
    print(f"energy_initial={float(e0.detach().cpu()):.12g}")
    print(f"energy_exact={float(e_exact.detach().cpu()):.12g}")
    print(f"energy_ct_projection={float(e_ct.detach().cpu()):.12g}")
    print(f"energy_drift_ct_vs_exact={float((e_ct - e_exact).abs().detach().cpu()):.12g}")

    print("\nTop projected probabilities after CTNet step:")
    probs = proj_ct.probabilities[0].detach()
    top = torch.topk(probs, k=min(10, probs.numel()))
    for value, idx in zip(top.values.cpu(), top.indices.cpu()):
        branch = "".join(proj_ct.branches[int(idx)])
        print(f"{branch}  P={float(value):.10f}")


if __name__ == "__main__":
    main()
