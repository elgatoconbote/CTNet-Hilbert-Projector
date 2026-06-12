#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal CTNet Hilbert Projector demo.

Example:
    python examples/demo_small_projector.py --n 3 --steps 1
"""

from __future__ import annotations

import argparse

import torch

from ctnet_hilbert_projector import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    HilbertProjectorConfig,
    UPPauli,
    UPHilbertProjector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet u/p Hilbert Projector demo")
    parser.add_argument("--n", type=int, default=3, help="number of u/p degrees; keep small for full enumeration")
    parser.add_argument("--steps", type=int, default=1, help="CTNet forward steps before projection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    layout = FoldLayout(N=64, d=16, z_tokens=32, z_dim=16, mem_slots=8, mem_dim=16, rel_edges=8, rel_dim=16)
    omega_core = FoldedCTNetOmegaCubo26(layout=layout).to(device)
    projector = UPHilbertProjector(HilbertProjectorConfig(n_qubits=args.n)).to(device)

    state = omega_core.random_state(batch=1, seed=args.seed, device=device)
    for _ in range(max(0, args.steps)):
        state = omega_core.forward_state(state)

    projection = projector.project(omega_core, state)
    print("=== CTNet Hilbert Projector / u-p demo ===")
    print(f"device={device}")
    print(f"n={args.n} branches={len(projection.branches)}")
    print(f"normalization_error={float(projection.normalization_error.max().detach().cpu()):.12g}")
    if args.n >= 1:
        z0 = UPPauli.z_expectation(projection, 0)
        print(f"<Z_0>_up={float(z0.detach().cpu()):.12g}")
    if args.n >= 2:
        zz01 = UPPauli.zz_correlation(projection, 0, 1)
        print(f"corr(Z_0,Z_1)_up={float(zz01.detach().cpu()):.12g}")

    print("\nbranch | amplitude | prob | mass | phase")
    for row in projection.as_branch_table(batch_index=0):
        branch = "".join(row.branch)
        print(
            f"{branch:>{args.n}} | "
            f"{row.amplitude.real:+.6f}{row.amplitude.imag:+.6f}j | "
            f"{row.probability:.8f} | {row.mass:.8f} | {row.phase:+.6f}"
        )


if __name__ == "__main__":
    main()
