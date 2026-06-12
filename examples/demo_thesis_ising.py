#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch

from ctnet_hilbert_projector import (
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    IsingConfig,
    amplitude_l2_error,
    evolve_exact,
    probability_l1_error,
    transverse_field_ising_matrix,
)
from ctnet_hilbert_projector.state_preparation import prepare_quantum_atlas_state
from ctnet_hilbert_projector.thesis_dynamics import (
    ThesisDynamicsConfig,
    cocycle_nonseparability_score,
    thesis_project,
    thesis_quantum_step,
)


def main() -> None:
    p = argparse.ArgumentParser(description="CTNet thesis-level Ising demo")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-strength", type=float, default=0.05)
    p.add_argument("--cocycle", type=float, default=0.25)
    p.add_argument("--prep-memory", type=float, default=0.20)
    p.add_argument("--prep-relation", type=float, default=0.25)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16)).to(device)
    H, branches = transverse_field_ising_matrix(IsingConfig(args.n, args.J, args.h), device=device)

    cfg = ThesisDynamicsConfig(
        n_qubits=args.n,
        hamiltonian_state_strength=args.state_strength,
        cocycle_strength=args.cocycle,
    )

    state = core.random_state(batch=1, seed=args.seed, device=device)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=args.prep_memory,
        relation_strength=args.prep_relation,
    )
    p0 = thesis_project(core, state, H, cfg, dt=0.0)
    exact = evolve_exact(p0.amplitudes[0], H, dt=args.dt, steps=args.steps)

    current = state
    result = None
    for _ in range(max(1, args.steps)):
        result = thesis_quantum_step(core, current, H, cfg, dt=args.dt)
        current = result.next_state

    proj = result.projection
    psi = proj.amplitudes[0]

    print("=== CTNet thesis-level Ising ===")
    print(f"device={device} n={args.n} branches={2 ** args.n} steps={args.steps} dt={args.dt}")
    print(f"prep_memory={args.prep_memory} prep_relation={args.prep_relation}")
    print(f"normalization_error={float(proj.normalization_error.max().detach().cpu()):.12g}")
    print(f"amp_l2_error={float(amplitude_l2_error(exact, psi).detach().cpu()):.12g}")
    print(f"prob_l1_error={float(probability_l1_error(exact, psi).detach().cpu()):.12g}")
    print(f"coherence_mean={float(proj.coherence.mean().detach().cpu()):.12g}")
    print(f"coherence_std={float(proj.coherence.std().detach().cpu()):.12g}")
    print(f"residue_mean={float(proj.residue.mean().detach().cpu()):.12g}")
    print(f"memory_mean={float(proj.memory.mean().detach().cpu()):.12g}")
    print(f"memory_std={float(proj.memory.std().detach().cpu()):.12g}")
    print(f"relation_mean={float(proj.relation.mean().detach().cpu()):.12g}")
    print(f"relation_std={float(proj.relation.std().detach().cpu()):.12g}")
    print(f"cocycle_nonseparability={float(cocycle_nonseparability_score(proj).max().detach().cpu()):.12g}")
    if proj.atlas_gauge is not None:
        print(f"atlas_std={float(proj.atlas_gauge.std().detach().cpu()):.12g}")

    top = torch.topk(proj.probabilities[0].detach(), k=min(10, proj.probabilities.shape[-1]))
    print("\nTop projected probabilities:")
    for value, idx in zip(top.values.cpu(), top.indices.cpu()):
        branch = "".join(proj.branches[int(idx)])
        print(f"{branch}  P={float(value):.10f}")


if __name__ == "__main__":
    main()
