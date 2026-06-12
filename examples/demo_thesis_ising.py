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
from ctnet_hilbert_projector.presets import get_thesis_preset
from ctnet_hilbert_projector.state_preparation import prepare_quantum_atlas_state
from ctnet_hilbert_projector.thesis_dynamics import (
    ThesisDynamicsConfig,
    cocycle_nonseparability_score,
    thesis_project,
    thesis_quantum_step,
)


def top_spread(probabilities: torch.Tensor, k: int = 10) -> torch.Tensor:
    top = torch.topk(probabilities.detach(), k=min(k, probabilities.shape[-1])).values
    return top.max() - top.min()


def main() -> None:
    p = argparse.ArgumentParser(description="CTNet thesis-level Ising demo")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preset", default="", help="calibrated preset, e.g. ising_v1")
    p.add_argument("--state-strength", type=float, default=0.05)
    p.add_argument("--cocycle", type=float, default=0.25)
    p.add_argument("--prep-memory", type=float, default=0.20)
    p.add_argument("--prep-relation", type=float, default=0.25)
    p.add_argument("--atlas", type=float, default=0.20)
    p.add_argument("--mass-feedback", type=float, default=0.05)
    p.add_argument("--phase-strength", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16)).to(device)
    H, branches = transverse_field_ising_matrix(IsingConfig(args.n, args.J, args.h), device=device)

    prep_memory = args.prep_memory
    prep_relation = args.prep_relation
    preset_name = "manual"
    if args.preset:
        preset = get_thesis_preset(args.preset, args.n)
        cfg = preset.config
        prep_memory = preset.prep_memory
        prep_relation = preset.prep_relation
        preset_name = preset.name
    else:
        cfg = ThesisDynamicsConfig(
            n_qubits=args.n,
            beta_coherence=args.beta,
            gamma_residue=args.gamma,
            hamiltonian_state_strength=args.state_strength,
            hamiltonian_phase_strength=args.phase_strength,
            mass_feedback_strength=args.mass_feedback,
            atlas_strength=args.atlas,
            cocycle_strength=args.cocycle,
        )

    state = core.random_state(batch=1, seed=args.seed, device=device)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=prep_memory,
        relation_strength=prep_relation,
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
    exact_prob = exact.abs().pow(2)
    ct_prob = proj.probabilities[0]

    print("=== CTNet thesis-level Ising ===")
    print(f"device={device} n={args.n} branches={2 ** args.n} steps={args.steps} dt={args.dt}")
    print(f"preset={preset_name}")
    print(f"prep_memory={prep_memory} prep_relation={prep_relation}")
    print(f"beta={cfg.beta_coherence} gamma={cfg.gamma_residue} atlas={cfg.atlas_strength}")
    print(f"phase_strength={cfg.hamiltonian_phase_strength} state_strength={cfg.hamiltonian_state_strength} mass_feedback={cfg.mass_feedback_strength}")
    print(f"normalization_error={float(proj.normalization_error.max().detach().cpu()):.12g}")
    print(f"amp_l2_error={float(amplitude_l2_error(exact, psi).detach().cpu()):.12g}")
    print(f"prob_l1_error={float(probability_l1_error(exact, psi).detach().cpu()):.12g}")
    print(f"top_spread_ct={float(top_spread(ct_prob).detach().cpu()):.12g}")
    print(f"top_spread_exact={float(top_spread(exact_prob).detach().cpu()):.12g}")
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
