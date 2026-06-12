#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

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
from ctnet_hilbert_projector.thesis_dynamics import ThesisDynamicsConfig, thesis_project, thesis_quantum_step


@dataclass(frozen=True)
class Candidate:
    prep_memory: float
    prep_relation: float
    atlas_strength: float
    mass_feedback: float
    phase_strength: float
    state_strength: float
    hamiltonian_mass: float
    cardinal_mass: float
    beta: float
    gamma: float


def parse_grid(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def evaluate(args: argparse.Namespace, cand: Candidate, device: torch.device) -> dict[str, float]:
    torch.manual_seed(args.seed)
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16)).to(device)
    H, branches = transverse_field_ising_matrix(IsingConfig(args.n, args.J, args.h), device=device)
    cfg = ThesisDynamicsConfig(
        n_qubits=args.n,
        beta_coherence=cand.beta,
        gamma_residue=cand.gamma,
        hamiltonian_state_strength=cand.state_strength,
        hamiltonian_phase_strength=cand.phase_strength,
        hamiltonian_mass_strength=cand.hamiltonian_mass,
        cardinal_mass_strength=cand.cardinal_mass,
        mass_feedback_strength=cand.mass_feedback,
        atlas_strength=cand.atlas_strength,
    )
    state = core.random_state(batch=1, seed=args.seed, device=device)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=cand.prep_memory,
        relation_strength=cand.prep_relation,
    )
    p0 = thesis_project(core, state, H, cfg, dt=0.0)
    exact = evolve_exact(p0.amplitudes[0], H, dt=args.dt, steps=args.steps)
    current = state
    result = None
    for _ in range(max(1, args.steps)):
        result = thesis_quantum_step(core, current, H, cfg, dt=args.dt)
        current = result.next_state
    proj = result.projection
    amp = amplitude_l2_error(exact, proj.amplitudes[0], phase_invariant=True)
    prob = probability_l1_error(exact, proj.amplitudes[0])
    top = torch.topk(proj.probabilities[0].detach(), k=min(args.topk, proj.probabilities.shape[-1])).values
    exact_prob = exact.abs().pow(2)
    exact_top = torch.topk(exact_prob.detach(), k=min(args.topk, exact_prob.shape[-1])).values
    spread = (top.max() - top.min()).detach()
    exact_spread = (exact_top.max() - exact_top.min()).detach()
    spread_gap = (spread - exact_spread).abs()
    mass_contrast_std = proj.mass_contrast.std().detach() if proj.mass_contrast is not None else torch.zeros((), device=device)
    loss = amp + args.prob_weight * prob + args.spread_gap_weight * spread_gap - args.spread_weight * spread
    return {
        "loss": float(loss.detach().cpu()),
        "amp_l2": float(amp.detach().cpu()),
        "prob_l1": float(prob.detach().cpu()),
        "top_spread": float(spread.cpu()),
        "top_spread_exact": float(exact_spread.cpu()),
        "spread_gap": float(spread_gap.cpu()),
        "coh_mean": float(proj.coherence.mean().detach().cpu()),
        "coh_std": float(proj.coherence.std().detach().cpu()),
        "mem_std": float(proj.memory.std().detach().cpu()),
        "rel_std": float(proj.relation.std().detach().cpu()),
        "atlas_std": float(proj.atlas_gauge.std().detach().cpu()) if proj.atlas_gauge is not None else 0.0,
        "mass_contrast_std": float(mass_contrast_std.cpu()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-calibrate CTNet thesis Ising regime")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--prob-weight", type=float, default=0.25)
    p.add_argument("--spread-weight", type=float, default=0.0)
    p.add_argument("--spread-gap-weight", type=float, default=1.0)
    p.add_argument("--prep-memory", default="0.2,0.4,0.8")
    p.add_argument("--prep-relation", default="0.25,0.6,1.0")
    p.add_argument("--atlas", default="0.1,0.2,0.4")
    p.add_argument("--mass-feedback", default="0.02,0.05,0.1")
    p.add_argument("--phase-strength", default="0.5,1.0,1.5")
    p.add_argument("--state-strength", default="0.02,0.05,0.1")
    p.add_argument("--hamiltonian-mass", default="0.0,0.5,1.0")
    p.add_argument("--cardinal-mass", default="0.0,0.25,0.5")
    p.add_argument("--beta", default="0.5,1.0,1.5")
    p.add_argument("--gamma", default="0.5,1.0")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    grids = [
        parse_grid(args.prep_memory),
        parse_grid(args.prep_relation),
        parse_grid(args.atlas),
        parse_grid(args.mass_feedback),
        parse_grid(args.phase_strength),
        parse_grid(args.state_strength),
        parse_grid(args.hamiltonian_mass),
        parse_grid(args.cardinal_mass),
        parse_grid(args.beta),
        parse_grid(args.gamma),
    ]
    best: tuple[float, Candidate, dict[str, float]] | None = None
    checked = 0
    for vals in itertools.product(*grids):
        cand = Candidate(*vals)
        metrics = evaluate(args, cand, device)
        checked += 1
        if best is None or metrics["loss"] < best[0]:
            best = (metrics["loss"], cand, metrics)
            print(f"NEW_BEST checked={checked} cand={cand} metrics={metrics}")
        if args.limit and checked >= args.limit:
            break
    print("=== BEST ===")
    print(f"checked={checked}")
    print(best[1])
    print(best[2])


if __name__ == "__main__":
    main()
