#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace

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
from ctnet_hilbert_projector.thesis_dynamics import ThesisDynamicsConfig, thesis_project, thesis_quantum_step


@dataclass(frozen=True)
class Regime:
    prep_memory: float
    prep_relation: float
    beta: float
    gamma: float
    state_strength: float
    phase_strength: float
    hamiltonian_mass: float
    cardinal_mass: float
    mass_feedback: float
    atlas: float

    @staticmethod
    def from_config(prep_memory: float, prep_relation: float, cfg: ThesisDynamicsConfig) -> "Regime":
        return Regime(
            prep_memory=prep_memory,
            prep_relation=prep_relation,
            beta=cfg.beta_coherence,
            gamma=cfg.gamma_residue,
            state_strength=cfg.hamiltonian_state_strength,
            phase_strength=cfg.hamiltonian_phase_strength,
            hamiltonian_mass=cfg.hamiltonian_mass_strength,
            cardinal_mass=cfg.cardinal_mass_strength,
            mass_feedback=cfg.mass_feedback_strength,
            atlas=cfg.atlas_strength,
        )

    def config(self, n_qubits: int) -> ThesisDynamicsConfig:
        return ThesisDynamicsConfig(
            n_qubits=n_qubits,
            beta_coherence=self.beta,
            gamma_residue=self.gamma,
            hamiltonian_state_strength=self.state_strength,
            hamiltonian_phase_strength=self.phase_strength,
            hamiltonian_mass_strength=self.hamiltonian_mass,
            cardinal_mass_strength=self.cardinal_mass,
            mass_feedback_strength=self.mass_feedback,
            atlas_strength=self.atlas,
            cocycle_strength=0.25,
        )


def top_spread(probabilities: torch.Tensor, k: int) -> torch.Tensor:
    top = torch.topk(probabilities.detach(), k=min(k, probabilities.shape[-1])).values
    return top.max() - top.min()


def clamp_regime(r: Regime) -> Regime:
    return Regime(
        prep_memory=max(0.01, min(2.0, r.prep_memory)),
        prep_relation=max(0.01, min(2.0, r.prep_relation)),
        beta=max(0.05, min(5.0, r.beta)),
        gamma=max(0.05, min(3.0, r.gamma)),
        state_strength=max(0.001, min(0.3, r.state_strength)),
        phase_strength=max(0.05, min(3.0, r.phase_strength)),
        hamiltonian_mass=max(0.0, min(3.0, r.hamiltonian_mass)),
        cardinal_mass=max(0.0, min(3.0, r.cardinal_mass)),
        mass_feedback=max(0.001, min(0.3, r.mass_feedback)),
        atlas=max(0.01, min(2.0, r.atlas)),
    )


def mutate(r: Regime, sigma: float, rng: random.Random) -> Regime:
    def m(x: float, rel: float = 1.0) -> float:
        return x + rng.gauss(0.0, sigma * rel * max(abs(x), 0.1))

    return clamp_regime(
        Regime(
            prep_memory=m(r.prep_memory),
            prep_relation=m(r.prep_relation),
            beta=m(r.beta),
            gamma=m(r.gamma),
            state_strength=m(r.state_strength, 0.5),
            phase_strength=m(r.phase_strength),
            hamiltonian_mass=m(r.hamiltonian_mass),
            cardinal_mass=m(r.cardinal_mass),
            mass_feedback=m(r.mass_feedback, 0.5),
            atlas=m(r.atlas),
        )
    )


def evaluate(args: argparse.Namespace, core: FoldedCTNetOmegaCubo26, H: torch.Tensor, branches, regime: Regime, device: torch.device) -> dict[str, float]:
    cfg = regime.config(args.n)
    state = core.random_state(batch=1, seed=args.seed, device=device)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=regime.prep_memory,
        relation_strength=regime.prep_relation,
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
    spread = top_spread(proj.probabilities[0], args.topk)
    exact_spread = top_spread(exact.abs().pow(2), args.topk)
    spread_gap = (spread - exact_spread).abs()
    mass_contrast_std = proj.mass_contrast.std().detach() if proj.mass_contrast is not None else torch.zeros((), device=device)
    loss = amp + args.prob_weight * prob + args.spread_gap_weight * spread_gap - args.mass_contrast_weight * mass_contrast_std
    return {
        "loss": float(loss.detach().cpu()),
        "amp_l2": float(amp.detach().cpu()),
        "prob_l1": float(prob.detach().cpu()),
        "top_spread": float(spread.detach().cpu()),
        "top_spread_exact": float(exact_spread.detach().cpu()),
        "spread_gap": float(spread_gap.detach().cpu()),
        "coh_mean": float(proj.coherence.mean().detach().cpu()),
        "coh_std": float(proj.coherence.std().detach().cpu()),
        "mem_std": float(proj.memory.std().detach().cpu()),
        "rel_std": float(proj.relation.std().detach().cpu()),
        "atlas_std": float(proj.atlas_gauge.std().detach().cpu()) if proj.atlas_gauge is not None else 0.0,
        "mass_contrast_std": float(mass_contrast_std.detach().cpu()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Local optimizer for CTNet thesis Ising regimes")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--preset", default="ising_v3")
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--anneal", type=float, default=0.985)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--prob-weight", type=float, default=0.25)
    p.add_argument("--spread-gap-weight", type=float, default=1.0)
    p.add_argument("--mass-contrast-weight", type=float, default=0.0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=64, d=16)).to(device)
    H, branches = transverse_field_ising_matrix(IsingConfig(args.n, args.J, args.h), device=device)
    preset = get_thesis_preset(args.preset, args.n)
    best_regime = Regime.from_config(preset.prep_memory, preset.prep_relation, preset.config)
    best_metrics = evaluate(args, core, H, branches, best_regime, device)
    print(f"START regime={best_regime} metrics={best_metrics}")

    sigma = args.sigma
    for i in range(1, args.iters + 1):
        cand = mutate(best_regime, sigma, rng)
        metrics = evaluate(args, core, H, branches, cand, device)
        if metrics["loss"] < best_metrics["loss"]:
            best_regime = cand
            best_metrics = metrics
            print(f"NEW_BEST iter={i} sigma={sigma:.6g} regime={best_regime} metrics={best_metrics}")
        sigma *= args.anneal

    print("=== BEST ===")
    print(best_regime)
    print(best_metrics)
    print("\nDemo command:")
    print(
        f'"$PY" examples/demo_thesis_ising.py --n {args.n} --steps {args.steps} --dt {args.dt} '
        f'--J {args.J} --h {args.h} '
        f'{"--cuda " if device.type == "cuda" else ""}'
        f'--prep-memory {best_regime.prep_memory:.8g} --prep-relation {best_regime.prep_relation:.8g} '
        f'--atlas {best_regime.atlas:.8g} --mass-feedback {best_regime.mass_feedback:.8g} '
        f'--phase-strength {best_regime.phase_strength:.8g} --state-strength {best_regime.state_strength:.8g} '
        f'--hamiltonian-mass {best_regime.hamiltonian_mass:.8g} --cardinal-mass {best_regime.cardinal_mass:.8g} '
        f'--beta {best_regime.beta:.8g} --gamma {best_regime.gamma:.8g}'
    )


if __name__ == "__main__":
    main()
