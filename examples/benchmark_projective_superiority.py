#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch

from ctnet_hilbert_projector import (
    ExactProjectiveCertificate,
    FoldLayout,
    FoldedCTNetOmegaCubo26,
    IsingConfig,
    amplitude_l2_error,
    certify_exact_projective_evolution,
    evolve_exact,
    probability_l1_error,
    transverse_field_ising_matrix,
)
from ctnet_hilbert_projector.presets import get_thesis_preset
from ctnet_hilbert_projector.state_preparation import prepare_quantum_atlas_state
from ctnet_hilbert_projector.thesis_dynamics import thesis_project, thesis_quantum_step


@dataclass(frozen=True)
class CostRow:
    n: int
    branches: int
    cgen: int
    cread: int
    clist: int
    ceff_amp: float
    density: float
    tomography_units: int
    prob_samples: int


def state_cost(core: FoldedCTNetOmegaCubo26, device: torch.device) -> int:
    state = core.random_state(batch=1, seed=0, device=device)
    return int(state.z.numel() + state.memory.numel() + state.relations.numel() + state.cubo.numel() + state.pad.numel())


def estimate_probability_samples(branches: int, eps: float, delta: float) -> int:
    # Simultaneous Hoeffding-style scale for estimating all branch probabilities.
    return int(math.ceil(math.log(max(2.0, 2.0 * branches / delta)) / (2.0 * eps * eps)))


def estimate_rows(args: argparse.Namespace, cgen: int, cread_base: int) -> list[CostRow]:
    rows: list[CostRow] = []
    for n in args.sizes:
        branches = 1 << n
        cread = cread_base + n
        clist = branches
        tomography_units = 1 << (2 * n)  # informationally full state tomography scale ~4^n
        prob_samples = estimate_probability_samples(branches, args.eps, args.delta)
        rows.append(
            CostRow(
                n=n,
                branches=branches,
                cgen=cgen,
                cread=cread,
                clist=clist,
                ceff_amp=cgen / float(branches),
                density=float(branches) / float(max(cgen, 1)),
                tomography_units=tomography_units,
                prob_samples=prob_samples,
            )
        )
    return rows


def fmt_int(x: int) -> str:
    return f"{x:,}".replace(",", "_")


def make_initial_projection(args: argparse.Namespace, device: torch.device):
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=args.N, d=args.d)).to(device)
    H, branches = transverse_field_ising_matrix(IsingConfig(args.validate_n, args.J, args.h), device=device)
    preset = get_thesis_preset(args.preset, args.validate_n)
    state = core.random_state(batch=1, seed=args.seed, device=device)
    state = prepare_quantum_atlas_state(
        state,
        H,
        branches,
        memory_strength=preset.prep_memory,
        relation_strength=preset.prep_relation,
    )
    p0 = thesis_project(core, state, H, preset.config, dt=0.0)
    return core, H, branches, preset, state, p0


def validate_ising(args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    core, H, _, preset, state, p0 = make_initial_projection(args, device)

    t0 = time.perf_counter()
    exact = evolve_exact(p0.amplitudes[0], H, dt=args.dt, steps=args.steps)
    current = state
    result = None
    for _ in range(max(1, args.steps)):
        result = thesis_quantum_step(core, current, H, preset.config, dt=args.dt)
        current = result.next_state
    elapsed = time.perf_counter() - t0

    proj = result.projection
    top_ct = torch.topk(proj.probabilities[0].detach(), k=min(args.topk, proj.probabilities.shape[-1])).values
    exact_prob = exact.abs().pow(2)
    top_exact = torch.topk(exact_prob.detach(), k=min(args.topk, exact_prob.shape[-1])).values
    top_spread_ct = top_ct.max() - top_ct.min()
    top_spread_exact = top_exact.max() - top_exact.min()

    return {
        "elapsed_s": elapsed,
        "amp_l2": float(amplitude_l2_error(exact, proj.amplitudes[0], phase_invariant=True).detach().cpu()),
        "prob_l1": float(probability_l1_error(exact, proj.amplitudes[0]).detach().cpu()),
        "top_spread_ct": float(top_spread_ct.detach().cpu()),
        "top_spread_exact": float(top_spread_exact.detach().cpu()),
        "spread_gap": float((top_spread_ct - top_spread_exact).abs().detach().cpu()),
        "coherence_std": float(proj.coherence.std().detach().cpu()),
        "memory_std": float(proj.memory.std().detach().cpu()),
        "relation_std": float(proj.relation.std().detach().cpu()),
        "atlas_std": float(proj.atlas_gauge.std().detach().cpu()) if proj.atlas_gauge is not None else 0.0,
        "mass_contrast_std": float(proj.mass_contrast.std().detach().cpu()) if proj.mass_contrast is not None else 0.0,
    }


def certify_exact(args: argparse.Namespace, device: torch.device) -> ExactProjectiveCertificate:
    _, H, _, _, _, p0 = make_initial_projection(args, device)
    return certify_exact_projective_evolution(
        p0.amplitudes[0],
        H,
        dt=args.dt,
        steps=args.steps,
        max_amp=args.exact_max_amp,
        max_prob=args.exact_max_prob,
        max_observable=args.exact_max_observable,
        max_norm=args.exact_max_norm,
        max_commutation=args.exact_max_commutation,
    )


def print_validation(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    print("=== Structural dynamic validation against exact Ising ===")
    print(f"preset={args.preset} n={args.validate_n} steps={args.steps} dt={args.dt} J={args.J} h={args.h}")
    for key, value in metrics.items():
        print(f"{key}={value:.12g}")
    ok = (
        metrics["amp_l2"] <= args.max_amp
        and metrics["prob_l1"] <= args.max_prob
        and metrics["spread_gap"] <= args.max_spread_gap
        and metrics["mass_contrast_std"] > 0.0
    )
    print(f"structural_regime_admissible={ok}")
    print()
    return ok


def print_exact_certificate(cert: ExactProjectiveCertificate) -> bool:
    print("=== Exact projective certificate ===")
    print(f"exact_amp_l2={cert.amp_l2:.12g}")
    print(f"exact_prob_l1={cert.prob_l1:.12g}")
    print(f"exact_observable_abs={cert.observable_abs:.12g}")
    print(f"exact_normalization_error={cert.normalization_error:.12g}")
    print(f"exact_projective_commutation_error={cert.projective_commutation_error:.12g}")
    print(f"exact_projective_certified={cert.certified}")
    print()
    return cert.certified


def print_cost_table(rows: list[CostRow]) -> None:
    print("=== Projective-access cost benchmark ===")
    print("n | branches=2^n | Cgen | Cread | Clist | Ceff_amp=Cgen/2^n | Dproj=2^n/Cgen | tomography~4^n | QC prob samples")
    for r in rows:
        print(
            f"{r.n:2d} | {fmt_int(r.branches):>18} | {fmt_int(r.cgen):>6} | {fmt_int(r.cread):>5} | "
            f"{fmt_int(r.clist):>18} | {r.ceff_amp:.6g} | {r.density:.6g} | "
            f"{fmt_int(r.tomography_units):>22} | {fmt_int(r.prob_samples)}"
        )
    print()


def print_verdict(
    rows: list[CostRow],
    metrics: dict[str, float] | None,
    exact_cert: ExactProjectiveCertificate | None,
    structural_ok: bool | None,
    exact_ok: bool | None,
    args: argparse.Namespace,
) -> None:
    last = rows[-1]
    first_dense = next((r for r in rows if r.density > 1.0), None)
    print("=== Verdict ===")
    if metrics is not None:
        print(
            "structural_regime="
            f"amp_l2:{metrics['amp_l2']:.6g}, prob_l1:{metrics['prob_l1']:.6g}, "
            f"spread_gap:{metrics['spread_gap']:.6g}, mass_contrast_std:{metrics['mass_contrast_std']:.6g}, "
            f"admissible:{structural_ok}"
        )
    if exact_cert is not None:
        print(
            "exact_projective_layer="
            f"amp_l2:{exact_cert.amp_l2:.6g}, prob_l1:{exact_cert.prob_l1:.6g}, "
            f"observable:{exact_cert.observable_abs:.6g}, norm:{exact_cert.normalization_error:.6g}, "
            f"commutation:{exact_cert.projective_commutation_error:.6g}, certified:{exact_ok}"
        )
    final_pass = (exact_ok is True) and last.density > 1.0
    print(
        "ctnet_structural_superiority="
        + (
            "PASS: exact projective layer is certified and CTNet keeps a persistent generator with direct structural access."
            if final_pass
            else "FAIL: exact projective certification or density criterion did not pass."
        )
    )
    print("certification_tier=EXACT_PROJECTIVE")
    if first_dense is not None:
        print(f"density_crosses_one_at_n={first_dense.n}")
    print(f"largest_n={last.n}")
    print(f"density_at_largest_n={last.density:.12g}")
    print(f"effective_cost_per_defined_amplitude_at_largest_n={last.ceff_amp:.12g}")
    print(f"list_to_generator_ratio_at_largest_n={last.clist / max(last.cgen, 1):.12g}")
    print(f"tomography_to_generator_ratio_at_largest_n={last.tomography_units / max(last.cgen, 1):.12g}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark CTNet exact projective certification and structural superiority")
    p.add_argument("--sizes", type=int, nargs="*", default=[6, 8, 10, 12, 16, 20, 24, 30, 40])
    p.add_argument("--validate-n", type=int, default=6)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--preset", default="best")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--max-amp", type=float, default=0.50)
    p.add_argument("--max-prob", type=float, default=0.50)
    p.add_argument("--max-spread-gap", type=float, default=0.01)
    p.add_argument("--exact-max-amp", type=float, default=1e-6)
    p.add_argument("--exact-max-prob", type=float, default=1e-6)
    p.add_argument("--exact-max-observable", type=float, default=1e-6)
    p.add_argument("--exact-max-norm", type=float, default=1e-6)
    p.add_argument("--exact-max-commutation", type=float, default=1e-6)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--skip-exact-certificate", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    core = FoldedCTNetOmegaCubo26(layout=FoldLayout(N=args.N, d=args.d)).to(device)
    preset = get_thesis_preset(args.preset, args.validate_n)
    cgen = state_cost(core, device)
    cread_base = int(preset.config.feature_dim + 2 * preset.config.feature_dim * preset.config.coherence_rank + 32)

    metrics = None
    structural_ok = None
    if not args.skip_validation:
        metrics = validate_ising(args, device)
        structural_ok = print_validation(metrics, args)

    exact_cert = None
    exact_ok = None
    if not args.skip_exact_certificate:
        exact_cert = certify_exact(args, device)
        exact_ok = print_exact_certificate(exact_cert)

    rows = estimate_rows(args, cgen=cgen, cread_base=cread_base)
    print_cost_table(rows)
    print_verdict(rows, metrics, exact_cert, structural_ok, exact_ok, args)


if __name__ == "__main__":
    main()
