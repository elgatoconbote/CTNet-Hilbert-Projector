#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXAMPLES_ROOT = REPO_ROOT / "examples"
for p in (str(SRC_ROOT), str(EXAMPLES_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ctnet_hilbert_projector import (  # noqa: E402
    IsingConfig,
    amplitude_l2_error,
    basis_state,
    evolve_exact,
    expectation,
    probability_l1_error,
    transverse_field_ising_matrix,
)
from solve_ising_cubo6d_exact_fixed_point import (  # noqa: E402
    ROOT as CUBO_ROOT,
    cubo6d_exact_fixed_point,
    problem_as_observador,
    read_q_sigma,
)
from ctnet_omega_cubo6d_plegado_ctnet26 import FoldLayout, FoldedCTNetOmegaCubo26  # noqa: E402
from train_vram_up_coherence_ctnet import batch_to_state, all_perspective_up_loss  # noqa: E402


def parse_up_branch(text: str, n: int) -> tuple[str, ...]:
    branch = text.strip().replace(",", "").replace(" ", "")
    if branch == "":
        branch = "u" * n
    if len(branch) != n:
        raise ValueError(f"psi0 branch length {len(branch)} != n={n}")
    if any(ch not in {"u", "p"} for ch in branch):
        raise ValueError("psi0 branch must contain only 'u' and 'p'")
    return tuple(branch)


def unit_global_phase(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-15) -> tuple[torch.Tensor, torch.Tensor]:
    """Return e^{i phi} and phi such that candidate ~= e^{i phi} reference.

    With normalized states and the convention candidate = exp(i phi) * reference,
    the global phase is the argument of <reference|candidate>.
    """
    ref = reference.to(dtype=torch.complex128)
    cand = candidate.to(device=ref.device, dtype=torch.complex128)
    ref = ref / ref.norm().clamp_min(eps)
    cand = cand / cand.norm().clamp_min(eps)
    overlap = torch.vdot(ref, cand)
    eiphi = overlap / overlap.abs().clamp_min(eps)
    phi = torch.angle(eiphi)
    return eiphi, phi


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    two_pi = torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.real.dtype if x.is_complex() else x.dtype)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.real.dtype if x.is_complex() else x.dtype)
    return torch.remainder(x + pi, two_pi) - pi


def main() -> None:
    ap = argparse.ArgumentParser(description="Find the global phase phi using Cubo 6D Q_sigma(Xi_solution)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--psi0", default="", help="Initial u/p branch, e.g. uuuuuu. Default: all-u")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--max-amp", type=float, default=1e-6)
    ap.add_argument("--max-prob", type=float, default=1e-6)
    ap.add_argument("--max-obs", type=float, default=1e-6)
    ap.add_argument("--max-norm", type=float, default=1e-9)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    # Cubo 6D branch: build Xi_solution and read Q_sigma(Xi_solution).
    model = FoldedCTNetOmegaCubo26(layout=FoldLayout()).to(device=device, dtype=dtype)
    obs = problem_as_observador(args.n, args.J, args.h, args.dt, args.steps)
    state0, _, _ = batch_to_state(model, [obs], device=device, dtype=dtype, max_bytes=args.max_bytes)
    xi_solution = cubo6d_exact_fixed_point(model, state0)
    amp_cubo = read_q_sigma(xi_solution, args.n).to(torch.complex128)

    up_loss, up_metrics = all_perspective_up_loss(model, xi_solution, xi_solution)
    final_obs = model.cubo_observation(xi_solution)

    # External judge only: exact Hilbert evolution from the declared psi0.
    H, branches = transverse_field_ising_matrix(
        IsingConfig(n_qubits=args.n, J=args.J, h=args.h),
        device=device,
        dtype=torch.complex128,
    )
    psi0_branch = parse_up_branch(args.psi0, args.n)
    psi0 = basis_state(branches, psi0_branch, device=device, dtype=torch.complex128)
    target = evolve_exact(psi0, H, dt=args.dt, steps=args.steps).to(torch.complex128)

    eiphi, phi = unit_global_phase(target, amp_cubo)
    aligned_target = eiphi * target
    aligned_cubo_to_target = torch.conj(eiphi) * amp_cubo

    amp_l2 = amplitude_l2_error(aligned_target, amp_cubo, phase_invariant=False)
    amp_l2_cubo_to_target = amplitude_l2_error(target, aligned_cubo_to_target, phase_invariant=False)
    prob_l1 = probability_l1_error(target, amp_cubo)
    obs_abs = (expectation(target, H) - expectation(amp_cubo.to(device=device), H)).abs()
    norm_err = (amp_cubo.abs().pow(2).sum() - 1.0).abs()
    overlap = torch.vdot(target / target.norm().clamp_min(1e-15), amp_cubo / amp_cubo.norm().clamp_min(1e-15))
    phase_residue = wrap_to_pi(torch.angle(amp_cubo) - torch.angle(target) - phi)
    max_branch_phase_residue = phase_residue.abs().max()

    certified = (
        float(amp_l2.detach().cpu()) <= args.max_amp
        and float(prob_l1.detach().cpu()) <= args.max_prob
        and float(obs_abs.detach().cpu()) <= args.max_obs
        and float(norm_err.detach().cpu()) <= args.max_norm
    )

    print("=== Cubo 6D global phase finder ===")
    print(f"parent_root={CUBO_ROOT}")
    print("route=Observador->batch_to_state->Cubo6DObserver->contextual_drive->exact_fixed_point_u=p->Q_sigma(Xi_solution)->phi")
    print("solver_uses_exact_dense_evolution=False")
    print("exact_dense_used_only_as_external_judge=True")
    print("global_phase_convention=Q_sigma(Xi_solution)=exp(i*phi)*target_sigma")
    print(f"n={args.n}")
    print(f"J={args.J}")
    print(f"h={args.h}")
    print(f"dt={args.dt}")
    print(f"steps={args.steps}")
    print(f"psi0={''.join(psi0_branch)}")
    print(f"final_omega={float(final_obs['omega'].mean().detach().cpu()):.12g}")
    print(f"final_residual={float(final_obs['residual'].mean().detach().cpu()):.12g}")
    print(f"final_absorption={float(final_obs['absorption'].mean().detach().cpu()):.12g}")
    print(f"final_closure_score={float(final_obs['closure_score'].mean().detach().cpu()):.12g}")
    print(f"u_p_total={float(up_loss.detach().cpu()):.12g}")
    for k, v in sorted(up_metrics.items()):
        print(f"{k}={v:.12g}")
    print(f"overlap_real={float(overlap.real.detach().cpu()):.17g}")
    print(f"overlap_imag={float(overlap.imag.detach().cpu()):.17g}")
    print(f"overlap_abs={float(overlap.abs().detach().cpu()):.17g}")
    print(f"eiphi_real={float(eiphi.real.detach().cpu()):.17g}")
    print(f"eiphi_imag={float(eiphi.imag.detach().cpu()):.17g}")
    print(f"phi_rad={float(phi.detach().cpu()):.17g}")
    print(f"phi_over_pi={float((phi / torch.pi).detach().cpu()):.17g}")
    print(f"amp_l2_after_phase_alignment={float(amp_l2.detach().cpu()):.12g}")
    print(f"amp_l2_cubo_times_exp_minus_i_phi={float(amp_l2_cubo_to_target.detach().cpu()):.12g}")
    print(f"prob_l1={float(prob_l1.detach().cpu()):.12g}")
    print(f"observable_abs={float(obs_abs.detach().cpu()):.12g}")
    print(f"normalization_error={float(norm_err.detach().cpu()):.12g}")
    print(f"max_branch_phase_residue_after_global_phi={float(max_branch_phase_residue.detach().cpu()):.12g}")
    print(f"quantum_exact_certified={certified}")

    if args.save:
        payload = {
            "phi_rad": phi.detach().cpu(),
            "eiphi": eiphi.detach().cpu(),
            "amplitudes_cubo": amp_cubo.detach().cpu(),
            "target": target.detach().cpu(),
            "aligned_target": aligned_target.detach().cpu(),
            "aligned_cubo_to_target": aligned_cubo_to_target.detach().cpu(),
            "overlap": overlap.detach().cpu(),
            "n": args.n,
            "J": args.J,
            "h": args.h,
            "dt": args.dt,
            "steps": args.steps,
            "psi0_branch": "".join(psi0_branch),
            "amp_l2_after_phase_alignment": float(amp_l2.detach().cpu()),
            "prob_l1": float(prob_l1.detach().cpu()),
            "observable_abs": float(obs_abs.detach().cpu()),
            "normalization_error": float(norm_err.detach().cpu()),
            "max_branch_phase_residue_after_global_phi": float(max_branch_phase_residue.detach().cpu()),
            "quantum_exact_certified": certified,
        }
        path = Path(args.save)
        torch.save(payload, path)
        print(f"saved={path}")


if __name__ == "__main__":
    main()
