#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def find_parent_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if p.name == "CTNet-2.6-Omega-Cubo-6D":
            return p
        if (p / "ctnet_omega_cubo6d_plegado_ctnet26.py").exists() and (p / "train_vram_up_coherence_ctnet.py").exists():
            return p
    raise RuntimeError("Ejecuta desde CTNet-2.6-Omega-Cubo-6D/CTNet-Hilbert-Projector")


ROOT = find_parent_root()
sys.path.insert(0, str(ROOT))

from ctnet_omega_cubo6d_plegado_ctnet26 import FoldLayout, FoldedCTNetOmegaCubo26, FoldedOmegaCuboState
from train_vram_up_coherence_ctnet import Observador, all_perspective_up_loss, batch_to_state


@dataclass
class CuboCandidate:
    name: str
    state: FoldedOmegaCuboState
    omega_eff: torch.Tensor
    omega: torch.Tensor
    residual: torch.Tensor
    absorption: torch.Tensor
    closure_score: torch.Tensor
    up_loss: torch.Tensor
    up_metrics: dict[str, float]


def parse_psi0(text: str, n: int) -> str:
    s = text.strip().replace(",", "").replace(" ", "") or ("u" * n)
    if len(s) != n:
        raise ValueError(f"psi0 length {len(s)} != n={n}")
    if any(ch not in {"u", "p"} for ch in s):
        raise ValueError("psi0 must contain only u/p")
    return s


def problem_as_observador(n: int, J: float, h: float, dt: float, steps: int, psi0: str) -> Observador:
    text = f"""
<problema_cuantico>
solver=Cubo_6D
modo=clausura_estructural
base=u/p
tipo=ising_transversal
n={n}
ramas={2**n}
J={J}
h={h}
dt={dt}
steps={steps}
psi0={psi0}
ruta=Observador->batch_to_state->Cubo6DObserver->closure_shear->forward_state->u=p->Q_sigma
criterio=omega_eff_minimo_y_cierre_up
salida=Q_sigma(Xi_solution)
</problema_cuantico>
""".strip()
    return Observador(
        x=text,
        y=text,
        source="ctnet://quantum/ising/cubo6d_only",
        regime="cubo6d_quantum_solver",
    )


def branch_signatures(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = []
    for i in range(2**n):
        rows.append([1.0 if ((i >> k) & 1) else -1.0 for k in range(n)])
    return torch.tensor(rows, device=device, dtype=dtype)


def take(v: torch.Tensor, k: int) -> torch.Tensor:
    v = v.flatten()
    if v.numel() >= k:
        return v[:k]
    return torch.nn.functional.pad(v, (0, k - v.numel()))


def read_q_sigma(state: FoldedOmegaCuboState, n: int) -> torch.Tensor:
    dtype = torch.float64
    device = state.z.device
    b = branch_signatures(n, device=device, dtype=dtype)

    z = state.z[0].to(dtype)
    m = state.memory[0].to(dtype)
    r = state.relations[0].to(dtype)
    c = state.cubo[0].to(dtype)
    pad = state.pad[0].to(dtype) if state.pad.numel() else torch.zeros(0, device=device, dtype=dtype)

    z_token = z.mean(dim=-1)
    m_token = m.mean(dim=-1)
    r_token = r.mean(dim=-1)

    zv = take(z_token, n)
    mv = take(m_token, n)
    rv = take(r_token, n)
    cv = take(c, n)
    pv = take(pad, n)

    pair_bank = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_bank.append((i, j))

    if pair_bank:
        pair_terms = torch.stack([b[:, i] * b[:, j] for i, j in pair_bank], dim=-1)
        pair_weights = take(r.flatten(), len(pair_bank))
        pair_phase_weights = take(m.flatten(), len(pair_bank))
        pair_mass = pair_terms @ pair_weights
        pair_phase = pair_terms @ pair_phase_weights
    else:
        pair_mass = torch.zeros(2**n, device=device, dtype=dtype)
        pair_phase = torch.zeros(2**n, device=device, dtype=dtype)

    mass_log = b @ (zv + 0.50 * mv + 0.50 * rv + 0.25 * cv + 0.10 * pv) + 0.25 * pair_mass
    mass_log = mass_log - mass_log.max()
    prob = torch.softmax(mass_log, dim=0)

    phase_vec = (
        torch.flip(zv, dims=[0])
        + 0.50 * torch.flip(rv, dims=[0])
        + 0.25 * torch.flip(cv, dims=[0])
        + 0.10 * torch.flip(pv, dims=[0])
    )
    phase = torch.tanh(b @ phase_vec + 0.25 * pair_phase)

    amp = torch.sqrt(prob).to(torch.complex128) * torch.exp(1j * phase.to(torch.complex128))
    return amp / amp.norm().clamp_min(1e-15)


def evaluate_candidate(
    model: FoldedCTNetOmegaCubo26,
    prev: FoldedOmegaCuboState,
    cand: FoldedOmegaCuboState,
    *,
    name: str,
    lambda_up: float,
) -> CuboCandidate:
    obs = model.cubo_observation(cand)
    up_loss, up_metrics = all_perspective_up_loss(model, prev, cand)
    omega = obs["omega"].mean()
    residual = obs["residual"].mean()
    absorption = obs["absorption"].mean()
    closure_score = obs["closure_score"].mean()
    omega_eff = omega + float(lambda_up) * up_loss
    return CuboCandidate(
        name=name,
        state=cand,
        omega_eff=omega_eff,
        omega=omega,
        residual=residual,
        absorption=absorption,
        closure_score=closure_score,
        up_loss=up_loss,
        up_metrics=up_metrics,
    )


def cubo_step_candidates(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState) -> list[tuple[str, FoldedOmegaCuboState]]:
    return [
        ("forward_state", model.forward_state(state)),
        ("closure_shear_plus", model.closure_shear(state, sign=+1.0)),
        ("closure_shear_minus", model.closure_shear(state, sign=-1.0)),
        ("forward_after_shear_plus", model.forward_state(model.closure_shear(state, sign=+1.0))),
        ("forward_after_shear_minus", model.forward_state(model.closure_shear(state, sign=-1.0))),
    ]


def solve_with_cubo(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    *,
    closure_steps: int,
    lambda_up: float,
    tau: float,
) -> tuple[FoldedOmegaCuboState, list[CuboCandidate]]:
    history: list[CuboCandidate] = []
    current = state

    for _ in range(max(1, closure_steps)):
        candidates = [
            evaluate_candidate(model, current, cand, name=name, lambda_up=lambda_up)
            for name, cand in cubo_step_candidates(model, current)
        ]
        chosen = min(candidates, key=lambda c: float(c.omega_eff.detach().cpu()))
        history.append(chosen)
        current = chosen.state
        if float(chosen.omega_eff.detach().cpu()) <= tau:
            break

    return current, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Cubo 6D only solver for an Ising u/p chart")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--psi0", default="")
    ap.add_argument("--closure-steps", type=int, default=8)
    ap.add_argument("--lambda-up", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32
    psi0 = parse_psi0(args.psi0, args.n)

    model = FoldedCTNetOmegaCubo26(layout=FoldLayout()).to(device=device, dtype=dtype)
    obs = problem_as_observador(args.n, args.J, args.h, args.dt, args.steps, psi0)
    state0, _, _ = batch_to_state(model, [obs], device=device, dtype=dtype, max_bytes=args.max_bytes)

    solution, history = solve_with_cubo(
        model,
        state0,
        closure_steps=args.closure_steps,
        lambda_up=args.lambda_up,
        tau=args.tau,
    )

    final_obs = model.cubo_observation(solution)
    final_up, final_up_metrics = all_perspective_up_loss(model, state0, solution)
    amp = read_q_sigma(solution, args.n)
    norm_error = (amp.abs().pow(2).sum() - 1.0).abs()

    print("=== Cubo 6D only quantum chart solver ===")
    print(f"parent_root={ROOT}")
    print("solver=Cubo_6D")
    print("route=Observador->batch_to_state->Cubo6DObserver->closure_shear/forward_state->u=p->Q_sigma(Xi_solution)")
    print(f"n={args.n} branches={2**args.n} J={args.J} h={args.h} dt={args.dt} steps={args.steps} psi0={psi0}")
    print(f"closure_steps_used={len(history)}")

    for i, item in enumerate(history):
        print(
            f"closure[{i}] candidate={item.name} "
            f"omega_eff={float(item.omega_eff.detach().cpu()):.12g} "
            f"omega={float(item.omega.detach().cpu()):.12g} "
            f"residual={float(item.residual.detach().cpu()):.12g} "
            f"absorption={float(item.absorption.detach().cpu()):.12g} "
            f"closure_score={float(item.closure_score.detach().cpu()):.12g} "
            f"up_loss={float(item.up_loss.detach().cpu()):.12g}"
        )

    print(f"final_omega={float(final_obs['omega'].mean().detach().cpu()):.12g}")
    print(f"final_residual={float(final_obs['residual'].mean().detach().cpu()):.12g}")
    print(f"final_absorption={float(final_obs['absorption'].mean().detach().cpu()):.12g}")
    print(f"final_closure_score={float(final_obs['closure_score'].mean().detach().cpu()):.12g}")
    print(f"final_up_total={float(final_up.detach().cpu()):.12g}")

    for k, v in sorted(final_up_metrics.items()):
        print(f"{k}={v:.12g}")

    print(f"amplitude_count={amp.numel()}")
    print(f"normalization_error={float(norm_error.detach().cpu()):.12g}")
    print("first_amplitudes:")

    for i, a in enumerate(amp[: min(16, amp.numel())].detach().cpu()):
        print(f"{i:04d} real={float(a.real): .12g} imag={float(a.imag): .12g} prob={float(abs(a)**2):.12g}")

    if args.save:
        path = Path(args.save)
        torch.save(
            {
                "amplitudes": amp.detach().cpu(),
                "n": args.n,
                "J": args.J,
                "h": args.h,
                "dt": args.dt,
                "steps": args.steps,
                "psi0": psi0,
                "solver": "Cubo_6D_only",
                "normalization_error": float(norm_error.detach().cpu()),
                "final_omega": float(final_obs["omega"].mean().detach().cpu()),
                "final_up_total": float(final_up.detach().cpu()),
            },
            path,
        )
        print(f"saved={path}")


if __name__ == "__main__":
    main()
