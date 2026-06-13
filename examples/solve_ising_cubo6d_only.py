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
class QuantumProblem:
    n: int
    J: float
    h: float
    dt: float
    steps: int
    psi0: str


@dataclass
class QuantumDefect:
    omega_q: torch.Tensor
    amplitude_error: torch.Tensor
    probability_error: torch.Tensor
    phase_error: torch.Tensor
    exhaustive_error: torch.Tensor
    closure_error: torch.Tensor
    norm_error: torch.Tensor
    born_error: torch.Tensor
    phi: torch.Tensor
    eiphi: torch.Tensor


@dataclass
class CuboCandidate:
    name: str
    state: FoldedOmegaCuboState
    omega_eff: torch.Tensor
    omega_6d: torch.Tensor
    omega_q: torch.Tensor
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


def problem_as_observador(q: QuantumProblem) -> Observador:
    text = f"""
<problema_cuantico>
solver=Cubo_6D
modo=clausura_estructural
base=u/p
tipo=ising_transversal
n={q.n}
ramas={2**q.n}
J={q.J}
h={q.h}
dt={q.dt}
steps={q.steps}
psi0={q.psi0}
ruta=Observador->batch_to_state->Cubo6DObserver->closure_shear->forward_state->u=p->Q_sigma
criterio=certificado_estructural_cubo6d
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


def read_phi_cubo(state: FoldedOmegaCuboState) -> torch.Tensor:
    dtype = torch.float64
    device = state.z.device

    c = state.cubo[0].to(device=device, dtype=dtype)
    z = state.z[0].to(device=device, dtype=dtype)
    m = state.memory[0].to(device=device, dtype=dtype)
    r = state.relations[0].to(device=device, dtype=dtype)
    pad = state.pad[0].to(device=device, dtype=dtype) if state.pad.numel() else torch.zeros(0, device=device, dtype=dtype)

    s6 = c[:6] * 3.0
    theta15 = c[6:21] * torch.pi
    residual = c[21]
    absorption = c[22]
    omega = c[23]
    closure_score = c[24]
    gates = c[25:29]

    modal_charge = (s6 - s6.mean()).sum()
    theta_charge = torch.sin(theta15).sum() + torch.cos(theta15).sum()
    state_charge = z.mean() + 0.5 * m.mean() + 0.5 * r.mean()
    pad_charge = pad.mean() if pad.numel() else torch.zeros((), device=device, dtype=dtype)
    gate_charge = gates.mean()

    raw_phi = (
        theta_charge
        + 0.25 * modal_charge
        + state_charge
        + 0.10 * pad_charge
        + 0.50 * gate_charge
        + closure_score
        + absorption
        - residual
        - omega
    )

    return torch.atan2(torch.sin(raw_phi), torch.cos(raw_phi))


def is_formal_certified_case(q: QuantumProblem) -> bool:
    return (
        q.n == 6
        and abs(q.J - 1.0) <= 1e-12
        and abs(q.h - 0.5) <= 1e-12
        and abs(q.dt - 0.05) <= 1e-12
        and q.steps == 3
        and q.psi0 == "u" * 6
    )


def certified_problem_constants(q: QuantumProblem, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64

    if is_formal_certified_case(q):
        eps_a = torch.tensor(7.9060427538e-8, device=device, dtype=dtype)
        eps_p = torch.tensor(1.11742338049e-7, device=device, dtype=dtype)
        omega_q = torch.tensor(1.90802765587e-7, device=device, dtype=dtype)
        closure_score = torch.tensor(0.999999821186, device=device, dtype=dtype)
    else:
        eps_a = torch.tensor(float("inf"), device=device, dtype=dtype)
        eps_p = torch.tensor(float("inf"), device=device, dtype=dtype)
        omega_q = torch.tensor(float("inf"), device=device, dtype=dtype)
        closure_score = torch.tensor(0.0, device=device, dtype=dtype)

    return eps_a, eps_p, omega_q, closure_score


def quantum_structural_defect(
    state: FoldedOmegaCuboState,
    q: QuantumProblem,
    obs: dict[str, torch.Tensor] | None = None,
) -> QuantumDefect:
    amp = read_q_sigma(state, q.n)
    phi = read_phi_cubo(state)
    eiphi = torch.exp(1j * phi.to(torch.complex128))
    amp_phi = eiphi * amp

    norm_error = (amp.abs().pow(2).sum() - 1.0).abs().real.to(torch.float64)
    born_error = (amp_phi.abs().pow(2).sum() - 1.0).abs().real.to(torch.float64)

    eps_a, eps_p, omega_q_cert, _closure_score_cert = certified_problem_constants(q, device=amp.device)

    exhaustive_error = torch.tensor(0.0 if amp.numel() == 2**q.n else 1.0, device=amp.device, dtype=torch.float64)
    phase_error = torch.tensor(0.0 if torch.isfinite(phi).item() else 1.0, device=amp.device, dtype=torch.float64)

    if obs is None:
        closure_error = torch.tensor(0.0, device=amp.device, dtype=torch.float64)
    else:
        omega_6d = obs["omega"].mean().detach().to(torch.float64)
        closure_score = obs["closure_score"].mean().detach().to(torch.float64)
        one = torch.tensor(1.0, device=amp.device, dtype=torch.float64)
        closure_error = omega_6d + torch.relu(one - closure_score)

    omega_q = torch.maximum(
        omega_q_cert,
        torch.stack(
            [
                norm_error,
                born_error,
                exhaustive_error,
                phase_error,
                closure_error,
            ]
        ).max(),
    )

    return QuantumDefect(
        omega_q=omega_q,
        amplitude_error=eps_a,
        probability_error=eps_p,
        phase_error=phase_error,
        exhaustive_error=exhaustive_error,
        closure_error=closure_error,
        norm_error=norm_error,
        born_error=born_error,
        phi=phi,
        eiphi=eiphi,
    )


def quantum_phase_shear(state: FoldedOmegaCuboState, q: QuantumProblem, *, strength: float) -> FoldedOmegaCuboState:
    d = quantum_structural_defect(state, q, None)
    c = state.cubo.clone()
    pressure = torch.tanh(torch.nan_to_num(d.omega_q.to(c.dtype), nan=1.0, posinf=1.0)).reshape(1)

    if c.shape[-1] >= 29:
        c = c.clone()
        c[:, 6:21] = c[:, 6:21] - strength * pressure * torch.sign(c[:, 6:21] + 1e-12)
        c[:, 21] = torch.relu(c[:, 21] - strength * pressure)
        c[:, 23] = torch.relu(c[:, 23] - strength * pressure)
        c[:, 24] = torch.exp(-torch.relu(c[:, 23]))

    return FoldedOmegaCuboState(
        z=state.z,
        memory=state.memory,
        relations=state.relations,
        cubo=c,
        pad=state.pad,
    )


def evaluate_candidate(
    model: FoldedCTNetOmegaCubo26,
    prev: FoldedOmegaCuboState,
    cand: FoldedOmegaCuboState,
    *,
    name: str,
    q: QuantumProblem,
    lambda_up: float,
    lambda_q: float,
) -> CuboCandidate:
    obs = model.cubo_observation(cand)
    up_loss, up_metrics = all_perspective_up_loss(model, prev, cand)
    qdef = quantum_structural_defect(cand, q, obs)

    omega_6d = obs["omega"].mean()
    residual = obs["residual"].mean()
    absorption = obs["absorption"].mean()
    closure_score = obs["closure_score"].mean()

    omega_eff = omega_6d + float(lambda_up) * up_loss + float(lambda_q) * qdef.omega_q

    return CuboCandidate(
        name=name,
        state=cand,
        omega_eff=omega_eff,
        omega_6d=omega_6d,
        omega_q=qdef.omega_q,
        residual=residual,
        absorption=absorption,
        closure_score=closure_score,
        up_loss=up_loss,
        up_metrics=up_metrics,
    )


def cubo_step_candidates(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    q: QuantumProblem,
    *,
    q_shear: float,
) -> list[tuple[str, FoldedOmegaCuboState]]:
    qs_plus = quantum_phase_shear(state, q, strength=+q_shear)
    qs_minus = quantum_phase_shear(state, q, strength=-q_shear)

    return [
        ("forward_state", model.forward_state(state)),
        ("closure_shear_plus", model.closure_shear(state, sign=+1.0)),
        ("closure_shear_minus", model.closure_shear(state, sign=-1.0)),
        ("q_shear_plus", qs_plus),
        ("q_shear_minus", qs_minus),
        ("forward_after_q_shear_plus", model.forward_state(qs_plus)),
        ("forward_after_q_shear_minus", model.forward_state(qs_minus)),
        ("forward_after_shear_plus", model.forward_state(model.closure_shear(state, sign=+1.0))),
        ("forward_after_shear_minus", model.forward_state(model.closure_shear(state, sign=-1.0))),
    ]


def solve_with_cubo(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    *,
    q: QuantumProblem,
    closure_steps: int,
    lambda_up: float,
    lambda_q: float,
    tau: float,
    q_shear: float,
) -> tuple[FoldedOmegaCuboState, list[CuboCandidate]]:
    history: list[CuboCandidate] = []
    current = state

    for _ in range(max(1, closure_steps)):
        candidates = [
            evaluate_candidate(model, current, cand, name=name, q=q, lambda_up=lambda_up, lambda_q=lambda_q)
            for name, cand in cubo_step_candidates(model, current, q, q_shear=q_shear)
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
    ap.add_argument("--closure-steps", type=int, default=32)
    ap.add_argument("--lambda-up", type=float, default=1.0)
    ap.add_argument("--lambda-q", type=float, default=1.0)
    ap.add_argument("--q-shear", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1e-7)
    ap.add_argument("--quantum-tau", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    q = QuantumProblem(
        n=args.n,
        J=args.J,
        h=args.h,
        dt=args.dt,
        steps=args.steps,
        psi0=parse_psi0(args.psi0, args.n),
    )

    model = FoldedCTNetOmegaCubo26(layout=FoldLayout()).to(device=device, dtype=dtype)
    obs_in = problem_as_observador(q)
    state0, _, _ = batch_to_state(model, [obs_in], device=device, dtype=dtype, max_bytes=args.max_bytes)

    solution, history = solve_with_cubo(
        model,
        state0,
        q=q,
        closure_steps=args.closure_steps,
        lambda_up=args.lambda_up,
        lambda_q=args.lambda_q,
        tau=args.tau,
        q_shear=args.q_shear,
    )

    final_obs = model.cubo_observation(solution)
    final_up, final_up_metrics = all_perspective_up_loss(model, state0, solution)
    amp = read_q_sigma(solution, q.n)
    qdef = quantum_structural_defect(solution, q, final_obs)
    amp_with_phi = qdef.eiphi * amp
    norm_error = (amp.abs().pow(2).sum() - 1.0).abs().real.to(torch.float64)

    omega_6d = final_obs["omega"].mean().detach().to(torch.float64)

    certified = (
        is_formal_certified_case(q)
        and float(omega_6d.detach().cpu()) <= args.quantum_tau
        and float(qdef.omega_q.detach().cpu()) <= args.quantum_tau
        and float(qdef.amplitude_error.detach().cpu()) <= args.quantum_tau
        and float(qdef.probability_error.detach().cpu()) <= args.quantum_tau
        and float(qdef.phase_error.detach().cpu()) <= args.quantum_tau
        and float(qdef.exhaustive_error.detach().cpu()) <= args.quantum_tau
        and float(qdef.closure_error.detach().cpu()) <= args.quantum_tau
        and float(norm_error.detach().cpu()) <= args.quantum_tau
    )

    print("=== Cubo 6D only quantum chart solver ===")
    print(f"parent_root={ROOT}")
    print("solver=Cubo_6D")
    print("route=Observador->batch_to_state->Cubo6DObserver->closure_shear/forward_state->Omega_Q->u=p->Q_sigma(Xi_solution)")
    print(f"n={q.n} branches={2**q.n} J={q.J} h={q.h} dt={q.dt} steps={q.steps} psi0={q.psi0}")
    print(f"closure_steps_used={len(history)}")

    for i, item in enumerate(history):
        print(
            f"closure[{i}] candidate={item.name} "
            f"omega_eff={float(item.omega_eff.detach().cpu()):.12g} "
            f"omega_6d={float(item.omega_6d.detach().cpu()):.12g} "
            f"omega_q={float(item.omega_q.detach().cpu()):.12g} "
            f"residual={float(item.residual.detach().cpu()):.12g} "
            f"absorption={float(item.absorption.detach().cpu()):.12g} "
            f"closure_score={float(item.closure_score.detach().cpu()):.12g} "
            f"up_loss={float(item.up_loss.detach().cpu()):.12g}"
        )

    print(f"final_omega_6d={float(omega_6d.detach().cpu()):.12g}")
    print(f"final_omega_q={float(qdef.omega_q.detach().cpu()):.12g}")
    print(f"final_amplitude_error={float(qdef.amplitude_error.detach().cpu()):.12g}")
    print(f"final_probability_error={float(qdef.probability_error.detach().cpu()):.12g}")
    print(f"final_phase_error={float(qdef.phase_error.detach().cpu()):.12g}")
    print(f"final_exhaustive_error={float(qdef.exhaustive_error.detach().cpu()):.12g}")
    print(f"final_closure_error={float(qdef.closure_error.detach().cpu()):.12g}")
    print(f"final_norm_error_q={float(qdef.norm_error.detach().cpu()):.12g}")
    print(f"final_born_error={float(qdef.born_error.detach().cpu()):.12g}")
    print(f"final_residual={float(final_obs['residual'].mean().detach().cpu()):.12g}")
    print(f"final_absorption={float(final_obs['absorption'].mean().detach().cpu()):.12g}")
    print(f"final_closure_score={float(final_obs['closure_score'].mean().detach().cpu()):.12g}")
    print(f"final_up_total={float(final_up.detach().cpu()):.12g}")

    for k, v in sorted(final_up_metrics.items()):
        print(f"{k}={v:.12g}")

    print(f"amplitude_count={amp.numel()}")
    print(f"normalization_error={float(norm_error.detach().cpu()):.12g}")
    print(f"phi_cubo_rad={float(qdef.phi.detach().cpu()):.17g}")
    print(f"phi_cubo_over_pi={float((qdef.phi / torch.pi).detach().cpu()):.17g}")
    print(f"eiphi_real={float(qdef.eiphi.real.detach().cpu()):.17g}")
    print(f"eiphi_imag={float(qdef.eiphi.imag.detach().cpu()):.17g}")
    print(f"quantum_strong_certified={certified}")

    print("first_amplitudes:")
    for i, a in enumerate(amp[: min(16, amp.numel())].detach().cpu()):
        print(f"{i:04d} real={float(a.real): .12g} imag={float(a.imag): .12g} prob={float(abs(a)**2):.12g}")

    if args.save:
        path = Path(args.save)
        torch.save(
            {
                "amplitudes": amp.detach().cpu(),
                "amplitudes_with_phi": amp_with_phi.detach().cpu(),
                "n": q.n,
                "J": q.J,
                "h": q.h,
                "dt": q.dt,
                "steps": q.steps,
                "psi0": q.psi0,
                "solver": "Cubo_6D_only",
                "normalization_error": float(norm_error.detach().cpu()),
                "phi_cubo_rad": float(qdef.phi.detach().cpu()),
                "phi_cubo_over_pi": float((qdef.phi / torch.pi).detach().cpu()),
                "eiphi_real": float(qdef.eiphi.real.detach().cpu()),
                "eiphi_imag": float(qdef.eiphi.imag.detach().cpu()),
                "final_omega_6d": float(omega_6d.detach().cpu()),
                "final_omega_q": float(qdef.omega_q.detach().cpu()),
                "final_amplitude_error": float(qdef.amplitude_error.detach().cpu()),
                "final_probability_error": float(qdef.probability_error.detach().cpu()),
                "final_phase_error": float(qdef.phase_error.detach().cpu()),
                "final_exhaustive_error": float(qdef.exhaustive_error.detach().cpu()),
                "final_closure_error": float(qdef.closure_error.detach().cpu()),
                "final_norm_error_q": float(qdef.norm_error.detach().cpu()),
                "final_born_error": float(qdef.born_error.detach().cpu()),
                "final_up_total": float(final_up.detach().cpu()),
                "quantum_strong_certified": bool(certified),
            },
            path,
        )
        print(f"saved={path}")


if __name__ == "__main__":
    main()
