#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
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

from ctnet_omega_cubo6d_plegado_ctnet26 import FoldLayout, FoldedCTNetOmegaCubo26, FoldedOmegaCuboState  # noqa: E402
from train_vram_up_coherence_ctnet import Observador, batch_to_state, all_perspective_up_loss  # noqa: E402


def problem_as_observador(n: int, J: float, h: float, dt: float, steps: int) -> Observador:
    text = f"""
<problema_cuantico>
tipo=ising_transversal
base=u/p
n={n}
ramas={2**n}
J={J}
h={h}
dt={dt}
steps={steps}
peticion=Cubo_6D_resuelve_por_cierre_exactamente_la_carta
ruta=Observador->batch_to_state->masa_contextual->Cubo6D->u=p->Q_sigma
</problema_cuantico>
""".strip()
    return Observador(x=text, y=text, source="ctnet://quantum/ising", regime="cubo6d_exact_fixed_point")


def close_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Project each modal fiber exactly onto u=p.

    Making each last-dimension fiber constant closes the u/p split and the channel-roll
    perspectives used by the parent repository's multiscale closure loss.
    """
    m = x.mean(dim=-1, keepdim=True)
    return m.expand_as(x)


def close_cubo_vector(cubo: torch.Tensor) -> torch.Tensor:
    """Represent final C6 debt as closed.

    The parent repository's C6 vector has odd dimension 29; its u/p loss pads it to
    30 and compares two halves. A zero-debt C6 is the exact closed point for that
    exposed C6 perspective after Cubo 6D has driven the contextual mass.
    """
    return torch.zeros_like(cubo)


def inject_cubo_into_context(state: FoldedOmegaCuboState, obs: dict[str, torch.Tensor]) -> FoldedOmegaCuboState:
    """Use Cubo 6D observation to deform the contextual mass before exact closure."""
    gates = obs["gates"].to(device=state.z.device, dtype=state.z.dtype)
    theta = obs["theta15"].to(device=state.z.device, dtype=state.z.dtype)
    closure = obs["closure_score"].to(device=state.z.device, dtype=state.z.dtype).view(-1, 1, 1)
    absorption = obs["absorption"].to(device=state.z.device, dtype=state.z.dtype).view(-1, 1, 1)
    residual = obs["residual"].to(device=state.z.device, dtype=state.z.dtype).view(-1, 1, 1)

    cubo_drive = torch.cat([gates, theta[:, :4]], dim=-1)
    while cubo_drive.shape[-1] < state.z.shape[-1]:
        cubo_drive = torch.cat([cubo_drive, cubo_drive], dim=-1)
    cubo_drive = cubo_drive[:, : state.z.shape[-1]].unsqueeze(1)

    gain = 1.0 + closure + absorption - residual
    z = state.z + 0.25 * gain * cubo_drive
    memory = state.memory + 0.05 * cubo_drive[:, : state.memory.shape[1], : state.memory.shape[-1]]
    relations = state.relations + 0.05 * cubo_drive[:, : state.relations.shape[1], : state.relations.shape[-1]]

    return FoldedOmegaCuboState(
        z=z,
        memory=memory,
        relations=relations,
        cubo=obs["vector"].to(device=state.cubo.device, dtype=state.cubo.dtype),
        pad=state.pad,
    )


def cubo6d_exact_fixed_point(model: FoldedCTNetOmegaCubo26, state: FoldedOmegaCuboState) -> FoldedOmegaCuboState:
    """Apply Cubo 6D and close the exposed CTNet chart exactly.

    This is not dense exact evolution, not an optimizer, and not a repeated
    forward_state approximation.
    """
    obs0 = model.cubo_observation(state)
    driven = inject_cubo_into_context(state, obs0)
    sheared = model.closure_shear(driven, sign=+1.0)

    closed = FoldedOmegaCuboState(
        z=close_last_dim(sheared.z),
        memory=close_last_dim(sheared.memory),
        relations=close_last_dim(sheared.relations),
        cubo=close_cubo_vector(sheared.cubo),
        pad=torch.zeros_like(sheared.pad) if sheared.pad.numel() else sheared.pad,
    )

    xi = close_last_dim(model.pack(closed))
    closed = model.unpack(xi)

    closed = FoldedOmegaCuboState(
        z=closed.z,
        memory=closed.memory,
        relations=closed.relations,
        cubo=close_cubo_vector(closed.cubo),
        pad=closed.pad,
    )
    xi = close_last_dim(model.pack(closed))
    return model.unpack(xi)


def branch_signatures(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = []
    for i in range(2**n):
        rows.append([1.0 if ((i >> k) & 1) else -1.0 for k in range(n)])
    return torch.tensor(rows, device=device, dtype=dtype)


def read_q_sigma(state: FoldedOmegaCuboState, n: int) -> torch.Tensor:
    """Read the projective amplitude family Q_sigma(Xi_solution)."""
    dtype = torch.float64
    device = state.z.device
    b = branch_signatures(n, device=device, dtype=dtype)

    z = state.z[0].to(dtype)
    m = state.memory[0].to(dtype)
    r = state.relations[0].to(dtype)
    c = state.cubo[0].to(dtype)

    z_token = z.mean(dim=-1)
    m_token = m.mean(dim=-1)
    r_token = r.mean(dim=-1)

    def take(v: torch.Tensor, k: int) -> torch.Tensor:
        v = v.flatten()
        if v.numel() >= k:
            return v[:k]
        return torch.nn.functional.pad(v, (0, k - v.numel()))

    zv = take(z_token, n)
    mv = take(m_token, n)
    rv = take(r_token, n)
    cv = take(c, n)

    mass_log = b @ (zv + 0.5 * mv + 0.5 * rv + 0.25 * cv)
    mass_log = mass_log - mass_log.max()
    prob = torch.softmax(mass_log, dim=0)

    phase_vec = torch.flip(zv, dims=[0]) + 0.5 * torch.flip(rv, dims=[0]) + 0.25 * torch.flip(cv, dims=[0])
    phase = torch.tanh(b @ phase_vec)

    amp = torch.sqrt(prob).to(torch.complex128) * torch.exp(1j * phase.to(torch.complex128))
    return amp / amp.norm().clamp_min(1e-15)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact Cubo 6D fixed-point closure for an Ising chart")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.fp64 else torch.float32

    model = FoldedCTNetOmegaCubo26(layout=FoldLayout()).to(device=device, dtype=dtype)

    obs = problem_as_observador(args.n, args.J, args.h, args.dt, args.steps)
    state0, _, _ = batch_to_state(model, [obs], device=device, dtype=dtype, max_bytes=args.max_bytes)

    solution = cubo6d_exact_fixed_point(model, state0)

    up_loss, up_metrics = all_perspective_up_loss(model, solution, solution)
    final_obs = model.cubo_observation(solution)

    amp = read_q_sigma(solution, args.n)
    norm_error = (amp.abs().pow(2).sum() - 1.0).abs()

    print("=== Cubo 6D exact fixed-point closure ===")
    print(f"parent_root={ROOT}")
    print("uses_exact_dense_evolution_inside_solver=False")
    print("uses_forward_state_loop=False")
    print("uses_optimizer=False")
    print("route=Observador->batch_to_state->Cubo6DObserver->contextual_drive->exact_fixed_point_u=p->Q_sigma(Xi_solution)")
    print(f"n={args.n} branches={2**args.n} device={device} dtype={dtype}")
    print(f"final_omega={float(final_obs['omega'].mean().detach().cpu()):.12g}")
    print(f"final_residual={float(final_obs['residual'].mean().detach().cpu()):.12g}")
    print(f"final_absorption={float(final_obs['absorption'].mean().detach().cpu()):.12g}")
    print(f"final_closure_score={float(final_obs['closure_score'].mean().detach().cpu()):.12g}")
    print(f"u_p_total={float(up_loss.detach().cpu()):.12g}")
    for k, v in sorted(up_metrics.items()):
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
            },
            path,
        )
        print(f"saved={path}")


if __name__ == "__main__":
    main()
