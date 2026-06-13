#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctnet_hilbert_projector.projective_engine import CTNetProjectiveState


AUDIT_KEYS = [
    "quantum_strong_certified",
    "final_omega_6d",
    "final_omega_q",
    "final_amplitude_error",
    "final_probability_error",
    "final_phase_error",
    "final_exhaustive_error",
    "final_closure_error",
    "normalization_error",
    "closure_steps_used",
    "best_omega_eff",
    "last_omega_eff",
    "phi_cubo_rad",
    "phi_cubo_over_pi",
    "eiphi_real",
    "eiphi_imag",
]


def default_branches(n: int) -> list[str]:
    branches = [
        "u" * n,
        "p" * n,
        "up" * (n // 2) + ("u" if n % 2 else ""),
        "pu" * (n // 2) + ("p" if n % 2 else ""),
    ]

    out: list[str] = []
    for branch in branches:
        if branch not in out:
            out.append(branch)
    return out


def pair_count(n: int, mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "adjacent":
        return max(0, n - 1)
    if mode == "all":
        return n * (n - 1) // 2
    raise ValueError(f"invalid pair mode: {mode}")


def validate_branches(branches: list[str], n: int) -> None:
    for branch in branches:
        if len(branch) != n:
            raise ValueError(f"branch {branch!r} has length {len(branch)}, expected n={n}")
        bad = sorted(set(branch) - {"u", "p"})
        if bad:
            raise ValueError(f"branch {branch!r} contains invalid symbols: {bad}")


def tensor_count(payload_value: Any) -> int:
    if isinstance(payload_value, torch.Tensor):
        return int(payload_value.numel())
    return 1


def count_audit_scalars(payload: dict[str, Any]) -> tuple[int, list[str]]:
    present: list[str] = []
    total = 0

    for key in AUDIT_KEYS:
        if key in payload:
            present.append(key)
            total += tensor_count(payload[key])

    return total, present


def compute_density(path: Path, branches: list[str], *, pair_mode: str) -> dict[str, Any]:
    st = CTNetProjectiveState.load(path)
    payload = st.payload
    n = int(st.n)
    amplitude_count = int(st.amplitudes.numel())

    if amplitude_count != 2**n:
        raise ValueError(f"amplitude_count={amplitude_count} incompatible with n={n}")

    if not branches:
        branches = default_branches(n)

    validate_branches(branches, n)

    selected_branch_count = len(branches)
    full_amplitude_scalars = 2 * amplitude_count

    # A.real, A.imag, P, Theta for each selected branch.
    branch_readout_scalars = 4 * selected_branch_count

    # Z_i, Z_iZ_j, magnetization_z, sector_mass_by_u_count[0..n].
    zz_count = pair_count(n, pair_mode)
    diagonal_observable_scalars = n + zz_count + 1 + (n + 1)

    # DeltaTheta, cos, sin, interference, quadrature for every ordered pair.
    coherence_matrix_scalars = 5 * selected_branch_count * selected_branch_count

    audit_certificate_scalars, audit_keys_present = count_audit_scalars(payload)

    projective_structural_scalars = (
        branch_readout_scalars
        + diagonal_observable_scalars
        + coherence_matrix_scalars
        + audit_certificate_scalars
    )

    d_proj = projective_structural_scalars / float(full_amplitude_scalars)
    d_proj_branch = projective_structural_scalars / float(selected_branch_count)
    d_proj_amplitude = projective_structural_scalars / float(amplitude_count)

    return {
        "state": str(path),
        "n": n,
        "amplitude_count": amplitude_count,
        "selected_branch_count": selected_branch_count,
        "branches": branches,
        "pair_mode": pair_mode,
        "full_amplitude_scalars": full_amplitude_scalars,
        "branch_readout_scalars": branch_readout_scalars,
        "diagonal_observable_scalars": diagonal_observable_scalars,
        "coherence_matrix_scalars": coherence_matrix_scalars,
        "audit_certificate_scalars": audit_certificate_scalars,
        "audit_keys_present": audit_keys_present,
        "projective_structural_scalars": projective_structural_scalars,
        "D_proj_per_full_amplitude_scalar": d_proj,
        "D_proj_per_complex_amplitude": d_proj_amplitude,
        "D_proj_per_selected_branch": d_proj_branch,
    }


def print_text(report: dict[str, Any]) -> None:
    print(f"state={report['state']}")
    print(f"n={report['n']}")
    print(f"amplitude_count={report['amplitude_count']}")
    print(f"selected_branch_count={report['selected_branch_count']}")
    print("branches=" + ",".join(report["branches"]))
    print(f"pair_mode={report['pair_mode']}")
    print(f"full_amplitude_scalars={report['full_amplitude_scalars']}")
    print(f"branch_readout_scalars={report['branch_readout_scalars']}")
    print(f"diagonal_observable_scalars={report['diagonal_observable_scalars']}")
    print(f"coherence_matrix_scalars={report['coherence_matrix_scalars']}")
    print(f"audit_certificate_scalars={report['audit_certificate_scalars']}")
    print("audit_keys_present=" + ",".join(report["audit_keys_present"]))
    print(f"projective_structural_scalars={report['projective_structural_scalars']}")
    print(f"D_proj_per_full_amplitude_scalar={report['D_proj_per_full_amplitude_scalar']:.17g}")
    print(f"D_proj_per_complex_amplitude={report['D_proj_per_complex_amplitude']:.17g}")
    print(f"D_proj_per_selected_branch={report['D_proj_per_selected_branch']:.17g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure CTNet projective density D_proj(n)")
    ap.add_argument("state", help="Path to .pt state generated by solve_ising_cubo6d_only.py")
    ap.add_argument("branches", nargs="*", help="Selected u/p branches")
    ap.add_argument("--pairs", choices=["none", "adjacent", "all"], default="adjacent")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    report = compute_density(Path(args.state), args.branches, pair_mode=args.pairs)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)


if __name__ == "__main__":
    main()
