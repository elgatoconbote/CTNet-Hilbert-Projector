from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from ctnet_hilbert_projector.projective_engine import sigma_to_index


def test_projective_density_counts_controlled_state(tmp_path: Path):
    state = tmp_path / "density_state.pt"

    amplitudes = torch.zeros(4, dtype=torch.complex128)
    amplitudes[sigma_to_index("pp")] = 1 / math.sqrt(2)
    amplitudes[sigma_to_index("uu")] = 1j / math.sqrt(2)

    torch.save(
        {
            "amplitudes": amplitudes,
            "amplitudes_with_phi": amplitudes,
            "n": 2,
            "quantum_strong_certified": True,
            "final_omega_6d": 0.0,
            "normalization_error": 0.0,
        },
        state,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/measure_projective_density.py",
            str(state),
            "pp",
            "uu",
            "--pairs",
            "adjacent",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(result.stdout)

    assert data["n"] == 2
    assert data["amplitude_count"] == 4
    assert data["selected_branch_count"] == 2
    assert data["full_amplitude_scalars"] == 8

    # A.real,A.imag,P,Theta for two branches.
    assert data["branch_readout_scalars"] == 8

    # Z_i = 2, adjacent ZZ = 1, magnetization = 1, sector masses = n+1 = 3.
    assert data["diagonal_observable_scalars"] == 7

    # Delta, cos, sin, interference, quadrature for 2x2 ordered pairs.
    assert data["coherence_matrix_scalars"] == 20

    # quantum_strong_certified, final_omega_6d, normalization_error.
    assert data["audit_certificate_scalars"] == 3

    assert data["projective_structural_scalars"] == 38
    assert data["D_proj_per_full_amplitude_scalar"] == 38 / 8
    assert data["D_proj_per_complex_amplitude"] == 38 / 4
    assert data["D_proj_per_selected_branch"] == 38 / 2


def test_projective_density_text_output(tmp_path: Path):
    state = tmp_path / "density_text_state.pt"

    amplitudes = torch.zeros(4, dtype=torch.complex128)
    amplitudes[sigma_to_index("pp")] = 1.0 + 0.0j

    torch.save(
        {
            "amplitudes": amplitudes,
            "amplitudes_with_phi": amplitudes,
            "n": 2,
            "quantum_strong_certified": True,
        },
        state,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/measure_projective_density.py",
            str(state),
            "pp",
            "uu",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "D_proj_per_full_amplitude_scalar=" in out
    assert "projective_structural_scalars=" in out
    assert "full_amplitude_scalars=8" in out
