from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import torch

from ctnet_hilbert_projector.projective_engine import sigma_to_index


def test_projective_phase_interference_kernel_controlled_pair(tmp_path: Path):
    state = tmp_path / "phase_state.pt"

    amplitudes = torch.zeros(4, dtype=torch.complex128)
    amplitudes[sigma_to_index("uu")] = 1 / math.sqrt(2)
    amplitudes[sigma_to_index("pp")] = 1j / math.sqrt(2)

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
        [sys.executable, "scripts/measure_projective_phase.py", str(state), "uu:pp"],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "pair=uu:pp" in out
    assert "uu.P=0.49999999999999989" in out
    assert "pp.P=0.49999999999999989" in out
    assert "DeltaTheta(uu,pp)=-1.5707963267948966" in out
    assert "sqrt_P_sigma_P_tau(uu,pp)=0.49999999999999989" in out
    assert "interference_kernel(uu,pp)=3.0616169978683824e-17" in out
    assert "quadrature_kernel(uu,pp)=-0.49999999999999989" in out


def test_projective_phase_wraps_delta_theta(tmp_path: Path):
    state = tmp_path / "wrap_state.pt"

    amplitudes = torch.zeros(4, dtype=torch.complex128)
    amplitudes[sigma_to_index("uu")] = complex(math.cos(3.0), math.sin(3.0)) / math.sqrt(2)
    amplitudes[sigma_to_index("pp")] = complex(math.cos(-3.0), math.sin(-3.0)) / math.sqrt(2)

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
        [sys.executable, "scripts/measure_projective_phase.py", str(state), "uu:pp"],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "pair=uu:pp" in out
    assert "DeltaTheta(uu,pp)=-0.28318530717958645" in out
