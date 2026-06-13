from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from ctnet_hilbert_projector.projective_engine import sigma_to_index


def test_projective_observables_for_pure_u_state(tmp_path: Path):
    state = tmp_path / "u_state.pt"

    amplitudes = torch.zeros(64, dtype=torch.complex128)
    amplitudes[sigma_to_index("uuuuuu")] = 1.0 + 0.0j

    torch.save(
        {
            "amplitudes": amplitudes,
            "amplitudes_with_phi": amplitudes,
            "n": 6,
            "quantum_strong_certified": True,
        },
        state,
    )

    result = subprocess.run(
        [sys.executable, "scripts/measure_projective_observables.py", str(state), "--pairs", "adjacent"],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "amplitude_count=64" in out
    assert "normalization_error=0" in out
    assert "magnetization_z=1" in out
    assert "<Z_0>=1" in out
    assert "<Z_5>=1" in out
    assert "<Z_0Z_1>=1" in out
    assert "sector_mass_u_count_6=1" in out


def test_projective_observables_for_pure_p_state(tmp_path: Path):
    state = tmp_path / "p_state.pt"

    amplitudes = torch.zeros(64, dtype=torch.complex128)
    amplitudes[sigma_to_index("pppppp")] = 1.0 + 0.0j

    torch.save(
        {
            "amplitudes": amplitudes,
            "amplitudes_with_phi": amplitudes,
            "n": 6,
            "quantum_strong_certified": True,
        },
        state,
    )

    result = subprocess.run(
        [sys.executable, "scripts/measure_projective_observables.py", str(state), "--pairs", "adjacent"],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "amplitude_count=64" in out
    assert "normalization_error=0" in out
    assert "magnetization_z=-1" in out
    assert "<Z_0>=-1" in out
    assert "<Z_5>=-1" in out
    assert "<Z_0Z_1>=1" in out
    assert "sector_mass_u_count_0=1" in out
