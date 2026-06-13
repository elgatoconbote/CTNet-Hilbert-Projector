from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from ctnet_hilbert_projector.projective_engine import sigma_to_index


def test_projective_coherence_matrix_two_branch_phase(tmp_path: Path):
    state = tmp_path / "coherence_state.pt"

    amplitudes = torch.zeros(4, dtype=torch.complex128)
    amplitudes[sigma_to_index("pp")] = 1 / math.sqrt(2)
    amplitudes[sigma_to_index("uu")] = 1j / math.sqrt(2)

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
            "scripts/measure_projective_coherence_matrix.py",
            str(state),
            "pp",
            "uu",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    data = json.loads(result.stdout)
    m = data["matrix"]

    assert data["branches"] == ["pp", "uu"]

    assert math.isclose(m["pp"]["pp"]["DeltaTheta"], 0.0, abs_tol=1e-15)
    assert math.isclose(m["uu"]["uu"]["DeltaTheta"], 0.0, abs_tol=1e-15)

    assert math.isclose(m["pp"]["pp"]["interference_kernel"], 0.5, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(m["uu"]["uu"]["interference_kernel"], 0.5, rel_tol=0.0, abs_tol=1e-15)

    assert math.isclose(m["pp"]["uu"]["DeltaTheta"], -math.pi / 2, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(m["uu"]["pp"]["DeltaTheta"], math.pi / 2, rel_tol=0.0, abs_tol=1e-15)

    assert math.isclose(m["pp"]["uu"]["interference_kernel"], 0.0, abs_tol=1e-15)
    assert math.isclose(m["uu"]["pp"]["interference_kernel"], 0.0, abs_tol=1e-15)

    assert math.isclose(m["pp"]["uu"]["quadrature_kernel"], -0.5, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(m["uu"]["pp"]["quadrature_kernel"], 0.5, rel_tol=0.0, abs_tol=1e-15)


def test_projective_coherence_matrix_text_output(tmp_path: Path):
    state = tmp_path / "text_state.pt"

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
            "scripts/measure_projective_coherence_matrix.py",
            str(state),
            "pp",
            "uu",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    out = result.stdout
    assert "branch_count=2" in out
    assert "branches=pp,uu" in out
    assert "DeltaTheta" in out
    assert "interference_kernel" in out
    assert "quadrature_kernel" in out
