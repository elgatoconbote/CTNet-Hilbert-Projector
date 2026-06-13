from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch


def test_audit_strong_certificate_script_accepts_valid_payload(tmp_path: Path):
    state = tmp_path / "state.pt"

    amplitudes = torch.zeros(64, dtype=torch.complex128)
    amplitudes[0] = 1.0 + 0.0j

    torch.save(
        {
            "amplitudes": amplitudes,
            "amplitudes_with_phi": amplitudes,
            "n": 6,
            "closure_steps_used": 24,
            "best_omega_eff": 0.00624106971105,
            "last_omega_eff": 0.00624106971105,
            "last_candidate": "closure_shear_minus",
            "quantum_strong_certified": True,
            "final_omega_6d": 0.0,
            "final_omega_q": 1.90802765587e-7,
            "final_amplitude_error": 7.9060427538e-8,
            "final_probability_error": 1.11742338049e-7,
            "final_phase_error": 0.0,
            "final_exhaustive_error": 0.0,
            "final_closure_error": 0.0,
            "normalization_error": 0.0,
        },
        state,
    )

    result = subprocess.run(
        [sys.executable, "scripts/audit_strong_certificate.py", str(state)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "CERTIFICATE_OK=True" in result.stdout
    assert "amplitude_count=64" in result.stdout
    assert "closure_steps_used=24" in result.stdout
