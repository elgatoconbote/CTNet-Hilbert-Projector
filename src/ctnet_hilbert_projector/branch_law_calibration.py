from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BranchLawCalibration:
    mass_shift: torch.Tensor
    phase_shift: torch.Tensor


def fit_branch_law(base_probabilities: torch.Tensor, base_phases: torch.Tensor, reference_amplitudes: torch.Tensor, eps: float = 1e-15) -> BranchLawCalibration:
    ref = reference_amplitudes.to(device=base_probabilities.device, dtype=torch.complex128)
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(eps)
    ref_prob = ref.abs().pow(2).real.clamp_min(eps)
    base_prob = base_probabilities.to(dtype=ref_prob.dtype).clamp_min(eps)
    mass_shift = torch.log(ref_prob) - torch.log(base_prob)
    mass_shift = mass_shift - mass_shift.mean(dim=-1, keepdim=True)
    base_phase = base_phases.to(device=ref.device, dtype=ref_prob.dtype)
    ref_phase = torch.angle(ref).to(dtype=ref_prob.dtype)
    phase_shift = torch.atan2(torch.sin(ref_phase - base_phase), torch.cos(ref_phase - base_phase))
    return BranchLawCalibration(mass_shift=mass_shift.detach(), phase_shift=phase_shift.detach())


def apply_branch_law(base_probabilities: torch.Tensor, base_phases: torch.Tensor, calibration: BranchLawCalibration, eps: float = 1e-15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_prob = base_probabilities.to(dtype=torch.float64).clamp_min(eps)
    mass_shift = calibration.mass_shift.to(device=base_prob.device, dtype=torch.float64)
    phase_shift = calibration.phase_shift.to(device=base_prob.device, dtype=torch.float64)
    probabilities = torch.softmax(torch.log(base_prob) + mass_shift, dim=-1)
    phases = base_phases.to(dtype=torch.float64) + phase_shift
    amplitudes = torch.sqrt(probabilities).to(torch.complex128) * torch.exp(1j * phases.to(torch.complex128))
    return amplitudes, probabilities, phases
