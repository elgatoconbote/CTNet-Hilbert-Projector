#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrated CTNet thesis presets.

Presets are empirical regimes found by the calibration scripts. They are not new
axioms; they are reproducible operating points for the thesis dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .thesis_dynamics import ThesisDynamicsConfig


@dataclass(frozen=True)
class ThesisIsingPreset:
    name: str
    prep_memory: float
    prep_relation: float
    config: ThesisDynamicsConfig


def ising_v1(n_qubits: int) -> ThesisIsingPreset:
    """Best calibrated n=6 transverse-field Ising regime before mass contrast.

    Calibration target: n=6, J=1.0, h=0.5, dt=0.05, steps=3.
    Observed best in grid: amp_l2 ~= 0.6146, loss ~= 0.7276.
    """
    return ThesisIsingPreset(
        name="ising_v1",
        prep_memory=0.4,
        prep_relation=0.6,
        config=ThesisDynamicsConfig(
            n_qubits=n_qubits,
            beta_coherence=3.0,
            gamma_residue=1.0,
            hamiltonian_state_strength=0.1,
            hamiltonian_phase_strength=1.5,
            mass_feedback_strength=0.05,
            atlas_strength=0.8,
            cocycle_strength=0.25,
        ),
    )


def ising_v2(n_qubits: int) -> ThesisIsingPreset:
    """Mass-contrast Ising regime.

    This preset activates Hamiltonian and cardinal mass contrast so that Born
    masses are no longer almost flat when exact projected dynamics has visible
    probability spread.
    """
    return ThesisIsingPreset(
        name="ising_v2",
        prep_memory=0.4,
        prep_relation=0.6,
        config=ThesisDynamicsConfig(
            n_qubits=n_qubits,
            beta_coherence=3.0,
            gamma_residue=1.0,
            hamiltonian_state_strength=0.1,
            hamiltonian_phase_strength=1.5,
            hamiltonian_mass_strength=1.0,
            cardinal_mass_strength=0.5,
            mass_feedback_strength=0.05,
            atlas_strength=0.8,
            cocycle_strength=0.25,
        ),
    )


def get_thesis_preset(name: str, n_qubits: int) -> ThesisIsingPreset:
    key = name.strip().lower().replace("-", "_")
    if key in {"ising_v1", "ising"}:
        return ising_v1(n_qubits)
    if key in {"ising_v2", "ising_mass"}:
        return ising_v2(n_qubits)
    raise ValueError(f"unknown thesis preset: {name!r}")
