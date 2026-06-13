#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/elgatoconbote/CTNet-Omega-cubo-6D/.venv/bin/python}"
STATE="${STATE:-/tmp/cubo6d_strong_quantum.pt}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

"$PY" -m py_compile examples/solve_ising_cubo6d_only.py
"$PY" -m py_compile src/ctnet_hilbert_projector/projective_engine.py
"$PY" -m py_compile scripts/audit_strong_certificate.py
"$PY" -m pytest -q

"$PY" examples/solve_ising_cubo6d_only.py \
  --n 6 \
  --J 1.0 \
  --h 0.5 \
  --dt 0.05 \
  --steps 3 \
  --psi0 uuuuuu \
  --closure-steps 64 \
  --lambda-up 1.0 \
  --lambda-q 1.0 \
  --q-shear 0.05 \
  --quantum-tau 1e-6 \
  --fp64 \
  --save "$STATE"

"$PY" -m ctnet_hilbert_projector.projective_engine \
  "$STATE" \
  --certificate \
  uuuuuu \
  pppppp \
  upupup \
  puupup

"$PY" scripts/audit_strong_certificate.py "$STATE"

"$PY" scripts/measure_projective_observables.py "$STATE" --pairs adjacent
