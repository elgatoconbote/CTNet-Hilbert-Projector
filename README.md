# CTNet Hilbert Projector

Implementación experimental de simulación cuántica estructural mediante CTNet 2.6 Omega + Cubo 6D y base cardinal `u/p`.

```text
Xi_t = (Z_t, M_t, R_t, C6_t, rho_t, K_t, Omega_t)
A_t(sigma) = Q_sigma(Xi_t),    sigma in {u,p}^n
```

## Estado actual

```text
pytest: 15 passed
preset: best / ising_v4
structural_regime_admissible=True
raw_structural_exact=False
exact_projective_certified=True
ctnet_structural_superiority=CALIBRATION_REQUIRED
certification_tier=CALIBRATION_REQUIRED
```

El resultado estructural actual no es exactitud dinámica fuerte. Es admisibilidad de régimen.

```text
amp_l2=0.43822440505
prob_l1=0.456387639046
spread_gap=0.00284448266029
mass_contrast_std=1.28768014908
```

La condición fuerte queda reservada para la transición estructural cruda, sin cierre posterior contra el objetivo exacto:

```text
amp_l2 <= 1e-6
prob_l1 <= 1e-6
raw_structural_exact=True
```

Mientras esos valores no se alcancen por calibración real de la dinámica CTNet, el benchmark debe devolver:

```text
ctnet_structural_superiority=CALIBRATION_REQUIRED
certification_tier=CALIBRATION_REQUIRED
```

## Certificación proyectiva formal

La capa proyectiva formal sí está certificada:

```text
exact_amp_l2=1.95799630724e-08
exact_prob_l1=3.91599258314e-08
exact_observable_abs=1.11375948908e-07
exact_normalization_error=0
exact_projective_commutation_error=0
exact_projective_certified=True
```

Esto no sustituye la calibración estructural cruda.

## Validación

```bash
git pull
"$PY" -m pip install -e . --no-deps
"$PY" -m pytest -q
```

Salida esperada:

```text
15 passed
```

## Benchmark

```bash
"$PY" examples/benchmark_projective_superiority.py \
  --cuda \
  --preset best \
  --validate-n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --sizes 6 8 10 12 16 20 24 30 40
```

Salida clave esperada con el preset actual:

```text
structural_regime_admissible=True
raw_structural_exact=False
exact_projective_certified=True
ctnet_structural_superiority=CALIBRATION_REQUIRED
certification_tier=CALIBRATION_REQUIRED
```

## Arquitectura

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
src/ctnet_hilbert_projector/hilbert_projector.py
src/ctnet_hilbert_projector/hamiltonians.py
src/ctnet_hilbert_projector/exact_certification.py
src/ctnet_hilbert_projector/thesis_dynamics.py
src/ctnet_hilbert_projector/state_preparation.py
src/ctnet_hilbert_projector/presets.py
examples/benchmark_projective_superiority.py
```
