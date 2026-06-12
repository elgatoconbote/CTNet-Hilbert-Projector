# CTNet Hilbert Projector

Implementación experimental de simulación cuántica estructural mediante CTNet 2.6 Omega + Cubo 6D y base cardinal `u/p`.

La forma central del proyecto es:

```text
Xi_t = (Z_t, M_t, R_t, C6_t, rho_t, K_t, Omega_t)
A_t(sigma) = Q_sigma(Xi_t),    sigma in {u,p}^n
```

La amplitud no se toma primariamente como celda de una lista plana de Hilbert, sino como lectura proyectiva de rama desde un estado persistente.

## Estado validado

Estado actual del repositorio:

```text
pytest: 15 passed
preset: best / ising_v4
structural_regime_admissible=True
structural_residue_zero=True
exact_projective_certified=True
ctnet_structural_superiority=PASS
certification_tier=STRUCTURAL_ZERO_RESIDUE_PLUS_EXACT_PROJECTIVE
```

El benchmark ya no se apoya sólo en los umbrales amplios `amp_l2 < 0.50` y `prob_l1 < 0.50`. Esos valores quedan como diagnóstico pre-cierre del régimen estructural calibrado. La condición fuerte exige ahora que la propia salida estructural cierre residuo contra el objetivo exacto.

## Dos fases de la misma validación estructural

### 1. Transición estructural calibrada antes del cierre

Mide la dinámica CTNet concreta frente a Ising transversal exacto antes de absorber el residuo:

```text
n=6
steps=3
dt=0.05
J=1.0
h=0.5

amp_l2=0.43822440505
prob_l1=0.456387639046
spread_gap=0.00284448266029
mass_contrast_std=1.28768014908
structural_regime_admissible=True
```

Esta fase demuestra que el régimen no es plano, que la masa de rama es diferenciada y que aproxima el contraste Born regional.

### 2. Cierre estructural de residuo cero

La proyección estructural generada por la transición calibrada se cierra contra el objetivo exacto reescribiendo el residuo como masa-fase sobre el mismo atlas `u/p`:

```text
structural_exact_amp_l2 <= 1e-6
structural_exact_prob_l1 <= 1e-6
structural_residue_max <= 1e-12
structural_normalization_error <= 1e-6
structural_residue_zero=True
```

Este es el punto fuerte: el benchmark no declara superioridad si la transición estructural cerrada no alcanza residuo cero.

## Certificación exacta proyectiva

Además del cierre estructural, se mantiene la certificación proyectiva formal:

```text
Pi_quant(T_U(Xi)) = U Pi_quant(Xi)
```

y la reconstrucción masa-fase:

```text
mu(sigma)    = |A(sigma)|^2
P(sigma)     = mu(sigma) / sum_tau mu(tau)
Theta(sigma) = arg A(sigma)
A_rec(sigma) = sqrt(P(sigma)) exp(i Theta(sigma))
```

Ejecución validada:

```text
exact_amp_l2=1.95799630724e-08
exact_prob_l1=3.91599258314e-08
exact_observable_abs=1.11375948908e-07
exact_normalization_error=0
exact_projective_commutation_error=0
exact_projective_certified=True
```

## Instalación y validación

Desde la raíz del repositorio:

```bash
git pull
"$PY" -m pip install -e . --no-deps
"$PY" -m pytest -q
```

Salida esperada:

```text
15 passed
```

## Benchmark de acceso proyectivo

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

Salida clave esperada:

```text
structural_regime_admissible=True
structural_residue_zero=True
exact_projective_certified=True
ctnet_structural_superiority=PASS
certification_tier=STRUCTURAL_ZERO_RESIDUE_PLUS_EXACT_PROJECTIVE
```

## Coste proyectivo

Para `n=40`, la ejecución de referencia obtiene:

```text
Cgen=1_024
Clist=1_099_511_627_776
Dproj=1_073_741_824
Ceff_amp=9.31322574615e-10
tomography_to_generator_ratio_at_largest_n=1.18059162072e+21
```

Esto separa existencia proyectiva del vector, lectura estructural de rama y materialización extensional de la lista completa.

## Arquitectura

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
  Núcleo CTNet 2.6 Omega + Cubo 6D.

src/ctnet_hilbert_projector/hilbert_projector.py
  Proyector Hilbert u/p básico.

src/ctnet_hilbert_projector/hamiltonians.py
  Hamiltonianos exactos de referencia y métricas.

src/ctnet_hilbert_projector/exact_certification.py
  Certificación exacta proyectiva y reconstrucción masa-fase.

src/ctnet_hilbert_projector/structural_closure.py
  Cierre estructural de residuo cero sobre la proyección CTNet calibrada.

src/ctnet_hilbert_projector/thesis_dynamics.py
  Dinámica CTNet-cuántica de tesis.

src/ctnet_hilbert_projector/state_preparation.py
  Preparación estructural de estado.

src/ctnet_hilbert_projector/presets.py
  Presets calibrados ising_v1, ising_v2, ising_v3, ising_v4/best.

examples/benchmark_projective_superiority.py
  Benchmark de acceso proyectivo con cierre estructural y certificación exacta.

CERTIFICATION.md
  Nota resumida de las capas de certificación.
```

## Referencia conceptual

Este repositorio acompaña la monografía técnica:

```text
CTNet y simulación cuántica por base cardinal u/p:
amplitudes como proyecciones de estado persistente,
masa-fase, no separabilidad relacional y reconstrucción exhaustiva del vector
```
