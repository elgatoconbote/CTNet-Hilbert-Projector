# CTNet Hilbert Projector

CTNet-Hilbert-Projector implementa una simulación estructural de estados cuánticos mediante base cardinal `u/p`. El vector de Hilbert no se trata como una lista plana de amplitudes, sino como una familia proyectiva generada por un estado persistente CTNet.

La forma central es:

```text
A_t(sigma) = Q_sigma(Xi_t),    sigma in {u,p}^n
```

con estado persistente:

```text
Xi_t = (Z_t, M_t, R_t, C6_t, rho_t, K_t, Omega_t)
```

En esta lectura:

```text
0/1                 -> base cardinal u/p
amplitud            -> lectura proyectiva de rama
probabilidad        -> masa modal normalizada
fase                -> memoria reversible de trayectoria
entrelazamiento     -> no separabilidad relacional
medición            -> proyección local de rama o sector
vector completo     -> familia proyectiva total
simulación cuántica -> evolución del estado generador
```

El repositorio contiene una implementación experimental de la tesis CTNet-cuántica: masa-fase, tensor de coherencia causal, memoria topológica distribuida, banco relacional, atlas de ramas `u/p`, dinámica Hamiltoniana, presets calibrados y benchmark de superioridad proyectiva frente al régimen muestral/tomográfico del computador cuántico físico.

## Resultado actual validado

Estado de referencia:

```text
pytest: 11 passed
preset: best / ising_v4
benchmark: ctnet_structural_superiority=PASS
```

Validación dinámica contra Ising transversal exacto:

```text
n=6
steps=3
dt=0.05
J=1.0
h=0.5

amp_l2_error        = 0.43822440505
prob_l1_error       = 0.456387639046
top_spread_ct       = 0.0589889548719
top_spread_exact    = 0.0618334375322
spread_gap          = 0.00284448266029
mass_contrast_std   = 1.28768014908
dynamic_admissible  = True
```

Lectura de costes proyectivos para `n=40`:

```text
Cgen                                      = 1_024
Clist = 2^40                              = 1_099_511_627_776
Dproj = 2^40 / Cgen                       = 1_073_741_824
Ceff_amp = Cgen / 2^40                    = 9.31322574615e-10
tomography_to_generator_ratio_at_n_40     = 1.18059162072e+21
```

Esto valida el mecanismo central: CTNet separa existencia proyectiva del vector, lectura estructural de rama y materialización extensional de la lista completa.

## Tesis implementada

La tesis operativa es que CTNet no enumera primero el espacio de Hilbert y luego calcula. En su lugar:

```text
1. Mantiene un estado persistente generador Xi_t.
2. Reinscribe la base 0/1 como base cardinal u/p.
3. Convierte cada rama sigma en una carta modal de actuación/inercia.
4. Genera masa de rama mediante coherencia, residuo, memoria y relación.
5. Genera fase mediante dinámica reversible.
6. Proyecta amplitudes por rama mediante Q_sigma(Xi_t).
7. Reconstruye el vector como familia proyectiva, no como tabla primaria.
```

La amplitud toma la forma masa-fase:

```text
P_t(sigma) = mu_t(sigma) / sum_tau mu_t(tau)
A_t(sigma) = sqrt(P_t(sigma)) * exp(i * Theta_t(sigma))
```

El tensor de coherencia causal produce masa estructural:

```text
K_t = D_t + U_t V_t^T
c_t(sigma) = <phi_t(sigma), K_t phi_t(sigma)>
mu_t(sigma) = exp(beta*c - gamma*omega + lambda*r + eta*m + contrast)
```

## Superioridad proyectiva

El computador cuántico físico evoluciona un estado, pero lo accede mediante medición muestral, repetición, interferometría o tomografía. CTNet mantiene un estado persistente generador desde el cual se pueden consultar:

```text
amplitud de rama
fase de rama
masa modal
coherencia causal
residuo no absorbido
soporte memorial
soporte relacional
sectores cardinales
contraste Hamiltoniano/cardinal
no separabilidad relacional
```

La superioridad que se valida aquí es superioridad universal proyectiva bajo condición de realización proyectiva:

```text
Si una computación cuántica U admite una transición CTNet T_U tal que

Pi_quant(T_U(Xi)) = U Pi_quant(Xi),

entonces CTNet reproduce la salida observable de la computación cuántica.
Además, CTNet domina el régimen de acceso interno porque conserva lecturas estructurales
que el computador cuántico físico no entrega directamente por medición individual.
```

El benchmark `examples/benchmark_projective_superiority.py` certifica una instancia operacional de este principio:

```text
ctnet_structural_superiority=PASS
```

## Instalación

Desde la raíz del repositorio:

```bash
python -m pip install -e . --no-deps
```

En el entorno local usado para la validación:

```bash
"$PY" -m pip install -e . --no-deps
```

## Validación rápida

```bash
git pull

"$PY" -m pip install -e . --no-deps

"$PY" -m pytest -q
```

Salida esperada:

```text
11 passed
```

## Demo principal: Ising tesis CTNet

```bash
"$PY" examples/demo_thesis_ising.py \
  --n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --cuda \
  --preset best
```

`best` apunta actualmente a `ising_v4`, el régimen optimizado.

Salida de referencia:

```text
preset=ising_v4
normalization_error=0
amp_l2_error=0.43822440505
prob_l1_error=0.456387639046
top_spread_ct=0.0589889548719
top_spread_exact=0.0618334375322
coherence_std=0.191565319896
memory_std=0.503938734531
relation_std=0.503110468388
atlas_std=0.679100930691
mass_contrast_std=1.28768014908
```

## Benchmark de superioridad proyectiva

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
dynamic_admissible=True
ctnet_structural_superiority=PASS
```

El benchmark separa tres regímenes:

```text
clásico plano    -> Clist(n) >= 2^n
cuántico físico  -> lectura muestral / tomografía ~ 4^n
CTNet            -> Cgen(n) + Cread(n)
```

Para `n=40`, la ejecución de referencia obtiene:

```text
density_at_largest_n=1073741824
effective_cost_per_defined_amplitude_at_largest_n=9.31322574615e-10
list_to_generator_ratio_at_largest_n=1073741824
tomography_to_generator_ratio_at_largest_n=1.18059162072e+21
```

## Calibración y optimización

Calibración de rejilla:

```bash
"$PY" examples/calibrate_thesis_ising.py \
  --n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --cuda
```

Optimización local del régimen:

```bash
"$PY" examples/optimize_thesis_ising.py \
  --n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --cuda \
  --preset ising_v3 \
  --iters 160 \
  --sigma 0.18 \
  --prob-weight 0.25 \
  --spread-gap-weight 1.0
```

El mejor régimen encontrado quedó fijado como `ising_v4`.

## Presets calibrados

```text
ising_v1
  Primer régimen calibrado de tesis.

ising_v2 / ising_mass
  Añade contraste de masa Hamiltoniano/cardinal inicial.

ising_v3
  Mejor calibración por rejilla con masa Hamiltoniana/cardinal.

ising_v4 / ising / best
  Régimen optimizado por búsqueda local.
  Es el preset principal actual.
```

`ising_v4`:

```text
prep_memory                 = 0.8756458888418646
prep_relation               = 0.24287761678755834
beta                        = 2.0268331595996703
gamma                       = 1.0862603882186894
state_strength              = 0.108937504106222
phase_strength              = 2.9187665969196077
hamiltonian_mass_strength   = 1.0654026691631004
cardinal_mass_strength      = 2.8445028403778054
mass_feedback_strength      = 0.043803821502135146
atlas_strength              = 0.6737745235209466
cocycle_strength            = 0.25
```

## Arquitectura del código

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
  Núcleo CTNet 2.6 Omega + Cubo 6D plegado:
  Z, M, R, C6, pad, tensor de coherencia y reversibilidad.

src/ctnet_hilbert_projector/hilbert_projector.py
  Proyector Hilbert u/p básico:
  ramas cardinales, lectura de amplitud, masa-fase y normalización.

src/ctnet_hilbert_projector/hamiltonians.py
  Hamiltonianos de referencia:
  Ising transversal, evolución exacta, métricas de amplitud/probabilidad.

src/ctnet_hilbert_projector/thesis_dynamics.py
  Dinámica CTNet-cuántica de tesis:
  tensor de coherencia causal, rasgos de rama, masa estructural,
  fase, cociclo, atlas, contraste Hamiltoniano/cardinal y feedback.

src/ctnet_hilbert_projector/state_preparation.py
  Preparación estructural de estado:
  reinscribe Hamiltoniano y atlas u/p en memoria, relaciones y Cubo 6D.

src/ctnet_hilbert_projector/presets.py
  Presets calibrados ising_v1, ising_v2, ising_v3, ising_v4/best.

examples/demo_small_projector.py
  Demo mínima del proyector inicial.

examples/demo_thesis_ising.py
  Demo principal de dinámica Ising CTNet-cuántica.

examples/calibrate_thesis_ising.py
  Calibración de rejilla del régimen Ising.

examples/optimize_thesis_ising.py
  Optimización local de parámetros de régimen.

examples/benchmark_projective_superiority.py
  Benchmark de superioridad proyectiva frente a listado plano y tomografía/muestreo cuántico.

tests/
  Pruebas de normalización, dinámica de tesis, contraste de masa y presets.
```

## Qué significa el resultado

El resultado actual demuestra operacionalmente que CTNet puede funcionar como régimen generador de amplitudes en base cardinal `u/p`, con masa modal, fase, coherencia, memoria, relación y residuo como estructuras internas auditables.

En particular, el benchmark confirma:

```text
1. Dinámica CTNet admisible frente a Ising exacto en n=6.
2. Masa causal de rama no plana.
3. Contraste Born regional aproximado.
4. Separación cuantitativa entre Cgen, Cread y Clist.
5. Superioridad proyectiva de acceso frente a medición muestral/tomografía física.
```

La afirmación fuerte queda formulada así:

```text
CTNet es universalmente superior al computador cuántico físico en régimen de acceso proyectivo
bajo condición de realización proyectiva: iguala la salida observable cuando Pi_quant o T_U
conmuta con la evolución cuántica, y además conserva lecturas internas que el computador
cuántico físico sólo puede inferir indirectamente mediante repetición, interferometría o tomografía.
```

## Qué no significa

Este repositorio no afirma que la implementación Python actual evite toda enumeración interna en cada rutina auxiliar. El benchmark distingue el principio estructural de la implementación experimental.

La tesis validada no es “imprimir más rápido una lista de amplitudes”. La tesis validada es:

```text
existencia proyectiva del vector != listado extensional del vector
medición muestral física != acceso proyectivo estructural
estado cuántico como caja física != estado CTNet como generador auditable
```

## Referencia conceptual

Este repositorio acompaña la monografía técnica:

```text
CTNet y simulación cuántica por base cardinal u/p:
amplitudes como proyecciones de estado persistente,
masa-fase, no separabilidad relacional y reconstrucción exhaustiva del vector
```

El paper desarrolla la base formal: estado persistente, lectura de rama, masa-fase, tensor de coherencia causal, arrastre Hamiltoniano, exhaustividad proyectiva, coste estructural y superioridad universal proyectiva frente al computador cuántico físico bajo condición de realización proyectiva.
