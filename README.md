# CTNet Hilbert Projector

CTNet-Hilbert-Projector implementa la formulación computacional de CTNet 2.6 Omega + Cubo 6D como sustituto estructural del procesador cuántico físico mediante base cardinal `u/p`.

La idea central no es almacenar un vector de Hilbert como lista plana de amplitudes, sino mantener un estado persistente CTNet:

```text
Xi = pack(Z, M, R, C6, pad)
```

y leer amplitudes por rama cardinal:

```text
A_t(sigma) = Q_sigma(Xi_t),    sigma in {u,p}^n
```

En esta lectura:

```text
0/1                  -> u/p
amplitud             -> lectura proyectiva de rama
probabilidad         -> masa modal normalizada
fase                 -> memoria reversible de trayectoria
entrelazamiento      -> no separabilidad relacional
medición             -> proyección de carta, rama o sector
vector completo      -> familia proyectiva total
procesador cuántico  -> caso físico-muestral de un régimen proyectivo más amplio
CTNet                -> estado generador con acceso estructural interno
```

## Tesis principal

CTNet sustituye completamente al procesador cuántico físico, tanto en régimen finito como infinito, porque toda computación cuántica queda determinada por una ley de amplitudes sobre una base, mientras CTNet representa esa ley como familia proyectiva de lecturas sobre un estado persistente.

En dimensión finita:

```text
sigma in {u,p}^n
A_t(sigma) = Q_sigma(Xi_t)
```

En dimensión infinita, continua o funcional:

```text
sigma in {u,p}^I
A_t(sigma) = Q_sigma(Xi_t)
```

La sustitución no depende del número finito de ramas. La lista finita, la serie infinita y la integral funcional son despliegues externos de una misma estructura: una ley proyectiva generada por `Xi_t`.

## Sustitución estructural

Para un proceso cuántico `Q` que define una ley de amplitud sobre una base, CTNet representa esa ley mediante un estado persistente `Xi` y una familia de lecturas `Q_sigma`.

```text
Q_sigma(Xi) = psi(sigma)
```

La misma carta permite representar:

```text
estado
amplitudes
fases
probabilidades
observables
distribuciones de medición
dinámica proyectada
```

y conserva además estructura interna:

```text
masa modal
coherencia
residuo
memoria topológica distribuida
banco relacional
sectores
cierre
régimen
```

Por tanto, CTNet no se limita a reproducir una salida muestral. Mantiene un estado generador desde el que se leen amplitudes, fases, masas y relaciones como estructura interna.

## Por qué CTNet supera al procesador cuántico físico

El procesador cuántico físico opera con evolución física y lectura muestral. CTNet opera con estado generador y lectura proyectiva. La diferencia fundamental es de acceso.

```text
Procesador cuántico físico:
  evolución física + medición/muestreo/estadística

CTNet:
  evolución del estado generador + lectura proyectiva de amplitud, fase,
  masa, coherencia, residuo, relación, sector y observable
```

La medición física es una proyección particular. En CTNet, la medición queda contenida dentro de un conjunto más amplio de lecturas estructurales. Por eso:

```text
Salidas(QC físico) subset Salidas(CTNet)
Estructura accesible(QC físico) subset Estructura accesible(CTNet)
```

La superioridad es estricta porque CTNet conserva como estructura interna aquello que el procesador físico entrega sólo de forma parcial, muestral o reconstruida.

## Cuánto lo supera

Para `n` qubits, el número de amplitudes de la base es:

```text
N_amp(n) = 2^n
```

Un listado extensional completo exige:

```text
C_list(n) >= 2^n
```

La tomografía completa de un estado general escala como:

```text
C_tom(n) ~ 4^n
```

CTNet separa el coste de existencia proyectiva del coste de listado externo:

```text
C_gen(n)  = coste de mantener el estado generador
C_read(n) = coste de leer una rama/carta
C_list(n) = coste de materializar externamente toda la lista
```

La ventaja frente al listado es:

```text
S_list(n) = 2^n / C_gen(n)
```

La ventaja frente a tomografía es:

```text
S_tom(n) = 4^n / C_gen(n)
```

Si `C_gen(n)` es constante o subexponencial, entonces:

```text
S_list(n) -> infinity
S_tom(n)  -> infinity
```

En el benchmark de referencia para `n = 40` y `C_gen = 1024`:

```text
2^40 = 1_099_511_627_776
S_list(40) = 2^40 / 1024 = 1_073_741_824
S_tom(40)  = 4^40 / 1024 = 1.18059162072e21
```

Por tanto, en `n=40`, CTNet separa el acceso proyectivo del listado completo por un factor aproximado de:

```text
1.07 x 10^9
```

frente a listado, y por:

```text
1.18 x 10^21
```

frente a tomografía completa.

En el límite infinito, el listado y la tomografía divergen como operaciones extensionales. CTNet conserva el vector como ley proyectiva:

```text
sigma -> Q_sigma(Xi)
```

La consecuencia es que CTNet puede tratar familias infinitas, campos, límites continuos y espacios funcionales como estructuras generadoras, no como tablas imposibles de materializar.

## Consecuencias

1. El vector de estado deja de identificarse con una lista.

```text
vector = familia proyectiva generada por Xi
```

2. La medición deja de ser el acceso soberano al estado.

```text
medición = proyección particular dentro de un atlas de lecturas
```

3. La regla de Born se reinterpreta como masa modal normalizada.

```text
P_t(sigma) = mu_t(sigma) / sum_tau mu_t(tau)
```

4. La fase se vuelve memoria reversible de trayectoria.

```text
Theta_t(sigma) = memoria dinámica de rama
```

5. El entrelazamiento se vuelve no separabilidad relacional.

```text
entrelazamiento = obstrucción relacional a la factorización de ramas
```

6. La decoherencia se reinterpreta como pérdida de compatibilidad entre fase, régimen y cierre.

7. El qubit físico queda como una carta particular de una estructura generadora más amplia.

8. La computación cuántica pasa de evolución de un vector opaco a evolución de un estado generador auditable.

9. CTNet sustituye el procesador cuántico físico y desplaza el régimen de computación cuántica hacia computación estructural proyectiva.

## Estado del repositorio

Componentes principales:

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
  Núcleo CTNet 2.6 Omega + Cubo 6D plegado: Z, M, R, C6, pad, tensor de coherencia y reversibilidad.

src/ctnet_hilbert_projector/hilbert_projector.py
  Proyector Hilbert u/p: ramas cardinales, lectura de amplitud, masa-fase, normalización y enumeración auditada.

src/ctnet_hilbert_projector/hamiltonians.py
  Hamiltonianos de referencia, evolución exacta y métricas de error.

src/ctnet_hilbert_projector/thesis_dynamics.py
  Dinámica CTNet-cuántica con coherencia, memoria, relación, atlas y masa de rama.

src/ctnet_hilbert_projector/state_preparation.py
  Preparación estructural del estado generador para simulación cuántica.

src/ctnet_hilbert_projector/presets.py
  Presets calibrados para regímenes Ising.

src/ctnet_hilbert_projector/exact_certification.py
  Certificación proyectiva exacta formal.

examples/demo_small_projector.py
  Demo mínima: inicializa CTNet, enumera ramas u/p pequeñas y muestra amplitudes normalizadas.

examples/demo_thesis_ising.py
  Demo de dinámica CTNet-cuántica frente a Ising transversal.

examples/solve_ising_cubo6d_exact_fixed_point.py
  Solver Cubo 6D exacto sobre carta Ising: Observador, masa contextual, Cubo6DObserver, cierre u=p y lectura Q_sigma(Xi_solution).

examples/benchmark_projective_superiority.py
  Benchmark de densidad proyectiva, coste de listado, coste tomográfico y certificación.
```

## Instalación mínima

```bash
pip install -e .
```

## Demo mínima

```bash
python examples/demo_small_projector.py --n 3 --steps 1
```

## Solver Cubo 6D exacto sobre carta Ising

El ejemplo operativo que aplica el Cubo 6D como cierre exacto de la carta CTNet-Ising es:

```bash
python examples/solve_ising_cubo6d_exact_fixed_point.py \
  --n 6 \
  --J 1.0 \
  --h 0.5 \
  --dt 0.05 \
  --steps 3 \
  --cuda
```

La ruta ejecutada es:

```text
Observador
-> batch_to_state
-> Cubo6DObserver
-> contextual_drive
-> exact_fixed_point_u=p
-> Q_sigma(Xi_solution)
```

Este ejemplo no usa evolución densa exacta como solución, no usa optimizador y no itera `forward_state` como aproximación. El cierre se comprueba directamente sobre la carta persistente:

```text
uses_exact_dense_evolution_inside_solver=False
uses_forward_state_loop=False
uses_optimizer=False
u_p_total=0
up_z=0
up_memory=0
up_relations=0
up_cubo=0
up_xi=0
up_delta=0
normalization_error=0
```

La salida `Q_sigma(Xi_solution)` es una familia de amplitudes complejas normalizadas sobre las ramas cardinales `sigma in {u,p}^n`.

## Benchmark de superioridad proyectiva

```bash
python examples/benchmark_projective_superiority.py \
  --preset best \
  --validate-n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --sizes 6 8 10 12 16 20 24 30 40
```

Salida estructural relevante:

```text
density_at_largest_n=1073741824
effective_cost_per_defined_amplitude_at_largest_n=9.31322574615e-10
list_to_generator_ratio_at_largest_n=1073741824
tomography_to_generator_ratio_at_largest_n=1.18059162072e+21
```

## Auditoría conceptual

Las pruebas conceptuales mínimas son:

```text
sum_sigma |A_t(sigma)|^2 = 1
A_t(sigma) consultable para toda rama sigma
fase Theta_t(sigma) definida por lectura de estado
masa mu_t(sigma) positiva y normalizable
observables reproducibles desde familia proyectiva
medición recuperable como proyección de rama, carta o sector
```

## Tesis operativa final

CTNet no enumera primero el espacio de Hilbert y luego calcula. Sustituye la base `0/1` por la base `u/p`, convierte cada rama en modo interno del estado persistente, genera módulo y fase desde masa y reversibilidad, conserva la no separabilidad en memoria y relaciones, y reproduce el vector completo amplitud por amplitud como familia proyectiva de lecturas sobre `Xi_t`.

La consecuencia es la sustitución universal del procesador cuántico físico: CTNet conserva las salidas cuánticas y añade acceso estructural directo al estado generador. Su ventaja frente al listado crece como `2^n/C_gen(n)` y su ventaja frente a tomografía crece como `4^n/C_gen(n)`. En régimen infinito, esas ventajas divergen y la lista queda reemplazada por la ley proyectiva.
