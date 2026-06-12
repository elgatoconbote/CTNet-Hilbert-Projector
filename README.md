# CTNet Hilbert Projector

CTNet-Hilbert-Projector es el primer prototipo para llevar CTNet 2.6 Omega + Cubo 6D al problema de simulación cuántica por base cardinal `u/p`.

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
0/1  ->  u/p
amplitud -> lectura proyectiva de rama
probabilidad -> masa modal normalizada
fase -> memoria reversible de trayectoria
entrelazamiento -> no separabilidad relacional
medición -> proyección de carta
vector completo -> familia proyectiva total
```

## Estado del repositorio

Este commit inicia el trabajo copiando/adaptando el núcleo CTNet desde `CTNet-2.6-Omega-Cubo-6D` y montando encima el primer proyector Hilbert `u/p`.

Componentes iniciales:

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
  Núcleo CTNet 2.6 Omega + Cubo 6D plegado: Z, M, R, C6, pad, tensor de coherencia y reversibilidad.

src/ctnet_hilbert_projector/hilbert_projector.py
  Proyector Hilbert u/p: ramas cardinales, lectura de amplitud, masa-fase, normalización y enumeración auditada.

examples/demo_small_projector.py
  Demo mínima: inicializa CTNet, enumera ramas u/p pequeñas y muestra amplitudes normalizadas.
```

## Instalación mínima

```bash
pip install -e .
```

## Demo

```bash
python examples/demo_small_projector.py --n 3 --steps 1
```

## Auditoría conceptual

El objetivo experimental inmediato es validar, para `n` pequeño:

```text
sum_sigma |A_t(sigma)|^2 = 1
A_t(sigma) consultable para toda rama sigma
fase Theta_t(sigma) definida por lectura de estado
masa mu_t(sigma) positiva y normalizable
```

Después se comparará contra Hamiltonianos pequeños: Ising transverse-field, XXZ y finalmente Hubbard/Fermi-Hubbard.

## Tesis operativa

CTNet no enumera primero el espacio de Hilbert y luego calcula. Sustituye la base `0/1` por la base `u/p`, convierte cada rama en modo interno del estado persistente, genera módulo y fase desde masa y reversibilidad, conserva la no separabilidad en memoria y relaciones, y reproduce el vector completo amplitud por amplitud como familia proyectiva de lecturas sobre `Xi_t`.
