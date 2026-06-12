# Roadmap CTNet-Hilbert-Projector

## Fase 0 — Base funcional actual

- Copiar/adaptar núcleo CTNet 2.6 Omega + Cubo 6D.
- Mantener estado plegado fijo:

```text
Xi = pack(Z, M, R, C6, pad)
```

- Añadir proyector Hilbert por base cardinal:

```text
A_t(sigma) = Q_sigma(Xi_t), sigma in {u,p}^n
```

- Materializar tabla de amplitudes sólo para auditoría en `n` pequeño.

## Fase 1 — Auditoría de amplitud

- Normalización: `sum_sigma |A(sigma)|^2 = 1`.
- Positividad de masa: `mu(sigma) > 0`.
- Fase acotada y consultable.
- Lectura completa para `n <= 10`.
- Métrica inicial:

```text
C_amp_eff(n) = C_Xi(n) / 2^n
D_proj(n) = 2^n / C_Xi(n)
```

## Fase 2 — Hamiltonianos pequeños

Implementar generadores objetivo y pruebas contra diagonalización exacta:

1. Transverse-field Ising 1D

```text
H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
```

2. XXZ / Heisenberg

```text
H = sum_i Jx X_iX_{i+1} + Jy Y_iY_{i+1} + Jz Z_iZ_{i+1}
```

3. Bell, GHZ y W en base `u/p`.

## Fase 3 — No separabilidad relacional

- Añadir cociclo relacional `chi(sigma_A, sigma_B)`.
- Medir no factorización:

```text
A(sigma_A, sigma_B) != A_A(sigma_A) A_B(sigma_B)
```

- Añadir métricas de correlación y entropía por partición.

## Fase 4 — Simulación estructural

- Evolución proyectiva:

```text
Xi_{t+1} = T_rho(Xi_t, H_t, K_t, Omega_t)
A_{t+1}(sigma) = Q_sigma(Xi_{t+1})
```

- Comparar observables contra evolución exacta en `n` pequeño.
- Medir error de amplitud:

```text
epsilon_A = sqrt(sum_sigma |A_CT(sigma) - A_exact(sigma)|^2)
```

## Fase 5 — Sistemas de muchos cuerpos

- Cadenas de espines grandes con lectura de observables.
- Redes 2D pequeñas.
- Primer prototipo Hubbard/Fermi-Hubbard.

## Principio de no degeneración

`u/p` no debe interpretarse como `1/0` renombrado. La codificación numérica `u=+1`, `p=-1` sólo es una carta de cálculo. La tesis estructural es:

```text
u = actuación / presión de transición
p = inercia / continuidad estructural
```
