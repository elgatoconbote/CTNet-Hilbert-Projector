# CTNet Hilbert Projector: estado del hito

## Hito cerrado

Primer simulador cuántico estructural CTNet/Cubo 6D en base u/p con lectura amplitud-por-amplitud.

## Ruta certificada

P_Q
-> Observador
-> batch_to_state
-> Xi_t
-> Cubo6DObserver
-> closure_shear / forward_state
-> Omega_Q
-> u=p
-> Q_sigma(Xi_solution)

## Instancia certificada

- Modelo: Ising transverse-field 1D
- n: 6
- J: 1.0
- h: 0.5
- dt: 0.05
- steps: 3
- psi0: uuuuuu

## Certificado

- quantum_strong_certified: True
- Omega_6D: 0.0
- Omega_Q: 1.90802765587e-07
- epsilon_A: 7.9060427538e-08
- epsilon_P: 1.11742338049e-07
- phase_error: 0.0
- exhaustive_error: 0.0
- closure_error: 0.0
- normalization_error: 2.220446049250313e-16

## Regla de arquitectura

El solver no usa rutas externas de evolución exacta, matriz exponencial, vector Hilbert externo ni oráculo de comparación. La solución se lee como familia proyectiva:

A_t(sigma) = Q_sigma(Xi_t)

## Producto

Nombre operativo:

CTNet Hilbert Projector

Nombre descriptivo:

CTNet-(u/p) Quantum Projective Simulator

## Siguiente expansión

1. API de consulta: A(sigma), P(sigma), Theta(sigma)
2. Observables internos
3. Correlaciones C_ij
4. Entrelazamiento como no separabilidad relacional
5. Segundo modelo: Heisenberg XXZ
6. Tercer modelo: Fermi-Hubbard
7. Benchmark de coste efectivo D_proj(n)
