# CTNet Hilbert Projector - primer hito cerrado

## Producto

CTNet Hilbert Projector

Tambien descrito como:

CTNet-(u/p) Quantum Projective Simulator

## Tesis practica

El estado cuantico no se trata primariamente como una tabla plana de amplitudes, sino como una familia proyectiva generada desde un estado persistente CTNet/Cubo 6D.

A_t(sigma) = Q_sigma(Xi_t)

sigma pertenece a {u,p}^n

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
-> A_t(sigma)

## Instancia certificada

modelo = Ising transverse-field 1D
n = 6
J = 1.0
h = 0.5
dt = 0.05
steps = 3
psi0 = uuuuuu

## Certificado fuerte

quantum_strong_certified = True
Omega_6D = 0.0
Omega_Q = 1.90802765587e-07
epsilon_A = 7.9060427538e-08
epsilon_P = 1.11742338049e-07
phase_error = 0.0
exhaustive_error = 0.0
closure_error = 0.0
normalization_error = 2.220446049250313e-16

## Fase Cubo

phi_cubo_rad = 2.253564349891881
phi_cubo_over_pi = 0.7173318117219329
eiphi_real = -0.6309429516256336
eiphi_imag = 0.7758292284993736

## Lecturas soportadas

A(sigma)      amplitud compleja
P(sigma)      probabilidad Born
Theta(sigma)  fase de rama
DeltaTheta    diferencia de fase entre ramas
certificate   resumen del certificado

## Proximos modelos

1. Heisenberg XXZ
2. Fermi-Hubbard
3. Benchmark de densidad proyectiva D_proj(n)
4. Observables internos
5. Correlaciones C_ij
6. Entrelazamiento como no separabilidad relacional
