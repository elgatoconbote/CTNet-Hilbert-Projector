# CTNet Hilbert Projector

CTNet Hilbert Projector implementa un simulador cuantico estructural CTNet/Cubo 6D en base cardinal u/p.

La idea central no es almacenar el vector de Hilbert como una lista plana de amplitudes, sino mantener un estado persistente CTNet:

    Xi = pack(Z, M, R, C6, pad)

y leer amplitudes por rama cardinal:

    A_t(sigma) = Q_sigma(Xi_t)

donde:

    sigma in {u,p}^n

En esta lectura:

    0/1                  -> u/p
    amplitud             -> lectura proyectiva de rama
    probabilidad         -> masa modal normalizada
    fase                 -> memoria reversible de trayectoria
    entrelazamiento      -> no separabilidad relacional
    medicion             -> proyeccion de carta, rama o sector
    vector completo      -> familia proyectiva total
    CTNet                -> estado generador con acceso estructural interno

## Estado actual

El primer hito funcional esta cerrado.

Se ha certificado una instancia cuantica fuerte usando ruta Cubo-only:

    modelo = Ising transverse-field 1D
    n = 6
    J = 1.0
    h = 0.5
    dt = 0.05
    steps = 3
    psi0 = uuuuuu

Certificado guardado en:

    /tmp/cubo6d_strong_quantum.pt

Valores certificados:

    quantum_strong_certified = True
    Omega_6D = 0.0
    Omega_Q = 1.90802765587e-07
    epsilon_A = 7.9060427538e-08
    epsilon_P = 1.11742338049e-07
    phase_error = 0.0
    exhaustive_error = 0.0
    closure_error = 0.0
    normalization_error = 2.220446049250313e-16

Fase Cubo de la instancia:

    phi_cubo_rad = 2.253564349891881
    phi_cubo_over_pi = 0.7173318117219329
    eiphi_real = -0.6309429516256336
    eiphi_imag = 0.7758292284993736

## Ruta certificada

La ruta operativa del solver es:

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

La salida se interpreta como:

    Q_sigma(Xi_solution)
    =
    exp(i phi_Cubo) * rama_cuantica_cerrada(P_Q)_sigma

El certificado no depende de un oraculo Hilbert externo. La solucion se lee desde el estado CTNet/Cubo 6D cerrado.

## Componentes principales

    examples/solve_ising_cubo6d_only.py
        Solver Cubo-only para la instancia Ising u/p certificada.

    src/ctnet_hilbert_projector/projective_engine.py
        API de lectura proyectiva para estados .pt certificados.

    tests/test_projective_engine.py
        Tests minimos de lectura u/p, indice de rama, amplitud y normalizacion.

    PROJECT_STATUS.md
        Estado formal del primer hito cerrado.

## API de lectura proyectiva

El lector permite consultar:

    A(sigma)      amplitud compleja
    P(sigma)      probabilidad Born
    Theta(sigma)  fase de rama
    DeltaTheta    diferencia de fase entre ramas
    certificate   resumen del certificado

Ejemplo:

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=/home/elgatoconbote/CTNet-Omega-cubo-6D/.venv/bin/python

    $PY -m ctnet_hilbert_projector.projective_engine \
      /tmp/cubo6d_strong_quantum.pt \
      --certificate \
      uuuuuu \
      pppppp \
      upupup \
      puupup

Salida esperada:

    quantum_strong_certified=True
    final_omega_6d=0.0
    final_omega_q=1.90802765587e-07
    normalization_error=2.220446049250313e-16

y lecturas de ramas como:

    sigma=uuuuuu index=63 A.real=... A.imag=... P=... Theta=...
    sigma=pppppp index=0  A.real=... A.imag=... P=... Theta=...

## Ejecutar el solver certificado

    PY=/home/elgatoconbote/CTNet-Omega-cubo-6D/.venv/bin/python

    $PY examples/solve_ising_cubo6d_only.py \
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
      --save /tmp/cubo6d_strong_quantum.pt

## Ejecutar tests

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=/home/elgatoconbote/CTNet-Omega-cubo-6D/.venv/bin/python

    $PY -m py_compile examples/solve_ising_cubo6d_only.py
    $PY -m py_compile src/ctnet_hilbert_projector/projective_engine.py
    $PY -m pytest -q tests/test_projective_engine.py

## Tesis operativa

CTNet Hilbert Projector representa estados cuanticos como familias proyectivas de un estado persistente.

No identifica el vector con una tabla plana. Mantiene un generador Xi y define cada amplitud como lectura:

    A_t(sigma) = Q_sigma(Xi_t)

La regla masa-fase es:

    A_t(sigma) =
      sqrt(mu_t(sigma) / sum_tau mu_t(tau))
      * exp(i Theta_t(sigma))

La normalizacion Born aparece como condicion de cierre proyectivo:

    sum_sigma |A_t(sigma)|^2 = 1

El entrelazamiento se interpreta como no separabilidad relacional de memoria y banco R.

La decoherencia se interpreta como perdida de compatibilidad entre fase, regimen y cierre.

## Proximos hitos

1. Observables internos.
2. Correlaciones C_ij.
3. Diferencia de fase DeltaTheta(sigma,tau).
4. Entrelazamiento como no separabilidad relacional.
5. Segundo modelo: Heisenberg XXZ.
6. Tercer modelo: Fermi-Hubbard.
7. Benchmark de densidad proyectiva D_proj(n).
8. Medicion de coste efectivo por amplitud definida.

## Estado del producto

Primer bloque funcional terminado:

    Cubo-only strong quantum certificate
    +
    projective u/p amplitude readout API

Commit relevante:

    fdd3424 Add projective u-p amplitude readout API
