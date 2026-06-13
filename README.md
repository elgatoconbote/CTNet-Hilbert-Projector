# CTNet Hilbert Projector

CTNet Hilbert Projector implementa un simulador estructural CTNet/Cubo 6D en base cardinal u/p.

La idea central no es almacenar el estado como una lista plana de amplitudes, sino mantener un estado persistente:

    Xi = pack(Z, M, R, C6, pad)

Las amplitudes se leen por rama cardinal:

    A_t(sigma) = Q_sigma(Xi_t)

con:

    sigma in {u,p}^n

## Estado actual

El primer bloque funcional esta cerrado.

La arquitectura integra el nucleo CTNet/Cubo 6D, el puente Observador -> batch_to_state y los shims de compatibilidad dentro del paquete.

Instancia de referencia:

    modelo = Ising transverse-field 1D
    n = 6
    J = 1.0
    h = 0.5
    dt = 0.05
    steps = 3
    psi0 = uuuuuu

Valores de referencia:

    quantum_strong_certified = True
    Omega_6D = 0.0
    Omega_Q = 1.90802765587e-07
    epsilon_A = 7.9060427538e-08
    epsilon_P = 1.11742338049e-07
    normalization_error = 2.220446049250313e-16

## Ruta operativa

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

## Reproduccion

Instalacion local recomendada:

    python3 -m venv .venv
    . .venv/bin/activate
    python3 -m pip install -e '.[dev]'

Comando unico:

    STATE=/tmp/cubo6d_strong_quantum.pt PY=python3 scripts/reproduce_strong_certificate.sh

Criterio esperado:

    18 passed
    closure_steps_used=24
    quantum_strong_certified=True
    CERTIFICATE_OK=True
    amplitude_count=64
    expected_amplitude_count=64
    normalization_error=2.220446049250313e-16

## Componentes

    examples/solve_ising_cubo6d_only.py
        Solver Cubo-only.

    src/ctnet_hilbert_projector/ctnet_omega_core.py
        Nucleo CTNet/Cubo 6D integrado en el paquete.

    src/ctnet_hilbert_projector/observer_bridge.py
        Puente Observador -> batch_to_state -> Xi.

    src/ctnet_hilbert_projector/projective_engine.py
        API de lectura proyectiva.

    ctnet_omega_cubo6d_plegado_ctnet26.py
    train_vram_up_coherence_ctnet.py
        Shims de compatibilidad para imports historicos.

    scripts/reproduce_strong_certificate.sh
        Reproduccion completa extremo a extremo.

    scripts/audit_strong_certificate.py
        Auditoria del artefacto .pt.

    scripts/measure_projective_observables.py
    scripts/measure_projective_phase.py
    scripts/measure_projective_coherence_matrix.py
    scripts/measure_projective_density.py
        Lecturas proyectivas, observables, fase, coherencia y densidad.

    STANDALONE.md
        Guia de reproduccion local.

    PROJECT_STATUS.md
        Estado formal del bloque.

## Lectura proyectiva

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=python3

    $PY -m ctnet_hilbert_projector.projective_engine \
      /tmp/cubo6d_strong_quantum.pt \
      --certificate \
      uuuuuu \
      pppppp \
      upupup \
      puupup

## Tests

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=python3

    $PY -m py_compile examples/solve_ising_cubo6d_only.py
    $PY -m py_compile src/ctnet_hilbert_projector/ctnet_omega_core.py
    $PY -m py_compile src/ctnet_hilbert_projector/observer_bridge.py
    $PY -m py_compile src/ctnet_hilbert_projector/projective_engine.py
    $PY -m pytest -q

## Tesis operativa

El vector completo no es el objeto primario. El objeto primario es el estado generador Xi y su ley proyectiva de lectura:

    sigma -> Q_sigma(Xi)

La lista completa de amplitudes es una salida posible, no la estructura primaria.
