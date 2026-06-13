# CTNet Hilbert Projector - estado del bloque

## Estado actual

El proyecto tiene cerrado un bloque funcional para la instancia Ising transverse-field n=6 en base cardinal u/p.

La arquitectura integra el nucleo CTNet/Cubo 6D, el puente Observador -> batch_to_state y los shims de compatibilidad dentro del paquete.

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

## Instancia de referencia

    modelo = Ising transverse-field 1D
    n = 6
    branches = 64
    J = 1.0
    h = 0.5
    dt = 0.05
    steps = 3
    psi0 = uuuuuu

## Valores de referencia

    quantum_strong_certified = True
    Omega_6D = 0.0
    Omega_Q = 1.90802765587e-07
    epsilon_A = 7.9060427538e-08
    epsilon_P = 1.11742338049e-07
    phase_error = 0.0
    exhaustive_error = 0.0
    closure_error = 0.0
    normalization_error = 2.220446049250313e-16
    amplitude_count = 64
    expected_amplitude_count = 64

## Cierre por meseta

    closure_steps_used = 24
    best_omega_eff = 0.006241069711053901
    last_omega_eff = 0.006241069711053901
    last_candidate = closure_shear_minus

La carta Cubo 6D alcanza una orbita de cierre de dos ciclos alrededor del minimo operativo de omega_eff.

## Reproduccion oficial

Comando unico:

    STATE=/tmp/cubo6d_strong_quantum.pt PY=python3 scripts/reproduce_strong_certificate.sh

Criterio de exito:

    18 passed
    quantum_strong_certified=True
    CERTIFICATE_OK=True
    closure_steps_used=24
    amplitude_count=64
    normalization_error ~= 0

## Componentes cerrados

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

    STANDALONE.md
        Guia de reproduccion local.

    scripts/audit_strong_certificate.py
        Auditoria del artefacto .pt.

    scripts/reproduce_strong_certificate.sh
        Reproduccion extremo a extremo.

    scripts/measure_projective_observables.py
    scripts/measure_projective_phase.py
    scripts/measure_projective_coherence_matrix.py
    scripts/measure_projective_density.py
        Lecturas proyectivas, observables, fase, coherencia y densidad.

## Lecturas soportadas

    A(sigma)
    P(sigma)
    Theta(sigma)
    DeltaTheta(sigma,tau)
    certificate
    Z_i
    Z_i Z_j
    magnetization_z
    sector_mass_by_u_count
    interference_kernel(sigma,tau)
    quadrature_kernel(sigma,tau)
    coherence_matrix(sigma,tau)
    D_proj(n)

## Densidad proyectiva D_proj(n)

Resultado validado para n=6 con cuatro ramas seleccionadas:

    projective_structural_scalars = 131
    full_amplitude_scalars = 128
    D_proj_per_full_amplitude_scalar = 1.0234375
    D_proj_per_complex_amplitude = 2.046875
    D_proj_per_selected_branch = 32.75

## Proximos hitos

1. Barrido de ramas para mapa de coherencia ampliado.
2. Observables no diagonales efectivos en carta u/p.
3. Entrelazamiento como no separabilidad relacional.
4. Segundo modelo: Heisenberg XXZ.
5. Tercer modelo: Fermi-Hubbard.
6. Benchmark de coste efectivo por amplitud definida.
7. Comparativa frente a reconstruccion/tomografia Hilbert externa.
