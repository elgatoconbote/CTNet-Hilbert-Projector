# CTNet Hilbert Projector - bloque funcional reproducible, auditable y observable

## Producto

CTNet Hilbert Projector

Tambien descrito como:

    CTNet-(u/p) Quantum Projective Simulator

## Estado actual

El proyecto tiene cerrado un bloque funcional completo para la instancia Ising transverse-field n=6 en base cardinal u/p.

El estado actual no es solo un primer certificado. La ruta ya es:

    reproducible
    auditable
    observable
    relacional
    coherente por fase

La reproduccion oficial ejecuta:

    tests
    solver Cubo-only
    lectura proyectiva Q_sigma
    auditoria del artefacto .pt
    observables diagonales
    fase/interferencia por pares
    matriz de coherencia proyectiva

## Tesis practica

El estado cuantico no se trata primariamente como una tabla plana de amplitudes, sino como una familia proyectiva generada desde un estado persistente CTNet/Cubo 6D.

    A_t(sigma) = Q_sigma(Xi_t)

con:

    sigma in {u,p}^n

El vector completo no es el objeto primario. El objeto primario es el estado generador Xi y su ley proyectiva de lectura:

    sigma -> Q_sigma(Xi)

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

La ruta extendida actual es:

    Xi_solution
    -> Q_sigma(Xi_solution)
    -> A_sigma
    -> P_sigma
    -> Theta_sigma
    -> observables diagonales
    -> DeltaTheta(sigma,tau)
    -> interference_kernel(sigma,tau)
    -> coherence_matrix(sigma,tau)

## Instancia certificada

    modelo = Ising transverse-field 1D
    n = 6
    branches = 64
    J = 1.0
    h = 0.5
    dt = 0.05
    steps = 3
    psi0 = uuuuuu

Artefacto de referencia:

    /tmp/cubo6d_strong_quantum.pt

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
    normalization_from_amplitudes = 2.220446049250313e-16
    born_with_phi_from_amplitudes = 2.220446049250313e-16
    amplitude_count = 64
    expected_amplitude_count = 64

## Fase Cubo

    phi_cubo_rad = 2.253564349891881
    phi_cubo_over_pi = 0.7173318117219329
    eiphi_real = -0.6309429516256336
    eiphi_imag = 0.7758292284993736

## Mejora de cierre: parada por meseta

Se ha añadido parada por meseta/oscilacion en solve_with_cubo.

Resultado validado:

    closure_steps_used = 24
    best_omega_eff = 0.006241069711053901
    last_omega_eff = 0.006241069711053901
    last_candidate = closure_shear_minus

Interpretacion:

La carta Cubo 6D alcanza una orbita de cierre de dos ciclos alrededor del minimo operativo de omega_eff. La parada por meseta evita gastar pasos de cierre sin perdida de exactitud cuantica certificada.

## Reproduccion oficial

Comando unico:

    PY=/home/elgatoconbote/CTNet-Omega-cubo-6D/.venv/bin/python \
    STATE=/tmp/cubo6d_strong_quantum.pt \
    scripts/reproduce_strong_certificate.sh

Criterio de exito:

    tests passed
    quantum_strong_certified=True
    CERTIFICATE_OK=True
    closure_steps_used=24
    amplitude_count=64
    normalization_error ~= 0

## Componentes cerrados

    examples/solve_ising_cubo6d_only.py
        Solver Cubo-only para la instancia Ising u/p certificada.

    src/ctnet_hilbert_projector/projective_engine.py
        API de lectura proyectiva A(sigma), P(sigma), Theta(sigma), DeltaTheta y certificate.

    scripts/audit_strong_certificate.py
        Auditoria independiente del artefacto .pt.

    scripts/reproduce_strong_certificate.sh
        Reproduccion completa extremo a extremo.

    scripts/measure_projective_observables.py
        Observables diagonales internos en base cardinal u/p.

    scripts/measure_projective_phase.py
        Fase, DeltaTheta e interferencia por pares de ramas.

    scripts/measure_projective_coherence_matrix.py
        Matriz relacional de coherencia cardinal.

    scripts/measure_projective_density.py
        Densidad proyectiva D_proj(n): estructura proyectiva por unidad de amplitud materializada.

## Lecturas soportadas

    A(sigma)
        amplitud compleja

    P(sigma)
        probabilidad Born

    Theta(sigma)
        fase de rama

    DeltaTheta(sigma,tau)
        diferencia de fase envuelta

    certificate
        resumen del certificado fuerte

    Z_i
        observable cardinal local

    Z_i Z_j
        correlacion diagonal cardinal

    magnetization_z
        magnetizacion media cardinal

    sector_mass_by_u_count
        masa sectorial por numero de simbolos u

    interference_kernel(sigma,tau)
        sqrt(P_sigma P_tau) * cos(DeltaTheta)

    quadrature_kernel(sigma,tau)
        sqrt(P_sigma P_tau) * sin(DeltaTheta)

    coherence_matrix(sigma,tau)
        mapa relacional de fase e interferencia sobre ramas seleccionadas

    D_proj(n)
        densidad de lectura estructural proyectiva frente a amplitudes materializadas

## Observables validados

Sobre /tmp/cubo6d_strong_quantum.pt:

    magnetization_z = -0.20809576248576475

    <Z_0> = -0.22494105045989862
    <Z_1> = -0.17434720894014868
    <Z_2> = -0.23282804851731762
    <Z_3> = -0.20287818379781389
    <Z_4> = -0.28194010604323916
    <Z_5> = -0.13163997715617046

    <Z_0Z_1> = 0.037139260328163906
    <Z_1Z_2> = 0.039774889634908425
    <Z_2Z_3> = 0.046263564021826839
    <Z_3Z_4> = 0.056351875945613923
    <Z_4Z_5> = 0.036208761647536399

Masa sectorial por numero de u:

    u_count=0  0.047748779772536595
    u_count=1  0.19053282358588583
    u_count=2  0.31447223002019076
    u_count=3  0.2748014146406797
    u_count=4  0.1340955299912478
    u_count=5  0.034646266907348601
    u_count=6  0.003702955082111025

Interpretacion:

El estado certificado esta inclinado hacia sectores de baja-media actuacion u. La masa maxima aparece en u_count=2, seguida por u_count=3. La magnetizacion negativa indica predominio modal de p sobre u.

## Fase e interferencia validadas

Par extremo:

    pair = uuuuuu:pppppp
    Theta(uuuuuu) = -0.85713605096373779
    Theta(pppppp) = 0.84589943799066292
    DeltaTheta = -1.7030354889544008
    cos_DeltaTheta = -0.13185408365280632
    sin_DeltaTheta = -0.99126913632175528
    sqrt_P_sigma_P_tau = 0.013297051805694167
    interference_kernel = -0.0017532705811236979
    quadrature_kernel = -0.013180957059056093

Par interno:

    pair = upupup:puupup
    Theta(upupup) = 0.24176084877850115
    Theta(puupup) = -0.068334976115374577
    DeltaTheta = 0.31009582489387572
    cos_DeltaTheta = 0.95230433330193631
    sin_DeltaTheta = 0.30514989230598566
    sqrt_P_sigma_P_tau = 0.011285641378477838
    interference_kernel = 0.010747365188816083
    quadrature_kernel = 0.0034438122512464881

Interpretacion:

    uuuuuu:pppppp
        relacion casi cuadratural y debilmente destructiva en la parte real.

    upupup:puupup
        relacion constructiva con fuerte alineacion de fase.

## Matriz de coherencia proyectiva

Ramas de referencia:

    uuuuuu
    pppppp
    upupup
    puupup

La diagonal de interference_kernel recupera la masa de rama:

    interference_kernel(uuuuuu,uuuuuu) = 0.003702955082111025
    interference_kernel(pppppp,pppppp) = 0.047748779772536595
    interference_kernel(upupup,upupup) = 0.010705676763146351
    interference_kernel(puupup,puupup) = 0.011897024741309202

Relaciones reales destacadas:

    K(uuuuuu,pppppp) = -0.0017532705811236979
    K(upupup,puupup) = 0.010747365188816083
    K(pppppp,upupup) = 0.018607313438026917
    K(pppppp,puupup) = 0.014548319615170807

Relaciones cuadraturales destacadas:

    Q(uuuuuu,pppppp) = -0.013180957059056093
    Q(pppppp,uuuuuu) = 0.013180957059056093

    Q(upupup,puupup) = 0.0034438122512464881
    Q(puupup,upupup) = -0.0034438122512464881

Interpretacion:

La matriz separa masa diagonal, interferencia real y orientacion relacional de fase. Esto convierte lecturas de amplitud por rama en un mapa cardinal de coherencia.

## Densidad proyectiva D_proj(n)

Definicion operativa:

    D_proj(n) =
        projective_structural_scalars
        /
        full_amplitude_scalars

con:

    full_amplitude_scalars = 2 * 2^n

y:

    projective_structural_scalars =
        branch_readout_scalars
        + diagonal_observable_scalars
        + coherence_matrix_scalars
        + audit_certificate_scalars

Resultado validado para n=6 con cuatro ramas seleccionadas:

    projective_structural_scalars = 131
    full_amplitude_scalars = 128
    D_proj_per_full_amplitude_scalar = 1.0234375
    D_proj_per_complex_amplitude = 2.046875
    D_proj_per_selected_branch = 32.75

Interpretacion:

D_proj(n) mide cuanta estructura proyectiva, observable, relacional y auditable se obtiene alrededor de Q_sigma(Xi) por unidad de amplitud compleja materializada externamente.

No sustituye al certificado fuerte. Lo complementa como metrica de densidad estructural.

## Commits funcionales recientes

    b284a44 Add plateau stop for Cubo 6D closure
    0ad6f89 Add strong certificate audit script
    ead09e2 Add strong certificate reproduction script
    209de4e Document strong certificate reproduction command
    4fafd1a Add projective observable measurement script
    c58e433 Document projective observables in reproduction flow
    4d7db25 Add projective phase interference measurement
    00c6833 Add projective coherence matrix measurement

## Estado del bloque

Bloque funcional cerrado:

    Cubo-only strong quantum certificate
    +
    projective u/p amplitude readout API
    +
    plateau stop
    +
    independent .pt audit
    +
    one-command reproduction
    +
    internal projective observables
    +
    phase/interference measurement
    +
    projective coherence matrix
    +
    projective density D_proj(n)

## Proximos hitos

1. Barrido de ramas para mapa de coherencia ampliado.
2. Observables no diagonales efectivos en carta u/p.
3. Entrelazamiento como no separabilidad relacional.
4. Segundo modelo: Heisenberg XXZ.
5. Tercer modelo: Fermi-Hubbard.
6. Benchmark de coste efectivo por amplitud definida.
7. Comparativa frente a reconstruccion/tomografia Hilbert externa.
