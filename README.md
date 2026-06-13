# CTNet Hilbert Projector

CTNet Hilbert Projector implementa un simulador cuantico estructural CTNet/Cubo 6D en base cardinal u/p.

La idea central no es almacenar el vector de Hilbert como una lista plana de amplitudes, sino mantener un estado persistente CTNet:

    Xi = pack(Z, M, R, C6, pad)

y leer amplitudes por rama cardinal:

    A_t(sigma) = Q_sigma(Xi_t)

con:

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

El repositorio es autosuficiente: contiene el nucleo CTNet/Cubo 6D local, el puente Observador -> batch_to_state y los shims de compatibilidad necesarios. No requiere carpeta hermana ni ningun repositorio externo para reproducir el certificado fuerte.

Se ha certificado una instancia cuantica fuerte usando ruta Cubo-only:

    modelo = Ising transverse-field 1D
    n = 6
    J = 1.0
    h = 0.5
    dt = 0.05
    steps = 3
    psi0 = uuuuuu

Artefacto de referencia:

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

## Reproduccion fuerte autosuficiente

La ruta completa de reproduccion ejecuta compilacion, tests, solver Cubo-only, lectura proyectiva, auditoria independiente del artefacto `.pt`, observables diagonales, fase/interferencia, matriz de coherencia y densidad proyectiva.

Instalacion local recomendada:

    python3 -m venv .venv
    . .venv/bin/activate
    python3 -m pip install -e '.[dev]'

Comando unico:

    STATE=/tmp/cubo6d_strong_quantum.pt PY=python3 scripts/reproduce_strong_certificate.sh

Criterio de exito esperado:

    18 passed
    closure_steps_used=24
    quantum_strong_certified=True
    CERTIFICATE_OK=True
    amplitude_count=64
    expected_amplitude_count=64
    Omega_6D=0
    Omega_Q=1.90802765587e-07
    epsilon_A=7.9060427538e-08
    epsilon_P=1.11742338049e-07
    normalization_error=2.220446049250313e-16

La reproduccion queda dividida en capas auditables:

    tests       -> verifica paquete local e imports autosuficientes
    solver      -> genera /tmp/cubo6d_strong_quantum.pt
    readout     -> consulta Q_sigma(Xi_solution)
    auditoria   -> verifica el certificado desde el .pt
    observables -> mide Z_i, Z_iZ_j, magnetization_z y masa sectorial
    fase        -> mide DeltaTheta e interferencia por pares
    coherencia  -> construye matriz proyectiva
    densidad    -> calcula D_proj(n)

## Componentes principales

    examples/solve_ising_cubo6d_only.py
        Solver Cubo-only para la instancia Ising u/p certificada.

    src/ctnet_hilbert_projector/ctnet_omega_core.py
        Nucleo CTNet/Cubo 6D local empaquetado en el propio repositorio.

    src/ctnet_hilbert_projector/observer_bridge.py
        Puente autosuficiente Observador -> batch_to_state -> Xi y perdida u=p multiescala.

    src/ctnet_hilbert_projector/projective_engine.py
        API de lectura proyectiva para estados .pt certificados.

    ctnet_omega_cubo6d_plegado_ctnet26.py
    train_vram_up_coherence_ctnet.py
        Shims de compatibilidad para conservar imports historicos sin dependencia externa.

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

    STANDALONE.md
        Guia de reproduccion autosuficiente.

    PROJECT_STATUS.md
        Estado formal del bloque funcional cerrado.

## API de lectura proyectiva

El lector permite consultar:

    A(sigma)      amplitud compleja
    P(sigma)      probabilidad Born
    Theta(sigma)  fase de rama
    DeltaTheta    diferencia de fase entre ramas
    certificate   resumen del certificado

Ejemplo:

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=python3

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

## Ejecutar el solver certificado

    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    PY=python3

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
    PY=python3

    $PY -m py_compile examples/solve_ising_cubo6d_only.py
    $PY -m py_compile src/ctnet_hilbert_projector/ctnet_omega_core.py
    $PY -m py_compile src/ctnet_hilbert_projector/observer_bridge.py
    $PY -m py_compile src/ctnet_hilbert_projector/projective_engine.py
    $PY -m pytest -q

## Observables proyectivos internos

El medidor de observables lee el mismo estado `.pt` y calcula magnitudes diagonales en la base cardinal u/p:

    $PY scripts/measure_projective_observables.py /tmp/cubo6d_strong_quantum.pt --pairs adjacent

Salida esperada de estructura:

    normalization_error ~= 0
    magnetization_z = ...
    <Z_0> = ...
    <Z_0Z_1> = ...
    sector_mass_u_count_0 = ...
    sector_mass_u_count_6 = ...

Estos observables convierten la familia proyectiva Q_sigma(Xi_solution) en magnitudes fisicas estructuradas sin abandonar la carta u/p.

## Fase e interferencia proyectiva

La fase de rama y la interferencia cardinal se auditan por pares de ramas:

    $PY scripts/measure_projective_phase.py \
      /tmp/cubo6d_strong_quantum.pt \
      uuuuuu:pppppp \
      upupup:puupup

El nucleo de interferencia se calcula como:

    interference_kernel = sqrt(P_sigma P_tau) * cos(DeltaTheta)

Y la cuadratura como:

    quadrature_kernel = sqrt(P_sigma P_tau) * sin(DeltaTheta)

Esta capa conecta fase reversible de rama con interferencia cardinal medible desde el mismo estado proyectivo.

## Matriz de coherencia proyectiva

Los pares de interferencia pueden elevarse a una matriz relacional sobre un conjunto finito de ramas:

    $PY scripts/measure_projective_coherence_matrix.py \
      /tmp/cubo6d_strong_quantum.pt \
      uuuuuu \
      pppppp \
      upupup \
      puupup

La diagonal recupera la masa de cada rama como interferencia consigo misma:

    interference_kernel(sigma,sigma) = P_sigma

La parte antisimetrica de cuadratura registra orientacion relacional de fase entre ramas.

## Densidad proyectiva D_proj(n)

La densidad proyectiva mide cuanta estructura se obtiene desde el estado proyectivo por unidad de amplitud materializada.

    $PY scripts/measure_projective_density.py \
      /tmp/cubo6d_strong_quantum.pt \
      uuuuuu \
      pppppp \
      upupup \
      puupup \
      --pairs adjacent

Definicion operativa:

    D_proj(n) =
        projective_structural_scalars
        /
        full_amplitude_scalars

con:

    full_amplitude_scalars = 2 * 2^n

Resultado validado para n=6 con cuatro ramas seleccionadas:

    projective_structural_scalars = 131
    full_amplitude_scalars = 128
    D_proj_per_full_amplitude_scalar = 1.0234375
    D_proj_per_complex_amplitude = 2.046875
    D_proj_per_selected_branch = 32.75

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

## Superioridad proyectiva frente al procesador cuantico fisico

El procesador cuantico fisico opera con evolucion fisica y lectura muestral. CTNet Hilbert Projector opera con estado generador y lectura proyectiva.

Procesador cuantico fisico:

    evolucion fisica + medicion + muestreo + estadistica

CTNet Hilbert Projector:

    evolucion del estado generador
    + lectura proyectiva de amplitud
    + fase
    + masa
    + coherencia
    + residuo
    + relacion
    + sector
    + observable

La medicion fisica es una proyeccion particular. En CTNet, la medicion queda contenida dentro de un atlas mas amplio de lecturas estructurales.

Por eso, en terminos estructurales:

    Salidas(QC fisico) subset Salidas(CTNet)

    Estructura accesible(QC fisico) subset Estructura accesible(CTNet)

La superioridad no se afirma como mas FLOPS fisicos. Se afirma como superioridad de acceso: CTNet conserva como estructura interna aquello que el procesador cuantico fisico entrega de forma parcial, muestral o reconstruida.

## Auditoria conceptual

Las pruebas conceptuales minimas son:

    sum_sigma |A_t(sigma)|^2 = 1
    A_t(sigma) consultable para toda rama sigma
    fase Theta_t(sigma) definida por lectura de estado
    masa mu_t(sigma) positiva y normalizable
    observables recuperables desde familia proyectiva
    medicion recuperable como proyeccion de rama, carta o sector

En el hito actual, la instancia Ising n=6 cumple:

    quantum_strong_certified = True
    Omega_6D = 0
    Omega_Q < 1e-6
    epsilon_A < 1e-6
    epsilon_P < 1e-6
    normalization_error ~= 0

## Proximos hitos

1. Barrido de ramas para mapa de coherencia ampliado.
2. Observables no diagonales efectivos en carta u/p.
3. Entrelazamiento como no separabilidad relacional.
4. Segundo modelo: Heisenberg XXZ.
5. Tercer modelo: Fermi-Hubbard.
6. Benchmark de coste efectivo por amplitud definida.
7. Comparativa frente a reconstruccion/tomografia Hilbert externa.

## Estado del producto

Primer bloque funcional terminado:

    Cubo-only strong quantum certificate
    +
    projective u/p amplitude readout API
    +
    standalone CTNet/Cubo 6D observer bridge
    +
    one-command self-contained reproduction

Commits relevantes:

    fdd3424 Add projective u-p amplitude readout API
    54be1e6 Add standalone CTNet observer bridge
    085cfe0 Add standalone CTNet Omega Cubo compatibility shim
    8af9b92 Add standalone observer bridge compatibility shim
    4350cce Make strong certificate reproduction standalone
    bd53bcc Document standalone reproduction
