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

## Cuanto lo supera

Para n grados cuanticos, el numero de amplitudes de la base es:

    N_amp(n) = 2^n

Un listado completo de amplitudes exige:

    C_list(n) >= 2^n

La tomografia completa de un estado general escala como:

    C_tom(n) ~ 4^n

CTNet separa tres costes:

    C_gen(n)  = coste de mantener el estado generador Xi
    C_read(n) = coste de leer una rama, carta o sector
    C_list(n) = coste de materializar externamente toda la lista

La ventaja frente al listado es:

    S_list(n) = 2^n / C_gen(n)

La ventaja frente a tomografia es:

    S_tom(n) = 4^n / C_gen(n)

Si C_gen(n) es constante o subexponencial, entonces:

    S_list(n) -> infinity
    S_tom(n)  -> infinity

Ejemplo de referencia para n = 40 y C_gen = 1024:

    2^40 = 1_099_511_627_776
    S_list(40) = 2^40 / 1024 = 1_073_741_824
    S_tom(40)  = 4^40 / 1024 = 1.18059162072e21

En n = 40, CTNet separa el acceso proyectivo del listado completo por un factor aproximado de:

    1.07 x 10^9

frente a listado, y por:

    1.18 x 10^21

frente a tomografia completa.

El punto central es que CTNet conserva el vector como ley proyectiva:

    sigma -> Q_sigma(Xi)

La lista completa es una salida posible, no el objeto primario.

## Consecuencias estructurales

1. El vector de estado deja de identificarse con una lista.

    vector = familia proyectiva generada por Xi

2. La medicion deja de ser el acceso soberano al estado.

    medicion = proyeccion particular dentro de un atlas de lecturas

3. La regla de Born se reinterpreta como masa modal normalizada.

    P_t(sigma) = mu_t(sigma) / sum_tau mu_t(tau)

4. La fase se vuelve memoria reversible de trayectoria.

    Theta_t(sigma) = memoria dinamica de rama

5. El entrelazamiento se vuelve no separabilidad relacional.

    entrelazamiento = obstruccion relacional a la factorizacion de ramas

6. La decoherencia se reinterpreta como perdida de compatibilidad entre fase, regimen y cierre.

7. El qubit fisico queda como una carta particular de una estructura generadora mas amplia.

8. La computacion cuantica pasa de evolucion de un vector opaco a evolucion de un estado generador auditable.

9. CTNet desplaza el regimen de simulacion cuantica hacia computacion estructural proyectiva.

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
