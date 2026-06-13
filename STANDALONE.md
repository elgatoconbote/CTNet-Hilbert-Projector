# Ejecucion autosuficiente

`CTNet-Hilbert-Projector` ya contiene localmente el nucleo CTNet/Cubo 6D necesario para reproducir el certificado fuerte. No hace falta ejecutar desde `CTNet-2.6-Omega-Cubo-6D` ni tener una carpeta hermana externa.

## Ruta local integrada

Los imports historicos se conservan por compatibilidad mediante shims locales:

```text
ctnet_omega_cubo6d_plegado_ctnet26.py
train_vram_up_coherence_ctnet.py
```

Ambos redirigen a la implementacion empaquetada dentro de:

```text
src/ctnet_hilbert_projector/ctnet_omega_core.py
src/ctnet_hilbert_projector/observer_bridge.py
```

## Reproduccion fuerte

Desde la raiz del repositorio:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
STATE=/tmp/cubo6d_strong_quantum.pt scripts/reproduce_strong_certificate.sh
```

La variable `PY` es opcional. Si no se define, el script usa `python` del entorno activo:

```bash
PY=.venv/bin/python STATE=/tmp/cubo6d_strong_quantum.pt scripts/reproduce_strong_certificate.sh
```

## Capas auditadas

La reproduccion ejecuta:

```text
py_compile
pytest
solver Cubo-only
lectura proyectiva Q_sigma
auditoria independiente del .pt
observables diagonales
fase e interferencia
matriz de coherencia proyectiva
densidad proyectiva D_proj(n)
```

## Frontera actual

El bloque autosuficiente reproduce el certificado fuerte de la instancia cerrada:

```text
modelo = Ising transverse-field 1D
n = 6
J = 1.0
h = 0.5
dt = 0.05
steps = 3
psi0 = uuuuuu
```

El siguiente endurecimiento formal es convertir la comparacion Hilbert externa en referencia local interna para n pequeno, de modo que el repo certifique no solo cierre estructural CTNet, sino tambien distancia exhaustiva contra una evolucion Hilbert calculada dentro del propio paquete.
