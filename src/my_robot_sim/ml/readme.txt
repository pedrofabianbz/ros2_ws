
# Navegación híbrida PRM + Pure Pursuit + PPO (evasión local)

Este módulo implementa y evalúa una arquitectura híbrida para navegación de un robot diferencial:

1) **Planificación global** con **PRM + A\*** sobre obstáculos estáticos (mapas SDF).
2) **Ejecución nominal** con seguidor geométrico **Pure Pursuit**.
3) **Evasión local** con una política **PPO discreta** que **solo interviene** cuando hay **riesgo** con obstáculos dinámicos (con histéresis y criterio por distancia/TTC).

El objetivo es comparar el baseline clásico (**PRM+PP**) frente al sistema integrado (**PRM+PP+RL**) en varios mapas y escenarios con obstáculos dinámicos.

---

## Estructura del proyecto (carpetas principales)

- `run/`
  - `prm_interactive.py`: interfaz interactiva para construir PRM, planear y ejecutar (incluye modo RL si se carga modelo).
- `sim/`
  - `core/`: geometría, mundo 2D, colisiones, utilidades base.
  - `runtime/`: integración/ejecución del sistema (control nominal + conmutador RL).
  - `training/`: entornos Gym para entrenamiento
    - `env_avoid_train.py`: entorno **empty** (obs 7D)
    - `env_avoid_corridor_train.py`: entorno **corridor** (obs 10D)
- `train/`
  - `train_avoid.py`, `train_avoid1.py`: entrenamiento PPO
  - `eval_avoid.py`, `eval_avoid1.py`: evaluación cuantitativa
  - `eval_avoid_empty.py`: evaluación en entorno abierto
  - `play_avoid_debug.py`: ejecución/debug de política
- `models/`: modelos entrenados (`.zip`) y/o checkpoints
- `tb_avoid/`: logs de TensorBoard
- `worlds/`: mapas SDF
  - `Boxes.sdf`, `corridor.sdf`, `corridor_s.sdf`, `diff_drive_empty.sdf`, `maze.sdf`
- `integration/`: scripts auxiliares (“glue”) de integración del pipeline (si aplica)
- `config/`, `launch/`: artefactos del proyecto (si aplica)
- `.venv/`: entorno virtual local (si lo usas dentro de esta carpeta)

---

## Requisitos

- Python 3.10+ recomendado
- Dependencias típicas:
  - `numpy`, `gymnasium`
  - `stable-baselines3`
  - `matplotlib` (si graficas)
  - `torch` (según tu instalación CUDA/CPU)

