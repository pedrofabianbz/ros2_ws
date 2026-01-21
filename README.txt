# ros2_ws — Workspace ROS 2 (Gazebo) + módulo de ML en Python

Este repositorio contiene un workspace de **ROS 2** (con paquetes de simulación/launch para **Gazebo**) y, adicionalmente, un módulo en **Python** para navegación híbrida y entrenamiento RL.

## Estructura (alto nivel)

- `src/`
  - paquetes ROS 2 del proyecto
  - **módulo Python para navegación/entrenamiento**:  
    `src/my_robot_sim/ml/`

Todo lo relacionado con:
- simulador 2D en Python,
- planificación PRM + A*,
- seguidor Pure Pursuit,
- entornos Gym para entrenamiento PPO,
- scripts de entrenamiento/evaluación,
- logs de TensorBoard,
- modelos entrenados,

======================================================
GUÍA DE USO (EVIDENCIAS) — FLUJO RECOMENDADO DESDE CERO
======================================================

Ubicación de trabajo:
ros2_ws/src/my_robot_sim/ml


0) Preparación (una sola vez)
-----------------------------
1. Crear entorno y activar:
   python3 -m venv .venv
   source .venv/bin/activate

2. Instalar dependencias:
   pip install -U pip
   pip install -r requirements.txt

   (Si no hay requirements.txt, mínimo:)
   pip install numpy gymnasium stable-baselines3 torch matplotlib tensorboard


1) Verificar que el baseline funciona (PRM + Pure Pursuit) — SIN RL
-------------------------------------------------------------------
Objetivo: confirmar mapas, PRM, A* y seguimiento nominal antes de RL.

1. Activar entorno:
   source .venv/bin/activate

2. Correr interactivo:
   python3 -m run.prm_interactive

3. Dentro del UI:
   - elegir un world (worlds/*.sdf)
   - click para GOAL
   - construir PRM + planear (A*)
   - ejecutar seguimiento nominal (Pure Pursuit)

Resultado esperado: sigue la ruta y evita obstáculos estáticos.


2) Entrenar PPO (generar modelos)
---------------------------------
Objetivo: entrenar en dos etapas (recomendado).

2.1 Entrenamiento en entorno abierto (empty, obs 7D):
   source .venv/bin/activate
   python3 -m train.train_avoid

Salida esperada:
- logs: tb_avoid/
- modelo: models/ (o donde lo guarde el script)

2.2 Fine-tuning en pasillo (corridor, obs 10D):
   source .venv/bin/activate
   python3 -m train.train_avoid1

Salida esperada:
- logs: tb_avoid/
- modelo: models/


3) Ver entrenamiento con TensorBoard (opcional)
-----------------------------------------------
source .venv/bin/activate
tensorboard --logdir tb_avoid


4) Evaluación cuantitativa (métricas / evidencias)
--------------------------------------------------
source .venv/bin/activate

Evaluación en entorno abierto:
   python3 -m train.eval_avoid_empty

Evaluación general:
   python3 -m train.eval_avoid

Evaluación pasillo/corridor:
   python3 -m train.eval_avoid1

Resultado esperado: métricas impresas y/o archivos exportados (según scripts).


5) Debug visual de la política (opcional)
-----------------------------------------
source .venv/bin/activate
python3 -m train.play_avoid_debug


6) Integración completa (PRM + PP + RL) en prm_interactive
----------------------------------------------------------
Objetivo: correr el sistema final donde RL SOLO interviene bajo riesgo.

1. Asegurar modelo en models/ (ej: models/modelo.zip)

2. Correr interactivo cargando RL:
   source .venv/bin/activate
   python3 -m run.prm_interactive --rl-model models/<modelo>.zip

3. Dentro del UI:
   - elegir world
   - click GOAL
   - build PRM + plan (A*)
   - ejecutar y observar intervención RL solo cuando hay riesgo


7) Dónde quedan las evidencias
------------------------------
- Modelos entrenados: models/
- Logs TensorBoard: tb_avoid/
- Mapas: worlds/
- Scripts de entrenamiento/eval: train/
- Entornos Gym: sim/training/
- Integración interactiva: run/prm_interactive.py
