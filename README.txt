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

está en:

**`src/my_robot_sim/ml/`**

> Para instalar dependencias y ejecutar experimentos en Python, sigue el README dentro de esa carpeta:  
> `src/my_robot_sim/ml/README.md`

## ROS 2 / Gazebo (workspace)

Este workspace se construye y ejecuta como cualquier workspace estándar de ROS 2.


colcon build --symlink-install
source install/setup.bash
