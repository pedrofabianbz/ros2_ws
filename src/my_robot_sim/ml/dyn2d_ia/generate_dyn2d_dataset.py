#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------- Parámetros del mundo -----------------

WORLD_SIZE = 5.0     # 5 m x 5 m
BOX_SIZE   = 0.5     # cubos 0.5 m
HALF_BOX   = BOX_SIZE / 2.0

# Grid tipo occupancy (para que se vea como 100x100 en RViz)
GRID_SIZE = 100
CELL = WORLD_SIZE / GRID_SIZE   # 0.05 m por celda

# Cubos estáticos (mapeados como en tu mundo Gazebo → [0,5]x[0,5])
STATIC_BOXES = [
    (2.5, 1.5),  # box1
    (2.5, 3.5),  # box2
    (1.5, 2.5),  # box3
]

# Cubo dinámico "box4"
BOX4_CENTER_X = 3.5
BOX4_CENTER_Y = 2.5
V_OBJ_MAX     = 0.3   # m/s (como en tu nodo)
R_OBS         = 0.25  # radio aprox del cubo (0.5/2)

# Robot y goal
R_ROBOT = 0.07        # ~ tamaño del robot según URDF (0.1 x 0.1)
R_GOAL  = 0.12

ROBOT_START = (0.7, 2.5)   # izquierda
GOAL_POS    = (4.3, 2.5)   # derecha

# Dinámica temporal
DT         = 0.05    # s por paso
EP_STEPS   = 600     # más pasos → episodios ~30 s
N_EPISODES = 9999    # hasta que pares con Ctrl+C

# Control
V_MAX_ROBOT       = 0.5   # m/s
WP_REACHED_DIST   = 0.1   # umbral para decir "llegué al waypoint"

# PRM
PRM_N_SAMPLES   = 200    # puntos aleatorios
PRM_K_NEIGHBORS = 10     # vecinos máximo
PRM_MAX_EDGE    = 1.0    # longitud máxima de arista (m)

# Inflado estático (safe zone con cubos)
SAFE_MARGIN_STATIC = 0.05
R_INFLATED = R_OBS + R_ROBOT + SAFE_MARGIN_STATIC  # obstáculo "inflado"

# Safe radius para evitar el cubo dinámico (predicción)
SAFE_RADIUS_DYNAMIC = R_OBS + R_ROBOT + 0.05  # algo mayor que el contacto


# ----------------- Utilidades grid estático -----------------

def world_to_cell(x, y):
    """(x,y) en [0, WORLD_SIZE] -> (i,j) en [0, GRID_SIZE-1]."""
    j = int(np.clip(x / CELL, 0, GRID_SIZE - 1))
    i = int(np.clip(y / CELL, 0, GRID_SIZE - 1))
    return i, j


def build_static_grid():
    """
    Construye un occupancy grid 100x100:
    - bordes = pared
    - cubos estáticos = valores altos
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    # bordes como paredes
    grid[0, :]  = 80
    grid[-1, :] = 80
    grid[:, 0]  = 80
    grid[:, -1] = 80

    # cubos estáticos pintados como cuadrados
    for (cx, cy) in STATIC_BOXES:
        half = BOX_SIZE / 2.0
        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half

        i_min, j_min = world_to_cell(x_min, y_min)
        i_max, j_max = world_to_cell(x_max, y_max)

        grid[i_min:i_max+1, j_min:j_max+1] = 200

    return grid


# ----------------- Colisiones (estático / dinámico) -----------------

def collides_with_static(x, y):
    """Robot colisiona con alguno de los tres cubos estáticos o con paredes."""
    for (cx, cy) in STATIC_BOXES:
        d = math.hypot(x - cx, y - cy)
        if d < R_INFLATED:   # usamos obstáculo inflado
            return True
    # bordes del mapa
    if x - R_ROBOT < 0.0 or x + R_ROBOT > WORLD_SIZE:
        return True
    if y - R_ROBOT < 0.0 or y + R_ROBOT > WORLD_SIZE:
        return True
    return False


def collides_with_dynamic(x, y, xo, yo):
    d = math.hypot(x - xo, y - yo)
    return d < (R_ROBOT + R_OBS)


def edge_is_free(x1, y1, x2, y2, step=0.02):
    """
    Comprueba si el segmento (x1,y1)-(x2,y2) está libre de colisión
    con los obstáculos estáticos, muestreando puntos intermedios.
    """
    dist = math.hypot(x2 - x1, y2 - y1)
    n = max(2, int(dist / step))
    for k in range(n + 1):
        t = k / float(n)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if collides_with_static(x, y):
            return False
    return True


# ----------------- Control “experto” PREDICTIVO dinámico -----------------

def compute_expert_action(xr, yr, xg, yg, xo, yo, vox, voy):
    """
    Acción (vx,vy) para ir al "goal local" (aquí: waypoint del PRM)
    pero evitando colisión futura con el cubo dinámico (xo,yo,vox,voy).

    Es básicamente el mismo esquema que usabas en Gazebo:
    - v_des = vector hacia el goal (o waypoint)
    - se predice cercanía futura con el objeto
    - si posible choque -> se añade componente lateral de evasión
    """
    # Vector base al goal (o waypoint)
    to_goal = np.array([xg - xr, yg - yr], dtype=float)
    dist_goal = np.linalg.norm(to_goal)
    if dist_goal < 1e-6:
        return 0.0, 0.0

    v_des = (to_goal / dist_goal) * V_MAX_ROBOT

    # Posición relativa del objeto
    p = np.array([xo - xr, yo - yr], dtype=float)
    v_obj = np.array([vox, voy], dtype=float)

    # Objeto casi quieto → ir al goal sin liarnos
    if np.linalg.norm(v_obj) < 1e-6:
        return float(v_des[0]), float(v_des[1])

    v_rel = v_obj - v_des
    v_rel_norm2 = float(np.dot(v_rel, v_rel))
    if v_rel_norm2 < 1e-6:
        # misma velocidad → seguimos v_des
        return float(v_des[0]), float(v_des[1])

    # tiempo de máxima cercanía
    COLLISION_HORIZON = 3.0  # s
    t_star = -float(np.dot(p, v_rel)) / v_rel_norm2
    t_star = np.clip(t_star, 0.0, COLLISION_HORIZON)

    closest = p + v_rel * t_star
    d_min = float(np.linalg.norm(closest))

    if d_min > SAFE_RADIUS_DYNAMIC:
        # sin riesgo serio → solo seguir al waypoint
        return float(v_des[0]), float(v_des[1])

    # riesgo → componente lateral
    perp = np.array([-p[1], p[0]], dtype=float)
    n = float(np.linalg.norm(perp))
    if n < 1e-6:
        perp = np.array([-v_obj[1], v_obj[0]], dtype=float)
        n = float(np.linalg.norm(perp))
        if n < 1e-6:
            return float(v_des[0]), float(v_des[1])
    perp /= n

    # cuanto más cerca, más peso de evasión
    w_avoid = np.clip((SAFE_RADIUS_DYNAMIC - d_min) / SAFE_RADIUS_DYNAMIC, 0.3, 1.0)
    v_avoid = v_des + w_avoid * V_MAX_ROBOT * perp

    s = float(np.linalg.norm(v_avoid))
    if s > V_MAX_ROBOT:
        v_avoid *= (V_MAX_ROBOT / s)

    return float(v_avoid[0]), float(v_avoid[1])


# ----------------- PRM: construcción del grafo -----------------

def build_prm():
    """
    Construye un PRM sencillo:
    - Muestras en espacio libre (usando collides_with_static con R_INFLATED)
    - Conexión por vecinos más cercanos sin colisión
    - Añade start y goal como nodos
    Devuelve: nodes (N x 2), edges (lista de listas de (neighbor_idx, cost))
    """
    np.random.seed(0)  # para que sea reproducible

    samples = []

    # 1) muestrear puntos libres
    while len(samples) < PRM_N_SAMPLES:
        x = np.random.uniform(R_ROBOT, WORLD_SIZE - R_ROBOT)
        y = np.random.uniform(R_ROBOT, WORLD_SIZE - R_ROBOT)
        if not collides_with_static(x, y):
            samples.append((x, y))

    # 2) añadir start y goal
    samples.append(ROBOT_START)  # índice start = N
    start_idx = len(samples) - 1
    samples.append(GOAL_POS)     # índice goal = N+1
    goal_idx = len(samples) - 1

    nodes = np.array(samples, dtype=float)
    N = nodes.shape[0]

    # 3) construir grafo con vecinos
    # precomputar distancias
    dists = np.linalg.norm(nodes[None, :, :] - nodes[:, None, :], axis=2)

    edges = [[] for _ in range(N)]

    for i in range(N):
        # ordenar vecinos por distancia (ignorando él mismo)
        order = np.argsort(dists[i])
        neighbors_added = 0
        for j in order[1:]:  # saltar i
            if neighbors_added >= PRM_K_NEIGHBORS:
                break
            if dists[i, j] > PRM_MAX_EDGE:
                break
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            if edge_is_free(x1, y1, x2, y2):
                cost = dists[i, j]
                edges[i].append((j, cost))
                edges[j].append((i, cost))
                neighbors_added += 1

    return nodes, edges, start_idx, goal_idx


# ----------------- PRM: búsqueda de camino (Dijkstra) -----------------

def shortest_path(nodes, edges, start_idx, goal_idx):
    """
    Camino más corto en el grafo PRM con Dijkstra.
    Devuelve lista de índices de nodos [start_idx, ..., goal_idx]
    o None si no hay camino.
    """
    N = len(nodes)
    dist = [math.inf] * N
    prev = [-1] * N
    visited = [False] * N

    dist[start_idx] = 0.0

    for _ in range(N):
        # elegir nodo no visitado con menor distancia
        u = -1
        best = math.inf
        for i in range(N):
            if not visited[i] and dist[i] < best:
                best = dist[i]
                u = i
        if u == -1:
            break
        visited[u] = True
        if u == goal_idx:
            break
        for v, cost in edges[u]:
            if visited[v]:
                continue
            alt = dist[u] + cost
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    if dist[goal_idx] == math.inf:
        return None

    # reconstruir camino
    path_idx = []
    u = goal_idx
    while u != -1:
        path_idx.append(u)
        u = prev[u]
    path_idx.reverse()
    return path_idx


# ----------------- Simulación + visualización -----------------

def main():
    # ----- construir PRM una sola vez -----
    print("Construyendo PRM...")
    nodes, edges, start_idx, goal_idx = build_prm()
    path_idx = shortest_path(nodes, edges, start_idx, goal_idx)
    if path_idx is None:
        print("❌ No se encontró camino PRM start→goal. Sube muestras / vecinos.")
        return

    path_waypoints = nodes[path_idx]   # array [M, 2]
    print(f"Camino PRM con {len(path_waypoints)} waypoints.")

    # ----- figura -----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.0, WORLD_SIZE)
    ax.set_ylim(0.0, WORLD_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Occupancy grid estático 100x100 (fondo estilo RViz)
    static_grid = build_static_grid()
    ax.imshow(
        static_grid,
        extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
        origin='lower',
        cmap='gray',
        vmin=0,
        vmax=255,
        interpolation='nearest',  # se ven los cuadritos 100x100
    )

    # paredes (borde negro)
    border = patches.Rectangle(
        (0.0, 0.0),
        WORLD_SIZE,
        WORLD_SIZE,
        fill=False,
        edgecolor='black',
        linewidth=2.0
    )
    ax.add_patch(border)

    # cubos estáticos (gris)
    for (cx, cy) in STATIC_BOXES:
        rect = patches.Rectangle(
            (cx - HALF_BOX, cy - HALF_BOX),
            BOX_SIZE,
            BOX_SIZE,
            facecolor='0.7',
            edgecolor='black',
        )
        ax.add_patch(rect)

    # goal (verde)
    goal_circle = plt.Circle(GOAL_POS, R_GOAL, color='green', alpha=0.8)
    ax.add_patch(goal_circle)

    # cubo dinámico (rojo)
    dyn_rect = patches.Rectangle(
        (BOX4_CENTER_X - HALF_BOX, BOX4_CENTER_Y - HALF_BOX),
        BOX_SIZE,
        BOX_SIZE,
        facecolor='red',
        edgecolor='black'
    )
    ax.add_patch(dyn_rect)

    # robot (azul)
    robot_circle = plt.Circle(ROBOT_START, R_ROBOT, color='blue', alpha=0.9)
    ax.add_patch(robot_circle)

    # trayectoria PRM (en naranja, fija)
    ax.plot(path_waypoints[:, 0], path_waypoints[:, 1],
            'o-', color='orange', linewidth=1.5, label='PRM path')
    ax.legend(loc='upper left')

    # trayectorias de episodios anteriores
    past_paths = []  # lista de arrays [ [x0,y0], [x1,y1], ... ]

    # línea de trayectoria actual
    current_traj_line, = ax.plot([], [], 'b-', linewidth=2)

    plt.ion()
    plt.show(block=False)

    t_global = 0.0

    for ep in range(N_EPISODES):
        # reset de episodio
        xr, yr = ROBOT_START
        xo, yo = BOX4_CENTER_X, BOX4_CENTER_Y

        path_x = [xr]
        path_y = [yr]

        success = False
        reason = "timeout"

        # empezamos desde el primer waypoint del PRM
        wp_idx = 0

        for step in range(EP_STEPS):
            # waypoint actual del PRM → será nuestro "goal local"
            xw, yw = path_waypoints[wp_idx]

            # velocidad del cubo dinámico (como en tu nodo: vy = 0.3*sin(0.5*t))
            vox = 0.0
            voy = V_OBJ_MAX * math.sin(0.5 * t_global)

            # control PREDICTIVO: ir hacia (xw,yw) evitando el cubo dinámico
            vx, vy = compute_expert_action(
                xr, yr,
                xw, yw,
                xo, yo,
                vox, voy
            )

            # avanzar robot
            xr_new = np.clip(xr + vx * DT, 0.0, WORLD_SIZE)
            yr_new = np.clip(yr + vy * DT, 0.0, WORLD_SIZE)

            # avanzar cubo dinámico
            yo_new = yo + voy * DT
            # limitar para no salir del mapa
            y_min = HALF_BOX
            y_max = WORLD_SIZE - HALF_BOX
            yo_new = max(y_min, min(y_max, yo_new))

            xr, yr = xr_new, yr_new
            yo = yo_new

            path_x.append(xr)
            path_y.append(yr)

            t_global += DT

            # ¿llegué al waypoint?
            if math.hypot(xr - xw, yr - yw) < WP_REACHED_DIST:
                if wp_idx < len(path_waypoints) - 1:
                    wp_idx += 1  # siguiente waypoint

            # ¿llegué (cerca) al goal?
            xg, yg = GOAL_POS
            if math.hypot(xr - xg, yr - yg) < 0.15:
                success = True
                reason = "goal"
                break

            # comprobar colisiones
            if collides_with_static(xr, yr):
                success = False
                reason = "static_collision"
                break
            if collides_with_dynamic(xr, yr, xo, yo):
                success = False
                reason = "dynamic_collision"
                break

            # actualizar dibujo
            robot_circle.center = (xr, yr)
            dyn_rect.set_xy((BOX4_CENTER_X - HALF_BOX, yo - HALF_BOX))

            # limpiar líneas viejas (dejamos la trayectoria PRM y la actual)
            for ln in ax.lines[:]:
                if ln is not current_traj_line and ln.get_label() != 'PRM path':
                    ln.remove()

            # trayectorias pasadas (en gris)
            for traj in past_paths:
                ax.plot(traj[:, 0], traj[:, 1], color='0.5', linewidth=1, alpha=0.4)

            # trayectoria actual en azul
            current_traj_line.set_data(path_x, path_y)

            ax.set_title(f"Episodio {ep+1} | step {step+1} | t={t_global:.1f}s")
            plt.pause(0.001)

        # episodio terminado
        traj = np.stack([np.array(path_x), np.array(path_y)], axis=1)
        past_paths.append(traj)

        print(f"[EP {ep+1}] terminado por: {reason}, pasos={len(path_x)}")

        # mostrar un poco la trayectoria final
        robot_circle.center = ROBOT_START
        dyn_rect.set_xy((BOX4_CENTER_X - HALF_BOX, BOX4_CENTER_Y - HALF_BOX))
        for ln in ax.lines[:]:
            if ln is not current_traj_line and ln.get_label() != 'PRM path':
                ln.remove()
        for traj in past_paths:
            ax.plot(traj[:, 0], traj[:, 1], color='0.5', linewidth=1, alpha=0.4)
        current_traj_line.set_data(path_x, path_y)
        ax.set_title(f"Episodio {ep+1} terminado por: {reason}")
        plt.pause(0.5)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
