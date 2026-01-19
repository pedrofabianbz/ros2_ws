#!/usr/bin/env python3
import math
import os
import contextlib

import numpy as np
import rclpy
from rclpy.node import Node as RclpyNode

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist

import tf2_ros
from tf_transformations import euler_from_quaternion

import torch
import torch.nn as nn
from torch.amp import autocast

# --- PARA EL LAUNCH ---
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


# ----------------- Constantes del grid NN -----------------

WORLD_SIZE = 5.0          # 5 m x 5 m
GRID_SIZE = 100           # 100 x 100
CELL = WORLD_SIZE / GRID_SIZE

NN_FREE = 0
NN_OBS = 200
NN_ROBOT = 150
NN_GOAL = 255


# ----------------- Modelo (igual al de entrenamiento) -----------------

class PolicyNet(nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100 -> 50

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50 -> 25

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 -> 12
        )

        cnn_out_dim = 64 * 12 * 12

        self.fc_grid = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # vx, vy
        )

    def forward(self, grid, state):
        x = self.cnn(grid)
        x = torch.flatten(x, 1)
        x = self.fc_grid(x)
        x = torch.cat([x, state], dim=1)
        out = self.fc(x)
        return out


# ----------------- Utilidades -----------------

def normalize_angle(angle):
    a = math.fmod(angle + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def occupancy_to_nn_grid(grid_msg: OccupancyGrid, rx, ry, gx, gy):
    """
    Convierte OccupancyGrid + pose de robot/goal a grid 100x100 listo para la IA.
    Ventana 5x5 m centrada en el robot.
    """
    grid = np.full((GRID_SIZE, GRID_SIZE), NN_FREE, dtype=np.uint8)

    w = grid_msg.info.width
    h = grid_msg.info.height
    res = grid_msg.info.resolution
    ox = grid_msg.info.origin.position.x
    oy = grid_msg.info.origin.position.y

    data = np.array(grid_msg.data, dtype=np.int16).reshape((h, w))

    half = WORLD_SIZE / 2.0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = rx - half + (j + 0.5) * CELL
            y = ry - half + (i + 0.5) * CELL

            cj = int((x - ox) / res)
            ci = int((y - oy) / res)

            if ci < 0 or cj < 0 or ci >= h or cj >= w:
                val = -1
            else:
                val = data[ci, cj]

            if val == -1:
                grid[i, j] = NN_FREE
            elif val >= 50:
                grid[i, j] = NN_OBS
            else:
                grid[i, j] = NN_FREE

    # Pintar robot y goal
    def paint_disc(xc, yc, radius, value):
        j_c = int((xc - (rx - half)) / CELL)
        i_c = int((yc - (ry - half)) / CELL)

        r_cells = int(radius / CELL) + 1
        H, W = grid.shape
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ii = i_c + di
                jj = j_c + dj
                if ii < 0 or jj < 0 or ii >= H or jj >= W:
                    continue
                dx = (jj - j_c) * CELL
                dy = (ii - i_c) * CELL
                if dx * dx + dy * dy <= radius * radius:
                    grid[ii, jj] = value

    paint_disc(rx, ry, 0.1, NN_ROBOT)
    paint_disc(gx, gy, 0.12, NN_GOAL)

    return grid.astype(np.float32) / 255.0


# ----------------- Nodo ROS2 -----------------

class Dyn2DPolicyNode(RclpyNode):
    def __init__(self):
        super().__init__('dyn2d_policy_node')

        # Parámetros
        self.declare_parameter('map_topic', '/dynamic_map')
        self.declare_parameter('goal_topic', '/goal')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('model_path', 'policy_expert_v1.pt')

        self.declare_parameter('max_linear', 0.4)
        self.declare_parameter('max_angular', 1.2)
        self.declare_parameter('k_ang', 2.0)

        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        goal_topic = self.get_parameter('goal_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.max_linear = self.get_parameter('max_linear').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular').get_parameter_value().double_value
        self.k_ang = self.get_parameter('k_ang').get_parameter_value().double_value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs/Pubs
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.last_map = None
        self.last_goal = None

        # Modelo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolicyNet(state_dim=8).to(self.device)

        # Resolver ruta del modelo
        model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        if os.path.isabs(model_path_param):
            model_path = model_path_param
        else:
            here = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(here, model_path_param)

        if not os.path.exists(model_path):
            self.get_logger().error(f"No se encontró el modelo en {model_path}")
        else:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.get_logger().info(f"Modelo cargado desde {model_path} en {self.device}")

        # AMP
        self.use_cuda_amp = (self.device.type == 'cuda')
        self.autocast_ctx = (lambda: autocast('cuda')) if self.use_cuda_amp else contextlib.nullcontext

        # Timer control (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    # ---- Callbacks ----

    def map_callback(self, msg: OccupancyGrid):
        self.last_map = msg

    def goal_callback(self, msg: PoseStamped):
        self.last_goal = msg

    # ---- Bucle principal ----

    def control_loop(self):
        if self.last_map is None or self.last_goal is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn_throttle(2000, f"TF {self.global_frame}->{self.base_frame} no disponible: {e}")
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        gx = self.last_goal.pose.position.x
        gy = self.last_goal.pose.position.y

        nn_grid = occupancy_to_nn_grid(self.last_map, rx, ry, gx, gy)
        state_vec = np.array([rx, ry, gx, gy, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        grid_t = torch.from_numpy(nn_grid).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,100,100)
        state_t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)            # (1,8)

        with torch.no_grad():
            with self.autocast_ctx():
                action_t = self.model(grid_t, state_t)

        vx_map, vy_map = action_t[0].cpu().numpy().tolist()

        speed = math.hypot(vx_map, vy_map)
        if speed < 0.01:
            self.publish_twist(0.0, 0.0)
            return

        desired_yaw = math.atan2(vy_map, vx_map)
        yaw_err = normalize_angle(desired_yaw - yaw)

        v = min(speed, self.max_linear)
        if abs(yaw_err) > math.pi / 2.0:
            v = 0.0

        w = self.k_ang * yaw_err
        w = max(-self.max_angular, min(self.max_angular, w))

        self.publish_twist(v, w)

    def publish_twist(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


# ----------------- main() para ros2 run (o ejecución directa) -----------------

def main(args=None):
    rclpy.init(args=args)
    node = Dyn2DPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# ----------------- Launch: para ros2 launch my_robot_sim dyn2d_policy_node.py -----------------

def generate_launch_description():
    """
    Permite usar ESTE MISMO ARCHIVO como launch:

      ros2 launch my_robot_sim dyn2d_policy_node.py
    """
    pkg_share = get_package_share_directory('my_robot_sim')
    script_path = os.path.join(pkg_share, 'launch', 'dyn2d_policy_node.py')

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_path],
            output='screen'
        )
    ])

if __name__ == '__main__':
    main()