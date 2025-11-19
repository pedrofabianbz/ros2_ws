#!/usr/bin/env python3
import math
from typing import Optional, Tuple
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
import tf2_ros

# Para launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


class DynamicSubmap(Node):
    def __init__(self):
        super().__init__('dynamic_submap')

        # Parámetros
        self.declare_parameter('input_map_topic', '/dynamic_map')
        self.declare_parameter('output_map_topic', '/dynamic_submap')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('sub_width', 100)
        self.declare_parameter('sub_height', 100)

        self.input_map_topic = self.get_parameter(
            'input_map_topic').get_parameter_value().string_value
        self.output_map_topic = self.get_parameter(
            'output_map_topic').get_parameter_value().string_value
        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter(
            'robot_frame').get_parameter_value().string_value
        self.sub_w = self.get_parameter(
            'sub_width').get_parameter_value().integer_value
        self.sub_h = self.get_parameter(
            'sub_height').get_parameter_value().integer_value

        # Estado
        self.last_map: Optional[OccupancyGrid] = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs / pubs
        self.create_subscription(
            OccupancyGrid,
            self.input_map_topic,
            self.map_callback,
            10
        )
        self.pub_submap = self.create_publisher(
            OccupancyGrid,
            self.output_map_topic,
            10
        )

        # Timer para publicar submap periódicamente
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"✅ DynamicSubmap inicializado: in='{self.input_map_topic}', out='{self.output_map_topic}', "
            f"frame='{self.global_frame}', robot_frame='{self.robot_frame}', "
            f"size={self.sub_w}x{self.sub_h}"
        )

    # ---------------- Callbacks ----------------

    def map_callback(self, msg: OccupancyGrid):
        self.last_map = msg

    def timer_callback(self):
        if self.last_map is None:
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        rx, ry, _ = robot_pose
        submap = self.build_submap(self.last_map, rx, ry)

        if submap is not None:
            self.pub_submap.publish(submap)

    # ---------------- Utilidades ----------------

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """Devuelve (x, y, yaw) de base_link en 'map'."""
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return None

        x = t.transform.translation.x
        y = t.transform.translation.y
        q = t.transform.rotation
        yaw = self.quaternion_to_yaw(q)
        return x, y, yaw

    def quaternion_to_yaw(self, q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def world_to_cell(self, info, x: float, y: float) -> Optional[Tuple[int, int]]:
        """(x,y) mundo → (i,j) índices en el grid."""
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        mx = (x - ox) / res
        my = (y - oy) / res

        j = int(math.floor(mx))
        i = int(math.floor(my))

        if j < 0 or i < 0 or j >= info.width or i >= info.height:
            return None
        return i, j

    def build_submap(self, full_map: OccupancyGrid,
                     robot_x: float, robot_y: float) -> Optional[OccupancyGrid]:
        info = full_map.info
        data = list(full_map.data)  # 1D list

        # Celda del robot en el grid completo
        cell = self.world_to_cell(info, robot_x, robot_y)
        if cell is None:
            return None

        i_robot, j_robot = cell

        half_h = self.sub_h // 2
        half_w = self.sub_w // 2

        # Rango en el mapa global
        i_min = i_robot - half_h
        i_max = i_robot + half_h
        j_min = j_robot - half_w
        j_max = j_robot + half_w

        # Preparamos datos del submap (relleno con -1 = desconocido por defecto)
        UNKNOWN = -1
        sub_data = [UNKNOWN] * (self.sub_w * self.sub_h)

        # Copiamos solo la parte que cae dentro del mapa global
        for si in range(self.sub_h):
            gi = i_min + si
            if gi < 0 or gi >= info.height:
                continue
            for sj in range(self.sub_w):
                gj = j_min + sj
                if gj < 0 or gj >= info.width:
                    continue

                # índice en mapa global
                g_idx = gi * info.width + gj
                val = data[g_idx]

                # índice en submap
                s_idx = si * self.sub_w + sj
                sub_data[s_idx] = val

        # Construimos el mensaje OccupancyGrid del submap
        sub = OccupancyGrid()
        sub.header.frame_id = full_map.header.frame_id  # normalmente 'map'
        sub.header.stamp = self.get_clock().now().to_msg()

        sub.info.resolution = info.resolution
        sub.info.width = self.sub_w
        sub.info.height = self.sub_h

        # Origen del submap en el mismo frame 'map'
        # Celda (0,0) del submap corresponde a (i_min, j_min) del mapa grande
        sub.info.origin.position.x = info.origin.position.x + j_min * info.resolution
        sub.info.origin.position.y = info.origin.position.y + i_min * info.resolution
        sub.info.origin.position.z = 0.0
        sub.info.origin.orientation = info.origin.orientation

        sub.data = sub_data
        return sub


def main(args=None):
    rclpy.init(args=args)
    node = DynamicSubmap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---- MODO LAUNCH (ros2 launch TU_PAQUETE dynamic_submap.py) ----
def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run = ExecuteProcess(cmd=['python3', this_file], output='screen')
    return LaunchDescription([TimerAction(period=1.0, actions=[run])])


if __name__ == '__main__':
    main()
