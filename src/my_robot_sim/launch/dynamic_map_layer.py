#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
import tf2_ros


class DynamicMapLayer(Node):
    def __init__(self):
        super().__init__('dynamic_map_layer')

        # ---------------- Parámetros ----------------
        self.declare_parameter('static_map_topic', '/map')
        self.declare_parameter('dynamic_map_topic', '/dynamic_map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('laser_frame', 'lidar')
        self.declare_parameter('decay_time', 3.0)          # segundos antes de borrar obstáculo
        self.declare_parameter('obs_min_occupancy', 80)    # valor para marcar obstáculo
        self.declare_parameter('free_threshold', 20)       # si el estático >= esto, asumimos pared estática

        # NUEVO → Congelar el mapa estático base
        self.declare_parameter('freeze_static_map', True)
        self.freeze_static_map = self.get_parameter(
            'freeze_static_map').get_parameter_value().bool_value
        self.static_frozen = False

        # Sim time: solo declárala si no existe (para evitar ParameterAlreadyDeclared)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        self.static_map_topic = self.get_parameter(
            'static_map_topic').get_parameter_value().string_value
        self.dynamic_map_topic = self.get_parameter(
            'dynamic_map_topic').get_parameter_value().string_value
        self.scan_topic = self.get_parameter(
            'scan_topic').get_parameter_value().string_value
        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.laser_frame = self.get_parameter(
            'laser_frame').get_parameter_value().string_value
        self.decay_time = self.get_parameter(
            'decay_time').get_parameter_value().double_value
        self.obs_value = int(self.get_parameter(
            'obs_min_occupancy').get_parameter_value().integer_value or 80)
        self.free_threshold = int(self.get_parameter(
            'free_threshold').get_parameter_value().integer_value or 20)

        # ---------------- Estado interno ----------------
        self.map_info = None            # OccupancyGrid.info
        self.base_data = None           # lista de ints (mapa estático)
        self.dyn_data = None            # lista de ints (-1: sin obs dinámica, 100: obs dinámica)
        self.last_hits = {}             # idx -> tiempo_último_impacto (segundos)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs / pubs
        self.create_subscription(OccupancyGrid, self.static_map_topic,
                                 self.map_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic,
                                 self.scan_callback, 10)
        self.pub_dynamic = self.create_publisher(
            OccupancyGrid, self.dynamic_map_topic, 10)

        # Timer de decaimiento + publicación
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info(
            "✅ DynamicMapLayer inicializado: "
            f"static='{self.static_map_topic}', dynamic='{self.dynamic_map_topic}', scan='{self.scan_topic}'"
        )

    # ------------------- Callbacks -------------------

    def map_callback(self, msg: OccupancyGrid):
        """Toma solo el PRIMER mapa recibido si freeze_static_map=True."""

        # Si ya lo congelamos, ignoramos mapas nuevos
        if self.freeze_static_map and self.static_frozen:
            return

        # Guardamos info del mapa
        self.map_info = msg.info
        self.base_data = list(msg.data)

        size = self.map_info.width * self.map_info.height

        if self.dyn_data is None or len(self.dyn_data) != size:
            self.dyn_data = [-1] * size
            self.last_hits.clear()

        # Si estamos congelando, marcamos que ya capturamos el primer mapa
        if self.freeze_static_map and not self.static_frozen:
            self.static_frozen = True
            self.get_logger().info("📌 Mapa estático BASE congelado.")

    def scan_callback(self, msg: LaserScan):
        """Procesa el scan y marca obstáculos dinámicos sobre zonas que
        el mapa estático considera libres.
        """
        if self.map_info is None or self.base_data is None or self.dyn_data is None:
            return

        # Obtener pose del láser en el frame 'map'
        laser_pose = self.get_laser_pose()
        if laser_pose is None:
            return

        lx, ly, lyaw = laser_pose
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            if r < msg.range_min or r > msg.range_max * 0.98:
                angle += msg.angle_increment
                continue

            # Punto final del rayo en coordenadas de 'map'
            ex = lx + r * math.cos(lyaw + angle)
            ey = ly + r * math.sin(lyaw + angle)

            idx = self.world_to_index(ex, ey)
            if idx is None:
                angle += msg.angle_increment
                continue

            # si el mapa estático ya marca algo muy ocupado, lo consideramos pared fija
            base_val = self.base_data[idx]
            if base_val >= self.free_threshold:
                angle += msg.angle_increment
                continue

            # Marcamos obstáculo dinámico (registramos último impacto)
            self.last_hits[idx] = now_sec

            angle += msg.angle_increment

    def timer_callback(self):
        """Decaimiento de obstáculos y publicación del mapa dinámico fusionado."""
        if self.map_info is None or self.base_data is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # ---- RECONSTRUIR SIEMPRE LA MÁSCARA DINÁMICA DESDE CERO ----
        size = self.map_info.width * self.map_info.height
        new_dyn = [-1] * size

        # Mantener solo celdas con impactos recientes y marcarlas como 100
        to_delete = []
        for idx, t in self.last_hits.items():
            if now_sec - t > self.decay_time:
                to_delete.append(idx)
            else:
                if 0 <= idx < size:
                    new_dyn[idx] = 100

        for idx in to_delete:
            self.last_hits.pop(idx, None)

        self.dyn_data = new_dyn

        # ---- Fusión estático + dinámico ----
        fused = []
        for i, base_val in enumerate(self.base_data):
            if self.dyn_data[i] == 100:
                fused.append(100)
            else:
                fused.append(base_val)

        out = OccupancyGrid()
        out.header.frame_id = self.global_frame
        out.header.stamp = self.get_clock().now().to_msg()
        out.info = self.map_info
        out.data = fused

        self.pub_dynamic.publish(out)

    # --------------- Utilidades ----------------

    def get_laser_pose(self) -> Optional[Tuple[float, float, float]]:
        """Devuelve (x, y, yaw) del láser en 'map'. Si falla TF, devuelve None."""
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.laser_frame,
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
        """Convierte un quaternion a yaw (Z)."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def world_to_index(self, x: float, y: float) -> Optional[int]:
        """Convierte (x,y) en mundo (frame map) a índice lineal en el grid."""
        info = self.map_info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        mx = (x - ox) / res
        my = (y - oy) / res

        j = int(math.floor(mx))
        i = int(math.floor(my))

        if j < 0 or i < 0 or j >= info.width or i >= info.height:
            return None

        return i * info.width + j


# --------------- main normal (ros2 run) ----------------

def main(args=None):
    rclpy.init(args=args)
    node = DynamicMapLayer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# --------------- modo launch (ros2 launch my_robot_sim dynamic_map_layer.py) ----------------
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
import os


def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run_node = ExecuteProcess(
        cmd=['python3', this_file],
        output='screen'
    )
    return LaunchDescription([TimerAction(period=1.0, actions=[run_node])])
