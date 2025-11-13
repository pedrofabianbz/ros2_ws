#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
import tf2_ros
from tf_transformations import euler_from_quaternion  # si no lo tienes, te doy versión sin esto abajo


class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # Parámetros
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lookahead_dist', 0.25)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('max_lin', 0.15)
        self.declare_parameter('max_ang', 0.8)

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.lookahead_dist = self.get_parameter('lookahead_dist').get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.max_lin = self.get_parameter('max_lin').get_parameter_value().double_value
        self.max_ang = self.get_parameter('max_ang').get_parameter_value().double_value

        # Suscripciones
        self.create_subscription(Path, '/plan', self.path_callback, 10)

        # Publicador
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Estado
        self.current_path = []  # lista de (x,y)
        self.target_idx = 0
        self.goal_reached = False

        # Control loop (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("✅ PathFollower (pure pursuit, frame map) inicializado")

    def path_callback(self, msg: Path):
        self.current_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.target_idx = 0
        self.goal_reached = False
        if self.current_path:
            self.get_logger().info(f"Nuevo path recibido: {len(self.current_path)} puntos")

    # ---- Utilidades ----
    def get_robot_pose_map(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time()
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

    def quaternion_to_yaw(self, q):
        # usando tf_transformations
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # ---- Control ----
    def control_loop(self):
        # Si no hay path o ya llegamos, quieto.
        if not self.current_path or self.goal_reached:
            self.publish_stop()
            return

        pose = self.get_robot_pose_map()
        if pose is None:
            # no TF todavía
            return

        x, y, yaw = pose

        # ¿Objetivo global alcanzado?
        gx, gy = self.current_path[-1]
        dist_goal = math.hypot(gx - x, gy - y)
        if dist_goal < self.goal_tolerance:
            self.get_logger().info("🎯 Goal alcanzado, deteniendo robot")
            self.goal_reached = True
            self.publish_stop()
            return

        # Buscar punto de lookahead sobre el path
        lookahead = self.lookahead_dist
        idx = self.target_idx
        best_idx = idx

        while idx < len(self.current_path):
            px, py = self.current_path[idx]
            d = math.hypot(px - x, py - y)
            if d >= lookahead:
                best_idx = idx
                break
            idx += 1

        # Si no se encontró suficientemente lejos, usar último
        if best_idx == self.target_idx and best_idx < len(self.current_path) - 1:
            best_idx += 1
        self.target_idx = best_idx

        tx, ty = self.current_path[self.target_idx]

        # Transformar objetivo a frame del robot
        dx = tx - x
        dy = ty - y
        # ángulo al objetivo en global
        target_angle = math.atan2(dy, dx)
        # error de orientación
        angle_error = self.normalize_angle(target_angle - yaw)
        dist = math.hypot(dx, dy)

        cmd = Twist()

        # Ley de control tipo pure pursuit simplificada
        # avanzar proporcional a distancia, girar proporcional al error angular
        k_lin = 0.8
        k_ang = 1.5

        cmd.linear.x = min(self.max_lin, k_lin * dist)
        cmd.angular.z = max(-self.max_ang, min(self.max_ang, k_ang * angle_error))

        # Pequeña zona muerta angular para evitar temblores
        if abs(angle_error) < 0.03:
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# ---- MODO LAUNCH (ros2 launch my_robot_sim path_follower.py) ----
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
import os
def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run_pf = ExecuteProcess(cmd=['python3', this_file], output='screen')
    return LaunchDescription([TimerAction(period=1.0, actions=[run_pf])])