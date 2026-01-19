#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped


class Box4Oscillator(Node):
    def __init__(self):
        super().__init__('box4_oscillator')

        # Publicador de velocidad
        self.pub_cmd = self.create_publisher(Twist, '/box4/cmd_vel', 10)

        # Publicador de la pose calculada
        self.pub_pose = self.create_publisher(PoseStamped, '/box4/pose', 10)

        # Tiempo interno
        self.t = 0.0
        self.dt = 0.01

        # POSICIÓN INICIAL (coincidir con tu SDF)
        self.x = 1.0     # <pose> 1 0 0.25 0 0 0  -> x = 1
        self.y = 0.0
        self.z = 0.25

        # Límites opcionales para que no se vaya al carajo
        self.y_min = -2.0
        self.y_max = 2.0

        # Timer
        self.timer = self.create_timer(self.dt, self.tick)

        self.get_logger().info("✅ Box4Oscillator inicializado (publica /box4/cmd_vel y /box4/pose)")

    def tick(self):
        # ---- 1) Velocidad oscilatoria en Y ----
        vy = 0.3 * math.sin(0.5 * self.t)

        cmd = Twist()
        cmd.linear.y = vy
        self.pub_cmd.publish(cmd)

        # ---- 2) Integrar posición (simple) ----
        self.y += vy * self.dt
        self.y = max(self.y_min, min(self.y_max, self.y))

        # ---- 3) Publicar la pose ----
        pose = PoseStamped()
        pose.header.frame_id = "map"  # Debe ser el mismo que tu global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(self.x)
        pose.pose.position.y = float(self.y)
        pose.pose.position.z = float(self.z)

        pose.pose.orientation.w = 1.0  # sin rotación

        self.pub_pose.publish(pose)

        # avanzar tiempo
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = Box4Oscillator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# -------- Launch embebido --------
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


if __name__ == '__main__':
    main()
