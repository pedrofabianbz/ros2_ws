#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Box4Oscillator(Node):
    def __init__(self):
        super().__init__('box4_oscillator')
        self.pub = self.create_publisher(Twist, '/box4/cmd_vel', 10)
        self.t = 0.0
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.tick)

    def tick(self):
        cmd = Twist()
        # oscilación en Y (en coordenadas del modelo)
        cmd.linear.y = 0.3 * math.sin(0.5 * self.t)
        self.pub.publish(cmd)
        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    node = Box4Oscillator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

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
