#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class DecisionManager(Node):
    def __init__(self):
        super().__init__('decision_manager')

        # Suscripciones
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(PoseStamped, '/prm_candidate', self.candidate_callback, 10)

        # Publicadores
        self.prm_goal_pub = self.create_publisher(PoseStamped, '/prm_goal', 10)
        self.exec_goal_pub = self.create_publisher(PoseStamped, '/exec_goal', 10)

        # Estado interno
        self._plan_received = None
        self._candidate_received = None
        self._waiting_goal = None
        self._start_time = None
        self.response_timeout = 2.0  # s

        # Timer para verificar respuestas (cada 100 ms)
        self.timer = self.create_timer(0.1, self.check_responses)

        self.get_logger().info("✅ DecisionManager inicializado")

    def goal_callback(self, msg: PoseStamped):
        self._plan_received = None
        self._candidate_received = None
        self._waiting_goal = msg
        self._start_time = time.time()
        self.get_logger().info(f"📌 Nuevo goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.prm_goal_pub.publish(msg)

    def plan_callback(self, msg: Path):
        if self._waiting_goal:
            self._plan_received = msg

    def candidate_callback(self, msg: PoseStamped):
        if self._waiting_goal:
            self._candidate_received = msg

    def check_responses(self):
        if not self._waiting_goal:
            return

        if self._plan_received is not None:
            self.get_logger().info("✅ PRM entregó plan completo → ejecutando goal final")
            self.exec_goal_pub.publish(self._waiting_goal)
            self._waiting_goal = None
            return

        if self._candidate_received is not None:
            cand = self._candidate_received
            self.get_logger().info(
                f"⚠️ Sin plan completo; uso candidato ({cand.pose.position.x:.2f}, {cand.pose.position.y:.2f})"
            )
            self.exec_goal_pub.publish(cand)
            self._waiting_goal = None
            return

        if time.time() - self._start_time > self.response_timeout:
            self.get_logger().warn("⏱️ Timeout esperando PRM. Envío goal original.")
            self.exec_goal_pub.publish(self._waiting_goal)
            self._waiting_goal = None

# ---- MODO SCRIPT (python3 decision_manager.py) ----
def main(args=None):
    rclpy.init(args=args)
    node = DecisionManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# ---- MODO LAUNCH (ros2 launch my_robot_sim decision_manager.py) ----
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
import os

def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run_dm = ExecuteProcess(cmd=['python3', this_file], output='screen')
    return LaunchDescription([TimerAction(period=1.0, actions=[run_dm])])
