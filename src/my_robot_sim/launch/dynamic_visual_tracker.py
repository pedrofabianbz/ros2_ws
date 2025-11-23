#!/usr/bin/env python3
import os
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge


class DynamicVisualTracker(Node):
    def __init__(self):
        super().__init__('dynamic_visual_tracker')

        # Parámetros
        self.declare_parameter('visual_topic', '/dynamic_visual')
        self.declare_parameter('info_map_topic', '/dynamic_submap')
        self.declare_parameter('frame_id', 'map')

        # mínimo píxeles rojos para considerar que hay objeto
        self.declare_parameter('min_region_pixels', 10)
        # horizonte de la flecha (segundos) para visualización
        self.declare_parameter('arrow_horizon', 1.0)
        # salto máximo permitido del centroide entre frames (en metros)
        self.declare_parameter('max_blob_jump', 1.0)
        # topic de imagen debug
        self.declare_parameter('debug_visual_topic', '/dynamic_visual_centroid')

        # NUEVO: parámetros de suavizado de velocidad
        # cuántas muestras de historial usar (>=2)
        self.declare_parameter('vel_history_len', 15)
        # tiempo mínimo total entre la más vieja y la más nueva para calcular vel
        self.declare_parameter('vel_min_dt', 0.15)
        # alfa del filtro exponencial (1 = sin filtro, 0.2–0.5 suele ir bien)
        self.declare_parameter('vel_smoothing', 0.4)

        self.visual_topic = self.get_parameter('visual_topic').value
        self.info_topic = self.get_parameter('info_map_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.min_region_pixels = int(self.get_parameter('min_region_pixels').value)
        self.arrow_horizon = float(self.get_parameter('arrow_horizon').value)
        self.max_blob_jump = float(self.get_parameter('max_blob_jump').value)
        self.debug_visual_topic = self.get_parameter('debug_visual_topic').value

        self.vel_history_len = max(2, int(self.get_parameter('vel_history_len').value))
        self.vel_min_dt = float(self.get_parameter('vel_min_dt').value)
        self.vel_smoothing = float(self.get_parameter('vel_smoothing').value)
        # clamp por si acaso
        if self.vel_smoothing < 0.0:
            self.vel_smoothing = 0.0
        if self.vel_smoothing > 1.0:
            self.vel_smoothing = 1.0

        # Estado
        self.map_info: Optional[object] = None
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_time: Optional[float] = None

        # Historial para velocidad: lista de (t, x, y)
        self.traj_history = []
        # velocidad suavizada
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0

        self.bridge = CvBridge()

        # Subs
        self.sub_info = self.create_subscription(
            OccupancyGrid,
            self.info_topic,
            self.map_info_callback,
            10,
        )

        self.sub_img = self.create_subscription(
            Image,
            self.visual_topic,
            self.image_callback,
            10,
        )

        # Publicadores lógicos
        self.pub_pose = self.create_publisher(PoseStamped, '/dyn_object_pose', 10)
        self.pub_twist = self.create_publisher(Twist, '/dyn_object_twist', 10)

        # Visualización en RViz
        self.pub_markers = self.create_publisher(MarkerArray, '/dyn_tracker_markers', 10)

        # Imagen debug
        self.pub_debug_img = self.create_publisher(Image, self.debug_visual_topic, 10)

        self.get_logger().info(
            f"✅ DynamicVisualTracker (rojo → bounding box cuadrado + vel suavizada). "
            f"visual='{self.visual_topic}', info='{self.info_topic}', "
            f"debug='{self.debug_visual_topic}', "
            f"vel_history_len={self.vel_history_len}, vel_min_dt={self.vel_min_dt}, "
            f"vel_smoothing={self.vel_smoothing}"
        )

    # ---------------- Callbacks ----------------

    def map_info_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info

    def image_callback(self, msg: Image):
        if self.map_info is None:
            return

        try:
            # dynamic_map_layer publica 'rgb8', aquí pedimos BGR
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Error cv_bridge: {e}")
            return

        h, w, _ = cv_img.shape

        # Canales BGR
        b = cv_img[:, :, 0].astype(np.int16)
        g = cv_img[:, :, 1].astype(np.int16)
        r = cv_img[:, :, 2].astype(np.int16)

        # Rojo = dinámico (255,0,0 en RGB → (0,0,255) en BGR)
        red_mask = (r > 200) & (g < 80) & (b < 80)

        ys, xs = np.where(red_mask)

        # Nada rojo (o muy poco)
        if len(xs) < self.min_region_pixels:
            self.clear_state_and_markers()
            self.publish_debug_image(cv_img, None, None, None, None, red_mask)
            return

        # ---------- Bounding box de TODOS los píxeles rojos ----------
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())

        width_px = x_max - x_min + 1
        height_px = y_max - y_min + 1

        # Lado del cuadrado (incluye siempre al bounding box)
        side = max(width_px, height_px)

        # Centro del rectángulo original
        rect_cx = 0.5 * (x_min + x_max)
        rect_cy = 0.5 * (y_min + y_max)

        # Posición aproximada del robot en la imagen (centro del submap)
        robot_j = (w - 1) / 2.0
        robot_i = (h - 1) / 2.0

        # ---------- Colocamos el cuadrado alineado con la cara que mira al robot ----------
        # Eje X (columnas)
        if robot_j < x_min:
            # Robot a la izquierda → cara visible es x_min
            sq_x_min = x_min
            sq_x_max = x_min + side - 1
        elif robot_j > x_max:
            # Robot a la derecha → cara visible es x_max
            sq_x_max = x_max
            sq_x_min = x_max - side + 1
        else:
            # Robot más o menos alineado en X → centramos
            cx = rect_cx
            sq_x_min = int(round(cx - side / 2.0))
            sq_x_max = sq_x_min + side - 1

        # Eje Y (filas)
        if robot_i < y_min:
            # Robot arriba → cara visible es y_min
            sq_y_min = y_min
            sq_y_max = y_min + side - 1
        elif robot_i > y_max:
            # Robot abajo → cara visible es y_max
            sq_y_max = y_max
            sq_y_min = y_max - side + 1
        else:
            # Robot más o menos alineado en Y → centramos
            cy = rect_cy
            sq_y_min = int(round(cy - side / 2.0))
            sq_y_max = sq_y_min + side - 1

        # 1) Clampear dentro de la imagen
        sq_x_min = max(0, sq_x_min)
        sq_y_min = max(0, sq_y_min)
        sq_x_max = min(w - 1, sq_x_max)
        sq_y_max = min(h - 1, sq_y_max)

        # 2) Asegurarnos de que TODOS los píxeles rojos quedan dentro del cuadrado
        sq_x_min = min(sq_x_min, x_min)
        sq_x_max = max(sq_x_max, x_max)
        sq_y_min = min(sq_y_min, y_min)
        sq_y_max = max(sq_y_max, y_max)

        # Volvemos a clampear por si hemos tocado los bordes
        sq_x_min = max(0, sq_x_min)
        sq_y_min = max(0, sq_y_min)
        sq_x_max = min(w - 1, sq_x_max)
        sq_y_max = min(h - 1, sq_y_max)

        # Centro del cuadro definitivo (en píxeles)
        j_center = 0.5 * (sq_x_min + sq_x_max)
        i_center = 0.5 * (sq_y_min + sq_y_max)

        # Píxel -> mundo
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        x = ox + (j_center + 0.5) * res
        y = oy + (i_center + 0.5) * res

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Gate por salto máximo (para evitar brincar a ruido)
        if self.last_pos is not None:
            jump = float(np.hypot(x - self.last_pos[0], y - self.last_pos[1]))
            if jump > self.max_blob_jump:
                self.clear_state_and_markers()
                self.publish_debug_image(
                    cv_img, None, None,
                    (sq_x_min, sq_y_min, sq_x_max, sq_y_max),
                    None,
                    red_mask
                )
                return

        # ---------- ACTUALIZAR HISTORIAL DE POSICIONES ----------
        self.traj_history.append((now_sec, x, y))
        # recortar histórico
        if len(self.traj_history) > self.vel_history_len:
            self.traj_history.pop(0)

        # ---------- CÁLCULO DE VELOCIDAD SUAVIZADA ----------
        vx_raw = 0.0
        vy_raw = 0.0
        if len(self.traj_history) >= 2:
            t0, x0, y0 = self.traj_history[0]
            t1, x1, y1 = self.traj_history[-1]
            dt_hist = t1 - t0
            if dt_hist > self.vel_min_dt:
                vx_raw = (x1 - x0) / dt_hist
                vy_raw = (y1 - y0) / dt_hist

        # Filtro exponencial sobre la velocidad
        alpha = self.vel_smoothing
        self.smooth_vx = alpha * vx_raw + (1.0 - alpha) * self.smooth_vx
        self.smooth_vy = alpha * vy_raw + (1.0 - alpha) * self.smooth_vy

        # Actualizar último punto (para el gate de salto)
        self.last_pos = (x, y)
        self.last_time = now_sec

        # Publicar pose / twist (usamos velocidad suavizada)
        self.publish_pose(x, y)
        self.publish_twist(self.smooth_vx, self.smooth_vy)

        # Markers en RViz (con velocidad suavizada)
        self.publish_markers(x, y, self.smooth_vx, self.smooth_vy)

        # Imagen debug
        self.publish_debug_image(
            cv_img,
            i_center,
            j_center,
            (sq_x_min, sq_y_min, sq_x_max, sq_y_max),
            (i_center, j_center),
            red_mask
        )

    # ---------------- Imagen debug ----------------

    def publish_debug_image(
        self,
        cv_img,
        i_center,
        j_center,
        bbox,
        center_pix,
        red_mask
    ):
        debug = cv_img.copy()

        # Pinta rojo fuerte donde hay rojo
        if red_mask is not None:
            debug[red_mask] = [0, 0, 255]  # BGR

        # Dibuja bounding box cuadrado en amarillo
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(debug,
                          (x_min, y_min),
                          (x_max, y_max),
                          (0, 255, 255), 1)

        # Centroide del cuadrado
        if center_pix is not None:
            i_c, j_c = center_pix
            c = (int(round(j_c)), int(round(i_c)))
            cv2.circle(debug, c, 4, (255, 0, 0), thickness=-1)  # punto azul

        try:
            img_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Error al convertir imagen debug: {e}")
            return

        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = self.frame_id
        self.pub_debug_img.publish(img_msg)

    # ---------------- Publicadores ----------------

    def publish_pose(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_pose.publish(msg)

    def publish_twist(self, vx: float, vy: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        self.pub_twist.publish(msg)

    def publish_markers(self, x: float, y: float, vx: float, vy: float):
        ma = MarkerArray()

        # Esfera azul en el centroide
        sphere = Marker()
        sphere.header.frame_id = self.frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = 'dyn_tracker'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = 0.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.2
        sphere.scale.y = 0.2
        sphere.scale.z = 0.2
        sphere.color.r = 0.0
        sphere.color.g = 0.0
        sphere.color.b = 1.0
        sphere.color.a = 1.0
        ma.markers.append(sphere)

        # Flecha roja con la velocidad suavizada
        arrow = Marker()
        arrow.header.frame_id = self.frame_id
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'dyn_tracker'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        p_start = Point()
        p_start.x = float(x)
        p_start.y = float(y)
        p_start.z = 0.0

        p_end = Point()
        p_end.x = float(x + vx * self.arrow_horizon)
        p_end.y = float(y + vy * self.arrow_horizon)
        p_end.z = 0.0

        arrow.points = [p_start, p_end]
        arrow.scale.x = 0.05
        arrow.scale.y = 0.1
        arrow.scale.z = 0.1
        arrow.color.r = 1.0
        arrow.color.g = 0.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        ma.markers.append(arrow)

        self.pub_markers.publish(ma)

    def publish_clear_markers(self):
        ma = MarkerArray()
        for mid in [0, 1]:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'dyn_tracker'
            m.id = mid
            m.action = Marker.DELETE
            ma.markers.append(m)
        self.pub_markers.publish(ma)

    def clear_state_and_markers(self):
        self.last_pos = None
        self.last_time = None
        self.traj_history = []
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.publish_clear_markers()


# ---------------- main + launch ----------------

def main(args=None):
    rclpy.init(args=args)
    node = DynamicVisualTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run = ExecuteProcess(
        cmd=['python3', this_file],
        output='screen'
    )
    return LaunchDescription([
        TimerAction(period=1.0, actions=[run])
    ])


if __name__ == '__main__':
    main()
