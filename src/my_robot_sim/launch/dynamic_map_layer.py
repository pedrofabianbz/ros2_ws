#!/usr/bin/env python3
import math
from typing import Optional, Tuple, List
import os

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_ros

# Para launch
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


class DynamicLayerSubmap(Node):
    def __init__(self):
        super().__init__('dynamic_map_layer')

        # ---------- Parámetros ----------
        # Entrada
        self.declare_parameter('static_map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')

        # Salidas
        self.declare_parameter('dynamic_map_topic', '/dynamic_map')
        self.declare_parameter('submap_topic', '/dynamic_submap')
        self.declare_parameter('motion_submap_topic', '/dynamic_motion_submap')
        self.declare_parameter('visual_topic', '/dynamic_visual')

        # Frames
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('laser_frame', 'lidar')
        self.declare_parameter('robot_frame', 'base_link')

        # Objeto dinámico (radio y valor en el mapa)
        # La POSE VIENE DE /box4/pose, no de TF
        self.declare_parameter('box_radius', 0.25)      # m (cubo 0.5m)
        self.declare_parameter('box_value', 100)        # valor en occupancy

        # Dinámica básica
        self.declare_parameter('decay_time', 0.5)
        self.declare_parameter('obs_min_occupancy', 80)
        self.declare_parameter('free_threshold', 20)

        # Filtro extra: no marcar dinámico pegado a paredes
        self.declare_parameter('static_border_threshold', 60)
        self.declare_parameter('static_border_radius', 1)

        # Submap
        self.declare_parameter('sub_width', 100)
        self.declare_parameter('sub_height', 100)

        # Congelar mapa estático base
        self.declare_parameter('freeze_static_map', True)

        # Sim time
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # Leer parámetros
        self.static_map_topic = self.get_parameter(
            'static_map_topic').get_parameter_value().string_value
        self.scan_topic = self.get_parameter(
            'scan_topic').get_parameter_value().string_value

        self.dynamic_map_topic = self.get_parameter(
            'dynamic_map_topic').get_parameter_value().string_value
        self.submap_topic = self.get_parameter(
            'submap_topic').get_parameter_value().string_value
        self.motion_submap_topic = self.get_parameter(
            'motion_submap_topic').get_parameter_value().string_value
        self.visual_topic = self.get_parameter(
            'visual_topic').get_parameter_value().string_value

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.laser_frame = self.get_parameter(
            'laser_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter(
            'robot_frame').get_parameter_value().string_value

        # parámetros del cubo dinámico
        self.box_radius = self.get_parameter(
            'box_radius').get_parameter_value().double_value
        self.box_value = int(self.get_parameter(
            'box_value').get_parameter_value().integer_value or 100)

        self.decay_time = self.get_parameter(
            'decay_time').get_parameter_value().double_value
        self.obs_value = int(self.get_parameter(
            'obs_min_occupancy').get_parameter_value().integer_value or 80)
        self.free_threshold = int(self.get_parameter(
            'free_threshold').get_parameter_value().integer_value or 20)

        self.static_border_threshold = int(self.get_parameter(
            'static_border_threshold').get_parameter_value().integer_value or 60)
        self.static_border_radius = int(self.get_parameter(
            'static_border_radius').get_parameter_value().integer_value or 1)

        self.sub_w = self.get_parameter(
            'sub_width').get_parameter_value().integer_value
        self.sub_h = self.get_parameter(
            'sub_height').get_parameter_value().integer_value

        self.freeze_static_map = self.get_parameter(
            'freeze_static_map').get_parameter_value().bool_value
        self.static_frozen = False

        # ---------- Estado interno ----------
        self.map_info = None
        self.base_data: Optional[List[int]] = None
        self.dyn_data: Optional[List[int]] = None
        self.last_hits = {}               # idx -> tiempo_último_impacto

        # para movimiento GLOBAL (en el mapa completo)
        self.prev_dyn_mask: Optional[List[bool]] = None  # len = width*height

        # última pose conocida del cubo (x,y) en frame global_frame
        self.last_box_pose: Optional[Tuple[float, float]] = None

        # TF (solo para láser y robot)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- Subs / pubs ----------
        self.create_subscription(
            OccupancyGrid,
            self.static_map_topic,
            self.map_callback,
            10
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        # SUSCRIPCIÓN A LA POSE DEL CUBO PUBLICADA POR Box4Oscillator
        self.create_subscription(
            PoseStamped,
            '/box4/pose',
            self.box_pose_callback,
            10
        )

        self.pub_dynamic_map = self.create_publisher(
            OccupancyGrid,
            self.dynamic_map_topic,
            10
        )
        self.pub_submap = self.create_publisher(
            OccupancyGrid,
            self.submap_topic,
            10
        )
        self.pub_motion_submap = self.create_publisher(
            OccupancyGrid,
            self.motion_submap_topic,
            10
        )
        self.pub_visual = self.create_publisher(
            Image,
            self.visual_topic,
            10
        )

        # Timer
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            f"✅ DynamicLayerSubmap inicializado:"
            f" static='{self.static_map_topic}', scan='{self.scan_topic}',"
            f" dynamic_map='{self.dynamic_map_topic}',"
            f" submap='{self.submap_topic}', motion='{self.motion_submap_topic}',"
            f" visual='{self.visual_topic}',"
            f" frames: global='{self.global_frame}', laser='{self.laser_frame}', robot='{self.robot_frame}',"
            f" box_radius={self.box_radius},"
            f" sub_size={self.sub_w}x{self.sub_h}, freeze_static_map={self.freeze_static_map}, "
            f" static_border_threshold={self.static_border_threshold}, "
            f" static_border_radius={self.static_border_radius}"
        )

    # ---------- Callbacks ----------

    def map_callback(self, msg: OccupancyGrid):
        if self.freeze_static_map and self.static_frozen:
            return

        self.map_info = msg.info
        self.base_data = list(msg.data)

        size = self.map_info.width * self.map_info.height
        if self.dyn_data is None or len(self.dyn_data) != size:
            self.dyn_data = [-1] * size
            self.last_hits.clear()

        if self.freeze_static_map and not self.static_frozen:
            self.static_frozen = True
            self.get_logger().info("📌 Mapa estático BASE congelado (primer /map).")

    def scan_callback(self, msg: LaserScan):
        if self.map_info is None or self.base_data is None or self.dyn_data is None:
            return

        laser_pose = self.get_laser_pose()
        if laser_pose is None:
            return

        lx, ly, lyaw = laser_pose
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        res = self.map_info.resolution

        angle = msg.angle_min
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            if r < msg.range_min or r > msg.range_max * 0.98:
                angle += msg.angle_increment
                continue

            ex = lx + r * math.cos(lyaw + angle)
            ey = ly + r * math.sin(lyaw + angle)

            dx = ex - lx
            dy = ey - ly
            dist = math.hypot(dx, dy)
            steps = max(1, int(dist / res))

            for s in range(steps):
                t = s / float(steps)
                fx = lx + dx * t
                fy = ly + dy * t
                idx_free = self.world_to_index(fx, fy)
                if idx_free is None:
                    continue

                base_val_free = self.base_data[idx_free]
                if base_val_free < self.free_threshold:
                    self.last_hits.pop(idx_free, None)

            idx_hit = self.world_to_index(ex, ey)
            if idx_hit is not None:
                base_val_hit = self.base_data[idx_hit]
                if base_val_hit < self.free_threshold:
                    if not self.is_near_static(idx_hit):
                        self.last_hits[idx_hit] = now_sec

            angle += msg.angle_increment

    # Pose del cubo desde /box4/pose
    def box_pose_callback(self, msg: PoseStamped):
        # Asumimos que /box4/pose está en el mismo frame que global_frame (map)
        self.last_box_pose = (msg.pose.position.x, msg.pose.position.y)

    def timer_callback(self):
        if self.map_info is None or self.base_data is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        size = self.map_info.width * self.map_info.height

        new_dyn = [-1] * size
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

        if self.prev_dyn_mask is None or len(self.prev_dyn_mask) != size:
            self.prev_dyn_mask = [False] * size

        motion_global = [0] * size
        for idx in range(size):
            prev_occ = self.prev_dyn_mask[idx]
            cur_occ = (self.dyn_data[idx] == 100)
            if prev_occ != cur_occ:
                motion_global[idx] = 1
            self.prev_dyn_mask[idx] = cur_occ

        fused = []
        for i, base_val in enumerate(self.base_data):
            if self.dyn_data[i] == 100:
                fused.append(100)
            else:
                fused.append(base_val)

        # ---- pintar el objeto dinámico box4 usando /box4/pose ----
        box_pose = self.last_box_pose
        if box_pose is not None:
            bx, by = box_pose
            self.paint_disc_world(
                fused,
                bx,
                by,
                radius=self.box_radius,
                value=self.box_value
            )
            if self.dyn_data is not None:
                self.paint_disc_world(
                    self.dyn_data,
                    bx,
                    by,
                    radius=self.box_radius,
                    value=100
                )

        full = OccupancyGrid()
        full.header.frame_id = self.global_frame
        full.header.stamp = self.get_clock().now().to_msg()
        full.info = self.map_info
        full.data = fused
        self.pub_dynamic_map.publish(full)

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        rx, ry, _ = robot_pose
        submap_msg, sub_data, global_idx_list = self.build_submap(full, rx, ry)

        if submap_msg is not None:
            self.pub_submap.publish(submap_msg)

            motion_msg, motion_data = self.build_motion_submap(
                submap_msg, global_idx_list, motion_global
            )
            if motion_msg is not None:
                self.pub_motion_submap.publish(motion_msg)

            self.publish_visual(submap_msg, sub_data, motion_data, global_idx_list)

    # ---------- TF ----------

    def get_laser_pose(self) -> Optional[Tuple[float, float, float]]:
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

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)
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

    # ---------- Utils mapa ----------

    def world_to_index(self, x: float, y: float) -> Optional[int]:
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

    def world_to_cell(self, info, x: float, y: float) -> Optional[Tuple[int, int]]:
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

    def paint_disc_world(self,
                         data_list: List[int],
                         cx: float,
                         cy: float,
                         radius: float,
                         value: int):
        if self.map_info is None:
            return

        info = self.map_info
        res = info.resolution
        cell = self.world_to_cell(info, cx, cy)
        if cell is None:
            return

        i_c, j_c = cell
        width = info.width
        height = info.height

        r_cells = int(radius / res) + 1

        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ii = i_c + di
                jj = j_c + dj
                if ii < 0 or jj < 0 or ii >= height or jj >= width:
                    continue
                dx = dj * res
                dy = di * res
                if dx * dx + dy * dy <= radius * radius:
                    idx = ii * width + jj
                    if 0 <= idx < len(data_list):
                        data_list[idx] = value

    def is_near_static(self, idx: int) -> bool:
        if self.base_data is None or self.map_info is None:
            return False

        width = self.map_info.width
        height = self.map_info.height

        i = idx // width
        j = idx % width

        r = self.static_border_radius
        thr = self.static_border_threshold

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni = i + di
                nj = j + dj
                if ni < 0 or nj < 0 or ni >= height or nj >= width:
                    continue
                n_idx = ni * width + nj
                if self.base_data[n_idx] >= thr:
                    return True
        return False

    def build_submap(self, full_map: OccupancyGrid,
                     robot_x: float, robot_y: float):
        info = full_map.info
        data = list(full_map.data)

        cell = self.world_to_cell(info, robot_x, robot_y)
        if cell is None:
            return None, None, None

        i_robot, j_robot = cell

        half_h = self.sub_h // 2
        half_w = self.sub_w // 2

        i_min = i_robot - half_h
        i_max = i_robot + half_h
        j_min = j_robot - half_w
        j_max = j_robot + half_w

        UNKNOWN = -1
        sub_data = [UNKNOWN] * (self.sub_w * self.sub_h)
        global_idx_list = [-1] * (self.sub_w * self.sub_h)

        for si in range(self.sub_h):
            gi = i_min + si
            if gi < 0 or gi >= info.height:
                continue
            for sj in range(self.sub_w):
                gj = j_min + sj
                if gj < 0 or gj >= info.width:
                    continue

                g_idx = gi * info.width + gj
                val = data[g_idx]

                s_idx = si * self.sub_w + sj
                sub_data[s_idx] = val
                global_idx_list[s_idx] = g_idx

        sub = OccupancyGrid()
        sub.header.frame_id = full_map.header.frame_id
        sub.header.stamp = self.get_clock().now().to_msg()

        sub.info.resolution = info.resolution
        sub.info.width = self.sub_w
        sub.info.height = self.sub_h

        sub.info.origin.position.x = info.origin.position.x + j_min * info.resolution
        sub.info.origin.position.y = info.origin.position.y + i_min * info.resolution
        sub.info.origin.position.z = 0.0
        sub.info.origin.orientation = info.origin.orientation

        sub.data = sub_data
        return sub, sub_data, global_idx_list

    def build_motion_submap(self,
                            submap_msg: OccupancyGrid,
                            global_idx_list: list,
                            motion_global: list):
        if motion_global is None:
            return None, None

        motion_data = [0] * len(global_idx_list)

        for s_idx, g_idx in enumerate(global_idx_list):
            if g_idx < 0:
                continue
            if motion_global[g_idx] == 1:
                motion_data[s_idx] = 100
            else:
                motion_data[s_idx] = 0

        motion = OccupancyGrid()
        motion.header = submap_msg.header
        motion.info = submap_msg.info
        motion.data = motion_data

        return motion, motion_data

    def publish_visual(self, submap_msg: OccupancyGrid,
                       sub_data: list,
                       motion_data: Optional[list],
                       global_idx_list: list):
        if self.base_data is None or self.dyn_data is None:
            return

        h = submap_msg.info.height
        w = submap_msg.info.width

        if motion_data is None:
            motion_data = [0] * len(sub_data)

        # info para localizar el cubo en la imagen
        has_box = self.last_box_pose is not None
        if has_box:
            bx, by = self.last_box_pose
            res = submap_msg.info.resolution
            ox = submap_msg.info.origin.position.x
            oy = submap_msg.info.origin.position.y
            r2 = self.box_radius * self.box_radius

        rgb_bytes = bytearray()
        for idx in range(len(sub_data)):
            g_idx = global_idx_list[idx]
            cur = sub_data[idx]
            mot = motion_data[idx]

            r = 255
            g = 255
            b = 255

            if g_idx >= 0:
                base_val = self.base_data[g_idx]
                dyn_val = self.dyn_data[g_idx]

                occ = (cur >= 50)
                static_occ = (base_val >= self.free_threshold and dyn_val != 100)
                dynamic_occ = (dyn_val == 100)
                moving = (mot == 100)

                if moving:
                    r, g, b = 0, 0, 255      # azul
                elif dynamic_occ:
                    r, g, b = 255, 0, 0      # rojo
                elif static_occ and occ:
                    r, g, b = 0, 0, 0        # negro
                else:
                    r, g, b = 255, 255, 255  # blanco

            # override si esta celda está dentro del radio del cubo
            if has_box:
                si = idx // w
                sj = idx % w
                cx = ox + (sj + 0.5) * res
                cy = oy + (si + 0.5) * res
                dx = cx - bx
                dy = cy - by
                if dx*dx + dy*dy <= r2:
                    # Pinta el cubo en VERDE para distinguirlo bien
                    r, g, b = 0, 255, 0

            rgb_bytes.extend([r, g, b])

        img = Image()
        img.header = submap_msg.header
        img.height = h
        img.width = w
        img.encoding = 'rgb8'
        img.is_bigendian = 0
        img.step = 3 * w
        img.data = bytes(rgb_bytes)

        self.pub_visual.publish(img)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicLayerSubmap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def generate_launch_description():
    this_file = os.path.realpath(__file__)
    run = ExecuteProcess(cmd=['python3', this_file], output='screen')
    return LaunchDescription([TimerAction(period=1.0, actions=[run])])


if __name__ == '__main__':
    main()
