import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    pkg = get_package_share_directory('my_robot_sim')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = os.path.join(pkg, 'worlds', 'diff_drive_empty.sdf')
    urdf  = os.path.join(pkg, 'resource', 'diff_drive.urdf')
    with open(urdf, 'r') as f:
        robot_description = f.read()

    # Robot State Publisher (TF de URDF)
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
    )

    # Gazebo (Ignition/Fortress)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r ' + world}.items(),
    )

    # Spawn del modelo usando /robot_description
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', '/robot_description', '-z', '0.05'],
        output='screen'
    )

    # Bridge ROS <-> GZ (sin TF)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/model/diff_drive/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/world/diff_drive_world/model/diff_drive/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',

            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        ],
        remappings=[
            ('/model/diff_drive/odometry', '/odom'),
            ('/world/diff_drive_world/model/diff_drive/joint_state', '/joint_states'),
            ('/world/diff_drive_world/model/diff_drive/link/imu_link/sensor/imu/imu', '/imu'),
        ],
        output='screen'
    )

    # === PUENTES TF ESTÁTICOS (IDENTIDAD) ===
    # diff_drive/base_link -> base_link
    static_bl_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # x y z qx qy qz qw  parent                 child
        arguments=['0','0','0','0','0','0','1','diff_drive/base_link','base_link'],
        output='screen'
    )

    # UKF
    config_dir = os.path.join(pkg, 'config')
    ukf_yaml  = os.path.join(config_dir, 'ukf.yaml')
    slam_yaml = os.path.join(config_dir, 'slam.yaml')

    ukf_node = Node(
        package='robot_localization',
        executable='ukf_node',
        name='ukf_filter_node',
        parameters=[ukf_yaml],
        output='screen'
    )

    # SLAM Toolbox (arranca con pequeño delay)
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_yaml],
    )
    slam_delayed = TimerAction(period=2.0, actions=[slam_node])

    # RViz
    rviz_cfg = os.path.join(pkg, 'resource', 'diff_drive_urdf.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        rsp,
        gz_sim,
        spawn,
        bridge,
        static_bl_bridge,
        ukf_node,        # odom(prefijo)->base_link(prefijo) por UKF
        slam_delayed,    # map->diff_drive/odom por SLAM
        rviz,
    ])
