# tb4_fortress_spawn.launch.py  (ROS 2 Humble + Gazebo Fortress)
import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable, ExecuteProcess, GroupAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    # -------- Args --------
    args = []
    for a, dv in [
        ('rviz','false'), ('use_sim_time','true'),
        ('model','standard'), ('namespace',''),
        ('spawn_dock','true'),  # puedes desactivar si tu mundo ya trae dock
        ('x','0.0'), ('y','0.0'), ('z','0.05'), ('yaw','0.0'),
        ('world','maze.sdf'),   # mundo dentro de my_robot_sim/worlds/
        ('robot_name','tb4_1'), # evitar choque con "turtlebot4"
        ('dock_name','standard_dock_1')
    ]:
        args.append(DeclareLaunchArgument(a, default_value=dv))

    ns          = LaunchConfiguration('namespace')
    use_sim     = LaunchConfiguration('use_sim_time')
    model       = LaunchConfiguration('model')
    world       = LaunchConfiguration('world')
    x = LaunchConfiguration('x'); y = LaunchConfiguration('y')
    z = LaunchConfiguration('z'); yaw = LaunchConfiguration('yaw')
    robot_name  = LaunchConfiguration('robot_name')
    dock_name   = LaunchConfiguration('dock_name')
    spawn_dock  = LaunchConfiguration('spawn_dock')

    # -------- Paths --------
    pkg_my = get_package_share_directory('my_robot_sim')
    pkg_tb4_ign_bringup   = get_package_share_directory('turtlebot4_ignition_bringup')
    pkg_tb4_desc          = get_package_share_directory('turtlebot4_description')
    pkg_tb4_viz           = get_package_share_directory('turtlebot4_viz')
    pkg_tb4_nav           = get_package_share_directory('turtlebot4_navigation')
    pkg_create_common     = get_package_share_directory('irobot_create_common_bringup')
    pkg_create_ign_bring  = get_package_share_directory('irobot_create_ignition_bringup')

    world_path = PathJoinSubstitution([pkg_my, 'worlds', world])

    # Launches de otros paquetes
    ros_ign_bridge_launch = PathJoinSubstitution([pkg_tb4_ign_bringup, 'launch', 'ros_ign_bridge.launch.py'])
    robot_description_launch = PathJoinSubstitution([pkg_tb4_desc, 'launch', 'robot_description.launch.py'])
    dock_description_launch  = PathJoinSubstitution([pkg_create_common, 'launch', 'dock_description.launch.py'])
    tb4_nodes_launch         = PathJoinSubstitution([pkg_tb4_ign_bringup, 'launch', 'turtlebot4_nodes.launch.py'])
    create3_nodes_launch     = PathJoinSubstitution([pkg_create_common, 'launch', 'create3_nodes.launch.py'])
    create3_ign_nodes_launch = PathJoinSubstitution([pkg_create_ign_bring, 'launch', 'create3_ignition_nodes.launch.py'])
    rviz_launch              = PathJoinSubstitution([pkg_tb4_viz, 'launch', 'view_robot.launch.py'])

    # -------- Entorno (Fortress) --------
    set_res = SetEnvironmentVariable(
        'IGN_GAZEBO_RESOURCE_PATH',
        ':'.join([
            pkg_tb4_desc,
            pkg_create_common,
            os.path.join(pkg_my, 'worlds'),
            os.path.join(pkg_my, 'models')
        ])
    )
    # Asegura carga de systems (p.ej. ign_ros2_control)
    set_plugins = SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '/opt/ros/humble/lib')

    # (Opcional) evitar segfaults de OGRE2 mientras depuras: comenta para volver a OGRE2
    # set_render = SetEnvironmentVariable('IGN_RENDERING_ENGINE', 'ogre')

    # -------- Servidor Ignition (Fortress) --------
    ign = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '4', world_path],
        output='screen'
    )

    # -------- Grupo de spawn --------
    spawn_group = GroupAction([
        PushRosNamespace(ns),

        # Descriptions
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([robot_description_launch]),
            launch_arguments={'model': model, 'use_sim_time': use_sim}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([dock_description_launch]),
            # dock en Ignition; si tu mundo ya trae dock, pon spawn_dock:=false
            launch_arguments={'gazebo': 'ignition'}.items(),
            condition=IfCondition(spawn_dock)
        ),

        # Spawn TB4 (Fortress -> ros_ign_gazebo)
        Node(
            package='ros_ign_gazebo', executable='create', output='screen',
            arguments=['-name', robot_name, '-x', x, '-y', y, '-z', z, '-Y', yaw, '-topic', 'robot_description']
        ),

        # Spawn Dock (opcional)
        Node(
            package='ros_ign_gazebo', executable='create', output='screen',
            arguments=['-name', dock_name, '-x', x, '-y', y, '-z', z, '-Y', yaw, '-topic', 'standard_dock_description'],
            condition=IfCondition(spawn_dock)
        ),

        # Bridge (Ignition) + nodos
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([ros_ign_bridge_launch]),
            launch_arguments={
                'model': model, 'robot_name': robot_name, 'dock_name': dock_name, 'namespace': ns
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([tb4_nodes_launch]),
            launch_arguments={'model': model}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_nodes_launch]),
            launch_arguments={'namespace': ns}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([create3_ign_nodes_launch]),
            launch_arguments={'robot_name': robot_name, 'dock_name': dock_name}.items()
        ),
    ])

    # (Opcional) RViz
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments={'namespace': ns, 'use_sim_time': use_sim}.items(),
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    ld = LaunchDescription(args)
    ld.add_action(set_res)
    ld.add_action(set_plugins)
    # ld.add_action(set_render)  # descomenta si quieres forzar OGRE1
    ld.add_action(ign)
    ld.add_action(spawn_group)
    ld.add_action(rviz)
    return ld
