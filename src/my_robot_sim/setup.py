from setuptools import find_packages, setup

package_name = 'my_robot_sim'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name, ['package.xml']))
## Worlds
data_files.append(('share/' + package_name + '/worlds', ['worlds/diff_drive_empty.sdf']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/maze.sdf']))
## Launch
data_files.append(('share/' + package_name + '/launch', ['launch/diff_drive_urdf_launch.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/tb4_fortress_spawn.launch.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/prm_planner.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/decision_manager.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/path_follower.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/rrt_planner.py']))
## Resource
data_files.append(('share/' + package_name + '/resource', ['resource/diff_drive.urdf']))
data_files.append(('share/' + package_name + '/resource', ['resource/diff_drive_urdf.rviz']))
##config
data_files.append(('share/' + package_name + '/config',
                   ['config/slam.yaml', 'config/ukf.yaml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pedro',
    maintainer_email='p.bello@uniandes.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)