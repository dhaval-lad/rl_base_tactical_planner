import os # Operating system library
from glob import glob # Handles file path names
from setuptools import setup # Facilitates the building of packages

package_name = 'outdoor_robot_spawner'

# Path of the current directory
cur_directory_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Path to the launch file      
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),

        # Path to the config file
        (os.path.join('share', package_name,'config'), glob('config/*.yaml')),

        # Path to the world file
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),

        # Path to the mobile robot sdf and config file
        (os.path.join('share', package_name,'models/4WD_robot/'), glob('./models/4WD_robot/*')),
        
        # Path to the pioneer sdf file
        (os.path.join('share', package_name,'models/4WD_robot/'), glob('./models/4WD_robot/model.sdf')),

        # Path to the pioneer config file
        (os.path.join('share', package_name,'models/4WD_robot/'), glob('./models/4WD_robot/model.config')),

        # Path to the target sdf file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.sdf')),

        # Path to the target config file
        (os.path.join('share', package_name,'models/Target/'), glob('./models/Target/model.config')),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tommaso Van Der Meer, Dhaval Lad (modified for tactical decision maker with discrete action space)',
    maintainer_email='tommaso.vandermeer@student.unisi.it, dhaval_lad@todo.todo',
    description='This package creates a simulation in Gazebo which includes a differential drive robot with Lidar and a Outdoor world',
    license='I am a student, I dont know what to put here :)',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'spawn_demo = outdoor_robot_spawner.spawn_demo:main',
          'start_training = outdoor_robot_spawner.start_training:main',
          'trained_agent = outdoor_robot_spawner.trained_agent:main',
        ],
    },
)