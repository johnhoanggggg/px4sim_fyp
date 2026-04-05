from setuptools import setup
import os
from glob import glob

package_name = 'px4_tof_avoidance'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.algorithms'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'tof_aggregator = px4_tof_avoidance.tof_aggregator_node:main',
            'avoidance = px4_tof_avoidance.avoidance_node:main',
        ],
    },
)
