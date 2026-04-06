# PX4 Sim FYP - Quadrotor Autonomous Navigation

PX4 SITL Gazebo simulation for quadrotor corridor navigation with Time-of-Flight sensors and propeller guards.

## File Structure

```
px4sim_fyp/
├── airframes/
│   └── 4022_gz_x500_tof        # Airframe config: sets PX4_SIM_MODEL=propguards_tof, sources 4001_gz_x500
├── models/
│   ├── propguards/              # Base X500 with cylindrical prop guards (no sensors)
│   │   ├── model.sdf            # mass=2.0kg, 4 guard cylinders, uses model://x500_base meshes
│   │   └── model.config
│   ├── propguards_tof/          # Propguards + 12 ToF sensors (ACTIVE MODEL)
│   │   ├── model.sdf            # Includes propguards via <uri>propguards</uri>, adds 12 gpu_lidar sensors
│   │   └── model.config
│   └── x500_tof/                # X500 (no guards) + 10 horizontal ToF sensors (LEGACY)
│       ├── model.sdf
│       └── model.config
├── worlds/
│   ├── truss.sdf                # Corridor: 8m x 2m x 1m
│   ├── truss2.sdf               # Extended truss corridor
│   └── pillars.sdf              # Pillar field for avoidance testing
├── scripts/
│   ├── fly_truss.py             # Main corridor navigation
│   ├── fly_truss2_vfh.py        # VFH corridor nav
│   ├── fly_truss2_vfh_yaw.py    # VFH with yaw control
│   ├── fly_truss2_fgm.py        # FGM corridor nav
│   ├── fly_truss2_dwa.py        # DWA corridor nav
│   ├── fly_pillars_fgm.py       # Pillar field with FGM avoidance
│   ├── vfh3d.py / vfh3d2.py / vfh3dyaw.py  # Vector Field Histogram 3D implementations
│   ├── fgm2d.py / fgm3d.py     # Follow the Gap Method implementations
│   ├── dwa3d.py                 # Dynamic Window Approach 3D
│   ├── tof_reader.py            # ToF sensor data reader utility
│   ├── viz2d.py                 # 2D visualization
│   └── gz_markers.py            # Gazebo visualization markers
├── ros2_ws/src/
│   ├── px4_sim_bringup/         # Launch files and setup
│   ├── px4_vslam/               # Visual SLAM integration
│   └── px4_tof_avoidance/       # ToF-based obstacle avoidance ROS2 node
├── setup.sh                     # Creates symlinks into ~/PX4-Autopilot (run once)
└── README.md
```

## How PX4-Autopilot Integration Works

`setup.sh` symlinks models, worlds, and airframes into `~/PX4-Autopilot/`:
- `models/*` -> `~/PX4-Autopilot/Tools/simulation/gz/models/`
- `worlds/*.sdf` -> `~/PX4-Autopilot/Tools/simulation/gz/worlds/`
- `airframes/4022_gz_x500_tof` -> `~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/`

## Running the Simulation

```bash
# First time: create symlinks
cd ~/px4sim_fyp && bash setup.sh

# Run with propguards_tof model (default for gz_x500_tof target):
cd ~/PX4-Autopilot && PX4_GZ_WORLD=truss2 make px4_sitl gz_x500_tof

# The gz_x500_tof make target loads airframe 4022 which sets PX4_SIM_MODEL=propguards_tof
# To use x500_tof instead: PX4_SIM_MODEL=x500_tof before make
```

## ToF Sensor Layout (propguards_tof)

12 sensors total: 10 horizontal ring at 36-degree spacing + 1 up + 1 down.
- Topics: `/tof/0` through `/tof/9` (horizontal), `/tof/up`, `/tof/down`
- Type: gpu_lidar, 8x8 rays, 45deg FOV, range 0.02-2.0m, 12Hz, noise stddev=0.015m

## Avoidance Algorithms

- **VFH3D**: Vector Field Histogram - builds polar histogram of obstacle density, finds best gap
- **FGM**: Follow the Gap Method - finds largest gap in sensor ring, steers toward it
- **DWA3D**: Dynamic Window Approach - samples velocity space, scores trajectories

## Key Technical Details

- Propguards model depends on `model://x500_base` meshes from PX4-Autopilot
- Motor plugin: `gz-sim-multicopter-motor-model-system`
- Physics: ODE, step=0.004s, 250Hz
- Flight scripts use MAVSDK Python for drone control
