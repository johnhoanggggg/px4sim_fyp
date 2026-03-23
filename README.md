# PX4 Truss Corridor Simulator

A PX4 simulation environment featuring a quadrotor (X500) equipped with 12 Time-of-Flight sensors navigating through a ceiling truss corridor structure in Gazebo.

## Overview

This project provides:

- **Custom X500 quadrotor model** with 12 VL53L8CX-approximation ToF sensors (10 horizontal at 36-degree increments + up/down)
- **Truss corridor world** — an 8m long, 2m wide, 1m tall structure at z=1.5–2.5m
- **Autonomous flight script** that flies the drone through the corridor using MAVLink offboard control

## Repository Structure

```
px4sim_fyp/
├── setup.sh                      # Symlinks project files into PX4-Autopilot
├── airframes/
│   └── 4022_gz_x500_tof          # PX4 airframe config for X500 + ToF
├── models/
│   └── x500_tof/
│       ├── model.config
│       └── model.sdf             # Vehicle model with 12 ToF sensors
├── scripts/
│   └── fly_truss.py              # Autonomous corridor flight script
└── worlds/
    └── truss.sdf                 # Gazebo truss corridor world
```

## Prerequisites

- **Ubuntu** (20.04 or 22.04 recommended)
- **PX4-Autopilot** — [installation guide](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html)
- **Gazebo** (Harmonic or Garden) — installed as part of PX4 setup
- **Python 3** with `pymavlink`:
  ```bash
  pip install pymavlink
  ```

## Setup

1. **Clone PX4-Autopilot** (if not already installed):
   ```bash
   git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
   bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh
   ```

2. **Clone this repository**:
   ```bash
   git clone <repo-url> ~/px4sim_fyp
   ```

3. **Run the setup script** to symlink models, worlds, and airframes into PX4:
   ```bash
   cd ~/px4sim_fyp
   chmod +x setup.sh
   ./setup.sh ~/PX4-Autopilot
   ```

   This creates symlinks for:
   - `models/x500_tof/` → `PX4-Autopilot/Tools/simulation/gz/models/`
   - `worlds/truss.sdf` → `PX4-Autopilot/Tools/simulation/gz/worlds/`
   - `airframes/4022_gz_x500_tof` → `PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/`

## Running the Simulation

1. **Start PX4 SITL with the truss world**:
   ```bash
   cd ~/PX4-Autopilot
   PX4_GZ_WORLD=truss make px4_sitl gz_x500_tof
   ```

2. **Run the autonomous flight script** (in a separate terminal):
   ```bash
   python3 ~/px4sim_fyp/scripts/fly_truss.py
   ```

   The script will:
   1. Stream setpoints and switch to OFFBOARD mode
   2. Arm the vehicle
   3. Take off to z=2.0m (center of the truss corridor)
   4. Fly through waypoints along the corridor (x = -3, -1, +1, +3m)
   5. Return to center, land, and disarm

## Sensor Configuration

The X500 model includes 12 GPU-based LIDAR sensors simulating VL53L8CX ToF sensors:

| Sensor | Direction | Topic |
|--------|-----------|-------|
| ToF 0–9 | Horizontal ring (36-degree spacing) | `/tof/0` – `/tof/9` |
| ToF Up | Upward | `/tof/up` |
| ToF Down | Downward | `/tof/down` |

Each sensor has:
- **Resolution**: 8x8 rays
- **FOV**: ~45 degrees
- **Range**: 0.02–4.0m
- **Update rate**: 12 Hz
- **Noise**: Gaussian, stddev=0.015m

## Truss Corridor Dimensions

- **Length**: 8m (x = -4.0 to 4.0m)
- **Width**: 2m (y = -1.0 to 1.0m)
- **Height**: 1m (z = 1.5 to 2.5m)
- **Structure**: Top/bottom chords, vertical posts every 2m, cross braces
