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
│   ├── fly_truss.py              # Autonomous corridor flight script
│   ├── fly_pillars.py            # Pillar-field navigation with VFH2D avoidance
│   ├── fly_avoid.py              # General obstacle-avoidance flight script
│   ├── vfh2d.py                  # 2D Vector Field Histogram algorithm
│   ├── vfh3d.py                  # 3D Vector Field Histogram algorithm
│   ├── viz2d.py                  # Real-time 2D histogram visualizer
│   ├── tof_reader.py             # ToF sensor data reader utility
│   └── gz_markers.py             # Gazebo marker visualization helpers
└── worlds/
    ├── truss.sdf                 # Gazebo truss corridor world
    └── pillars.sdf               # Gazebo pillar-field world
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

## VFH2D Obstacle Avoidance

The `vfh2d.py` module implements a **2D Vector Field Histogram** for horizontal-plane obstacle avoidance using data from the ToF sensor ring.

### How It Works

1. **Histogram construction** — 3D obstacle points are projected onto the horizontal plane and binned into a 36-bin polar histogram (10° resolution). Closer obstacles receive higher weight.
2. **Hysteresis thresholding** — bins transition between free/blocked using dual thresholds (`threshold_low` / `threshold_high`) to prevent flickering.
3. **Obstacle enlargement** — blocked bins are dilated by ±`enlarge_bins` to account for the drone's physical radius (0.35m). Default is 3 bins (±30°).
4. **Clearance-biased direction selection** — the algorithm picks the free bin closest to the goal, but applies a **clearance penalty** to bins near blocked sectors (up to 40° of extra cost). This prevents the drone from grazing obstacles by steering it toward directions with more open space on both sides.
5. **Proximity speed scaling** — forward speed is reduced proportionally as obstacles get closer, down to 20% of max speed at the safe-distance threshold.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution_deg` | 10° | Angular width of each histogram bin |
| `threshold_low` | 0.15 | Hysteresis low threshold (bin stays blocked) |
| `threshold_high` | 0.3 | Hysteresis high threshold (bin becomes blocked) |
| `safe_distance` | 1.2m | Distance at which speed scaling begins |
| `max_speed` | 0.4 m/s | Maximum horizontal speed |
| `drone_radius` | 0.35m | Drone body radius for obstacle enlargement |
| `enlarge_bins` | 3 | Blocked bins dilated ± this many bins |

### Pillar-Field Navigation

`fly_pillars.py` navigates through the `pillars.sdf` world using VFH2D:

```bash
# Terminal 1 — start PX4 SITL with pillar world
cd ~/PX4-Autopilot
PX4_GZ_WORLD=pillars make px4_sitl gz_x500_tof

# Terminal 2 — run the avoidance script
python3 ~/px4sim_fyp/scripts/fly_pillars.py
```

A real-time visualizer (`viz2d.py`) displays the polar histogram and chosen heading direction in a separate window during flight.

## Truss Corridor Dimensions

- **Length**: 8m (x = -4.0 to 4.0m)
- **Width**: 2m (y = -1.0 to 1.0m)
- **Height**: 1m (z = 1.5 to 2.5m)
- **Structure**: Top/bottom chords, vertical posts every 2m, cross braces
