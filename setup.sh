#!/bin/bash
REPO_DIR=$(dirname "$(realpath "$0")")
PX4_DIR=${1:-$HOME/PX4-Autopilot}

ln -s $REPO_DIR/models/x500_tof \
  $PX4_DIR/Tools/simulation/gz/models/x500_tof

ln -s $REPO_DIR/worlds/truss.sdf \
  $PX4_DIR/Tools/simulation/gz/worlds/truss.sdf

ln -sf $REPO_DIR/worlds/pillars.sdf \
  $PX4_DIR/Tools/simulation/gz/worlds/pillars.sdf

ln -s $REPO_DIR/airframes/4022_gz_x500_tof \
  $PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4022_gz_x500_tof

echo "Done."
