#!/bin/bash
REPO_DIR=$(dirname "$(realpath "$0")")
PX4_DIR=${1:-$HOME/PX4-Autopilot}

ln -sfn $REPO_DIR/models/x500_tof \
  $PX4_DIR/Tools/simulation/gz/models/x500_tof

ln -sf $REPO_DIR/worlds/truss.sdf \
  $PX4_DIR/Tools/simulation/gz/worlds/truss.sdf

ln -sf $REPO_DIR/worlds/pillars.sdf \
  $PX4_DIR/Tools/simulation/gz/worlds/pillars.sdf

ln -sf $REPO_DIR/worlds/truss2.sdf \
  $PX4_DIR/Tools/simulation/gz/worlds/truss2.sdf

ln -sf $REPO_DIR/airframes/4022_gz_x500_tof \
  $PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4022_gz_x500_tof

echo "Done."
