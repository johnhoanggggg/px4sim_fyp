#!/usr/bin/env python3
"""
FGM3D — 3-D Follow-the-Gap Method for obstacle avoidance.

Extends FGM2D with vertical (Z-axis) avoidance so the drone can navigate
multi-layer truss lattices where obstacles exist above and below.

Horizontal avoidance:
    Uses FGM2D on obstacle points filtered near the drone's altitude.

Vertical avoidance:
    Computes safe vertical velocity by measuring clearance above and below,
    applying repulsive forces when ceiling/floor obstacles are close, and
    blending with the desired altitude from the waypoint.

Returns (vx, vy, vz) in body FLU frame.
"""

import math
import numpy as np

from fgm2d import FGM2D


class FGM3D:
    """
    3-D Follow-the-Gap planner.

    Parameters
    ----------
    n_rays, max_range, bubble_radius, safe_distance, max_speed,
    gap_weight_goal, gap_weight_width, min_gap_width_deg, edge_margin_deg :
        Passed through to FGM2D for horizontal avoidance.
    z_band : float
        Vertical half-band (m) for filtering obstacle points that count
        as horizontal-plane threats.  Points with |Z| > z_band are ignored
        by the 2-D planner (but still used for vertical clearance).
    vert_safe : float
        Minimum safe vertical clearance (m).  Below this, repulsive
        vertical velocity kicks in.
    max_vz : float
        Maximum vertical speed (m/s).
    vz_gain : float
        P-gain for altitude tracking (vz = gain * altitude_error).
    """

    def __init__(
        self,
        # FGM2D params
        n_rays: int = 72,
        max_range: float = 1.5,
        bubble_radius: float = 0.3,
        safe_distance: float = 0.8,
        max_speed: float = 0.6,
        gap_weight_goal: float = 2.0,
        gap_weight_width: float = 0.3,
        min_gap_width_deg: float = 10.0,
        edge_margin_deg: float = 10.0,
        # 3-D specific
        z_band: float = 0.8,
        vert_safe: float = 0.6,
        max_vz: float = 0.5,
        vz_gain: float = 1.2,
    ):
        self.fgm2d = FGM2D(
            n_rays=n_rays,
            max_range=max_range,
            bubble_radius=bubble_radius,
            safe_distance=safe_distance,
            max_speed=max_speed,
            gap_weight_goal=gap_weight_goal,
            gap_weight_width=gap_weight_width,
            min_gap_width_deg=min_gap_width_deg,
            edge_margin_deg=edge_margin_deg,
        )
        self.z_band = z_band
        self.vert_safe = vert_safe
        self.max_vz = max_vz
        self.vz_gain = vz_gain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        obstacle_pts: np.ndarray,
        goal_body: tuple,
    ) -> tuple:
        """
        Compute a safe 3-D velocity in body FLU frame.

        Args:
            obstacle_pts: (N, 3) body-FLU points (X fwd, Y left, Z up).
            goal_body:    (gx, gy, gz) desired direction in body FLU.
                          gz > 0 means the goal is above the drone.

        Returns:
            (vx, vy, vz) velocity in body FLU.
        """
        gx, gy = float(goal_body[0]), float(goal_body[1])
        gz = float(goal_body[2]) if len(goal_body) > 2 else 0.0

        # --- Horizontal avoidance (XY plane) ---
        if len(obstacle_pts) > 0:
            z_mask = (obstacle_pts[:, 2] > -self.z_band) & \
                     (obstacle_pts[:, 2] < self.z_band)
            xy_pts = obstacle_pts[z_mask]
        else:
            xy_pts = obstacle_pts

        vx, vy = self.fgm2d.update(xy_pts, (gx, gy))

        # --- Vertical avoidance ---
        vz = self._compute_vz(obstacle_pts, gz)

        return (vx, vy, vz)

    def get_histogram(self):
        return self.fgm2d.get_histogram()

    def get_chosen_direction(self):
        return self.fgm2d.get_chosen_direction()

    def reset(self):
        self.fgm2d.reset()

    @property
    def bubble_radius(self):
        return self.fgm2d.bubble_radius

    # ------------------------------------------------------------------
    # Vertical clearance
    # ------------------------------------------------------------------

    def _compute_vz(self, pts: np.ndarray, goal_z: float) -> float:
        """
        Compute safe vertical velocity.

        goal_z : positive = goal is above, negative = below.
        """
        ceil_dist = float('inf')
        floor_dist = float('inf')

        if len(pts) > 0:
            # Only consider points horizontally close (within 1.0 m)
            horiz_dist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            close = pts[horiz_dist < 1.0]

            if len(close) > 0:
                above = close[close[:, 2] > 0.15]
                below = close[close[:, 2] < -0.15]
                if len(above) > 0:
                    ceil_dist = float(np.min(above[:, 2]))
                if len(below) > 0:
                    floor_dist = float(-np.max(below[:, 2]))

        # Desired vz toward goal altitude
        if abs(goal_z) > 0.15:
            vz = self.vz_gain * np.clip(goal_z, -1.0, 1.0)
            vz = float(np.clip(vz, -self.max_vz, self.max_vz))
        else:
            vz = 0.0

        # Repulsion from ceiling
        if ceil_dist < self.vert_safe:
            ratio = 1.0 - ceil_dist / self.vert_safe
            repulsion = -self.max_vz * ratio
            if vz > 0:
                # Trying to go up toward ceiling — override
                vz = min(vz, repulsion)
            else:
                vz += repulsion

        # Repulsion from floor
        if floor_dist < self.vert_safe:
            ratio = 1.0 - floor_dist / self.vert_safe
            repulsion = self.max_vz * ratio
            if vz < 0:
                # Trying to go down toward floor — override
                vz = max(vz, repulsion)
            else:
                vz += repulsion

        return float(np.clip(vz, -self.max_vz, self.max_vz))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fgm = FGM3D(
        n_rays=72,
        max_range=3.0,
        bubble_radius=0.55,
        safe_distance=1.2,
        max_speed=0.5,
    )

    # Beam above at 0.6 m — should push down
    beam_above = np.array([
        [0.5, -0.2, 0.6], [0.5, 0.0, 0.6], [0.5, 0.2, 0.6],
        [-0.2, 0.0, 0.6], [0.0, 0.3, 0.6],
    ])
    vel = fgm.update(beam_above, (2.0, 0.0, 0.3))
    print(f"Beam above → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  vz should be negative (pushed down from ceiling)")

    # Beam below at 0.5 m — should push up
    fgm.reset()
    beam_below = np.array([
        [0.3, -0.2, -0.5], [0.3, 0.0, -0.5], [0.3, 0.2, -0.5],
    ])
    vel = fgm.update(beam_below, (2.0, 0.0, -0.3))
    print(f"\nBeam below → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  vz should be positive (pushed up from floor)")

    # No obstacles — straight to goal
    fgm.reset()
    vel = fgm.update(np.empty((0, 3)), (2.0, 0.5, 1.0))
    print(f"\nNo obstacles → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  vz should be positive (toward goal above)")
