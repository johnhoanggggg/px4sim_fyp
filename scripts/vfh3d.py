#!/usr/bin/env python3
"""
VFH3D (Vector Field Histogram 3D) obstacle avoidance.

Takes 3D obstacle points in body frame and a goal direction,
produces a safe velocity vector that steers toward the goal
while avoiding obstacles.
"""

import math
import numpy as np


class VFH3D:
    """
    3D Vector Field Histogram for obstacle avoidance.

    Builds a spherical polar histogram (azimuth × elevation) from obstacle
    points, identifies free sectors, and selects the best steering direction
    closest to the goal.
    """

    def __init__(
        self,
        resolution_deg: float = 10.0,
        threshold_low: float = 0.3,
        threshold_high: float = 0.6,
        safe_distance: float = 0.5,
        max_speed: float = 0.5,
    ):
        """
        Args:
            resolution_deg: histogram bin size in degrees.
            threshold_low:  bins below this are considered free.
            threshold_high: bins above this are considered blocked.
            safe_distance:  distance (m) at which obstacle weight is maximum.
            max_speed:      maximum output velocity (m/s).
        """
        self.res = math.radians(resolution_deg)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.safe_distance = safe_distance
        self.max_speed = max_speed

        # Histogram dimensions
        self.n_az = int(round(2 * math.pi / self.res))       # azimuth bins
        self.n_el = int(round(math.pi / self.res))            # elevation bins

        # Persistent binary histogram for hysteresis
        self._blocked = np.zeros((self.n_el, self.n_az), dtype=bool)

    def update(self, obstacle_pts: np.ndarray, goal_ned: tuple) -> tuple:
        """
        Compute a safe velocity vector.

        Args:
            obstacle_pts: (N, 3) array of obstacle positions in body frame
                          (X=forward, Y=left, Z=up).
            goal_ned:     (vx, vy, vz) desired direction in NED frame.
                          Does not need to be unit length; only direction matters.

        Returns:
            (vx, vy, vz) velocity command in NED frame (m/s).
            Returns (0, 0, 0) if fully blocked.
        """
        histogram = self._build_histogram(obstacle_pts)
        self._apply_hysteresis(histogram)
        best_dir = self._select_direction(goal_ned)

        if best_dir is None:
            return (0.0, 0.0, 0.0)

        # Scale speed: slow down when obstacles are near
        speed = self._compute_speed(obstacle_pts)
        return (best_dir[0] * speed, best_dir[1] * speed, best_dir[2] * speed)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_histogram(self, pts: np.ndarray) -> np.ndarray:
        """Build density histogram from obstacle points."""
        hist = np.zeros((self.n_el, self.n_az), dtype=np.float32)

        if len(pts) == 0:
            return hist

        # Convert body-frame points to spherical coords
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        dist = np.sqrt(x**2 + y**2 + z**2)
        dist = np.clip(dist, 0.01, None)

        azimuth = np.arctan2(y, x)                        # -pi..pi
        elevation = np.arcsin(np.clip(z / dist, -1, 1))   # -pi/2..pi/2

        # Obstacle weight: inverse of distance (closer = more dangerous)
        weight = np.clip(1.0 - dist / (self.safe_distance * 4), 0, 1)

        # Bin indices
        az_idx = ((azimuth + math.pi) / self.res).astype(int) % self.n_az
        el_idx = ((elevation + math.pi / 2) / self.res).astype(int)
        el_idx = np.clip(el_idx, 0, self.n_el - 1)

        # Accumulate
        np.add.at(hist, (el_idx, az_idx), weight)

        return hist

    def _apply_hysteresis(self, histogram: np.ndarray):
        """Update blocked mask with hysteresis thresholds."""
        # Currently blocked cells stay blocked until below low threshold
        still_blocked = self._blocked & (histogram >= self.threshold_low)
        # Currently free cells become blocked above high threshold
        newly_blocked = ~self._blocked & (histogram >= self.threshold_high)
        self._blocked = still_blocked | newly_blocked

    def _select_direction(self, goal_ned: tuple) -> tuple | None:
        """
        Find the free direction closest to the goal.

        Works in body frame (assumes heading ≈ 0 for now, so body ≈ NED).
        Returns unit direction vector in NED or None if fully blocked.
        """
        goal = np.array(goal_ned, dtype=np.float64)
        goal_norm = np.linalg.norm(goal)
        if goal_norm < 1e-6:
            return (0.0, 0.0, 0.0)
        goal = goal / goal_norm

        # Build list of candidate directions (centers of free bins)
        best_dir = None
        best_cost = float("inf")

        for ei in range(self.n_el):
            for ai in range(self.n_az):
                if self._blocked[ei, ai]:
                    continue
                # Bin center angles
                az = -math.pi + (ai + 0.5) * self.res
                el = -math.pi / 2 + (ei + 0.5) * self.res
                # Direction vector (body frame ≈ NED for now)
                dx = math.cos(el) * math.cos(az)
                dy = math.cos(el) * math.sin(az)
                dz = math.sin(el)
                d = np.array([dx, dy, dz])
                # Cost = angle to goal (lower is better)
                cos_angle = np.dot(d, goal)
                cost = 1.0 - cos_angle  # 0 = aligned, 2 = opposite
                if cost < best_cost:
                    best_cost = cost
                    best_dir = d

        if best_dir is None:
            return None
        return (float(best_dir[0]), float(best_dir[1]), float(best_dir[2]))

    def _compute_speed(self, pts: np.ndarray) -> float:
        """Scale speed based on closest obstacle in the horizontal plane.
        Only considers obstacles within ±30° of horizontal so that
        ground/ceiling detections don't throttle lateral speed."""
        if len(pts) == 0:
            return self.max_speed

        # Filter to near-horizontal obstacles only
        horiz_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        vert = np.abs(pts[:, 2])
        elevation = np.arctan2(vert, np.clip(horiz_dist, 0.01, None))
        horiz_mask = elevation < math.radians(30)

        if not np.any(horiz_mask):
            return self.max_speed

        dists = np.linalg.norm(pts[horiz_mask], axis=1)
        min_dist = float(np.min(dists))

        if min_dist < self.safe_distance:
            return self.max_speed * max(min_dist / self.safe_distance, 0.3)
        return self.max_speed

    def get_blocked_bins(self):
        """Return list of (azimuth, elevation, blocked) for each bin center."""
        bins = []
        for ei in range(self.n_el):
            for ai in range(self.n_az):
                az = -math.pi + (ai + 0.5) * self.res
                el = -math.pi / 2 + (ei + 0.5) * self.res
                bins.append((az, el, bool(self._blocked[ei, ai])))
        return bins

    def reset(self):
        """Clear the hysteresis state."""
        self._blocked[:] = False


# ---------------------------------------------------------------------------
# Standalone test with synthetic data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vfh = VFH3D(resolution_deg=10, safe_distance=0.5, max_speed=0.5)

    # Simulate wall on the right (positive Y in body = left, so negative Y = right)
    wall_pts = []
    for z in np.linspace(-0.3, 0.3, 5):
        for x in np.linspace(-1, 1, 10):
            wall_pts.append([x, -0.8, z])  # wall 0.8m to the right
    pts = np.array(wall_pts)

    # Goal: fly forward (+X in NED)
    goal = (1.0, 0.0, 0.0)
    vel = vfh.update(pts, goal)
    print(f"Wall on right, goal forward:")
    print(f"  Velocity: vx={vel[0]:.3f}, vy={vel[1]:.3f}, vz={vel[2]:.3f}")
    print(f"  Should steer slightly left (positive vy)")

    # Goal: fly right (into wall)
    goal2 = (0.0, -1.0, 0.0)
    vel2 = vfh.update(pts, goal2)
    print(f"\nWall on right, goal right (into wall):")
    print(f"  Velocity: vx={vel2[0]:.3f}, vy={vel2[1]:.3f}, vz={vel2[2]:.3f}")
    print(f"  Should deflect away from wall")
