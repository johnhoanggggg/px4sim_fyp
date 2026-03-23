#!/usr/bin/env python3
"""
VFH2D — 2D Vector Field Histogram for horizontal-plane obstacle avoidance.

Projects 3D obstacle points onto the horizontal plane, builds a 1D azimuth
histogram, and selects the free direction closest to the goal.
"""

import math
import numpy as np


class VFH2D:
    """
    2D Vector Field Histogram.

    Builds a polar histogram (azimuth only) from obstacle points projected
    onto the horizontal plane.  Applies hysteresis and selects the safest
    direction toward the goal.
    """

    def __init__(
        self,
        resolution_deg: float = 10.0,
        threshold_low: float = 0.3,
        threshold_high: float = 0.6,
        safe_distance: float = 0.5,
        max_speed: float = 0.5,
    ):
        self.res = math.radians(resolution_deg)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.safe_distance = safe_distance
        self.max_speed = max_speed

        self.n_bins = int(round(2 * math.pi / self.res))
        self._blocked = np.zeros(self.n_bins, dtype=bool)

        # Cache bin centre angles (body frame: 0 = forward, +ve = left)
        self._bin_centres = np.array(
            [-math.pi + (i + 0.5) * self.res for i in range(self.n_bins)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obstacle_pts: np.ndarray, goal_body: tuple) -> tuple:
        """
        Compute a safe 2D velocity in body FLU frame.

        Args:
            obstacle_pts: (N, 3) body-FLU points (X fwd, Y left, Z up).
            goal_body:    (gx, gy) or (gx, gy, gz) desired direction in
                          body FLU.  Only the horizontal (x, y) part is used.

        Returns:
            (vx, vy) velocity in body FLU (horizontal plane).
            (0, 0) if fully blocked.
        """
        histogram = self._build_histogram(obstacle_pts)
        self._apply_hysteresis(histogram)
        best_az = self._select_direction(goal_body)

        if best_az is None:
            return (0.0, 0.0)

        speed = self._compute_speed(obstacle_pts)
        vx = speed * math.cos(best_az)
        vy = speed * math.sin(best_az)
        return (vx, vy)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Return list of (azimuth_rad, is_blocked) for every bin."""
        return list(zip(self._bin_centres.tolist(), self._blocked.tolist()))

    def get_chosen_direction(self) -> float | None:
        """Return the last chosen azimuth (body frame) or None."""
        return self._last_chosen

    def reset(self):
        self._blocked[:] = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    _last_chosen: float | None = None

    def _build_histogram(self, pts: np.ndarray) -> np.ndarray:
        hist = np.zeros(self.n_bins, dtype=np.float32)
        if len(pts) == 0:
            return hist

        x, y = pts[:, 0], pts[:, 1]
        horiz_dist = np.sqrt(x**2 + y**2)
        # Ignore points that are almost directly above/below
        mask = horiz_dist > 0.05
        if not np.any(mask):
            return hist

        x, y, horiz_dist = x[mask], y[mask], horiz_dist[mask]
        azimuth = np.arctan2(y, x)  # body FLU azimuth
        weight = np.clip(1.0 - horiz_dist / (self.safe_distance * 4), 0, 1)

        idx = ((azimuth + math.pi) / self.res).astype(int) % self.n_bins
        np.add.at(hist, idx, weight)
        return hist

    def _apply_hysteresis(self, histogram: np.ndarray):
        still = self._blocked & (histogram >= self.threshold_low)
        newly = ~self._blocked & (histogram >= self.threshold_high)
        self._blocked = still | newly

    def _select_direction(self, goal_body: tuple) -> float | None:
        gx, gy = float(goal_body[0]), float(goal_body[1])
        goal_az = math.atan2(gy, gx)

        free = ~self._blocked
        if not np.any(free):
            self._last_chosen = None
            return None

        # Angular difference (wrapped to [-pi, pi])
        diff = self._bin_centres[free] - goal_az
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        best_idx = np.argmin(np.abs(diff))
        chosen = float(self._bin_centres[free][best_idx])
        self._last_chosen = chosen
        return chosen

    def _compute_speed(self, pts: np.ndarray) -> float:
        if len(pts) == 0:
            return self.max_speed
        horiz_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        # Only consider near-horizontal points (body |z| < 0.5m)
        mask = np.abs(pts[:, 2]) < 0.5
        if not np.any(mask):
            return self.max_speed
        min_d = float(np.min(horiz_dist[mask]))
        if min_d < self.safe_distance:
            return self.max_speed * max(min_d / self.safe_distance, 0.3)
        return self.max_speed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vfh = VFH2D(resolution_deg=10, safe_distance=0.5, max_speed=0.5)
    # Wall to the right in body FLU (negative Y = right)
    wall = np.array([[x, -0.8, 0.0] for x in np.linspace(-1, 1, 20)])
    vel = vfh.update(wall, (1.0, 0.0))
    print(f"Wall right, goal fwd → vel=({vel[0]:.2f}, {vel[1]:.2f})  (expect slight left)")
    for az, blk in vfh.get_histogram():
        if blk:
            print(f"  BLOCKED {math.degrees(az):+6.1f}°")
