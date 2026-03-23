#!/usr/bin/env python3
"""
VFH2D — 2D Vector Field Histogram for horizontal-plane obstacle avoidance.

Projects 3D obstacle points onto the horizontal plane, builds a 1D azimuth
histogram, enlarges obstacles by the drone radius, and selects the free
direction closest to the goal.
"""

import math
import numpy as np


class VFH2D:
    """
    2D Vector Field Histogram with obstacle enlargement.

    Builds a polar histogram (azimuth only) from obstacle points projected
    onto the horizontal plane.  Enlarges blocked sectors by the drone radius
    so thin obstacles are properly avoided.
    """

    def __init__(
        self,
        resolution_deg: float = 10.0,
        threshold_low: float = 0.2,
        threshold_high: float = 0.35,
        safe_distance: float = 0.8,
        max_speed: float = 0.5,
        drone_radius: float = 0.35,
        enlarge_bins: int = 2,
    ):
        self.res = math.radians(resolution_deg)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.drone_radius = drone_radius
        self.enlarge_bins = enlarge_bins  # spread blocked bins ± this many

        self.n_bins = int(round(2 * math.pi / self.res))
        self._blocked = np.zeros(self.n_bins, dtype=bool)
        self._last_chosen: float | None = None

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
            goal_body:    (gx, gy) desired direction in body FLU.

        Returns:
            (vx, vy) velocity in body FLU (horizontal plane).
            (0, 0) if fully blocked.
        """
        histogram = self._build_histogram(obstacle_pts)
        self._apply_hysteresis(histogram)
        self._enlarge_obstacles()
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
        self._last_chosen = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_histogram(self, pts: np.ndarray) -> np.ndarray:
        hist = np.zeros(self.n_bins, dtype=np.float32)
        if len(pts) == 0:
            return hist

        x, y = pts[:, 0], pts[:, 1]
        horiz_dist = np.sqrt(x**2 + y**2)
        # Ignore points directly above/below
        mask = horiz_dist > 0.05
        if not np.any(mask):
            return hist

        x, y, horiz_dist = x[mask], y[mask], horiz_dist[mask]
        azimuth = np.arctan2(y, x)

        # More aggressive weight: close obstacles get high weight
        weight = np.clip(1.0 - horiz_dist / (self.safe_distance * 2.5), 0, 1)

        idx = ((azimuth + math.pi) / self.res).astype(int) % self.n_bins
        np.add.at(hist, idx, weight)
        return hist

    def _apply_hysteresis(self, histogram: np.ndarray):
        still = self._blocked & (histogram >= self.threshold_low)
        newly = ~self._blocked & (histogram >= self.threshold_high)
        self._blocked = still | newly

    def _enlarge_obstacles(self):
        """Spread blocked bins to neighbors (obstacle enlargement).

        This accounts for the drone's physical radius — a direction that
        just barely misses a pillar in the histogram would still clip it
        with the drone body.
        """
        if self.enlarge_bins <= 0:
            return
        enlarged = self._blocked.copy()
        for offset in range(1, self.enlarge_bins + 1):
            enlarged |= np.roll(self._blocked, offset)
            enlarged |= np.roll(self._blocked, -offset)
        self._blocked = enlarged

    def _select_direction(self, goal_body: tuple) -> float | None:
        gx, gy = float(goal_body[0]), float(goal_body[1])
        goal_az = math.atan2(gy, gx)

        free_mask = ~self._blocked
        if not np.any(free_mask):
            self._last_chosen = None
            return None

        free_indices = np.where(free_mask)[0]
        free_centres = self._bin_centres[free_indices]

        # Angular difference to goal (wrapped to [-pi, pi])
        diff = free_centres - goal_az
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        goal_cost = np.abs(diff)

        # Clearance penalty: prefer directions far from blocked bins.
        # For each free bin, count how many bins away the nearest blocked
        # bin is.  Bins adjacent to obstacles get a high penalty.
        clearance = np.full(len(free_indices), self.n_bins // 2, dtype=float)
        for k, fi in enumerate(free_indices):
            for offset in range(1, self.n_bins // 2 + 1):
                if self._blocked[(fi + offset) % self.n_bins] or \
                   self._blocked[(fi - offset) % self.n_bins]:
                    clearance[k] = offset
                    break
        # Normalize: 0 = right next to obstacle, 1 = maximally clear
        clearance_norm = clearance / (self.n_bins // 2)
        # Penalty in radians: bins with clearance=1 bin get ~40° penalty
        clearance_penalty = (1.0 - clearance_norm) * math.radians(40)

        cost = goal_cost + clearance_penalty

        best_idx = np.argmin(cost)
        chosen = float(free_centres[best_idx])
        self._last_chosen = chosen
        return chosen

    def _compute_speed(self, pts: np.ndarray) -> float:
        if len(pts) == 0:
            return self.max_speed
        horiz_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        # Only near-horizontal obstacles
        mask = np.abs(pts[:, 2]) < 0.8
        if not np.any(mask):
            return self.max_speed
        min_d = float(np.min(horiz_dist[mask]))
        if min_d < self.safe_distance:
            return self.max_speed * max(min_d / self.safe_distance, 0.2)
        return self.max_speed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vfh = VFH2D(resolution_deg=10, safe_distance=0.8, max_speed=0.5)
    # Thin pillar ahead-right
    pillar = np.array([[1.0, -0.3, 0.0], [1.0, -0.2, 0.0], [1.0, -0.1, 0.0]])
    vel = vfh.update(pillar, (1.0, 0.0))
    print(f"Pillar ahead-right → vel=({vel[0]:.2f}, {vel[1]:.2f})")
    blocked = [(math.degrees(az), b) for az, b in vfh.get_histogram() if b]
    print(f"Blocked bins: {len(blocked)}")
    for deg, _ in blocked:
        print(f"  {deg:+6.1f} deg")
