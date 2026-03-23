#!/usr/bin/env python3
"""
APF2D — 2-D Artificial Potential Field obstacle avoidance.

Computes a velocity command by summing:
  - An attractive force toward the goal.
  - Repulsive forces away from each nearby obstacle point (inverse-square).

No angular binning — the force field is continuous, which eliminates the
"threading between bins" problem that causes VFH to clip pillars.

The class exposes the same public API as VFH2DStar so it can be swapped
in without changing the flight script or visualizer.
"""

import math
import numpy as np


class APF2D:
    """
    Artificial Potential Field planner (2-D horizontal plane).

    Parameters
    ----------
    repulsive_gain : float
        Strength of the repulsive field.  Higher = wider berth.
    attractive_gain : float
        Strength of the attractive (goal) field.
    influence_distance : float
        Obstacles beyond this distance (m) produce zero repulsion.
    safe_distance : float
        Distance at which speed starts to reduce.
    max_speed : float
        Maximum output speed (m/s).
    drone_radius : float
        Effective radius (m) subtracted from obstacle distance so repulsion
        kicks in earlier.
    max_repulsive_force : float
        Clamp per-obstacle repulsive force magnitude to avoid singularities.
    """

    def __init__(
        self,
        repulsive_gain: float = 0.6,
        attractive_gain: float = 1.0,
        influence_distance: float = 2.5,
        safe_distance: float = 1.2,
        max_speed: float = 0.4,
        drone_radius: float = 0.35,
        max_repulsive_force: float = 3.0,
    ):
        self.rep_gain = repulsive_gain
        self.att_gain = attractive_gain
        self.influence_dist = influence_distance
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.drone_radius = drone_radius
        self.max_rep_force = max_repulsive_force

        # For visualizer compatibility — build a pseudo-histogram each cycle
        self._n_bins = 36
        self._res = 2 * math.pi / self._n_bins
        self._bin_centres = np.array(
            [-math.pi + (i + 0.5) * self._res for i in range(self._n_bins)]
        )
        self._blocked = np.zeros(self._n_bins, dtype=bool)
        self._last_chosen: float | None = None
        self._obs_xy = np.empty((0, 2))

    # ------------------------------------------------------------------
    # Public API (matches VFH2DStar interface)
    # ------------------------------------------------------------------

    def update(self, obstacle_pts: np.ndarray, goal_body: tuple) -> tuple:
        """
        Compute a safe 2-D velocity in body FLU frame.

        Args:
            obstacle_pts: (N, 3) body-FLU points (X fwd, Y left, Z up).
            goal_body:    (gx, gy) desired direction in body FLU.

        Returns:
            (vx, vy) velocity in body FLU (horizontal plane).
        """
        gx, gy = float(goal_body[0]), float(goal_body[1])

        # --- Attractive force (capped so it doesn't dominate far away) ---
        goal_dist = math.sqrt(gx * gx + gy * gy)
        if goal_dist > 1e-3:
            att_x = self.att_gain * gx / goal_dist
            att_y = self.att_gain * gy / goal_dist
        else:
            att_x = att_y = 0.0

        # --- Repulsive forces ---
        rep_x = rep_y = 0.0
        min_obs_dist = float('inf')

        if len(obstacle_pts) > 0:
            xy = obstacle_pts[:, :2]
            self._obs_xy = xy.copy()
            dists = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)

            # Only consider obstacles within influence distance
            mask = (dists > 0.05) & (dists < self.influence_dist)
            if np.any(mask):
                xy_m = xy[mask]
                d_m = dists[mask]
                min_obs_dist = float(np.min(d_m))

                # Effective distance (shrink by drone radius)
                d_eff = np.maximum(d_m - self.drone_radius, 0.05)

                # Repulsive magnitude: gain * (1/d_eff - 1/d_inf) / d_eff^2
                d_inf = self.influence_dist - self.drone_radius
                mag = self.rep_gain * (1.0 / d_eff - 1.0 / d_inf) / (d_eff ** 2)
                mag = np.minimum(mag, self.max_rep_force)

                # Direction: away from obstacle (negate unit vector toward obs)
                ux = -xy_m[:, 0] / d_m
                uy = -xy_m[:, 1] / d_m

                rep_x = float(np.sum(mag * ux))
                rep_y = float(np.sum(mag * uy))
        else:
            self._obs_xy = np.empty((0, 2))

        # --- Sum forces ---
        fx = att_x + rep_x
        fy = att_y + rep_y
        f_mag = math.sqrt(fx * fx + fy * fy)

        if f_mag < 1e-6:
            self._last_chosen = None
            self._build_pseudo_histogram()
            return (0.0, 0.0)

        # Normalize to unit direction, then scale by speed
        direction = math.atan2(fy, fx)
        self._last_chosen = direction

        speed = self._compute_speed(min_obs_dist)
        vx = speed * fx / f_mag
        vy = speed * fy / f_mag

        # Build pseudo-histogram for visualizer
        self._build_pseudo_histogram()

        return (vx, vy)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Return list of (azimuth_rad, is_blocked) for visualizer."""
        return list(zip(self._bin_centres.tolist(), self._blocked.tolist()))

    def get_chosen_direction(self) -> float | None:
        return self._last_chosen

    def reset(self):
        self._blocked[:] = False
        self._last_chosen = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_speed(self, min_obs_dist: float) -> float:
        """Reduce speed near obstacles."""
        if min_obs_dist >= self.safe_distance:
            return self.max_speed
        ratio = max(min_obs_dist / self.safe_distance, 0.2)
        return self.max_speed * ratio

    def _build_pseudo_histogram(self):
        """Build a VFH-style blocked histogram for the visualizer."""
        self._blocked[:] = False
        if len(self._obs_xy) == 0:
            return

        dists = np.sqrt(self._obs_xy[:, 0] ** 2 + self._obs_xy[:, 1] ** 2)
        mask = (dists > 0.05) & (dists < self.influence_dist)
        if not np.any(mask):
            return

        xy_m = self._obs_xy[mask]
        d_m = dists[mask]
        azimuth = np.arctan2(xy_m[:, 1], xy_m[:, 0])
        idx = ((azimuth + math.pi) / self._res).astype(int) % self._n_bins

        # Accumulate weight (closer = heavier)
        weight = np.clip(1.0 - d_m / self.influence_dist, 0, 1)
        hist = np.zeros(self._n_bins, dtype=np.float32)
        np.add.at(hist, idx, weight)
        self._blocked = hist > 0.2


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    apf = APF2D(
        repulsive_gain=0.6,
        attractive_gain=1.0,
        influence_distance=2.5,
        safe_distance=1.2,
        max_speed=0.5,
        drone_radius=0.35,
    )

    # Pillar straight ahead
    pillar = np.array([
        [1.0, -0.1, 0.0], [1.0, 0.0, 0.0], [1.0, 0.1, 0.0],
    ])
    vel = apf.update(pillar, (2.0, 0.0))
    print(f"Pillar ahead → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    print(f"  Chosen: {math.degrees(apf.get_chosen_direction()):.1f}°")

    # Pillar ahead-right, goal ahead
    apf.reset()
    pillar_r = np.array([
        [1.0, -0.3, 0.0], [1.0, -0.2, 0.0], [1.0, -0.1, 0.0],
    ])
    vel = apf.update(pillar_r, (2.0, 0.0))
    print(f"\nPillar ahead-right → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    print(f"  Chosen: {math.degrees(apf.get_chosen_direction()):.1f}°")
    print(f"  (Should steer left to avoid)")

    # U-shaped trap
    apf.reset()
    trap = np.array([
        [1.0, -0.3, 0.0], [1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.3, 0.0],
        [0.8, -0.5, 0.0], [0.6, -0.5, 0.0], [0.4, -0.5, 0.0],
        [0.8, 0.5, 0.0], [0.6, 0.5, 0.0], [0.4, 0.5, 0.0],
    ])
    vel = apf.update(trap, (1.0, 0.0))
    print(f"\nU-trap, goal ahead → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    chosen = apf.get_chosen_direction()
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°")
        print(f"  (Should back away from trap)")
