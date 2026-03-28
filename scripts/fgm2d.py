#!/usr/bin/env python3
"""
FGM2D — Follow-the-Gap Method for 2-D obstacle avoidance.

Algorithm:
  1. Build a 1-D range profile around the drone (like a lidar scan).
  2. Inflate each obstacle by the drone's physical radius so that any
     ray that would let the drone body clip an edge is marked blocked.
  3. Find contiguous free arcs ("gaps").
  4. For each gap, compute a steering point: the point in the gap
     closest to the goal direction, pulled inward from the gap edges
     by a margin so the drone doesn't shave the obstacle at the edge.
  5. Score gaps by (a) proximity to goal, (b) width, and pick best.
  6. Slow down proportionally to nearest obstacle distance.

Key advantage over VFH/APF: the drone always steers through clearance-
checked openings with an explicit physical-radius bubble, and picks the
steering point closest to the goal *within* that opening.

Exposes the same public API as VFH2DStar for drop-in replacement.
"""

import math
import numpy as np


class FGM2D:
    """
    Follow-the-Gap planner (2-D horizontal plane).

    Parameters
    ----------
    n_rays : int
        Number of rays in the virtual range scan (default 72 = 5° each).
    max_range : float
        Maximum sensing range (m).  Rays with no obstacle get this value.
    bubble_radius : float
        Safety bubble inflated around each obstacle point (m).  Should be
        >= physical drone radius + margin.
    safe_distance : float
        Distance below which speed is reduced.
    max_speed : float
        Maximum output speed (m/s).
    gap_weight_goal : float
        Weight for angular distance from gap steering point to goal.
    gap_weight_width : float
        Bonus per radian of gap width (prefer wider gaps).
    min_gap_width_deg : float
        Minimum passable gap width in degrees.
    edge_margin_deg : float
        Margin in degrees to pull the steering point away from gap edges
        so the drone doesn't shave obstacles at the edge of a gap.
    """

    def __init__(
        self,
        n_rays: int = 72,
        max_range: float = 3.0,
        bubble_radius: float = 0.55,
        safe_distance: float = 1.2,
        max_speed: float = 0.4,
        gap_weight_goal: float = 1.0,
        gap_weight_width: float = 0.3,
        min_gap_width_deg: float = 20.0,
        edge_margin_deg: float = 10.0,
    ):
        self.n_rays = n_rays
        self.max_range = max_range
        self.bubble_radius = bubble_radius
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.gap_weight_goal = gap_weight_goal
        self.gap_weight_width = gap_weight_width
        self.min_gap_rad = math.radians(min_gap_width_deg)
        self.edge_margin = math.radians(edge_margin_deg)

        # Ray angles: evenly spaced over [-pi, pi)
        self._res = 2 * math.pi / n_rays
        self._ray_angles = np.array(
            [-math.pi + (i + 0.5) * self._res for i in range(n_rays)]
        )

        self._blocked = np.zeros(n_rays, dtype=bool)
        self._last_chosen: float | None = None
        self._range_profile = np.full(n_rays, max_range)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obstacle_pts: np.ndarray, goal_body: tuple) -> tuple:
        """
        Compute a safe 2-D velocity in body FLU frame.

        Args:
            obstacle_pts: (N, 3) body-FLU points (X fwd, Y left, Z up).
            goal_body:    (gx, gy) desired direction in body FLU.

        Returns:
            (vx, vy) velocity in body FLU.
        """
        gx, gy = float(goal_body[0]), float(goal_body[1])
        goal_az = math.atan2(gy, gx)

        # 1. Build range profile
        self._range_profile[:] = self.max_range
        min_obs_dist = float('inf')

        if len(obstacle_pts) > 0:
            xy = obstacle_pts[:, :2]
            dists = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
            mask = dists > 0.05
            if np.any(mask):
                xy_m = xy[mask]
                d_m = dists[mask]
                min_obs_dist = float(np.min(d_m))
                azimuth = np.arctan2(xy_m[:, 1], xy_m[:, 0])
                ray_idx = ((azimuth + math.pi) / self._res).astype(int) % self.n_rays
                np.minimum.at(self._range_profile, ray_idx, d_m)

        # 2. Block rays: any ray whose range < max_range gets bubble-expanded
        self._blocked[:] = False
        if len(obstacle_pts) > 0:
            xy = obstacle_pts[:, :2]
            dists = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
            mask = (dists > 0.05) & (dists < self.max_range)
            if np.any(mask):
                d_m = dists[mask]
                azimuth = np.arctan2(xy[mask, 1], xy[mask, 0])

                for i in range(len(d_m)):
                    # Angular half-width the drone body subtends at this range
                    half_ang = math.asin(min(self.bubble_radius / d_m[i], 1.0))
                    az = azimuth[i]
                    ray_diff = _wrap_array(self._ray_angles - az)
                    self._blocked |= (np.abs(ray_diff) <= half_ang)

        # 3. Find gaps
        gaps = self._find_gaps()

        # 4. No passable gap → stop
        if not gaps:
            self._last_chosen = None
            return (0.0, 0.0)

        # 5. Pick best gap and compute steering angle
        best_az = self._select_gap(gaps, goal_az)
        self._last_chosen = best_az

        # 6. Speed
        speed = self._compute_speed(min_obs_dist)
        vx = speed * math.cos(best_az)
        vy = speed * math.sin(best_az)
        return (vx, vy)

    def get_histogram(self) -> list[tuple[float, bool]]:
        return list(zip(self._ray_angles.tolist(), self._blocked.tolist()))

    def get_chosen_direction(self) -> float | None:
        return self._last_chosen

    def reset(self):
        self._blocked[:] = False
        self._last_chosen = None
        self._range_profile[:] = self.max_range

    # ------------------------------------------------------------------
    # Gap finding (index-based to handle wrap correctly)
    # ------------------------------------------------------------------

    def _find_gaps(self):
        """
        Return list of (start_idx, end_idx, width_rays) for contiguous
        runs of free rays.  Indices are in [0, n_rays) and the run goes
        CCW from start to end (inclusive), wrapping if end < start.
        Only gaps whose angular width >= min_gap_rad are returned.
        """
        free = ~self._blocked
        if not np.any(free):
            return []
        if np.all(free):
            # Everything open
            return [(0, self.n_rays - 1, self.n_rays)]

        n = self.n_rays
        min_rays = max(1, int(math.ceil(self.min_gap_rad / self._res)))

        # Find transitions blocked→free (gap start) and free→blocked (gap end)
        # Work on a linear scan, then handle the one possible wrap-around gap.
        gaps = []
        in_gap = False
        start = 0

        for i in range(n):
            if free[i] and not in_gap:
                start = i
                in_gap = True
            elif not free[i] and in_gap:
                length = i - start
                if length >= min_rays:
                    gaps.append((start, i - 1, length))
                in_gap = False

        # Gap that runs off the end
        if in_gap:
            tail_start = start
            tail_end = n - 1
            # Check if it wraps into the first gap
            if len(gaps) > 0 and gaps[0][0] == 0:
                # Merge: the wrap-around gap is tail_start→(first gap end)
                first = gaps.pop(0)
                length = (n - tail_start) + (first[1] + 1)
                if length >= min_rays:
                    gaps.append((tail_start, first[1], length))
            else:
                length = n - tail_start
                if length >= min_rays:
                    gaps.append((tail_start, tail_end, length))

        return gaps

    def _select_gap(self, gaps, goal_az: float) -> float:
        """
        For each gap, compute the best steering point (the point inside
        the gap closest to goal_az, pulled inward from edges by margin).
        Return the azimuth of the best-scoring gap's steering point.
        """
        best_score = float('inf')
        best_az = goal_az

        for start_idx, end_idx, length in gaps:
            width_rad = length * self._res

            # Gap edge angles (inward by margin so we don't shave edges)
            margin = min(self.edge_margin, width_rad * 0.4)
            lo = self._ray_angles[start_idx] + margin   # pull CW edge inward
            hi = self._ray_angles[end_idx]   - margin   # pull CCW edge inward

            # Clamp goal into [lo, hi] range of the gap
            steer = self._clamp_angle_to_arc(goal_az, lo, hi)

            # Score: angular distance from goal + width bonus
            dist_to_goal = abs(_wrap_scalar(steer - goal_az))
            score = self.gap_weight_goal * dist_to_goal - self.gap_weight_width * width_rad

            if score < best_score:
                best_score = score
                best_az = steer

        return best_az

    def _clamp_angle_to_arc(self, angle, lo, hi):
        """
        Clamp `angle` to the shortest arc from `lo` to `hi` (going CCW).
        If angle is inside the arc, return it unchanged.  Otherwise
        return whichever edge is closer.
        """
        # Normalize everything relative to lo
        span = _wrap_scalar(hi - lo)
        if span < 0:
            span += 2 * math.pi  # arc goes CCW from lo to hi

        d = _wrap_scalar(angle - lo)
        if d < 0:
            d += 2 * math.pi

        if d <= span:
            return angle  # inside the arc

        # Outside — pick nearest edge
        d_lo = abs(_wrap_scalar(angle - lo))
        d_hi = abs(_wrap_scalar(angle - hi))
        return lo if d_lo <= d_hi else hi

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_speed(self, min_obs_dist: float) -> float:
        if min_obs_dist >= self.safe_distance:
            return self.max_speed
        ratio = max(min_obs_dist / self.safe_distance, 0.15)
        return self.max_speed * ratio


def _wrap_scalar(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def _wrap_array(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fgm = FGM2D(
        n_rays=72,
        max_range=3.0,
        bubble_radius=0.55,
        safe_distance=1.2,
        max_speed=0.5,
    )

    # Single pillar ahead — should steer around it
    pillar = np.array([
        [1.2, -0.1, 0.0], [1.2, 0.0, 0.0], [1.2, 0.1, 0.0],
    ])
    vel = fgm.update(pillar, (2.0, 0.0))
    chosen = fgm.get_chosen_direction()
    print(f"Pillar ahead → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°  (should be ~±30°)")

    # Pillar ahead-right — should steer left
    fgm.reset()
    pillar_r = np.array([
        [1.0, -0.3, 0.0], [1.0, -0.2, 0.0], [1.0, -0.1, 0.0],
    ])
    vel = fgm.update(pillar_r, (2.0, 0.0))
    chosen = fgm.get_chosen_direction()
    print(f"\nPillar ahead-right → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°  (should steer left, ~+20-40°)")

    # Two pillars forming a gap — should fly through centre
    fgm.reset()
    gap_pts = np.array([
        # Left pillar
        [1.5, 0.6, 0.0], [1.5, 0.7, 0.0], [1.5, 0.8, 0.0],
        # Right pillar
        [1.5, -0.6, 0.0], [1.5, -0.7, 0.0], [1.5, -0.8, 0.0],
    ])
    vel = fgm.update(gap_pts, (2.0, 0.0))
    chosen = fgm.get_chosen_direction()
    print(f"\nTwo pillars w/ gap → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°  (should be ~0° through gap)")

    # U-shaped trap
    fgm.reset()
    trap = np.array([
        [1.0, -0.3, 0.0], [1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.3, 0.0],
        [0.8, -0.5, 0.0], [0.6, -0.5, 0.0], [0.4, -0.5, 0.0],
        [0.8, 0.5, 0.0], [0.6, 0.5, 0.0], [0.4, 0.5, 0.0],
    ])
    vel = fgm.update(trap, (1.0, 0.0))
    chosen = fgm.get_chosen_direction()
    print(f"\nU-trap → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°  (should go backward ~180°)")

    # No obstacles — straight to goal
    fgm.reset()
    vel = fgm.update(np.empty((0, 3)), (2.0, 1.0))
    chosen = fgm.get_chosen_direction()
    print(f"\nNo obstacles → vel=({vel[0]:.3f}, {vel[1]:.3f})")
    if chosen is not None:
        print(f"  Chosen: {math.degrees(chosen):.1f}°  (should be ~26.6°)")
