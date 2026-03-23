#!/usr/bin/env python3
"""
VFH*2D — 2D Vector Field Histogram Star with multi-step lookahead.

Extends VFH2D by projecting the drone forward along each candidate free
direction, rebuilding the histogram at the projected position, and
recursively searching a tree of possible paths.  The first direction of the
lowest-cost branch is chosen, which avoids dead-ends that greedy VFH walks
into.

The obstacle point cloud is kept in a fixed reference frame so that
projected positions can query it without re-sensing.
"""

import math
import numpy as np


class VFH2DStar:
    """
    VFH* with 2D horizontal-plane histograms and tree lookahead.

    Parameters
    ----------
    resolution_deg : float
        Bin width in degrees (default 10 → 36 bins).
    threshold_low / threshold_high : float
        Hysteresis thresholds for blocking bins.
    safe_distance : float
        Distance (m) at which obstacles start affecting speed.
    max_speed : float
        Maximum horizontal speed (m/s).
    drone_radius : float
        Physical radius for obstacle enlargement.
    enlarge_bins : int
        Spread blocked bins ± this many neighbours.
    lookahead_depth : int
        How many steps to project forward (1 = plain VFH, 2-3 recommended).
    step_distance : float
        How far (m) to project forward per lookahead level.
    goal_weight : float
        Cost weight for angular deviation from goal.
    smooth_weight : float
        Cost weight for direction change between consecutive levels.
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
        lookahead_depth: int = 3,
        step_distance: float = 0.8,
        goal_weight: float = 1.0,
        smooth_weight: float = 0.3,
    ):
        self.res = math.radians(resolution_deg)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.drone_radius = drone_radius
        self.enlarge_bins = enlarge_bins
        self.lookahead_depth = lookahead_depth
        self.step_distance = step_distance
        self.goal_weight = goal_weight
        self.smooth_weight = smooth_weight

        self.n_bins = int(round(2 * math.pi / self.res))
        self._blocked = np.zeros(self.n_bins, dtype=bool)
        self._last_chosen: float | None = None

        # Bin centre angles [-pi, pi)
        self._bin_centres = np.array(
            [-math.pi + (i + 0.5) * self.res for i in range(self.n_bins)]
        )

        # Only expand a limited number of children per level to keep it fast.
        # We pick the best `_max_children` free bins by goal cost at each node.
        self._max_children = 8

    # ------------------------------------------------------------------
    # Public API  (same interface as VFH2D for drop-in replacement)
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
        # Store obstacle cloud for lookahead queries (in body frame of
        # current position — we treat it as a local 2D map).
        self._obs_xy = obstacle_pts[:, :2].copy() if len(obstacle_pts) > 0 else np.empty((0, 2))

        gx, gy = float(goal_body[0]), float(goal_body[1])
        goal_az = math.atan2(gy, gx)

        # --- Level 0: build histogram at current position ----------------
        blocked_0 = self._histogram_at(np.zeros(2))
        self._apply_hysteresis(blocked_0)
        self._blocked = blocked_0.copy()  # store for get_histogram()

        free_mask = ~blocked_0
        if not np.any(free_mask):
            self._last_chosen = None
            return (0.0, 0.0)

        # --- Tree search -------------------------------------------------
        best_az = self._tree_search(
            pos=np.zeros(2),
            blocked=blocked_0,
            goal_az=goal_az,
            prev_az=None,
            depth=0,
        )

        if best_az is None:
            self._last_chosen = None
            return (0.0, 0.0)

        self._last_chosen = best_az
        speed = self._compute_speed(obstacle_pts)
        vx = speed * math.cos(best_az)
        vy = speed * math.sin(best_az)
        return (vx, vy)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Return list of (azimuth_rad, is_blocked) for every bin."""
        return list(zip(self._bin_centres.tolist(), self._blocked.tolist()))

    def get_chosen_direction(self) -> float | None:
        return self._last_chosen

    def reset(self):
        self._blocked[:] = False
        self._last_chosen = None

    # ------------------------------------------------------------------
    # Tree search
    # ------------------------------------------------------------------

    def _tree_search(self, pos, blocked, goal_az, prev_az, depth):
        """
        Recursively search the lookahead tree and return the best
        first-level azimuth.

        Returns the best azimuth at the *current* level that leads to
        the lowest total branch cost, or None if all blocked.
        """
        free_indices = np.where(~blocked)[0]
        if len(free_indices) == 0:
            return None

        free_centres = self._bin_centres[free_indices]

        # Score candidates by goal alignment to prune search
        diff = self._wrap(free_centres - goal_az)
        goal_cost = np.abs(diff)

        # Clearance penalty (same as VFH2D)
        clearance = np.full(len(free_indices), self.n_bins // 2, dtype=float)
        for k, fi in enumerate(free_indices):
            for offset in range(1, self.n_bins // 2 + 1):
                if blocked[(fi + offset) % self.n_bins] or \
                   blocked[(fi - offset) % self.n_bins]:
                    clearance[k] = offset
                    break
        clearance_norm = clearance / (self.n_bins // 2)
        clearance_penalty = (1.0 - clearance_norm) * math.radians(40)

        level_cost = self.goal_weight * goal_cost + clearance_penalty

        # Smoothness: penalize direction change from previous level
        if prev_az is not None:
            smooth_cost = np.abs(self._wrap(free_centres - prev_az))
            level_cost += self.smooth_weight * smooth_cost

        # If at max depth or depth == lookahead_depth-1, return best here
        if depth >= self.lookahead_depth - 1:
            best_idx = np.argmin(level_cost)
            return float(free_centres[best_idx])

        # Otherwise, expand top candidates deeper
        n_expand = min(self._max_children, len(free_indices))
        top_indices = np.argpartition(level_cost, n_expand)[:n_expand]

        best_total = float('inf')
        best_az_first = None

        for ci in top_indices:
            az = float(free_centres[ci])
            cost_here = float(level_cost[ci])

            # Project position forward
            next_pos = pos + self.step_distance * np.array([math.cos(az), math.sin(az)])

            # Build histogram at projected position
            next_blocked = self._histogram_at(next_pos)

            # Recurse
            child_az = self._tree_search(
                pos=next_pos,
                blocked=next_blocked,
                goal_az=goal_az,
                prev_az=az,
                depth=depth + 1,
            )

            if child_az is None:
                # Dead end — penalize heavily
                child_cost = math.pi
            else:
                # Evaluate child's branch cost from the child's perspective
                child_cost = abs(self._wrap_scalar(child_az - goal_az))

            total = cost_here + child_cost
            if total < best_total:
                best_total = total
                best_az_first = az

        return best_az_first

    # ------------------------------------------------------------------
    # Histogram construction at an arbitrary 2D position
    # ------------------------------------------------------------------

    def _histogram_at(self, pos: np.ndarray) -> np.ndarray:
        """
        Build a blocked-bin mask as if the drone were at `pos` (2D offset
        from the current body origin in the local XY plane).
        """
        hist = np.zeros(self.n_bins, dtype=np.float32)
        blocked = np.zeros(self.n_bins, dtype=bool)

        if len(self._obs_xy) == 0:
            return blocked

        # Shift obstacles relative to projected position
        rel = self._obs_xy - pos  # (N, 2)
        dist = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
        mask = dist > 0.05
        if not np.any(mask):
            return blocked

        rel, dist = rel[mask], dist[mask]
        azimuth = np.arctan2(rel[:, 1], rel[:, 0])
        weight = np.clip(1.0 - dist / (self.safe_distance * 2.5), 0, 1)

        idx = ((azimuth + math.pi) / self.res).astype(int) % self.n_bins
        np.add.at(hist, idx, weight)

        # Apply threshold (no hysteresis for projected positions — we
        # don't have temporal state there)
        blocked = hist >= self.threshold_high

        # Obstacle enlargement
        if self.enlarge_bins > 0:
            enlarged = blocked.copy()
            for offset in range(1, self.enlarge_bins + 1):
                enlarged |= np.roll(blocked, offset)
                enlarged |= np.roll(blocked, -offset)
            blocked = enlarged

        return blocked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_hysteresis(self, histogram_blocked: np.ndarray):
        """In-place hysteresis on the level-0 histogram using stored state."""
        # Re-run the full histogram to get continuous values for hysteresis
        hist = np.zeros(self.n_bins, dtype=np.float32)
        if len(self._obs_xy) > 0:
            dist = np.sqrt(self._obs_xy[:, 0]**2 + self._obs_xy[:, 1]**2)
            mask = dist > 0.05
            if np.any(mask):
                rel = self._obs_xy[mask]
                d = dist[mask]
                azimuth = np.arctan2(rel[:, 1], rel[:, 0])
                weight = np.clip(1.0 - d / (self.safe_distance * 2.5), 0, 1)
                idx = ((azimuth + math.pi) / self.res).astype(int) % self.n_bins
                np.add.at(hist, idx, weight)

        still = self._blocked & (hist >= self.threshold_low)
        newly = ~self._blocked & (hist >= self.threshold_high)
        result = still | newly

        # Enlarge
        if self.enlarge_bins > 0:
            enlarged = result.copy()
            for offset in range(1, self.enlarge_bins + 1):
                enlarged |= np.roll(result, offset)
                enlarged |= np.roll(result, -offset)
            result = enlarged

        histogram_blocked[:] = result

    def _compute_speed(self, pts: np.ndarray) -> float:
        if len(pts) == 0:
            return self.max_speed
        horiz_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        mask = np.abs(pts[:, 2]) < 0.8
        if not np.any(mask):
            return self.max_speed
        min_d = float(np.min(horiz_dist[mask]))
        if min_d < self.safe_distance:
            return self.max_speed * max(min_d / self.safe_distance, 0.2)
        return self.max_speed

    @staticmethod
    def _wrap(angles):
        """Wrap array of angles to [-pi, pi)."""
        return (angles + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _wrap_scalar(a):
        return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vfh = VFH2DStar(
        resolution_deg=10, safe_distance=0.8, max_speed=0.5,
        lookahead_depth=3, step_distance=0.8,
    )

    # U-shaped trap: pillars ahead and to the sides, open behind-left
    trap = np.array([
        # Wall ahead
        [1.0, -0.3, 0.0], [1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.3, 0.0],
        # Wall right
        [0.8, -0.5, 0.0], [0.6, -0.5, 0.0], [0.4, -0.5, 0.0],
        # Wall left
        [0.8, 0.5, 0.0], [0.6, 0.5, 0.0], [0.4, 0.5, 0.0],
    ])

    # Goal is straight ahead (through the wall)
    vel = vfh.update(trap, (1.0, 0.0))
    print(f"U-trap, goal ahead → vel=({vel[0]:.2f}, {vel[1]:.2f})")
    chosen = vfh.get_chosen_direction()
    if chosen is not None:
        print(f"  Chosen direction: {math.degrees(chosen):.1f}°")
        print(f"  (Should route around, not into the wall)")

    # Simple pillar test
    vfh.reset()
    pillar = np.array([[1.0, -0.3, 0.0], [1.0, -0.2, 0.0], [1.0, -0.1, 0.0]])
    vel = vfh.update(pillar, (1.0, 0.0))
    print(f"\nPillar ahead-right → vel=({vel[0]:.2f}, {vel[1]:.2f})")
    blocked = [(math.degrees(az), b) for az, b in vfh.get_histogram() if b]
    print(f"  Blocked bins: {len(blocked)}")
