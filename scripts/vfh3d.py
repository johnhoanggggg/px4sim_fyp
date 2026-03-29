#!/usr/bin/env python3
"""
VFH3D — 3-D Vector Field Histogram on a spherical scan.

Based on VFH+/VFH* extended to 3D:
  1. Build obstacle density histogram on spherical grid (azimuth × elevation)
  2. Threshold to create binary blocked/free histogram
  3. Enlarge blocked sectors by safety margin (robot radius + clearance)
  4. Score all candidate (free) directions with weighted cost function:
     - Goal alignment
     - Obstacle clearance (inverse min-range in neighbourhood)
     - Heading smoothness (consistency with previous direction)
     - Reverse penalty
  5. Select lowest-cost candidate

Key difference from FGM3D: no discrete gap detection / flood-fill.
Directions near obstacles are never selected because the enlarged blocked
mask and clearance cost naturally steer the drone through gap centres.

Drop-in replacement for FGM3D — identical public API.
"""

import math
import numpy as np

# -----------------------------------------------------------------------
# Sensor geometry (matches x500_tof model.sdf / tof_reader.py)
# -----------------------------------------------------------------------
_FOV_HALF = 0.3927  # ±22.5 deg per sensor axis

# 8 horizontal sensors (was 10; tof_4 and tof_6 moved to tilted)
_HORIZONTAL_YAWS = [
    0.0, 0.6283, 1.2566, 1.8850,
    3.1416, -1.8850, -1.2566, -0.6283,
]
# 4 pitched/vertical sensors: forward-up 45°, forward-down 45°, up, down
_VERTICAL_PITCHES = [-math.pi / 4, math.pi / 4, -math.pi / 2, math.pi / 2]


def _build_coverage_mask(az_centres, el_centres, az_res, el_res, n_az, n_el):
    """Compute a boolean mask of which (el, az) cells are covered by sensors."""
    coverage = np.zeros((n_el, n_az), dtype=bool)

    az_grid, el_grid = np.meshgrid(az_centres, el_centres)
    cos_el = np.cos(el_grid)
    cell_dirs = np.stack([
        cos_el * np.cos(az_grid),
        cos_el * np.sin(az_grid),
        np.sin(el_grid),
    ], axis=-1)

    def _rotz(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _roty(pitch):
        c, s = math.cos(pitch), math.sin(pitch)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _mark(rot):
        R_inv = rot.T
        local = np.einsum('ij,mnj->mni', R_inv, cell_dirs)
        lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
        ha = np.arctan2(ly, lx)
        va = np.arctan2(lz, np.sqrt(lx**2 + ly**2))
        within = (lx > 0) & (np.abs(ha) <= _FOV_HALF) & (np.abs(va) <= _FOV_HALF)
        coverage[within] = True

    for yaw in _HORIZONTAL_YAWS:
        _mark(_rotz(yaw))
    for pitch in _VERTICAL_PITCHES:
        _mark(_roty(pitch))

    return coverage


class VFH3D:
    """
    3-D Vector Field Histogram planner on a spherical scan.

    Parameters
    ----------
    n_az : int
        Azimuth bins (horizontal, default 72 = 5 deg each).
    n_el : int
        Elevation bins (vertical, default 18).
    max_range : float
        Sensing range (m).
    bubble_radius : float
        Safety inflation around obstacles (m), >= drone radius.
    safe_distance : float
        Distance below which speed is reduced.
    max_speed : float
        Maximum output speed (m/s).
    w_goal : float
        Cost weight for angular distance to goal direction.
    w_obstacle : float
        Cost weight for inverse clearance (prefers directions far from obstacles).
    w_smooth : float
        Cost weight for consistency with previous chosen direction.
    w_reverse : float
        Cost weight penalising backward motion.
    safety_margin_cells : int
        Blocked sectors are dilated by this many cells in all directions.
        Larger values force candidates further from obstacle boundaries.
    clearance_radius_cells : int
        Radius (in cells) for min-range lookup around each candidate.
    el_max_deg : float
        Maximum elevation angle (deg).
    """

    def __init__(
        self,
        n_az: int = 72,
        n_el: int = 18,
        max_range: float = 1.5,
        bubble_radius: float = 0.3,
        safe_distance: float = 0.8,
        max_speed: float = 0.6,
        w_goal: float = 1.0,
        w_obstacle: float = 1.5,
        w_smooth: float = 0.3,
        w_reverse: float = 0.8,
        safety_margin_cells: int = 2,
        clearance_radius_cells: int = 3,
        el_max_deg: float = 70.0,
    ):
        self.n_az = n_az
        self.n_el = n_el
        self.max_range = max_range
        self._bubble_radius = bubble_radius
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.w_goal = w_goal
        self.w_obstacle = w_obstacle
        self.w_smooth = w_smooth
        self.w_reverse = w_reverse
        self._safety_margin_cells = safety_margin_cells
        self._clearance_radius_cells = clearance_radius_cells
        self.el_max = math.radians(el_max_deg)

        # Azimuth bin centres [-pi, pi)
        self._az_res = 2 * math.pi / n_az
        self._az_centres = np.array(
            [-math.pi + (i + 0.5) * self._az_res for i in range(n_az)]
        )

        # Elevation bin centres [-el_max, +el_max]
        self._el_res = 2 * self.el_max / n_el
        self._el_centres = np.array(
            [-self.el_max + (i + 0.5) * self._el_res for i in range(n_el)]
        )

        # Pre-compute unit direction for each cell (n_el, n_az, 3)
        az_grid, el_grid = np.meshgrid(self._az_centres, self._el_centres)
        cos_el = np.cos(el_grid)
        self._cell_dirs = np.stack([
            cos_el * np.cos(az_grid),   # X (forward)
            cos_el * np.sin(az_grid),   # Y (left)
            np.sin(el_grid),            # Z (up)
        ], axis=-1)

        # Sensor coverage mask
        self._coverage = _build_coverage_mask(
            self._az_centres, self._el_centres,
            self._az_res, self._el_res, n_az, n_el,
        )

        # State
        self._blocked = np.zeros((n_el, n_az), dtype=bool)
        self._range_map = np.full((n_el, n_az), max_range)
        self._last_chosen_az: float | None = None
        self._last_chosen_el: float | None = None
        self._last_total_cost: np.ndarray | None = None
        self._last_candidate: np.ndarray | None = None
        self._last_goal_az: float | None = None
        self._last_goal_el: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obstacle_pts: np.ndarray, goal_body: tuple) -> tuple:
        """
        Compute safe 3-D velocity in body FLU.

        Args:
            obstacle_pts: (N, 3) in body FLU (X fwd, Y left, Z up).
            goal_body:    (gx, gy, gz) goal direction in body FLU.

        Returns:
            (vx, vy, vz) velocity in body FLU.
        """
        gx = float(goal_body[0])
        gy = float(goal_body[1])
        gz = float(goal_body[2]) if len(goal_body) > 2 else 0.0

        goal_dist = math.sqrt(gx**2 + gy**2 + gz**2)
        goal_az = math.atan2(gy, gx)
        goal_el = math.atan2(gz, math.sqrt(gx**2 + gy**2)) if goal_dist > 0.01 else 0.0
        self._last_goal_az = goal_az
        self._last_goal_el = goal_el
        goal_dir = np.array([
            math.cos(goal_el) * math.cos(goal_az),
            math.cos(goal_el) * math.sin(goal_az),
            math.sin(goal_el),
        ])

        # 1. Build spherical range map and blocked mask
        self._build_map(obstacle_pts)

        # 2. Min obstacle distance (for speed scaling)
        min_obs_dist = float('inf')
        if len(obstacle_pts) > 0:
            dists = np.sqrt(np.sum(obstacle_pts**2, axis=1))
            valid = dists > 0.05
            if np.any(valid):
                min_obs_dist = float(np.min(dists[valid]))

        # 3. Enlarge blocked sectors — VFH safety margin
        candidate = self._compute_candidates()
        self._last_candidate = candidate

        if not np.any(candidate):
            self._last_chosen_az = None
            self._last_chosen_el = None
            self._last_total_cost = None
            return self._retreat(obstacle_pts, min_obs_dist)

        # 4. Score all candidate directions
        total_cost = self._score_candidates(candidate, goal_dir)
        self._last_total_cost = total_cost

        # 5. Select lowest-cost candidate
        best_idx = np.unravel_index(np.argmin(total_cost), total_cost.shape)
        best_ei, best_ai = best_idx
        best_az = float(self._az_centres[best_ai])
        best_el = float(self._el_centres[best_ei])

        self._last_chosen_az = best_az
        self._last_chosen_el = best_el

        # 6. Compute speed and output velocity
        speed = self._compute_speed(min_obs_dist)
        cos_el = math.cos(best_el)
        vx = speed * cos_el * math.cos(best_az)
        vy = speed * cos_el * math.sin(best_az)
        vz = speed * math.sin(best_el)

        return (vx, vy, vz)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Return horizontal-slice histogram for viz2d compatibility."""
        mid = self.n_el // 2
        band = max(1, self.n_el // 6)
        lo, hi = max(0, mid - band), min(self.n_el, mid + band + 1)
        horiz_blocked = np.any(self._blocked[lo:hi, :], axis=0)
        return list(zip(self._az_centres.tolist(), horiz_blocked.tolist()))

    def get_chosen_direction(self) -> float | None:
        """Return chosen azimuth for viz2d compatibility."""
        return self._last_chosen_az

    def get_sphere_data(self) -> dict:
        """Return spherical grid data for visualization."""
        return {
            "blocked": self._blocked.tolist(),
            "coverage": self._coverage.tolist(),
            "range_map": self._range_map.tolist(),
            "max_range": self.max_range,
            "az_centres": self._az_centres.tolist(),
            "el_centres": self._el_centres.tolist(),
            "chosen_az": self._last_chosen_az,
            "chosen_el": self._last_chosen_el,
            "total_cost": self._last_total_cost.tolist() if self._last_total_cost is not None else None,
            "candidate_mask": self._last_candidate.tolist() if self._last_candidate is not None else None,
            "goal_az": self._last_goal_az,
            "goal_el": self._last_goal_el,
        }

    def reset(self):
        self._blocked = ~self._coverage.copy()
        self._range_map[:] = self.max_range
        self._last_chosen_az = None
        self._last_chosen_el = None
        self._last_total_cost = None
        self._last_candidate = None
        self._last_goal_az = None
        self._last_goal_el = None

    @property
    def bubble_radius(self):
        return self._bubble_radius

    # ------------------------------------------------------------------
    # Spherical map building
    # ------------------------------------------------------------------

    def _build_map(self, pts: np.ndarray):
        """Project obstacle points onto the spherical grid and inflate."""
        self._blocked = ~self._coverage.copy()
        self._range_map[:] = self.max_range

        if len(pts) == 0:
            return

        dists = np.sqrt(np.sum(pts**2, axis=1))
        valid = (dists > 0.05) & (dists < self.max_range)
        if not np.any(valid):
            return

        pts_v = pts[valid]
        d_v = dists[valid]

        az = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        el = np.arctan2(pts_v[:, 2], np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2))

        az_idx = ((az + math.pi) / self._az_res).astype(int) % self.n_az
        el_idx = ((el + self.el_max) / self._el_res).astype(int)
        el_idx = np.clip(el_idx, 0, self.n_el - 1)

        np.minimum.at(self._range_map, (el_idx, az_idx), d_v)

        cell_min_dist = {}
        cell_min_dir = {}
        for i in range(len(d_v)):
            key = (int(el_idx[i]), int(az_idx[i]))
            if key not in cell_min_dist or d_v[i] < cell_min_dist[key]:
                cell_min_dist[key] = float(d_v[i])
                cell_min_dir[key] = pts_v[i] / d_v[i]

        for key, dist in cell_min_dist.items():
            half_ang = math.asin(min(self._bubble_radius / dist, 1.0))
            obs_dir = cell_min_dir[key]
            dots = np.sum(self._cell_dirs * obs_dir, axis=-1)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dist = np.arccos(dots)
            self._blocked |= (ang_dist <= half_ang)

    # ------------------------------------------------------------------
    # VFH-specific: candidate selection and cost scoring
    # ------------------------------------------------------------------

    def _compute_candidates(self) -> np.ndarray:
        """Dilate blocked mask by safety_margin_cells, return ~dilated.

        This is the VFH "enlarged" histogram step — ensures candidate
        directions are at least safety_margin_cells away from any blocked
        sector, keeping the drone well clear of obstacle boundaries.
        """
        dilated = self._blocked.copy()
        m = self._safety_margin_cells
        for de in range(-m, m + 1):
            for da in range(-m, m + 1):
                if de == 0 and da == 0:
                    continue
                shifted = np.roll(self._blocked, -da, axis=1)  # azimuth wraps
                if de > 0:
                    shifted = np.pad(shifted[de:], ((0, de), (0, 0)),
                                     constant_values=True)
                elif de < 0:
                    shifted = np.pad(shifted[:de], ((-de, 0), (0, 0)),
                                     constant_values=True)
                dilated |= shifted
        return ~dilated

    def _min_range_nearby(self) -> np.ndarray:
        """Compute minimum range in a neighbourhood around each cell.

        Used for clearance cost — directions where nearby cells have low
        range values get penalised even if the cell itself is free.
        """
        result = self._range_map.copy()
        r = self._clearance_radius_cells
        for de in range(-r, r + 1):
            for da in range(-r, r + 1):
                if de == 0 and da == 0:
                    continue
                shifted = np.roll(self._range_map, -da, axis=1)
                if de > 0:
                    shifted = np.pad(shifted[de:], ((0, de), (0, 0)),
                                     constant_values=self.max_range)
                elif de < 0:
                    shifted = np.pad(shifted[:de], ((-de, 0), (0, 0)),
                                     constant_values=self.max_range)
                np.minimum(result, shifted, out=result)
        return result

    def _score_candidates(self, candidate: np.ndarray,
                          goal_dir: np.ndarray) -> np.ndarray:
        """Score all cells. Non-candidate cells get inf cost.

        Cost = w_goal × goal_angle
             + w_obstacle × (max_range / nearby_clearance)
             + w_smooth × heading_change
             + w_reverse × backward_penalty
        """

        # --- cost_goal: angular distance to goal ---
        dots_goal = np.sum(self._cell_dirs * goal_dir, axis=-1)
        np.clip(dots_goal, -1.0, 1.0, out=dots_goal)
        cost_goal = np.arccos(dots_goal)

        # --- cost_obstacle: inverse clearance in neighbourhood ---
        min_nearby = self._min_range_nearby()
        np.clip(min_nearby, 0.1, None, out=min_nearby)
        cost_obstacle = self.max_range / min_nearby

        # --- cost_smooth: angular distance from previous direction ---
        if self._last_chosen_az is not None:
            prev_el = self._last_chosen_el if self._last_chosen_el is not None else 0.0
            prev_dir = np.array([
                math.cos(prev_el) * math.cos(self._last_chosen_az),
                math.cos(prev_el) * math.sin(self._last_chosen_az),
                math.sin(prev_el),
            ])
            dots_prev = np.sum(self._cell_dirs * prev_dir, axis=-1)
            np.clip(dots_prev, -1.0, 1.0, out=dots_prev)
            cost_smooth = np.arccos(dots_prev)
        else:
            cost_smooth = np.zeros((self.n_el, self.n_az))

        # --- cost_reverse: penalty for backward motion (X < 0 in body FLU) ---
        forward_component = self._cell_dirs[..., 0]
        cost_reverse = np.maximum(0.0, -forward_component)

        # --- total ---
        total = (self.w_goal * cost_goal
                 + self.w_obstacle * cost_obstacle
                 + self.w_smooth * cost_smooth
                 + self.w_reverse * cost_reverse)

        total[~candidate] = np.inf
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retreat(self, pts: np.ndarray, min_obs_dist: float) -> tuple:
        """When fully blocked, back away from the centroid of nearby obstacles."""
        speed = self._compute_speed(min_obs_dist) * 0.5
        if len(pts) == 0:
            return (0.0, 0.0, 0.0)

        dists = np.sqrt(np.sum(pts**2, axis=1))
        close = pts[dists < self.safe_distance * 1.5]
        if len(close) == 0:
            close = pts

        centroid = np.mean(close, axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 0.01:
            return (-speed, 0.0, 0.0)
        retreat_dir = -centroid / norm
        return (float(speed * retreat_dir[0]),
                float(speed * retreat_dir[1]),
                float(speed * retreat_dir[2]))

    def _compute_speed(self, min_obs_dist: float) -> float:
        if min_obs_dist >= self.safe_distance:
            return self.max_speed
        ratio = max(min_obs_dist / self.safe_distance, 0.15)
        return self.max_speed * ratio
