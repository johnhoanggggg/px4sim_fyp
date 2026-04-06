#!/usr/bin/env python3
"""
VFH3D — 3-D Vector Field Histogram on a spherical scan.

Drop-in replacement for FGM3D.  Uses the classic VFH+ approach extended
to 3-D:

  1. Build a **polar obstacle density** histogram on a (azimuth × elevation)
     spherical grid.  Each obstacle point votes into nearby cells weighted
     by inverse-square distance (closer obstacles → higher density).

  2. **Threshold** the density map into blocked / free using adaptive
     hysteresis (low/high thresholds to prevent chattering).

  3. Identify **candidate valleys** — contiguous free regions on the grid
     via connected-component flood fill (same topology as FGM3D gaps).

  4. **Cost-function selection** — score each valley by:
       • angular distance from the valley's best steering cell to goal
       • angular distance from current heading (smooth transitions)
       • valley width bonus (prefer wide openings)
     Pick the lowest-cost valley; steer toward its best cell.

  5. Speed is modulated by proximity to the nearest obstacle.

The API is identical to FGM3D so the flight script needs only to swap
the import.

Sphere discretisation matches FGM3D:
    - n_az   azimuth bins   covering [-π, π)
    - n_el   elevation bins covering [-el_max, el_max]

Sensor coverage mask is reused from fgm3d module.
"""

import math
import numpy as np

# Reuse the sensor coverage builder from the existing codebase
from fgm3d import _build_coverage_mask

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _wrap(a: float) -> float:
    """Wrap angle to [-π, π)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _angular_dist_sphere(az1, el1, az2, el2):
    """Great-circle angular distance between two spherical directions."""
    d1 = np.array([math.cos(el1)*math.cos(az1),
                    math.cos(el1)*math.sin(az1),
                    math.sin(el1)])
    d2 = np.array([math.cos(el2)*math.cos(az2),
                    math.cos(el2)*math.sin(az2),
                    math.sin(el2)])
    dot = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    return math.acos(dot)


# -----------------------------------------------------------------------
# VFH3D
# -----------------------------------------------------------------------

class VFH3D:
    """
    3-D Vector Field Histogram obstacle avoidance planner.

    Drop-in replacement for FGM3D — identical constructor signature,
    public methods, and return types.

    Parameters
    ----------
    n_az : int
        Azimuth bins (horizontal, default 72 = 5° each).
    n_el : int
        Elevation bins (vertical, default 18).
    max_range : float
        Sensing range (m).
    bubble_radius : float
        Safety inflation around obstacles (m).
    safe_distance : float
        Distance below which speed is reduced.
    max_speed : float
        Maximum output speed (m/s).
    gap_weight_goal : float
        Cost weight for angular distance to goal.
    gap_weight_width : float
        Bonus weight for wider valleys.
    min_gap_cells : int
        Minimum contiguous free cells to count as a valley.
    min_gap_metres : float
        Minimum physical valley width (m).
    edge_margin_deg : float
        Pull steering inward from valley boundary (deg).
    el_max_deg : float
        Maximum elevation angle (deg).
    heading_smooth : float
        EMA alpha for heading smoothing (0=full smooth, 1=instant).

    VFH-specific parameters
    -----------------------
    density_a : float
        Obstacle density weight  a − b·d  coefficient (constant part).
        Higher → obstacles produce more density.
    density_b : float
        Obstacle density weight  a − b·d  coefficient (distance part).
        Higher → density falls off faster with range.
    threshold_high : float
        Density above which a cell becomes blocked.
    threshold_low : float
        Density below which a blocked cell becomes free (hysteresis).
    cost_smooth : float
        Cost weight for angular distance from previous heading.
    """

    def __init__(
        self,
        n_az: int = 72,
        n_el: int = 18,
        max_range: float = 1.5,
        bubble_radius: float = 0.3,
        safe_distance: float = 0.8,
        max_speed: float = 0.6,
        gap_weight_goal: float = 2.0,
        gap_weight_width: float = 0.3,
        min_gap_cells: int = 2,
        min_gap_metres: float = 0.3,
        edge_margin_deg: float = 8.0,
        el_max_deg: float = 70.0,
        heading_smooth: float = 0.4,
        # VFH-specific
        density_a: float = 5.0,
        density_b: float = 2.5,
        threshold_high: float = 3.0,
        threshold_low: float = 1.5,
        cost_smooth: float = 1.0,
    ):
        self.n_az = n_az
        self.n_el = n_el
        self.max_range = max_range
        self._bubble_radius = bubble_radius
        self.safe_distance = safe_distance
        self.max_speed = max_speed
        self.gap_weight_goal = gap_weight_goal
        self.gap_weight_width = gap_weight_width
        self.min_gap_cells = min_gap_cells
        self.min_gap_metres = min_gap_metres
        self.edge_margin = math.radians(edge_margin_deg)
        self.el_max = math.radians(el_max_deg)
        self._heading_smooth = heading_smooth

        # VFH density params
        self._density_a = density_a
        self._density_b = density_b
        self._thresh_high = threshold_high
        self._thresh_low = threshold_low
        self._cost_smooth = cost_smooth

        # Azimuth bin centres [-π, π)
        self._az_res = 2 * math.pi / n_az
        self._az_centres = np.array(
            [-math.pi + (i + 0.5) * self._az_res for i in range(n_az)]
        )

        # Elevation bin centres [-el_max, +el_max]
        self._el_res = 2 * self.el_max / n_el
        self._el_centres = np.array(
            [-self.el_max + (i + 0.5) * self._el_res for i in range(n_el)]
        )

        # Unit direction for each cell (n_el, n_az, 3)
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

        # ---- State ----
        self._density = np.zeros((n_el, n_az), dtype=float)
        self._blocked = np.zeros((n_el, n_az), dtype=bool)
        self._range_map = np.full((n_el, n_az), max_range)
        self._last_chosen_az: float | None = None
        self._last_chosen_el: float | None = None
        self._last_valleys: list = []

        # Stuck detection (matches FGM3D behaviour)
        self._stuck_counter = 0
        self._prev_goal_dist = float('inf')

    # ------------------------------------------------------------------
    # Public API (identical to FGM3D)
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
        goal_el = (math.atan2(gz, math.sqrt(gx**2 + gy**2))
                   if goal_dist > 0.01 else 0.0)

        # 1. Build density histogram and threshold into blocked map
        self._build_density_map(obstacle_pts)
        self._apply_threshold()

        # 2. Min obstacle distances
        min_obs_dist = float('inf')
        min_fwd_obs_dist = float('inf')
        if len(obstacle_pts) > 0:
            dists = np.sqrt(np.sum(obstacle_pts**2, axis=1))
            valid = dists > 0.0001
            if np.any(valid):
                min_obs_dist = float(np.min(dists[valid]))
                fwd = valid & (obstacle_pts[:, 0] > 0)
                if np.any(fwd):
                    min_fwd_obs_dist = float(np.min(dists[fwd]))

        # 3. Stuck detection
        if goal_dist < self._prev_goal_dist - 0.05:
            self._stuck_counter = 0
        else:
            self._stuck_counter += 1
        self._prev_goal_dist = goal_dist

        if self._stuck_counter > 20 or min_fwd_obs_dist < self._bubble_radius:
            self._last_chosen_az = None
            self._last_chosen_el = None
            if self._stuck_counter > 20:
                self._stuck_counter = 0
            return self._retreat(obstacle_pts, min_obs_dist)

        # 4. Find candidate valleys (connected free regions)
        valleys = self._find_valleys()
        self._last_valleys = valleys

        # 5. No valley → retreat
        if not valleys:
            self._last_chosen_az = None
            self._last_chosen_el = None
            return self._retreat(obstacle_pts, min_obs_dist)

        # 6. Cost-function valley selection
        best_az, best_el = self._select_valley(valleys, goal_az, goal_el)

        # 7. Heading smoothing (angular EMA with shortest-arc)
        if self._last_chosen_az is not None and self._heading_smooth < 1.0:
            alpha = self._heading_smooth
            delta_az = _wrap(best_az - self._last_chosen_az)
            best_az = self._last_chosen_az + alpha * delta_az
            best_el = (1 - alpha) * self._last_chosen_el + alpha * best_el

        self._last_chosen_az = best_az
        self._last_chosen_el = best_el

        # 8. Speed modulation
        speed = self._compute_speed(min_obs_dist)

        # 9. Spherical → Cartesian velocity
        cos_el = math.cos(best_el)
        vx = speed * cos_el * math.cos(best_az)
        vy = speed * cos_el * math.sin(best_az)
        vz = speed * math.sin(best_el)

        return (vx, vy, vz)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Horizontal-slice histogram for viz2d compatibility."""
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
            "gaps": self._last_valleys,
            # VFH-specific: expose density for debugging / advanced viz
            "density": self._density.tolist(),
        }

    def reset(self):
        self._density[:] = 0.0
        self._blocked = ~self._coverage.copy()
        self._range_map[:] = self.max_range
        self._last_chosen_az = None
        self._last_chosen_el = None
        self._last_valleys = []
        self._stuck_counter = 0
        self._prev_goal_dist = float('inf')

    @property
    def bubble_radius(self):
        return self._bubble_radius

    # ------------------------------------------------------------------
    # VFH density histogram
    # ------------------------------------------------------------------

    def _build_density_map(self, pts: np.ndarray):
        """
        Build the VFH polar obstacle density histogram.

        Each obstacle point contributes density  h = a − b·d  (clamped ≥ 0)
        to all cells within its angular bubble.  The bubble angular radius
        is  arcsin(bubble_radius / d).  This produces a smooth density
        field where close obstacles dominate.

        Also fills the range map (min range per cell) used for valley
        width checks and speed modulation.
        """
        self._density[:] = 0.0
        self._range_map[:] = self.max_range

        if len(pts) == 0:
            return

        dists = np.sqrt(np.sum(pts**2, axis=1))
        valid = (dists > 0.00001) & (dists < self.max_range)
        if not np.any(valid):
            return

        pts_v = pts[valid]
        d_v = dists[valid]

        # Bin indices for range map
        az = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        el = np.arctan2(pts_v[:, 2], np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2))
        az_idx = ((az + math.pi) / self._az_res).astype(int) % self.n_az
        el_idx = ((el + self.el_max) / self._el_res).astype(int)
        el_idx = np.clip(el_idx, 0, self.n_el - 1)
        np.minimum.at(self._range_map, (el_idx, az_idx), d_v)

        # Deduplicate per cell — keep closest point only (same as FGM3D)
        cell_min_dist = {}
        cell_min_dir = {}
        for i in range(len(d_v)):
            key = (int(el_idx[i]), int(az_idx[i]))
            if key not in cell_min_dist or d_v[i] < cell_min_dist[key]:
                cell_min_dist[key] = float(d_v[i])
                cell_min_dir[key] = pts_v[i] / d_v[i]

        # Vote density into cells within the angular bubble
        for key, dist in cell_min_dist.items():
            # VFH density weight: closer → higher
            h = max(self._density_a - self._density_b * dist, 0.0)
            if h <= 0.0:
                continue

            # Angular bubble for safety inflation
            half_ang = math.asin(min(self._bubble_radius / dist, 1.0))

            obs_dir = cell_min_dir[key]
            dots = np.sum(self._cell_dirs * obs_dir, axis=-1)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dist = np.arccos(dots)

            # Smooth vote: linear fall-off within bubble
            within = ang_dist <= half_ang
            falloff = np.where(within,
                               h * (1.0 - ang_dist / (half_ang + 1e-9)),
                               0.0)
            self._density += falloff

    def _apply_threshold(self):
        """
        Hysteresis thresholding of density → blocked map.

        Cells above threshold_high become blocked.
        Cells below threshold_low  become free.
        Cells in between keep their previous state.
        Uncovered cells (no sensor) are always blocked.
        """
        newly_blocked = self._density >= self._thresh_high
        newly_free = self._density < self._thresh_low

        # Hysteresis: only change state when threshold is crossed
        self._blocked = np.where(newly_blocked, True,
                        np.where(newly_free, False,
                                 self._blocked))

        # Uncovered cells are always blocked
        self._blocked[~self._coverage] = True

    # ------------------------------------------------------------------
    # Valley finding (connected-component flood fill)
    # ------------------------------------------------------------------

    def _find_valleys(self):
        """
        Find candidate valleys — contiguous free regions on the spherical
        grid.  Identical topology to FGM3D._find_gaps().

        Returns list of valleys, each a list of (el_idx, az_idx) tuples.
        """
        # A cell is a candidate if (a) not density-blocked, AND
        # (b) range map shows no obstacle within safe_distance.
        free = (~self._blocked) & (self._range_map >= self.safe_distance)
        # Blind spots: traversable (same policy as FGM3D)
        free[~self._coverage] = True

        if not np.any(free):
            return []

        visited = np.zeros_like(free, dtype=bool)
        valleys = []

        for ei in range(self.n_el):
            for ai in range(self.n_az):
                if free[ei, ai] and not visited[ei, ai]:
                    cells = []
                    stack = [(ei, ai)]
                    visited[ei, ai] = True
                    while stack:
                        ce, ca = stack.pop()
                        cells.append((ce, ca))
                        for de, da in [(-1, -1), (-1, 0), (-1, 1),
                                       (0, -1),           (0, 1),
                                       (1, -1),  (1, 0),  (1, 1)]:
                            ne = ce + de
                            na = (ca + da) % self.n_az
                            if ne < 0 or ne >= self.n_el:
                                continue
                            if free[ne, na] and not visited[ne, na]:
                                visited[ne, na] = True
                                stack.append((ne, na))

                    if len(cells) < self.min_gap_cells:
                        continue
                    if not self._valley_wide_enough(cells):
                        continue
                    valleys.append(cells)

        return valleys

    def _valley_wide_enough(self, cells) -> bool:
        """Check minimum physical width — same logic as FGM3D."""
        ei_vals = [c[0] for c in cells]
        ai_vals = [c[1] for c in cells]

        az_span = (max(ai_vals) - min(ai_vals) + 1) * self._az_res
        el_span = (max(ei_vals) - min(ei_vals) + 1) * self._el_res

        cell_set = set(cells)
        min_border_range = self.max_range
        for ce, ca in cells:
            for de, da in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]:
                ne = ce + de
                na = (ca + da) % self.n_az
                if ne < 0 or ne >= self.n_el:
                    continue
                if (ne, na) not in cell_set and self._blocked[ne, na]:
                    r = self._range_map[ne, na]
                    if r < min_border_range:
                        min_border_range = r

        phys_w = min(az_span, el_span) * min_border_range
        return phys_w >= self.min_gap_metres

    # ------------------------------------------------------------------
    # VFH+ cost-function valley selection
    # ------------------------------------------------------------------

    def _select_valley(self, valleys, goal_az: float, goal_el: float) -> tuple:
        """
        VFH+ multi-objective cost function for valley selection.

        For each valley, find the best steering cell, then score the
        valley by:
            cost = w_goal · Δ(cell, goal)
                 + w_smooth · Δ(cell, prev_heading)
                 − w_width · valley_size

        where Δ is great-circle angular distance.

        Returns (az, el) of the best steering point.
        """
        goal_dir = np.array([
            math.cos(goal_el) * math.cos(goal_az),
            math.cos(goal_el) * math.sin(goal_az),
            math.sin(goal_el),
        ])

        # Previous heading direction (for smoothness cost)
        if self._last_chosen_az is not None:
            prev_az = self._last_chosen_az
            prev_el = self._last_chosen_el
        else:
            prev_az = goal_az
            prev_el = goal_el

        best_cost = float('inf')
        best_az = goal_az
        best_el = goal_el

        for cells in valleys:
            valley_size = len(cells)

            # Find best steering cell in this valley
            best_cell_cost = float('inf')
            steer_az, steer_el = goal_az, goal_el
            steer_goal_ang = float('inf')

            for ei, ai in cells:
                cell_dir = self._cell_dirs[ei, ai]
                # Angular distance to goal
                dot_goal = float(np.clip(np.dot(cell_dir, goal_dir), -1.0, 1.0))
                ang_goal = math.acos(dot_goal)

                # Clearance: prefer cells far from obstacles
                clearance = self._range_map[ei, ai]
                clearance_penalty = self.max_range / max(clearance, 0.1)

                cell_cost = ang_goal + 0.3 * clearance_penalty
                if cell_cost < best_cell_cost:
                    best_cell_cost = cell_cost
                    steer_az = self._az_centres[ai]
                    steer_el = self._el_centres[ei]
                    steer_goal_ang = ang_goal

            # Pull from boundary
            steer_az, steer_el = self._pull_from_boundary(
                cells, steer_az, steer_el, goal_dir
            )

            # VFH+ cost: goal proximity + heading smoothness − width bonus
            smooth_ang = _angular_dist_sphere(
                steer_az, steer_el, prev_az, prev_el)

            cost = (self.gap_weight_goal * steer_goal_ang
                    + self._cost_smooth * smooth_ang
                    - self.gap_weight_width * valley_size
                      * self._az_res * self._el_res)

            if cost < best_cost:
                best_cost = cost
                best_az = steer_az
                best_el = steer_el

        return best_az, best_el

    def _pull_from_boundary(self, cells, steer_az, steer_el, goal_dir):
        """
        Pull steering point inward from valley boundary toward centroid.
        Identical to FGM3D._pull_from_boundary().
        """
        cell_set = set(cells)

        ai = int((steer_az + math.pi) / self._az_res) % self.n_az
        ei = int((steer_el + self.el_max) / self._el_res)
        ei = max(0, min(self.n_el - 1, ei))

        at_boundary = False
        for de, da in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ne = ei + de
            na = (ai + da) % self.n_az
            if ne < 0 or ne >= self.n_el:
                at_boundary = True
                break
            if (ne, na) not in cell_set:
                at_boundary = True
                break

        if not at_boundary:
            return steer_az, steer_el

        # Gap centroid (circular mean for azimuth)
        sum_sin_az = 0.0
        sum_cos_az = 0.0
        sum_el = 0.0
        for ce, ca in cells:
            sum_cos_az += math.cos(self._az_centres[ca])
            sum_sin_az += math.sin(self._az_centres[ca])
            sum_el += self._el_centres[ce]

        n = len(cells)
        centroid_az = math.atan2(sum_sin_az / n, sum_cos_az / n)
        centroid_el = sum_el / n

        margin_frac = min(0.3, self.edge_margin / math.pi)
        out_az = steer_az + margin_frac * _wrap(centroid_az - steer_az)
        out_el = steer_el + margin_frac * (centroid_el - steer_el)
        return out_az, out_el

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retreat(self, pts: np.ndarray, min_obs_dist: float) -> tuple:
        """Back away from the centroid of nearby obstacles."""
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
        ratio = max(min_obs_dist / self.safe_distance, 0.1)
        return self.max_speed * ratio


# -----------------------------------------------------------------------
# Self-test (same scenarios as FGM3D)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    vfh = VFH3D(
        n_az=72, n_el=18,
        max_range=3.0,
        bubble_radius=0.55,
        safe_distance=1.2,
        max_speed=0.5,
    )

    # Beam ahead — should steer around
    beam = np.array([
        [1.2, -0.1, 0.0], [1.2, 0.0, 0.0], [1.2, 0.1, 0.0],
    ])
    vel = vfh.update(beam, (2.0, 0.0, 0.0))
    print(f"Beam ahead → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")

    # Beam above — should go under
    vfh.reset()
    beam_above = np.array([
        [0.5, -0.3, 0.5], [0.5, 0.0, 0.5], [0.5, 0.3, 0.5],
        [-0.2, 0.0, 0.5], [0.0, 0.3, 0.5],
    ])
    vel = vfh.update(beam_above, (2.0, 0.0, 0.3))
    print(f"\nBeam above, goal up → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should steer forward but avoid going up")

    # Rafter blocking forward-up, gap is forward-low
    vfh.reset()
    rafter = np.array([
        [1.0, y, 0.3 + abs(y) * 0.3]
        for y in np.linspace(-1.0, 1.0, 15)
    ])
    vel = vfh.update(rafter, (2.0, 0.0, -0.5))
    print(f"\nRafter blocking fwd-up → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should go forward-and-down (vz negative)")

    # No obstacles — straight to goal
    vfh.reset()
    vel = vfh.update(np.empty((0, 3)), (2.0, 0.5, 1.0))
    print(f"\nNo obstacles → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should point toward goal")