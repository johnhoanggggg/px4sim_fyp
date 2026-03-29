#!/usr/bin/env python3
"""
FGM3D — True 3-D Follow-the-Gap Method on a spherical scan.

Instead of separating horizontal and vertical avoidance, this projects all
obstacle points onto a unit sphere (azimuth + elevation), builds a 2-D
blocked/free map, finds 3-D gaps, and steers toward the best gap closest
to the goal direction.  The output is a full (vx, vy, vz) velocity in
body FLU frame.

Sphere discretisation:
    - n_az   azimuth bins   covering [-pi, pi)      (horizontal)
    - n_el   elevation bins covering [-el_max, el_max] (vertical)
    Each cell is a small solid-angle patch.  Obstacle points that fall
    within max_range are projected onto cells and mark them blocked
    (with bubble inflation).  Contiguous free regions are found via
    flood-fill, scored by proximity to the goal direction, and the
    best steering direction is picked.
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
    """Compute a boolean mask of which (el, az) cells are covered by sensors.

    For each sensor, projects every cell direction into the sensor's local
    frame and checks whether it falls within the rectangular ±FOV_HALF
    field of view.  This is exact and avoids gaps caused by discrete ray
    sampling when the ray spacing exceeds the grid bin width.
    """
    coverage = np.zeros((n_el, n_az), dtype=bool)

    # Precompute unit direction for each grid cell (n_el, n_az, 3)
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
        # Project all cell directions into sensor-local frame
        R_inv = rot.T
        local = np.einsum('ij,mnj->mni', R_inv, cell_dirs)  # (n_el, n_az, 3)
        lx = local[..., 0]
        ly = local[..., 1]
        lz = local[..., 2]
        # Cell is covered if in front of sensor and within rectangular FOV
        ha = np.arctan2(ly, lx)
        va = np.arctan2(lz, np.sqrt(lx**2 + ly**2))
        within = (lx > 0) & (np.abs(ha) <= _FOV_HALF) & (np.abs(va) <= _FOV_HALF)
        coverage[within] = True

    for yaw in _HORIZONTAL_YAWS:
        _mark(_rotz(yaw))
    for pitch in _VERTICAL_PITCHES:
        _mark(_roty(pitch))

    return coverage


class FGM3D:
    """
    True 3-D Follow-the-Gap planner on a spherical scan.

    Parameters
    ----------
    n_az : int
        Azimuth bins (horizontal, default 72 = 5 deg each).
    n_el : int
        Elevation bins (vertical, default 18 = ~10 deg each for ±90 deg).
    max_range : float
        Sensing range (m).
    bubble_radius : float
        Safety inflation around obstacles (m), >= drone radius.
    safe_distance : float
        Distance below which speed is reduced.
    max_speed : float
        Maximum output speed (m/s).
    gap_weight_goal : float
        Weight for angular distance from steering point to goal.
    gap_weight_width : float
        Bonus for larger gap regions.
    min_gap_cells : int
        Minimum number of contiguous free cells to count as a gap.
    min_gap_metres : float
        Minimum physical gap width (m).  Gaps whose angular span at
        the local obstacle range is narrower than this get rejected.
    edge_margin_deg : float
        Pull steering point inward from gap boundary (deg).
    el_max_deg : float
        Maximum elevation angle (deg).  90 = full hemisphere up/down.
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

        # Sensor coverage mask — True where at least one sensor ray falls
        self._coverage = _build_coverage_mask(
            self._az_centres, self._el_centres,
            self._az_res, self._el_res, n_az, n_el,
        )

        # State
        self._blocked = np.zeros((n_el, n_az), dtype=bool)
        self._range_map = np.full((n_el, n_az), max_range)
        self._last_chosen_az: float | None = None
        self._last_chosen_el: float | None = None
        self._last_gaps: list = []

        # Stuck detection
        self._stuck_counter = 0
        self._prev_goal_dist = float('inf')

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

        # 1. Build spherical range map and blocked mask
        self._build_map(obstacle_pts)

        # 2. Find min obstacle distance (for speed scaling)
        min_obs_dist = float('inf')
        if len(obstacle_pts) > 0:
            dists = np.sqrt(np.sum(obstacle_pts**2, axis=1))
            valid = dists > 0.05
            if np.any(valid):
                min_obs_dist = float(np.min(dists[valid]))

        # 3. Stuck detection: if not making progress toward goal, retreat
        if goal_dist < self._prev_goal_dist - 0.05:
            self._stuck_counter = 0
        else:
            self._stuck_counter += 1
        self._prev_goal_dist = goal_dist

        # Force retreat if stuck too long or inside bubble radius
        if self._stuck_counter > 20 or min_obs_dist < self._bubble_radius:
            self._last_chosen_az = None
            self._last_chosen_el = None
            if self._stuck_counter > 20:
                self._stuck_counter = 0
            return self._retreat(obstacle_pts, min_obs_dist)

        # 4. Find gaps via connected-component flood fill
        gaps = self._find_gaps()
        self._last_gaps = gaps

        # 5. No passable gap → back away from nearest obstacle
        if not gaps:
            self._last_chosen_az = None
            self._last_chosen_el = None
            return self._retreat(obstacle_pts, min_obs_dist)

        # 6. Pick best gap and steering direction
        best_az, best_el = self._select_gap(gaps, goal_az, goal_el)

        # 7. Heading smoothing — angular EMA with shortest-arc interpolation
        if self._last_chosen_az is not None and self._heading_smooth < 1.0:
            alpha = self._heading_smooth
            delta_az = (best_az - self._last_chosen_az + math.pi) % (2 * math.pi) - math.pi
            best_az = self._last_chosen_az + alpha * delta_az
            best_el = (1 - alpha) * self._last_chosen_el + alpha * best_el

        self._last_chosen_az = best_az
        self._last_chosen_el = best_el

        # 8. Compute speed (slow near obstacles)
        speed = self._compute_speed(min_obs_dist)

        # 9. Convert spherical steering direction to Cartesian velocity
        cos_el = math.cos(best_el)
        vx = speed * cos_el * math.cos(best_az)
        vy = speed * cos_el * math.sin(best_az)
        vz = speed * math.sin(best_el)

        return (vx, vy, vz)

    def get_histogram(self) -> list[tuple[float, bool]]:
        """Return horizontal-slice histogram for viz2d compatibility.

        Projects the spherical blocked map onto the azimuth axis:
        a ray is blocked if ANY elevation bin at that azimuth is blocked.
        """
        # Use the elevation band near horizontal (middle rows)
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
            "gaps": self._last_gaps,
        }

    def reset(self):
        self._blocked = ~self._coverage.copy()
        self._range_map[:] = self.max_range
        self._last_chosen_az = None
        self._last_chosen_el = None
        self._last_gaps = []
        self._stuck_counter = 0
        self._prev_goal_dist = float('inf')

    @property
    def bubble_radius(self):
        return self._bubble_radius

    # ------------------------------------------------------------------
    # Spherical map building
    # ------------------------------------------------------------------

    def _build_map(self, pts: np.ndarray):
        """Project obstacle points onto the spherical grid and inflate."""
        # Start with uncovered cells blocked (can't fly where you can't see)
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

        # Spherical coords of each point
        az = np.arctan2(pts_v[:, 1], pts_v[:, 0])
        el = np.arctan2(pts_v[:, 2], np.sqrt(pts_v[:, 0]**2 + pts_v[:, 1]**2))

        # Bin indices
        az_idx = ((az + math.pi) / self._az_res).astype(int) % self.n_az
        el_idx = ((el + self.el_max) / self._el_res).astype(int)
        el_idx = np.clip(el_idx, 0, self.n_el - 1)

        # Fill range map (min range per cell)
        np.minimum.at(self._range_map, (el_idx, az_idx), d_v)

        # Deduplicate: for each occupied cell, keep only the closest point
        # This avoids redundant inflation for multiple points on the same beam
        cell_min_dist = {}   # (ei, ai) → min distance
        cell_min_dir = {}    # (ei, ai) → unit direction of closest point
        for i in range(len(d_v)):
            key = (int(el_idx[i]), int(az_idx[i]))
            if key not in cell_min_dist or d_v[i] < cell_min_dist[key]:
                cell_min_dist[key] = float(d_v[i])
                cell_min_dir[key] = pts_v[i] / d_v[i]

        # Inflate per occupied cell (much fewer iterations than per-point)
        for key, dist in cell_min_dist.items():
            half_ang = math.asin(min(self.bubble_radius / dist, 1.0))
            obs_dir = cell_min_dir[key]
            dots = np.sum(self._cell_dirs * obs_dir, axis=-1)
            dots = np.clip(dots, -1.0, 1.0)
            ang_dist = np.arccos(dots)
            self._blocked |= (ang_dist <= half_ang)

    # ------------------------------------------------------------------
    # 3-D gap finding via flood fill on the spherical grid
    # ------------------------------------------------------------------

    def _find_gaps(self):
        """
        Find connected free regions on the spherical blocked grid.

        Returns list of gaps, each a list of (el_idx, az_idx) tuples.
        Gaps are filtered by both minimum cell count and minimum
        physical width (angular span × range to nearby obstacles).
        """
        # Erode blocked mask for gap-finding (self._blocked stays intact for
        # safety).  Thin blocked barriers from bubble inflation between truss
        # members are opened up so flood-fill can connect free regions.
        # Two passes: first remove isolated noise, then erode thin barriers.
        blocked = self._blocked.copy()
        n_el, n_az = blocked.shape

        # Count blocked 8-neighbours for each cell
        neighbour_count = np.zeros_like(blocked, dtype=int)
        for de, da in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(blocked, -de, axis=0)
            shifted = np.roll(shifted, -da, axis=1)
            if de == -1:
                shifted[-1, :] = False
            elif de == 1:
                shifted[0, :] = False
            neighbour_count += shifted.astype(int)

        # A covered blocked cell with ≤4 blocked neighbours (out of 8) is
        # likely a thin inflation barrier — reclassify as free for gap finding.
        thin_barrier = blocked & (neighbour_count <= 4) & self._coverage
        blocked[thin_barrier] = False

        free = ~blocked
        if not np.any(free):
            return []

        visited = np.zeros_like(free, dtype=bool)
        gaps = []

        for ei in range(self.n_el):
            for ai in range(self.n_az):
                if free[ei, ai] and not visited[ei, ai]:
                    # Flood fill
                    cells = []
                    stack = [(ei, ai)]
                    visited[ei, ai] = True
                    while stack:
                        ce, ca = stack.pop()
                        cells.append((ce, ca))
                        for de, da in [(-1,-1),(-1,0),(-1,1),
                                       (0,-1),        (0,1),
                                       (1,-1), (1,0), (1,1)]:
                            ne = ce + de
                            na = (ca + da) % self.n_az
                            if ne < 0 or ne >= self.n_el:
                                continue
                            if free[ne, na] and not visited[ne, na]:
                                visited[ne, na] = True
                                stack.append((ne, na))

                    if len(cells) < self.min_gap_cells:
                        continue

                    # Check physical width: estimate gap span in metres
                    if not self._gap_wide_enough(cells):
                        continue

                    gaps.append(cells)

        return gaps

    def _gap_wide_enough(self, cells) -> bool:
        """Check if gap is physically wide enough for the drone to fly through.

        Computes the angular span of the gap (bounding box on the grid)
        and multiplies by the range to the nearest obstacle boundary to
        get an approximate physical width in metres.
        """
        ei_vals = [c[0] for c in cells]
        ai_vals = [c[1] for c in cells]

        # Angular span in azimuth and elevation
        az_span = (max(ai_vals) - min(ai_vals) + 1) * self._az_res
        el_span = (max(ei_vals) - min(ei_vals) + 1) * self._el_res

        # Find the range to the nearest blocked cell bordering this gap
        # (tells us how far away the gap walls are)
        cell_set = set(cells)
        min_border_range = self.max_range
        for ce, ca in cells:
            for de, da in [(-1,-1),(-1,0),(-1,1),
                           (0,-1),        (0,1),
                           (1,-1), (1,0), (1,1)]:
                ne = ce + de
                na = (ca + da) % self.n_az
                if ne < 0 or ne >= self.n_el:
                    continue
                if (ne, na) not in cell_set and self._blocked[ne, na]:
                    r = self._range_map[ne, na]
                    if r < min_border_range:
                        min_border_range = r

        # Physical width ≈ angular_span × range
        phys_w = min(az_span, el_span) * min_border_range
        return phys_w >= self.min_gap_metres

    def _select_gap(self, gaps, goal_az: float, goal_el: float) -> tuple:
        """
        Pick the best gap and return (az, el) steering direction.

        For each gap, find the free cell closest to the goal direction.
        Score gaps by angular proximity to goal + size bonus + clearance.
        """
        best_score = float('inf')
        best_az = goal_az
        best_el = goal_el

        goal_dir = np.array([
            math.cos(goal_el) * math.cos(goal_az),
            math.cos(goal_el) * math.sin(goal_az),
            math.sin(goal_el),
        ])

        for cells in gaps:
            gap_size = len(cells)

            # Find cell in this gap closest to goal direction,
            # but penalise cells near obstacles (low range_map)
            best_cell_score = float('inf')
            steer_az, steer_el = goal_az, goal_el

            for ei, ai in cells:
                cell_dir = self._cell_dirs[ei, ai]
                dot = float(np.dot(cell_dir, goal_dir))
                dot = max(-1.0, min(1.0, dot))
                ang = math.acos(dot)
                # Clearance penalty: prefer cells far from obstacles
                clearance = self._range_map[ei, ai]
                clearance_penalty = self.max_range / max(clearance, 0.1)
                cell_score = ang + 0.3 * clearance_penalty
                if cell_score < best_cell_score:
                    best_cell_score = cell_score
                    min_ang = ang
                    steer_az = self._az_centres[ai]
                    steer_el = self._el_centres[ei]

            # Pull steering point inward from gap boundary
            steer_az, steer_el = self._pull_from_boundary(
                cells, steer_az, steer_el, goal_dir
            )

            score = (self.gap_weight_goal * min_ang
                     - self.gap_weight_width * gap_size * self._az_res * self._el_res)

            if score < best_score:
                best_score = score
                best_az = steer_az
                best_el = steer_el

        return best_az, best_el

    def _pull_from_boundary(self, cells, steer_az, steer_el, goal_dir):
        """
        If the chosen steering cell is at the edge of the gap, pull it
        inward toward the gap centre to avoid shaving obstacles.
        """
        # Build a set of gap cells for fast lookup
        cell_set = set(cells)

        # Find the cell index of the steering point
        ai = int((steer_az + math.pi) / self._az_res) % self.n_az
        ei = int((steer_el + self.el_max) / self._el_res)
        ei = max(0, min(self.n_el - 1, ei))

        # Check if it's near the boundary
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

        # Find the gap centroid and pull toward it
        sum_az = 0.0
        sum_el = 0.0
        # Use circular mean for azimuth
        sum_sin_az = 0.0
        sum_cos_az = 0.0
        for ce, ca in cells:
            sum_cos_az += math.cos(self._az_centres[ca])
            sum_sin_az += math.sin(self._az_centres[ca])
            sum_el += self._el_centres[ce]

        n = len(cells)
        centroid_az = math.atan2(sum_sin_az / n, sum_cos_az / n)
        centroid_el = sum_el / n

        # Blend: pull margin fraction toward centroid
        margin_frac = min(0.3, self.edge_margin / math.pi)
        out_az = steer_az + margin_frac * _wrap_scalar(centroid_az - steer_az)
        out_el = steer_el + margin_frac * (centroid_el - steer_el)

        return out_az, out_el

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retreat(self, pts: np.ndarray, min_obs_dist: float) -> tuple:
        """When fully blocked, back away from the centroid of nearby obstacles."""
        speed = self._compute_speed(min_obs_dist) * 0.5  # slow retreat
        if len(pts) == 0:
            return (0.0, 0.0, 0.0)

        dists = np.sqrt(np.sum(pts**2, axis=1))
        close = pts[dists < self.safe_distance * 1.5]
        if len(close) == 0:
            close = pts

        # Centroid of threats — retreat in opposite direction
        centroid = np.mean(close, axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 0.01:
            return (-speed, 0.0, 0.0)  # default: back up
        retreat_dir = -centroid / norm
        return (float(speed * retreat_dir[0]),
                float(speed * retreat_dir[1]),
                float(speed * retreat_dir[2]))

    def _compute_speed(self, min_obs_dist: float) -> float:
        if min_obs_dist >= self.safe_distance:
            return self.max_speed
        ratio = max(min_obs_dist / self.safe_distance, 0.1)
        return self.max_speed * ratio


def _wrap_scalar(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fgm = FGM3D(
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
    vel = fgm.update(beam, (2.0, 0.0, 0.0))
    print(f"Beam ahead → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")

    # Beam above — should go under
    fgm.reset()
    beam_above = np.array([
        [0.5, -0.3, 0.5], [0.5, 0.0, 0.5], [0.5, 0.3, 0.5],
        [-0.2, 0.0, 0.5], [0.0, 0.3, 0.5],
    ])
    vel = fgm.update(beam_above, (2.0, 0.0, 0.3))
    print(f"\nBeam above, goal up → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should steer forward but avoid going up")

    # Rafter blocking forward-up, gap is forward-low
    fgm.reset()
    rafter = np.array([
        [1.0, y, 0.3 + abs(y) * 0.3]
        for y in np.linspace(-1.0, 1.0, 15)
    ])
    vel = fgm.update(rafter, (2.0, 0.0, -0.5))
    print(f"\nRafter blocking fwd-up → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should go forward-and-down (vz negative)")

    # No obstacles — straight to goal
    fgm.reset()
    vel = fgm.update(np.empty((0, 3)), (2.0, 0.5, 1.0))
    print(f"\nNo obstacles → vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
    print(f"  Should point toward goal")
