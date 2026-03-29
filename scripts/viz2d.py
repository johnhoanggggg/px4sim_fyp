#!/usr/bin/env python3
"""
Lightweight 2D top-down visualizer for FGM obstacle avoidance.

Runs as a separate process (spawned by fly_*.py).  Reads state
from a multiprocessing.Queue and draws:
  - Left panel:  Top-down map with obstacles, waypoints, drone trail,
                  histogram bins, and chosen direction
  - Right panel: Azimuth-elevation spherical grid showing blocked cells
                  and chosen steering direction
"""

import math
import multiprocessing as mp
from queue import Empty

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np


# Pillar positions in NED (north, east) — must match pillars.sdf
# SDF ENU pose (x=east, y=north) → NED (north=enu_y, east=enu_x)
PILLAR_NED = [
    (2.0,  0.8),
    (2.5, -1.0),
    (4.0,  0.0),
    (4.5,  2.0),
    (4.0, -1.5),
    (6.0,  1.0),
    (6.5, -0.5),
    (7.0,  2.5),
    (8.0,  0.0),
    (8.5, -2.0),
    (9.5,  1.5),
    (10.0, -0.8),
    (11.0,  0.5),
    (11.5, -1.5),
    (3.0,  3.0),
]

BIN_LINE_LEN = 1.0     # length of bin indicator lines
ARROW_LEN = 1.4        # chosen direction arrow length
PILLAR_RADIUS = 0.15
DRONE_SIZE = 0.4
TRAIL_MAX = 2000


def _body_flu_to_ned(az_body, yaw):
    """Convert body-FLU azimuth to NED (north, east) unit direction."""
    bx = math.cos(az_body)
    by = math.sin(az_body)
    c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
    dn = bx * c_yaw + by * s_yaw
    de = bx * s_yaw - by * c_yaw
    return dn, de


def run_viz(queue: mp.Queue):
    """Entry point — call from a multiprocessing.Process."""
    fig, (ax_map, ax_sphere) = plt.subplots(
        1, 2, figsize=(14, 8),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )
    fig.canvas.manager.set_window_title("FGM3D Visualizer")

    # =====================================================================
    # LEFT PANEL — Top-down map
    # =====================================================================
    obstacles_drawn = {"done": False}
    for (pn, pe) in PILLAR_NED:
        circ = plt.Circle((pe, pn), PILLAR_RADIUS, color="0.45",
                          ec="0.3", lw=1, zorder=2)
        ax_map.add_patch(circ)

    ax_map.set_aspect("equal")
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.set_title("Top-Down View")
    ax_map.grid(True, alpha=0.3)
    ax_map.set_xlim(-4, 5)
    ax_map.set_ylim(-2, 14)

    trail_x, trail_y = [], []
    trail_line, = ax_map.plot([], [], "-", color="cornflowerblue", lw=1, alpha=0.5, zorder=1)
    drone_marker, = ax_map.plot([], [], "o", color="dodgerblue", ms=8,
                                mec="navy", mew=1.5, zorder=6)
    heading_line, = ax_map.plot([], [], "-", color="navy", lw=2.5, zorder=6)
    wp_scatter = ax_map.scatter([], [], marker="D", c="dodgerblue", s=60,
                                edgecolors="navy", linewidths=0.5, zorder=3)
    cur_wp_scatter = ax_map.scatter([], [], marker="*", c="red", s=200, zorder=4)
    chosen_line, = ax_map.plot([], [], "-", color="blue", lw=3, solid_capstyle="round", zorder=5)

    free_lc = LineCollection([], colors="limegreen", linewidths=1.5, alpha=0.6, zorder=3)
    blocked_lc = LineCollection([], colors="red", linewidths=3.0, alpha=0.8, zorder=4)
    ax_map.add_collection(free_lc)
    ax_map.add_collection(blocked_lc)

    # =====================================================================
    # RIGHT PANEL — Azimuth-Elevation spherical grid
    # =====================================================================
    ax_sphere.set_xlabel("Azimuth (deg)")
    ax_sphere.set_ylabel("Elevation (deg)")
    ax_sphere.set_title("Spherical Blocked Grid (body FLU)")

    # Camera panorama background (below obstacle overlay)
    cam_img = ax_sphere.imshow(
        np.zeros((140, 360, 3), dtype=np.uint8),
        aspect="auto",
        origin="lower",
        extent=[-180, 180, -70, 70],
        interpolation="bilinear",
        zorder=0,
    )
    cam_img.set_visible(False)

    # Obstacle overlay with alpha channel (above camera)
    # Black = blind spot, Green→Yellow→Red = far→close, Blue = detected gap
    sphere_img = ax_sphere.imshow(
        np.zeros((18, 72, 4)),  # RGBA
        aspect="auto",
        origin="lower",
        extent=[-180, 180, -70, 70],
        interpolation="nearest",
        zorder=1,
    )
    # Colorbar: use a scalar mappable for the legend
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_sphere, shrink=0.7, pad=0.02)
    cbar.set_label("Closeness (1 = touching, 0 = clear, black = blind)")
    # Crosshair for chosen direction
    chosen_dot, = ax_sphere.plot([], [], "x", color="blue", ms=14, mew=3, zorder=10)
    # Goal direction marker
    goal_dot, = ax_sphere.plot([], [], "+", color="cyan", ms=12, mew=2, zorder=9)
    # Forward direction reference line
    ax_sphere.axvline(0, color="white", alpha=0.3, lw=1, ls="--")
    ax_sphere.axhline(0, color="white", alpha=0.3, lw=1, ls="--")

    state = {"first": True}

    def _update(_frame):
        data = None
        try:
            while True:
                data = queue.get_nowait()
        except Empty:
            pass

        if data is None:
            return []

        # Replace static obstacles on first packet if dynamic list given
        if not obstacles_drawn["done"] and "obstacles" in data:
            while ax_map.patches:
                ax_map.patches[0].remove()
            obs_r = data.get("obstacle_radius", PILLAR_RADIUS)
            for (on, oe) in data["obstacles"]:
                circ = plt.Circle((oe, on), obs_r, color="0.45",
                                  ec="0.3", lw=1, zorder=2)
                ax_map.add_patch(circ)
            obstacles_drawn["done"] = True

        dn, de = data["drone_n"], data["drone_e"]
        yaw = data["yaw"]
        bins = data["bins"]
        chosen = data.get("chosen")
        wps = data["waypoints"]
        cur_wp = data["current_wp"]

        # --- Top-down panel ---
        trail_x.append(de)
        trail_y.append(dn)
        if len(trail_x) > TRAIL_MAX:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        drone_marker.set_data([de], [dn])

        h_dn, h_de = _body_flu_to_ned(0.0, yaw)
        heading_line.set_data(
            [de, de + DRONE_SIZE * h_de],
            [dn, dn + DRONE_SIZE * h_dn])

        if wps:
            wp_e = [w[1] for w in wps]
            wp_n = [w[0] for w in wps]
            wp_scatter.set_offsets(np.column_stack([wp_e, wp_n]))
            if 0 <= cur_wp < len(wps):
                cur_wp_scatter.set_offsets([[wps[cur_wp][1], wps[cur_wp][0]]])

        free_segs, blocked_segs = [], []
        for az_body, blocked in bins:
            b_dn, b_de = _body_flu_to_ned(az_body, yaw)
            seg = [(de, dn), (de + BIN_LINE_LEN * b_de, dn + BIN_LINE_LEN * b_dn)]
            if blocked:
                blocked_segs.append(seg)
            else:
                free_segs.append(seg)

        free_lc.set_segments(free_segs)
        blocked_lc.set_segments(blocked_segs)

        if chosen is not None:
            c_dn, c_de = _body_flu_to_ned(chosen, yaw)
            chosen_line.set_data(
                [de, de + ARROW_LEN * c_de],
                [dn, dn + ARROW_LEN * c_dn])
            chosen_line.set_visible(True)
        else:
            chosen_line.set_visible(False)

        # --- Camera panorama background ---
        cam_frame = data.get("camera_frame")
        if cam_frame is not None:
            # Horizontal flip only: camera longitude is right-positive
            # but FLU azimuth is left-positive. No vertical flip needed —
            # Gazebo equirectangular row 0 = looking down, matching origin="lower"
            cam_img.set_data(cam_frame[:, ::-1])
            cam_img.set_visible(True)
        else:
            cam_img.set_visible(False)

        # --- Spherical grid panel ---
        sphere = data.get("sphere")
        if sphere is not None:
            range_map = np.array(sphere["range_map"])
            max_range = sphere["max_range"]
            coverage = np.array(sphere["coverage"], dtype=bool)
            az_deg = np.array(sphere["az_centres"]) * 180 / math.pi
            el_deg = np.array(sphere["el_centres"]) * 180 / math.pi

            # Closeness: 1 = touching, 0 = far/free
            closeness = 1.0 - np.clip(range_map / max_range, 0, 1)

            # Build RGBA image (vectorized)
            cmap = plt.cm.RdYlGn_r
            rgba = cmap(closeness)        # (n_el, n_az, 4) all cells colored

            # Detected gaps → semi-transparent blue
            gaps = sphere.get("gaps", [])
            for gap_cells in gaps:
                for ei, ai in gap_cells:
                    rgba[ei, ai] = [0.2, 0.4, 1.0, 0.45]

            # Blind spots → opaque black
            blind = ~coverage
            rgba[blind] = [0.0, 0.0, 0.0, 1.0]

            # Alpha: obstacles opaque, clear areas semi-transparent (camera shows through)
            covered = coverage
            if cam_frame is not None:
                # With camera: clear cells become translucent so camera is visible
                rgba[covered, 3] = np.clip(closeness[covered] * 1.5 + 0.15, 0.15, 0.85)
                # Gap cells stay at their set alpha (0.45)
                for gap_cells in gaps:
                    for ei, ai in gap_cells:
                        if coverage[ei, ai]:
                            rgba[ei, ai, 3] = 0.45

            sphere_img.set_data(rgba)
            ext = [az_deg[0] - 2.5, az_deg[-1] + 2.5,
                   el_deg[0] - 4, el_deg[-1] + 4]
            sphere_img.set_extent(ext)
            cam_img.set_extent(ext)

            # Chosen steering direction
            s_az = sphere.get("chosen_az")
            s_el = sphere.get("chosen_el")
            if s_az is not None and s_el is not None:
                chosen_dot.set_data([math.degrees(s_az)], [math.degrees(s_el)])
                chosen_dot.set_visible(True)
            else:
                chosen_dot.set_visible(False)

        # Auto-scale on first real data
        if state["first"]:
            state["first"] = False
            if wps:
                obs_ne = data.get("obstacles", PILLAR_NED)
                all_e = [w[1] for w in wps] + [p[1] for p in obs_ne] + [de]
                all_n = [w[0] for w in wps] + [p[0] for p in obs_ne] + [dn]
                pad = 2.5
                ax_map.set_xlim(min(all_e) - pad, max(all_e) + pad)
                ax_map.set_ylim(min(all_n) - pad, max(all_n) + pad)

        fig.canvas.draw_idle()
        return []

    _anim = FuncAnimation(fig, _update, interval=100, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo mode with fake data
    q: mp.Queue = mp.Queue()
    import threading, time

    def _fake():
        t = 0.0
        n_az, n_el = 72, 18
        wps = [(0, 0), (4, 0), (8, 1), (11, 0)]
        az_c = [(-math.pi + (i + 0.5) * 2 * math.pi / n_az) for i in range(n_az)]
        el_c = [(-math.radians(70) + (i + 0.5) * 2 * math.radians(70) / n_el) for i in range(n_el)]
        while True:
            bins = [(math.radians(a - 180 + 5), a % 70 < 20) for a in range(0, 360, 10)]
            # Fake range map — beam across forward direction
            range_map = np.full((n_el, n_az), 3.0)
            range_map[n_el//2-1:n_el//2+2, n_az//4:3*n_az//4] = 0.6  # horizontal beam
            range_map[3:n_el-3, n_az//2-2:n_az//2+2] = 1.0  # vertical post
            blocked = range_map < 2.9
            # Fake coverage — blind spots at high/low elevation bands
            coverage = np.ones((n_el, n_az), dtype=bool)
            coverage[0:3, :] = False    # below -45 deg
            coverage[-3:, :] = False    # above +45 deg
            q.put({
                "drone_n": 2 + t * 0.3,
                "drone_e": math.sin(t * 0.5) * 1.5,
                "yaw": 0.0,
                "bins": bins,
                "chosen": 0.1,
                "waypoints": wps,
                "current_wp": min(int(t / 3), len(wps) - 1),
                "sphere": {
                    "blocked": blocked.tolist(),
                    "coverage": coverage.tolist(),
                    "range_map": range_map.tolist(),
                    "max_range": 3.0,
                    "az_centres": az_c,
                    "el_centres": el_c,
                    "chosen_az": 0.1,
                    "chosen_el": -0.2,
                },
            })
            t += 0.1
            time.sleep(0.1)

    threading.Thread(target=_fake, daemon=True).start()
    run_viz(q)
