#!/usr/bin/env python3
"""
Lightweight 2D top-down visualizer for VFH2D obstacle avoidance.

Runs as a separate process (spawned by fly_pillars.py).  Reads state
from a multiprocessing.Queue and draws:
  - Pillar positions (grey circles)
  - Waypoints and current target
  - Drone position + heading triangle
  - VFH2D histogram bins radiating from the drone (red=blocked, green=free)
  - Chosen steering direction (blue arrow)
  - Drone trail
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
    """Convert body-FLU azimuth to NED (north, east) unit direction.

    Body FLU: X=forward, Y=left.  NED: X=north, Y=east.
    Since left = -east, the Y signs flip vs the FRD→NED rotation.
    """
    bx = math.cos(az_body)
    by = math.sin(az_body)
    c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
    dn = bx * c_yaw + by * s_yaw
    de = bx * s_yaw - by * c_yaw
    return dn, de


def run_viz(queue: mp.Queue):
    """Entry point — call from a multiprocessing.Process."""
    fig, ax = plt.subplots(figsize=(7, 9))
    fig.canvas.manager.set_window_title("VFH2D Top-Down View")

    # Static elements — pillars
    for (pn, pe) in PILLAR_NED:
        circ = plt.Circle((pe, pn), PILLAR_RADIUS, color="0.45",
                          ec="0.3", lw=1, zorder=2)
        ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("VFH2D Obstacle Avoidance — Top Down")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 5)
    ax.set_ylim(-2, 14)

    # Dynamic artists
    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], "-", color="cornflowerblue", lw=1, alpha=0.5, zorder=1)
    drone_marker, = ax.plot([], [], "o", color="dodgerblue", ms=8,
                            mec="navy", mew=1.5, zorder=6)
    heading_line, = ax.plot([], [], "-", color="navy", lw=2.5, zorder=6)
    wp_scatter = ax.scatter([], [], marker="D", c="dodgerblue", s=60,
                            edgecolors="navy", linewidths=0.5, zorder=3)
    cur_wp_scatter = ax.scatter([], [], marker="*", c="red", s=200, zorder=4)
    chosen_line, = ax.plot([], [], "-", color="blue", lw=3, solid_capstyle="round", zorder=5)

    # Bin line collections (reused each frame)
    free_lc = LineCollection([], colors="limegreen", linewidths=1.5, alpha=0.6, zorder=3)
    blocked_lc = LineCollection([], colors="red", linewidths=3.0, alpha=0.8, zorder=4)
    ax.add_collection(free_lc)
    ax.add_collection(blocked_lc)

    state = {"first": True}

    def _update(_frame):
        # Drain queue, keep latest
        data = None
        try:
            while True:
                data = queue.get_nowait()
        except Empty:
            pass

        if data is None:
            return []

        dn, de = data["drone_n"], data["drone_e"]
        yaw = data["yaw"]
        bins = data["bins"]           # list of (az_body_rad, blocked)
        chosen = data.get("chosen")   # az in body frame or None
        wps = data["waypoints"]       # [(n, e), ...]
        cur_wp = data["current_wp"]

        # Trail
        trail_x.append(de)
        trail_y.append(dn)
        if len(trail_x) > TRAIL_MAX:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)

        # Drone position
        drone_marker.set_data([de], [dn])

        # Heading indicator
        h_dn, h_de = _body_flu_to_ned(0.0, yaw)  # forward direction
        heading_line.set_data(
            [de, de + DRONE_SIZE * h_de],
            [dn, dn + DRONE_SIZE * h_dn])

        # Waypoints
        if wps:
            wp_e = [w[1] for w in wps]
            wp_n = [w[0] for w in wps]
            wp_scatter.set_offsets(np.column_stack([wp_e, wp_n]))
            if 0 <= cur_wp < len(wps):
                cur_wp_scatter.set_offsets([[wps[cur_wp][1], wps[cur_wp][0]]])

        # Bin lines — separate into free and blocked segments
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

        # Chosen direction
        if chosen is not None:
            c_dn, c_de = _body_flu_to_ned(chosen, yaw)
            chosen_line.set_data(
                [de, de + ARROW_LEN * c_de],
                [dn, dn + ARROW_LEN * c_dn])
            chosen_line.set_visible(True)
        else:
            chosen_line.set_visible(False)

        # Auto-scale on first real data
        if state["first"]:
            state["first"] = False
            if wps:
                all_e = [w[1] for w in wps] + [p[1] for p in PILLAR_NED] + [de]
                all_n = [w[0] for w in wps] + [p[0] for p in PILLAR_NED] + [dn]
                pad = 2.5
                ax.set_xlim(min(all_e) - pad, max(all_e) + pad)
                ax.set_ylim(min(all_n) - pad, max(all_n) + pad)

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
        wps = [(0, 0), (4, 0), (8, 1), (11, 0)]
        while True:
            bins = [(math.radians(a - 180 + 5), a % 70 < 20) for a in range(0, 360, 10)]
            q.put({
                "drone_n": 2 + t * 0.3,
                "drone_e": math.sin(t * 0.5) * 1.5,
                "yaw": 0.0,
                "bins": bins,
                "chosen": 0.1,
                "waypoints": wps,
                "current_wp": min(int(t / 3), len(wps) - 1),
            })
            t += 0.1
            time.sleep(0.1)

    threading.Thread(target=_fake, daemon=True).start()
    run_viz(q)
