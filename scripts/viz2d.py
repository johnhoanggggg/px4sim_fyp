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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

BIN_LINE_LEN = 0.6   # length of histogram bin indicator lines
PILLAR_RADIUS = 0.15
DRONE_SIZE = 0.3
TRAIL_MAX = 500


def run_viz(queue: mp.Queue):
    """Entry point — call from a multiprocessing.Process."""
    fig, ax = plt.subplots(figsize=(7, 9))
    fig.canvas.manager.set_window_title("VFH2D Top-Down View")

    # Static elements — pillars
    for (pn, pe) in PILLAR_NED:
        circ = plt.Circle((pe, pn), PILLAR_RADIUS, color="0.55", zorder=2)
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
    trail_line, = ax.plot([], [], "b-", lw=0.8, alpha=0.4, zorder=1)
    drone_marker, = ax.plot([], [], "bs", ms=6, zorder=5)
    heading_line, = ax.plot([], [], "b-", lw=2, zorder=5)
    wp_scatter = ax.scatter([], [], marker="D", c="dodgerblue", s=50, zorder=3)
    cur_wp_scatter = ax.scatter([], [], marker="*", c="red", s=150, zorder=4)
    chosen_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                               zorder=6)
    bin_lines = []  # will be populated on first frame

    state = {"first": True}

    def _update(_frame):
        # Drain queue, keep latest
        data = None
        try:
            while True:
                data = queue.get_nowait()
        except Exception:
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
        hx = de + DRONE_SIZE * math.sin(yaw)  # yaw in NED: 0=north, CW
        hy = dn + DRONE_SIZE * math.cos(yaw)
        heading_line.set_data([de, hx], [dn, hy])

        # Waypoints
        if wps:
            wp_e = [w[1] for w in wps]
            wp_n = [w[0] for w in wps]
            wp_scatter.set_offsets(np.column_stack([wp_e, wp_n]))
            if 0 <= cur_wp < len(wps):
                cur_wp_scatter.set_offsets([[wps[cur_wp][1], wps[cur_wp][0]]])

        # Clear old bin lines
        for ln in bin_lines:
            ln.remove()
        bin_lines.clear()

        # Draw bin indicators
        for az_body, blocked in bins:
            # Body FLU azimuth → world azimuth
            # In body FLU: az=0 forward, +ve=left
            # Yaw in NED: 0=north, +ve CW
            # World north component = cos(yaw) * cos(az_body) - sin(yaw) * sin(az_body)
            #   (because body forward rotated by yaw, and body left is -sin_az in FRD)
            # Actually: body FLU fwd = (cos_yaw, sin_yaw) in NED (N, E)
            #           body FLU left = (-sin_yaw, cos_yaw) in NED
            # direction_ned = cos(az) * fwd + sin(az) * left
            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
            dx_n = math.cos(az_body) * c_yaw + math.sin(az_body) * (-s_yaw)
            dx_e = math.cos(az_body) * s_yaw + math.sin(az_body) * c_yaw
            color = "red" if blocked else "limegreen"
            lw = 2.5 if blocked else 1.0
            ln, = ax.plot(
                [de, de + BIN_LINE_LEN * dx_e],
                [dn, dn + BIN_LINE_LEN * dx_n],
                color=color, lw=lw, alpha=0.7, zorder=4,
            )
            bin_lines.append(ln)

        # Chosen direction arrow
        if chosen is not None:
            c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
            cx_n = math.cos(chosen) * c_yaw + math.sin(chosen) * (-s_yaw)
            cx_e = math.cos(chosen) * s_yaw + math.sin(chosen) * c_yaw
            arrow_len = BIN_LINE_LEN * 1.5
            chosen_arrow.xy = (de + arrow_len * cx_e, dn + arrow_len * cx_n)
            chosen_arrow.set_position((de, dn))
            chosen_arrow.set_visible(True)
        else:
            chosen_arrow.set_visible(False)

        # Auto-scale on first real data
        if state["first"]:
            state["first"] = False
            if wps:
                all_e = [w[1] for w in wps] + [p[1] for p in PILLAR_NED] + [de]
                all_n = [w[0] for w in wps] + [p[0] for p in PILLAR_NED] + [dn]
                pad = 2.0
                ax.set_xlim(min(all_e) - pad, max(all_e) + pad)
                ax.set_ylim(min(all_n) - pad, max(all_n) + pad)

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
