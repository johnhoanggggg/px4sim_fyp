#!/usr/bin/env python3
"""
Read equirectangular panoramic camera images from Gazebo Transport.

Subscribes to /camera/panoramic and exposes the latest frame as a numpy
RGB array.  Follows the same transport pattern as tof_reader.py.

Requires: gz-transport and gz-msgs Python bindings (ship with Gazebo Harmonic),
          numpy.
"""

import threading
import numpy as np

try:
    from gz.transport import Node
except ImportError:
    from gz.transport13 import Node

try:
    from gz.msgs.image_pb2 import Image
except ImportError:
    from gz.msgs10.image_pb2 import Image


class CameraReader:
    """Subscribe to a Gazebo camera topic and buffer the latest frame."""

    def __init__(self, topic: str = "/camera/panoramic"):
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._node = Node()
        self._node.subscribe(Image, topic, self._cb)

    def _cb(self, msg):
        """Gazebo transport callback — runs on internal transport thread."""
        w = msg.width
        h = msg.height
        fmt = msg.pixel_format_type

        # Determine bytes-per-pixel from pixel_format_type enum.
        # Common Gazebo values:  RGB_INT8 = 2,  RGBA_INT8 = ?
        # Fall back to inferring from data length.
        raw = msg.data
        total = len(raw)

        if total == 0 or w == 0 or h == 0:
            return

        bpp = total // (w * h)
        if bpp < 3:
            return  # unsupported format

        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, bpp)

        # Keep only RGB channels (drop alpha if present)
        rgb = arr[:, :, :3].copy()

        with self._lock:
            self._frame = rgb

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recent camera frame as (H, W, 3) uint8, or None."""
        with self._lock:
            return self._frame

    def close(self):
        """Release resources (Node is cleaned up by GC)."""
        with self._lock:
            self._frame = None
