"""Shared target definitions and HSV settings for the RoboMaster project."""

import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).with_name("hsv_ranges.json")
GIMBAL_HOME_PITCH_DEG = -3  # Raised 5 degrees for CAN bus clearance.
GIMBAL_MIN_PITCH_DEG = -15.0  # Shared lower limit for scanning and aiming.


def reset_gimbal(gimbal):
    """Return yaw to center, then tilt slightly down for the target row."""
    gimbal.recenter().wait_for_completed()
    gimbal.move(pitch=GIMBAL_HOME_PITCH_DEG, yaw=0).wait_for_completed()

DEFAULT_RANGES = {
    "r": [[0, 121, 97, 14, 255, 255]],
    "y": [[0, 255, 127, 40, 255, 255]],
    "g": [[48, 201, 0, 102, 255, 84]],
    "b": [[107, 83, 0, 151, 237, 90]],
}

COLOR_NAMES = {"r": "RED", "y": "YELLOW", "g": "GREEN", "b": "BLUE"}
SHAPE_NAMES = {"o": "CIRCLE 7cm", "i": "RECTANGLE 9x6cm",
               "p": "PORTRAIT RECTANGLE 6x9cm", "u": "SQUARE 7x7cm"}


def load_ranges():
    if not CONFIG_PATH.exists():
        save_ranges(DEFAULT_RANGES)
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for color, ranges in data.items():
        if color not in COLOR_NAMES or not isinstance(ranges, list):
            raise ValueError("Invalid HSV configuration")
        for values in ranges:
            if len(values) != 6 or any(not isinstance(v, int) for v in values):
                raise ValueError(f"Invalid HSV range for {color}")
            lo, hi = values[:3], values[3:]
            if any(lo[k] > hi[k] for k in range(3)):
                raise ValueError(f"HSV lower exceeds upper for {color}")
    return data


def save_ranges(data):
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def color_mask(hsv, color, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for values in ranges[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(
            hsv, np.array(values[:3], dtype=np.uint8),
            np.array(values[3:], dtype=np.uint8)))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # The supplied yellow range overlaps red (H 0..14 at S 255).
    # Assign shared pixels to red so a red shape cannot count as yellow.
    if color == "y" and "r" in ranges:
        red_mask = color_mask(hsv, "r", ranges)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(red_mask))
    return mask
