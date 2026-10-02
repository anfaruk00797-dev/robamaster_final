"""Rotate RoboMaster EP in place and map color/shape detections."""

import argparse
import json
import math
import time
from pathlib import Path
from queue import Empty
from threading import Lock

import cv2
import numpy as np
from robomaster import robot, blaster

from target_config import (COLOR_NAMES, SHAPE_NAMES, color_mask, load_ranges,
                           reset_gimbal, GIMBAL_HOME_PITCH_DEG, GIMBAL_MIN_PITCH_DEG)
from target_shooter import DISPLAY_COLORS, find_matches, parse_targets
from scan_motion import align_before_scan, OffsetMove


MAP_SIZE_M = 1.0
ROBOT_X_M = ROBOT_Y_M = MAP_SIZE_M / 2
CAMERA_HFOV_DEG = 96.0
SCAN_SPEED_DPS = 90.0
SLOW_SPEED_DPS = 24.0
SLOW_ZONE_DEG = 20.0
STOP_TOLERANCE_DEG = 2.0
MAX_SAMPLE_AGE_S = 1.0
MAX_SCAN_TIME_S = 90.0
MAX_CAMERA_GAP_S = 3.0
SKIP_BEARING_DEG = 5.0
MARK_MATCH_MARGIN_PX = 18
TRACK_BEARING_DEG = 7.0
INNER_HALF_WIDTH_FRACTION = 18 / 1280
INNER_HALF_HEIGHT_FRACTION = 18 / 720
ACCEPT_HALF_WIDTH_FRACTION = 80 / 1280
ACCEPT_HALF_HEIGHT_FRACTION = 55 / 720
AIM_TIMEOUT_S = 8.0
REACQUIRE_TIMEOUT_S = 3.0
TARGET_LOST_S = 0.7
YAW_GAIN = 0.16
PITCH_GAIN = 0.10
MAX_YAW_AIM_DPS = 40.0
MAX_PITCH_AIM_DPS = 24.0
HOME_PITCH_DEG = GIMBAL_HOME_PITCH_DEG
SCAN_PITCH_LOW_DEG = GIMBAL_MIN_PITCH_DEG
SCAN_PITCH_HIGH_DEG = GIMBAL_HOME_PITCH_DEG
SCAN_PITCH_SPEED_DPS = 12.0


def scan_pitch_velocity(pitch, direction):
    """Bounce between two downward angles using measured gimbal pitch."""
    if pitch <= SCAN_PITCH_LOW_DEG + 1.0:
        direction = 1
    elif pitch >= SCAN_PITCH_HIGH_DEG - 1.0:
        direction = -1
    target = SCAN_PITCH_HIGH_DEG if direction > 0 else SCAN_PITCH_LOW_DEG
    speed = min(SCAN_PITCH_SPEED_DPS, max(2.0, abs(target - pitch) * 2.0))
    return direction * speed, direction


def wrap_angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def ray_to_square(bearing_deg):
    """Intersect a bearing from the fixed robot pose with the 1 m square."""
    radians = math.radians(bearing_deg)
    dx, dy = math.sin(radians), math.cos(radians)
    limits = []
    if abs(dx) > 1e-9:
        limits.append(((MAP_SIZE_M if dx > 0 else 0.0) - ROBOT_X_M) / dx)
    if abs(dy) > 1e-9:
        limits.append(((MAP_SIZE_M if dy > 0 else 0.0) - ROBOT_Y_M) / dy)
    distance = min(t for t in limits if t >= 0)
    return round(ROBOT_X_M + distance * dx, 3), round(ROBOT_Y_M + distance * dy, 3)


def target_bearing(pixel_x, width, turn_deg):
    focal_px = width / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
    camera_offset_deg = math.degrees(math.atan((pixel_x - width / 2) / focal_px))
    return wrap_angle(turn_deg + camera_offset_deg)


def pick_in_sweep_order(candidates, scanning_right):
    if not candidates:
        return None
    return (min(candidates, key=lambda item: item[2]) if scanning_right
            else max(candidates, key=lambda item: item[2]))


def is_ahead(pixel_x, image_center_x, scanning_right):
    """Only inspect the half of the image leading the current sweep."""
    return pixel_x > image_center_x if scanning_right else pixel_x < image_center_x


def project_saved_mark(item, width, height, turn_deg, pitch_deg):
    """Project a stored world direction into the current camera frame."""
    focal_px = width / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
    horizontal = wrap_angle(item["bearing_deg"] - turn_deg)
    vertical = item["elevation_deg"] - pitch_deg
    if abs(horizontal) >= 80 or abs(vertical) >= 80:
        return None
    center_y = height // 2
    px = round(width / 2 + focal_px * math.tan(math.radians(horizontal)))
    py = round(center_y - focal_px * math.tan(math.radians(vertical)))
    if 0 <= px < width and 0 <= py < height:
        return px, py
    return None


def mark_inside_box(point, box, margin=0):
    if point is None:
        return False
    x, y, w, h = box
    return (x - margin <= point[0] <= x + w + margin and
            y - margin <= point[1] <= y + h + margin)


def selection_reason(item, requested_codes, marks, skipped, width, turn_deg, scanning_right):
    code, box, cx, _cy = item
    if code[1] not in {requested[1] for requested in requested_codes}:
        return "COLOR NOT REQUESTED"
    if any(mark_inside_box(point, box, MARK_MATCH_MARGIN_PX) for _saved, point in marks):
        return "ALREADY SAVED"
    if not is_ahead(cx, width // 2, scanning_right):
        return "OTHER HALF"
    bearing = target_bearing(cx, width, turn_deg)
    if any(old_code[1] == code[1]
           and abs(wrap_angle(old_bearing - bearing)) <= SKIP_BEARING_DEG
           for old_code, old_bearing in skipped):
        return "SKIPPED AFTER AIM LOSS"
    return "READY"


def match_locked_target(found, code, locked_bearing, width, turn_deg):
    same = []
    for item in found:
        if item[0] != code:
            continue
        bearing = target_bearing(item[2], width, turn_deg)
        error = abs(wrap_angle(bearing - locked_bearing))
        if error <= TRACK_BEARING_DEG:
            same.append((error, item))
    return min(same, key=lambda pair: pair[0])[1] if same else None


class HeadingTracker:
    def __init__(self):
        self.lock = Lock()
        self.last_yaw = None
        self.unwrapped = 0.0
        self.updated_at = None

    def callback(self, attitude):
        yaw = float(attitude[0])
        with self.lock:
            if self.last_yaw is not None:
                self.unwrapped += wrap_angle(yaw - self.last_yaw)
            self.last_yaw = yaw
            self.updated_at = time.monotonic()

    def read(self):
        with self.lock:
            if self.updated_at is None or time.monotonic() - self.updated_at > MAX_SAMPLE_AGE_S:
                raise RuntimeError("Chassis heading telemetry missing or stale")
            return self.last_yaw, self.unwrapped


class GimbalTracker:
    def __init__(self):
        self.lock = Lock()
        self.yaw = None
        self.pitch_angle = None
        self.updated_at = None

    def callback(self, angles):
        with self.lock:
            self.pitch_angle = float(angles[0])
            self.yaw = float(angles[1])  # yaw relative to the chassis
            self.updated_at = time.monotonic()

    def read(self):
        with self.lock:
            if self.updated_at is None or time.monotonic() - self.updated_at > MAX_SAMPLE_AGE_S:
                raise RuntimeError("Gimbal angle telemetry missing or stale")
            return self.yaw

    def pitch(self):
        with self.lock:
            if self.updated_at is None or time.monotonic() - self.updated_at > MAX_SAMPLE_AGE_S:
                raise RuntimeError("Gimbal angle telemetry missing or stale")
            return self.pitch_angle


class TargetMap:
    def __init__(self, initial_yaw):
        self.initial_yaw = initial_yaw
        self.targets = []

    def observe(self, code, box, pixel_x, pixel_y, width, height,
                chassis_yaw, gimbal_yaw, gimbal_pitch, turn_deg):
        # Positive image x is to the camera's right. The camera follows the gimbal.
        relative_bearing = target_bearing(pixel_x, width, turn_deg)
        map_x, map_y = ray_to_square(relative_bearing)
        focal_px = width / (2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
        image_center_y = height // 2
        elevation = gimbal_pitch + math.degrees(math.atan(
            (image_center_y - pixel_y) / focal_px))
        item = {
            "code": code,
            "shape": SHAPE_NAMES[code[0]],
            "color": COLOR_NAMES[code[1]],
            "bearing_deg": round(relative_bearing, 1),
            "elevation_deg": round(elevation, 2),
            "map_x_m": map_x,
            "map_y_m": map_y,
            "first_chassis_yaw_deg": round(chassis_yaw, 1),
            "last_chassis_yaw_deg": round(chassis_yaw, 1),
            "first_gimbal_yaw_deg": round(gimbal_yaw, 1),
            "gimbal_pitch_deg": round(gimbal_pitch, 1),
            "box_px": list(box),
        }
        self.targets.append(item)
        return item

    def save(self, path):
        document = {
            "map_size_m": [MAP_SIZE_M, MAP_SIZE_M],
            "robot_position_m": [ROBOT_X_M, ROBOT_Y_M],
            "robot_initial_yaw_deg": round(self.initial_yaw, 1),
            "heading_convention": "0=initial forward, positive=right, x=right, y=forward",
            "location_method": "Camera bearing intersected with 1 m square boundary; distance is assumed, not measured",
            "targets": self.targets,
        }
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def draw_map(target_map):
    canvas = np.full((560, 500, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (460, 460), (40, 40, 40), 2)
    for value in (0, 0.5, 1):
        px = 40 + round(value * 420)
        py = 460 - round(value * 420)
        cv2.line(canvas, (px, 40), (px, 460), (220, 220, 220), 1)
        cv2.line(canvas, (40, py), (460, py), (220, 220, 220), 1)
        cv2.putText(canvas, f"{value:g}", (max(25, px - 8), 478),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        cv2.putText(canvas, f"{value:g}", (5, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
    cv2.circle(canvas, (250, 250), 9, (0, 0, 0), -1)
    cv2.putText(canvas, "ROBOT / FRONT UP", (155, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for item in target_map.targets:
        px = 40 + round(item["map_x_m"] * 420)
        py = 460 - round(item["map_y_m"] * 420)
        color = DISPLAY_COLORS[item["code"][1]]
        cv2.circle(canvas, (px, py), 8, color, -1)
        cv2.putText(canvas, item["code"], (min(px + 10, 455), max(py - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"1 x 1 m | Targets: {len(target_map.targets)} | Q: stop | H: reload HSV",
                (20, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
    return canvas


def draw_detection_mask(image, ranges, found, marks, selected, state):
    height, width = image.shape[:2]
    roi_height = height
    hsv = cv2.cvtColor(image[:roi_height], cv2.COLOR_BGR2HSV)
    view = np.zeros_like(image)
    roi = view[:roi_height]
    for color in COLOR_NAMES:
        mask = color_mask(hsv, color, ranges)
        roi[mask > 0] = DISPLAY_COLORS[color]
    for code, box, _cx, _cy in found:
        x, y, w, h = box
        recorded = any(mark_inside_box(point, box, MARK_MATCH_MARGIN_PX)
                       for _saved, point in marks)
        if recorded:
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 0), -1)
        else:
            cv2.rectangle(view, (x, y), (x + w, y + h), DISPLAY_COLORS[code[1]], 2)
        cv2.putText(view, ("IGNORED " if recorded else "DETECT ") + code,
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
    if selected is not None:
        x, y, w, h = selected[1]
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.putText(view, "FOCUS " + selected[0], (x, min(y + h + 22, height - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(view, f"MASK | {state} | red yellow green blue", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def record_and_fire(bot, target_map, saved, output_path):
    """Persist the target before one IR attempt; never retry an uncertain shot."""
    saved["ir_status"] = "pending"
    target_map.save(output_path)
    try:
        result = bot.blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
    except Exception:
        saved["ir_status"] = "unknown"
        target_map.save(output_path)
        raise
    saved["ir_status"] = "confirmed" if result is True else "unconfirmed"
    target_map.save(output_path)
    if result is not True:
        raise RuntimeError("IR command not confirmed; stopped without retry")


def run(output_path, right_yaw_sign, requested_codes):
    ranges = load_ranges()
    all_codes = {shape + color for shape in SHAPE_NAMES for color in COLOR_NAMES}
    bot = robot.Robot()
    chassis_tracker = HeadingTracker()
    gimbal_tracker = GimbalTracker()
    chassis_subscribed = False
    gimbal_subscribed = False
    streaming = False
    target_map = None
    completed = False
    plot_path = output_path.with_suffix(".png")
    plot_saved = False
    bot.initialize(conn_type="ap")
    try:
        print("Alignment before detection...")
        alignment_results = align_before_scan(bot)
        reset_gimbal(bot.gimbal)
        chassis_subscribed = bot.chassis.sub_attitude(freq=20, callback=chassis_tracker.callback)
        if not chassis_subscribed:
            raise RuntimeError("Could not subscribe to chassis attitude")
        gimbal_subscribed = bot.gimbal.sub_angle(freq=20, callback=gimbal_tracker.callback)
        if not gimbal_subscribed:
            raise RuntimeError("Could not subscribe to gimbal angles")
        deadline = time.monotonic() + 5
        while (chassis_tracker.updated_at is None or gimbal_tracker.updated_at is None) and time.monotonic() < deadline:
            time.sleep(0.05)
        initial_yaw, _ = chassis_tracker.read()
        origin = gimbal_tracker.read()
        target_map = TargetMap(initial_yaw)
        if bot.camera.start_video_stream(display=False) is False:
            raise RuntimeError("Camera stream failed")
        streaming = True

        # Show a live camera image before sending the first turn command.
        cv2.namedWindow("RoboMaster scan", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detection mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("1x1 target map", cv2.WINDOW_NORMAL)
        deadline = time.monotonic() + 5
        while True:
            try:
                first_image = bot.camera.read_cv2_image(timeout=0.2, strategy="newest")
            except Empty:
                first_image = None
            if first_image is not None:
                height, width = first_image.shape[:2]
                first_found = find_matches(first_image, all_codes, ranges)
                first_mask = draw_detection_mask(first_image, ranges, first_found,
                                                 [], None, "READY")
                for code, box, center_x, _center_y in first_found:
                    x, y, w, h = box
                    paint = DISPLAY_COLORS[code[1]]
                    cv2.rectangle(first_image, (x, y), (x + w, y + h), paint, 2)
                    cv2.putText(first_image, f"{code} {SHAPE_NAMES[code[0]]} {COLOR_NAMES[code[1]]}",
                                (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, paint, 2)
                cv2.imshow("RoboMaster scan", first_image)
                cv2.imshow("Detection mask", first_mask)
                cv2.imshow("1x1 target map", draw_map(target_map))
                cv2.waitKey(1)
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Camera did not provide a frame before scan")

        # Gimbal yaw relative to the stationary chassis.
        waypoints = (origin + right_yaw_sign * 180,
                     origin,
                     origin - right_yaw_sign * 180)
        phase_names = ("RIGHT TO REAR", "LEFT TO FRONT", "LEFT TO REAR")
        phase = 0
        phase_started = time.monotonic()
        last_frame_at = time.monotonic()
        pitch_direction = -1
        state = "SCAN"
        selected_code = None
        selected_bearing = None
        aim_started = None
        last_seen_at = None
        skipped = []
        offset_move = OffsetMove(bot)
        home_observation = None
        pending_error = None
        selected_color = None
        last_event = "Waiting for a requested target"
        last_report_at = -float("inf")
        detected_counts = {}
        reason_counts = {}
        aim_count = 0
        stop_reason = "scan_complete"
        print("Gimbal scanning and centering targets. Q=stop, H=reload HSV.")
        for code in sorted(requested_codes):
            print(f"Requested: {code} = {SHAPE_NAMES[code[0]]} / {COLOR_NAMES[code[1]]}")
        while phase < len(waypoints):
            now = time.monotonic()
            chassis_yaw, _ = chassis_tracker.read()
            gimbal_yaw = gimbal_tracker.read()
            gimbal_pitch = gimbal_tracker.pitch()
            remaining = waypoints[phase] - gimbal_yaw
            try:
                image = bot.camera.read_cv2_image(timeout=0.2, strategy="newest")
            except Empty:
                image = None
            if image is not None:
                # Refresh telemetry after the potentially blocking frame read.
                now = time.monotonic()
                chassis_yaw, _ = chassis_tracker.read()
                gimbal_yaw = gimbal_tracker.read()
                gimbal_pitch = gimbal_tracker.pitch()
                remaining = waypoints[phase] - gimbal_yaw
                last_frame_at = time.monotonic()
                raw_image = image.copy()
                height, width = image.shape[:2]
                found = find_matches(image, all_codes, ranges)
                turn_deg = right_yaw_sign * (gimbal_yaw - origin)
                marks = [(item, project_saved_mark(item, width, height,
                                                    turn_deg, gimbal_pitch))
                         for item in target_map.targets]
                # Stored image rays belong to the home position, not the
                # temporarily translated camera. Restore them after return.
                if offset_move.away or state in ("MOVE_OUT", "MOVE_BACK"):
                    marks = []
                reports = set()
                for detection in found:
                    code = detection[0]
                    detected_counts[code] = detected_counts.get(code, 0) + 1
                    reason = selection_reason(detection, requested_codes, marks, skipped,
                                              width, turn_deg, right_yaw_sign * remaining > 0)
                    reports.add(code + ": " + (reason if state == "SCAN" else state))
                    if state == "SCAN":
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if now - last_report_at >= 1.0:
                    print("Detected: " + (" | ".join(sorted(reports)) if reports else "NONE") +
                          f" | yaw={gimbal_yaw:.1f} pitch={gimbal_pitch:.1f}")
                    last_report_at = now
                for code, box, center_x, center_y in found:
                    x, y, w, h = box
                    recorded = any(mark_inside_box(point, box, MARK_MATCH_MARGIN_PX)
                                   for _item, point in marks)
                    paint = (0, 0, 0) if recorded else DISPLAY_COLORS[code[1]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), paint,
                                  -1 if recorded else 2)
                    bearing = target_bearing(center_x, width, turn_deg)
                    label = (f"IGNORED {code}" if recorded else
                             f"{code} {SHAPE_NAMES[code[0]]} {COLOR_NAMES[code[1]]} {bearing:+.1f} deg")
                    reason = selection_reason((code, box, center_x, center_y), requested_codes,
                                              marks, skipped, width, turn_deg,
                                              right_yaw_sign * remaining > 0)
                    if state == "SCAN":
                        label += " | " + reason
                    cv2.putText(image, label,
                                (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255) if recorded else paint, 2)
                for item, point in marks:
                    if point is not None:
                        paint = DISPLAY_COLORS[item["code"][1]]
                        cv2.circle(image, point, 9, (0, 0, 0), 3)
                        cv2.circle(image, point, 7, paint, -1)
                        cv2.putText(image, "SAVED " + item["code"],
                                    (min(point[0] + 10, width - 110), max(point[1] - 10, 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, paint, 2)
                # The cropped detection ROI does not change the camera center.
                center = (width // 2, height // 2)
                inner_x = max(8, round(width * INNER_HALF_WIDTH_FRACTION))
                inner_y = max(8, round(height * INNER_HALF_HEIGHT_FRACTION))
                accept_x = max(24, round(width * ACCEPT_HALF_WIDTH_FRACTION))
                accept_y = max(20, round(height * ACCEPT_HALF_HEIGHT_FRACTION))
                cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.rectangle(image, (center[0] - accept_x, center[1] - accept_y),
                              (center[0] + accept_x, center[1] + accept_y),
                              (0, 165, 255), 2)
                cv2.rectangle(image, (center[0] - inner_x, center[1] - inner_y),
                              (center[0] + inner_x, center[1] + inner_y),
                              (0, 255, 0), 1)
                chosen = None
                if state in ("MOVE_OUT", "MOVE_BACK") and offset_move.poll():
                    if state == "MOVE_BACK":
                        if pending_error is not None:
                            raise pending_error
                        state = "RESUME_SCAN"
                        phase_started = now
                    else:
                        state = "REACQUIRE"
                        aim_started = now
                        last_seen_at = now
                if state == "REACQUIRE":
                    exact = [item for item in found if item[0] in requested_codes
                             and item[0][1] == selected_color
                             and abs(wrap_angle(target_bearing(item[2], width, turn_deg)
                                                - selected_bearing)) < 20]
                    if exact:
                        chosen = min(exact, key=lambda item: abs(wrap_angle(
                            target_bearing(item[2], width, turn_deg) - selected_bearing)))
                        selected_code = chosen[0]
                        selected_bearing = target_bearing(chosen[2], width, turn_deg)
                        state = "AIM"
                        last_seen_at = now
                    elif now - aim_started >= REACQUIRE_TIMEOUT_S:
                        last_event = "Requested shape not confirmed; returning without IR"
                        skipped.append((home_observation[0], home_observation[2]))
                        offset_move.begin(returning=True)
                        state = "MOVE_BACK"
                if state == "SCAN":
                    candidates = []
                    scanning_right = right_yaw_sign * remaining > 0
                    for item in found:
                        code, _box, cx, cy = item
                        if selection_reason(item, requested_codes, marks, skipped,
                                            width, turn_deg, scanning_right) != "READY":
                            continue
                        candidates.append(item)
                    if candidates:
                        # Sweep order: when turning right, center the leftmost
                        # visible target first; when turning left, do the reverse.
                        chosen = pick_in_sweep_order(candidates, scanning_right)
                        selected_code = chosen[0]
                        selected_color = selected_code[1]
                        selected_bearing = target_bearing(chosen[2], width,
                                                          right_yaw_sign * (gimbal_yaw - origin))
                        home_observation = (selected_code,
                            (chosen[1], chosen[2], chosen[3], width, height,
                             chassis_yaw, gimbal_yaw, gimbal_pitch, turn_deg),
                            selected_bearing)
                        aim_started = now
                        last_seen_at = now
                        state = "MOVE_OUT"
                        aim_count += 1
                        print(f"AIM START: {selected_code}")
                        last_event = "Aiming " + selected_code + " into orange box"
                        bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                        offset_move.begin(target_bearing(chosen[2], width,
                                                         right_yaw_sign * gimbal_yaw))
                        last_event = "Step away 10 cm, then confirm shape and aim"
                if state == "AIM":
                    # A target's map bearing stays nearly constant as the gimbal
                    # turns. Lock to that bearing so a neighbor with the same
                    # color and shape cannot steal the aim.
                    chosen = match_locked_target(
                        found, selected_code, selected_bearing, width,
                        right_yaw_sign * (gimbal_yaw - origin))
                    if chosen is not None and any(
                            mark_inside_box(point, chosen[1], MARK_MATCH_MARGIN_PX)
                            for _saved, point in marks):
                        bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                        offset_move.begin(returning=True)
                        state = "MOVE_BACK"
                        phase_started = now
                        chosen = None
                    if chosen is not None:
                        code, box, cx, cy = chosen
                        last_seen_at = now
                        ex, ey = cx - center[0], cy - center[1]
                        x, y, w, h = box
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
                        cv2.line(image, center, (cx, cy), (255, 0, 255), 2)
                        cv2.putText(image, f"LOCKED {code} | error {ex:+d},{ey:+d} px",
                                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 0, 255), 2)
                        if abs(ex) <= accept_x and abs(ey) <= accept_y:
                            bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                            # Keep the map ray at the original camera position;
                            # record the actual firing pose separately.
                            saved = target_map.observe(code, *home_observation[1])
                            saved["firing_gimbal_yaw_deg"] = gimbal_yaw
                            saved["firing_gimbal_pitch_deg"] = gimbal_pitch
                            saved["chassis_offset_m"] = list(offset_move.offset)
                            try:
                                record_and_fire(bot, target_map, saved, output_path)
                                print(f"Saved {code}; IR x1 confirmed; returning 10 cm")
                                last_event = "IR x1 CONFIRMED: " + code
                            except Exception as exc:
                                pending_error = exc
                                last_event = "IR unconfirmed; return then stop, no retry"
                            offset_move.begin(returning=True)
                            state = "MOVE_BACK"
                            phase_started = now
                        else:
                            yaw_speed = right_yaw_sign * max(-MAX_YAW_AIM_DPS,
                                min(MAX_YAW_AIM_DPS, ex * YAW_GAIN))
                            pitch_speed = max(-MAX_PITCH_AIM_DPS,
                                min(MAX_PITCH_AIM_DPS, -ey * PITCH_GAIN))
                            # Slow down near the lower clearance limit and stop
                            # downward aiming there; upward correction stays available.
                            if pitch_speed < 0:
                                pitch_speed = max(pitch_speed,
                                    -max(0.0, gimbal_pitch - GIMBAL_MIN_PITCH_DEG) * 2.0)
                            if abs(gimbal_yaw) > 245 and yaw_speed * gimbal_yaw > 0:
                                yaw_speed = 0
                            if bot.gimbal.drive_speed(pitch_speed=pitch_speed,
                                                      yaw_speed=yaw_speed) is False:
                                raise RuntimeError("Gimbal rejected aim command")
                    else:
                        bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    if state == "AIM" and (now - aim_started > AIM_TIMEOUT_S or
                                          now - last_seen_at > TARGET_LOST_S):
                        last_event = ("AIM TIMEOUT: " if now - aim_started > AIM_TIMEOUT_S
                                      else "TARGET LOST: ") + selected_code
                        print(last_event + "; IR not sent")
                        skipped.append((home_observation[0], home_observation[2]))
                        offset_move.begin(returning=True)
                        state = "MOVE_BACK"
                        phase_started = now
                cv2.putText(image, f"{state} | {phase_names[phase]} | yaw {gimbal_yaw:+.1f} | pitch {gimbal_pitch:+.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, "RECORD + IR x1: " + ",".join(sorted(requested_codes)),
                            (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.putText(image, last_event, (10, height - 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("RoboMaster scan", image)
                cv2.imshow("Detection mask", draw_detection_mask(
                    raw_image, ranges, found, marks, chosen, state))
            elif time.monotonic() - last_frame_at > MAX_CAMERA_GAP_S:
                raise RuntimeError("Camera frames stopped during scan")
            else:
                bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                cv2.imshow("1x1 target map", draw_map(target_map))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_reason = "user_quit"
                    break
                if key == ord("h"):
                    ranges = load_ranges()
                    print("HSV settings reloaded")
                continue

            if state == "RESUME_SCAN":
                bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                state = "SCAN"
                phase_started = now
            elif state == "SCAN":
                if abs(remaining) <= STOP_TOLERANCE_DEG:
                    bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    phase += 1
                    phase_started = now
                    if phase == 1:
                        skipped.clear()
                    if phase == 2:
                        skipped.clear()
                elif now - phase_started > MAX_SCAN_TIME_S:
                    raise RuntimeError("Scan turn timed out")
                else:
                    sign = 1 if remaining > 0 else -1
                    speed = SLOW_SPEED_DPS if abs(remaining) < SLOW_ZONE_DEG else SCAN_SPEED_DPS
                    pitch_speed, pitch_direction = scan_pitch_velocity(
                        gimbal_pitch, pitch_direction)
                    if bot.gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=sign * speed) is False:
                        raise RuntimeError("Gimbal rejected scan command")
            cv2.imshow("1x1 target map", draw_map(target_map))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_reason = "user_quit"
                break
            if key == ord("h"):
                ranges = load_ranges()
                print("HSV settings reloaded")
        completed = phase >= len(waypoints)
        diagnostic_path = output_path.with_name(output_path.stem + "_diagnostics.json")
        diagnostic_path.write_text(json.dumps({
            "requested_codes": sorted(requested_codes), "stop_reason": stop_reason,
            "completed": completed, "detected_frame_counts": detected_counts,
            "alignment_results": alignment_results,
            "selection_reason_counts": reason_counts, "aim_attempts": aim_count,
            "recorded_targets": len(target_map.targets),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"End: {stop_reason}; detected codes={','.join(sorted(detected_counts)) or 'NONE'}; aim attempts={aim_count}")
        print(f"Saved {len(target_map.targets)} mapped target(s) to {output_path}")
    finally:
        try:
            try:
                bot.chassis.drive_speed(x=0, y=0, z=0)
            finally:
                bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        finally:
            if target_map is not None:
                target_map.save(output_path)
                plot_saved = cv2.imwrite(str(plot_path), draw_map(target_map))
            if streaming:
                bot.camera.stop_video_stream()
            if gimbal_subscribed:
                bot.gimbal.unsub_angle()
            if chassis_subscribed:
                bot.chassis.unsub_attitude()
            bot.close()
            cv2.destroyAllWindows()
    if not plot_saved:
        raise IOError(f"Could not save map plot: {plot_path}")
    if completed:
        print(f"Map plot: {plot_path}")
        cv2.imshow("Final target map - press any key to close", draw_map(target_map))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("target_map.json"))
    parser.add_argument("--right-yaw-sign", type=int, choices=(-1, 1), default=1,
                        help="Use -1 if negative gimbal yaw speed turns right on your robot")
    parser.add_argument("--targets", help="Shape+color codes, e.g. ub,or,pg; prompts if omitted")
    args = parser.parse_args()
    if args.targets is not None:
        try:
            requested = set(parse_targets(args.targets))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        print("Shapes: u=square, o=circle, i=horizontal rectangle, p=vertical rectangle")
        print("Colors: r=red, y=yellow, g=green, b=blue")
        while True:
            try:
                requested = set(parse_targets(input("Targets to record + IR x1 (e.g. ub,or): ")))
                break
            except ValueError as exc:
                print(exc)
    run(args.output, args.right_yaw_sign, requested)
