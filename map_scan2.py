"""Standalone four-side target scan extracted from main2nos.py.

Run at a stationary target cell. Uses wheel odometry for 10 cm offsets,
Sharp only for scan safety/return centering, and map_scan.py for vision/IR.
It does not navigate the map or select grid goals.
"""
import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time

LOOP_PERIOD_S = .05
DATA_TIMEOUT_S = .75
SHARP_NAV_ENABLED = False
SHARP_WALL_CM = 26.0
SHARP_STOP_CM = 3.0
SHARP_RELEASE_CM = 3.5
SHARP_CLEARANCE_CM = 7.0
SHARP_CLEARANCE_SPEED_MPS = .035
SHARP_CLEARANCE_MAX_TRAVEL_M = .15
SHARP_CENTER_TOLERANCE_CM = 1.0
SHARP_MAX_AGE_S = 1.5
LOG = logging.getLogger("map_scan2")
VERSION = "map-scan2-continuous-30deg-2"
SCAN_SWEEP_OFFSET_DEG = 30.0
SCAN_SWEEP_SPEED_DPS = 35.0
SCAN_SWEEP_SLOW_DPS = 12.0
SCAN_SWEEP_STAGE_TIMEOUT_S = 12.0
SCAN_CAMERA_GAP_S = 3.0

class SafetyStop(RuntimeError):
    """A requested stop that ends navigation cleanly without an error traceback."""

class ScanOffsetBlocked(RuntimeError):
    """A side scan cannot continue safely; return its measured partial offset."""

    def __init__(self, travelled, left_cm, right_cm):
        self.travelled = travelled
        super().__init__('Sharp blocked target-scan offset after %.3f m (left=%s cm, right=%s cm)' % (travelled, left_cm, right_cm))

SHARP_CALIBRATION = {'LEFT_SENSOR': {'a': 0.00019024, 'b': -0.18564551, 'c': 46.21151396, 'min_adc': 103, 'max_adc': 635, 'noise_threshold': 219}, 'RIGHT_SENSOR': {'a': 0.00022938, 'b': -0.21975928, 'c': 55.31746333, 'min_adc': 158, 'max_adc': 584, 'noise_threshold': 368}}

def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0

def limit(value, maximum):
    return max(-maximum, min(maximum, value))

class Telemetry:

    def __init__(self):
        self.lock = threading.Lock()
        self.gimbal = self.position = None
        self.gimbal_time = self.position_time = 0.0
        self.chassis_attitude = None
        self.attitude_time = 0.0

    def on_gimbal(self, values):
        with self.lock:
            self.gimbal = tuple((float(v) for v in values))
            self.gimbal_time = time.monotonic()

    def on_position(self, values):
        with self.lock:
            self.position = tuple((float(v) for v in values))
            self.position_time = time.monotonic()

    def on_attitude(self, values):
        with self.lock:
            self.chassis_attitude = tuple((float(v) for v in values))
            self.attitude_time = time.monotonic()

    def read(self):
        with self.lock:
            now = time.monotonic()
            if self.gimbal is None or self.position is None or now - self.gimbal_time > DATA_TIMEOUT_S or (now - self.position_time > DATA_TIMEOUT_S):
                raise RuntimeError('Missing/stale gimbal or position telemetry')
            if not all((math.isfinite(v) for v in self.gimbal + self.position)):
                raise RuntimeError('Invalid telemetry')
            return (self.gimbal, self.position)

    def read_position(self):
        """Wheel telemetry stays valid while map_scan owns gimbal angles."""
        with self.lock:
            if self.position is None or time.monotonic() - self.position_time > DATA_TIMEOUT_S or (not all((math.isfinite(v) for v in self.position))):
                raise RuntimeError('Missing/stale wheel-position telemetry')
            return self.position

    def read_cached(self):
        """Use the last sensor values only for a stopped/error map snapshot."""
        with self.lock:
            if self.gimbal is None or self.position is None or (not all((math.isfinite(v) for v in self.gimbal + self.position))):
                raise RuntimeError('No valid cached telemetry for stopped map')
            return (self.gimbal, self.position)

    def wait_ready(self):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                return self.read()
            except RuntimeError:
                time.sleep(LOOP_PERIOD_S)
        raise RuntimeError('Telemetry did not become ready within 5 seconds')

def sharp_distance_cm(adc, sensor):
    """Use the supplied quadratic calibration; a missing ADC is unknown."""
    if adc is None or not math.isfinite(float(adc)) or float(adc) <= 0:
        return None
    cfg = SHARP_CALIBRATION[sensor]
    adc = max(cfg['min_adc'], min(float(adc), cfg['max_adc']))
    return max(0.0, cfg['a'] * adc * adc + cfg['b'] * adc + cfg['c'])

class SharpMonitor:
    """Read the side Sharp sensors without delaying the steering loop."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.left_cm = self.right_cm = None
        self.left_adc = self.right_adc = None
        self.sample_time = 0.0
        self.too_close = False
        self.correction_active = False

    def start(self):
        if getattr(self.robot, 'sensor_adaptor', None) is None:
            raise RuntimeError('Sharp sensor adaptor unavailable; cannot enable 3 cm brake')
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.sample_time > 0 and self.left_cm is not None and (self.right_cm is not None)
            if ready:
                return
            time.sleep(0.05)
        raise RuntimeError('No valid readings from both Sharp sensors')

    def _poll(self):
        adaptor = self.robot.sensor_adaptor
        while not self.stop_event.is_set():
            try:
                left_adc = adaptor.get_adc(id=2, port=1)
                left = sharp_distance_cm(left_adc, 'LEFT_SENSOR')
                time.sleep(0.03)
                right_adc = adaptor.get_adc(id=1, port=1)
                right = sharp_distance_cm(right_adc, 'RIGHT_SENSOR')
                close = any((value is not None and value <= SHARP_STOP_CM for value in (left, right)))
                with self.lock:
                    self.left_cm, self.right_cm = (left, right)
                    self.left_adc, self.right_adc = (left_adc, right_adc)
                    self.sample_time = time.monotonic()
                    was_braking = self.too_close
                    clear = left is not None and right is not None and (left > SHARP_RELEASE_CM) and (right > SHARP_RELEASE_CM)
                    self.too_close = close or (was_braking and (not clear))
                    braking = self.too_close
                    correction_active = self.correction_active
                if braking and (not correction_active) and SHARP_NAV_ENABLED:
                    self.robot.chassis.drive_speed(x=0, y=0, z=0)
                if braking and (not was_braking):
                    LOG.warning('SHARP BRAKE left=%s cm (ADC=%s) right=%s cm (ADC=%s), limit=%.1f cm', left, left_adc, right, right_adc, SHARP_STOP_CM)
                elif not braking and was_braking:
                    LOG.info('SHARP BRAKE released left=%.2f cm right=%.2f cm', left, right)
            except Exception as exc:
                LOG.warning('Sharp read unavailable: %s', exc)
            self.stop_event.wait(0.07)

    def read(self):
        with self.lock:
            if self.thread is None:
                return (None, None)
            if time.monotonic() - self.sample_time > SHARP_MAX_AGE_S or self.left_cm is None or self.right_cm is None:
                return (None, None)
            return (self.left_cm, self.right_cm)

    def braking(self):
        with self.lock:
            return self.too_close

    def allow_correction(self, enabled):
        with self.lock:
            self.correction_active = enabled

    def wait_clear(self, on_wait=None):
        while self.braking():
            self.robot.chassis.drive_speed(x=0, y=0, z=0)
            if on_wait is not None:
                on_wait()
            time.sleep(0.1)

    def wait_fresh(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            readings = self.read()
            if readings[0] is not None and readings[1] is not None:
                return readings
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop('Sharp data unavailable for %.1f s: stopped' % timeout)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

def sharp_clearance_speed(left_cm, right_cm, center_single_wall=False):
    """Robot-right is +y. Return zero only when side clearance is safe."""
    left_wall = left_cm <= SHARP_WALL_CM
    right_wall = right_cm <= SHARP_WALL_CM
    if left_wall and right_wall:
        if left_cm + right_cm < 2 * SHARP_RELEASE_CM:
            raise SafetyStop('Both side walls are too close to release Sharp brake; stopped')
        difference = left_cm - right_cm
        if min(left_cm, right_cm) >= SHARP_RELEASE_CM and abs(difference) <= SHARP_CENTER_TOLERANCE_CM:
            return 0.0
        direction = -1 if difference > 0 else 1
    elif left_wall:
        target = 18.0 if center_single_wall else SHARP_CLEARANCE_CM
        if left_cm >= target:
            return 0.0
        direction = 1
    elif right_wall:
        target = 18.0 if center_single_wall else SHARP_CLEARANCE_CM
        if right_cm >= target:
            return 0.0
        direction = -1
    else:
        if center_single_wall:
            raise SafetyStop('Sharp cannot center after target scan: no side wall detected')
        return 0.0
    return direction * SHARP_CLEARANCE_SPEED_MPS

def load_target_scanner(path):
    """Load map_scan.py and its adjacent helper modules only when needed."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError('map_scan.py not found: %s' % path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('main2_map_scan', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load map_scan.py: %s' % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _scan_move_10cm(nav, axis, metres):
    """Move one scan offset using wheel odometry; never rely on SDK move ACK."""
    start = nav.data.read_position()
    started = time.monotonic()
    requested = abs(metres)
    try:
        while True:
            position = nav.data.read_position()
            distance = math.hypot(position[0] - start[0], position[1] - start[1])
            if nav.sharp.braking():
                left_cm, right_cm = nav.sharp.read()
                moving_away = axis == 'y' and (metres > 0 and left_cm is not None and (left_cm <= SHARP_STOP_CM) and (right_cm is None or right_cm > SHARP_STOP_CM) or (metres < 0 and right_cm is not None and (right_cm <= SHARP_STOP_CM) and (left_cm is None or left_cm > SHARP_STOP_CM)))
                if not moving_away:
                    nav.stop_chassis()
                    clear_by = time.monotonic() + 0.35
                    while nav.sharp.braking() and time.monotonic() < clear_by:
                        time.sleep(LOOP_PERIOD_S)
                    position = nav.data.read_position()
                    distance = math.hypot(position[0] - start[0], position[1] - start[1])
                    if nav.sharp.braking():
                        left_cm, right_cm = nav.sharp.read()
                        raise ScanOffsetBlocked(distance, left_cm, right_cm)
            if distance >= requested - 0.005:
                if distance > requested + 0.03:
                    raise SafetyStop('Target-scan offset overshot %.3f/%.3f m' % (distance, requested))
                return distance
            if time.monotonic() - started > 6.0:
                raise SafetyStop('Target-scan offset stalled %.3f/%.3f m' % (distance, requested))
            speed = math.copysign(min(0.12, max(0.04, 1.5 * (requested - distance))), metres)
            nav.robot.chassis.drive_speed(x=speed if axis == 'x' else 0, y=speed if axis == 'y' else 0, z=0, timeout=0.5)
            time.sleep(LOOP_PERIOD_S)
    finally:
        nav.stop_chassis()

def _scan_center_between_two_walls(nav):
    """Sharp is consulted only after returning, and only for two side walls."""
    left_cm, right_cm = nav.sharp.read()
    if left_cm is None or right_cm is None or left_cm > SHARP_WALL_CM or (right_cm > SHARP_WALL_CM):
        LOG.info('TARGET SCAN Sharp centering skipped: two side walls not present')
        return
    start = nav.data.read_position()
    deadline = time.monotonic() + 6.0
    try:
        while time.monotonic() < deadline:
            left_cm, right_cm = nav.sharp.read()
            if left_cm is None or right_cm is None:
                raise SafetyStop('Sharp data lost while centering after target scan')
            if left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM:
                LOG.info('TARGET SCAN Sharp centering ended: two walls no longer detected')
                return
            speed = sharp_clearance_speed(left_cm, right_cm)
            if speed == 0:
                LOG.info('TARGET SCAN Sharp centered left=%.1f right=%.1f cm', left_cm, right_cm)
                return
            position = nav.data.read_position()
            if math.hypot(position[0] - start[0], position[1] - start[1]) > 0.15:
                raise SafetyStop('Sharp scan centering exceeded 15 cm')
            nav.robot.chassis.drive_speed(x=0, y=speed, z=0, timeout=0.5)
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop('Sharp scan centering timed out')
    finally:
        nav.stop_chassis()

SCAN_CAMERA_WINDOW = 'RoboMaster scan'

SCAN_MASK_WINDOW = 'Detection mask'

SCAN_MAP_WINDOW = '1x1 target map'

def _show_scan_camera(scanner, frame, found, direction, offset, ranges=None, target_map=None, requested_codes=None, selected=None, nav=None, state='SCAN'):
    """Render map_scan's three views and saved-target overlays."""
    cv2 = getattr(scanner, 'cv2', None)
    if cv2 is None:
        return
    view = frame.copy()
    height, width = view.shape[:2]
    turn_deg = offset
    pitch_deg = 0.0
    if nav is not None:
        angles, _ = nav.data.read()
        turn_deg = wrap(angles[3] - nav.main_yaw)
        pitch_deg = angles[0]
    marks = []
    if target_map is not None:
        marks = [(item, scanner.project_saved_mark(item, width, height, turn_deg, pitch_deg)) for item in target_map.targets if item.get('scan_direction') == direction]
    for code, (x, y, w, h), cx, _cy in found:
        recorded = any((scanner.mark_inside_box(point, (x, y, w, h), scanner.MARK_MATCH_MARGIN_PX) for _item, point in marks))
        paint = (0, 0, 0) if recorded else scanner.DISPLAY_COLORS[code[1]]
        cv2.rectangle(view, (x, y), (x + w, y + h), paint, -1 if recorded else 2)
        if requested_codes is None:
            label = code
        elif recorded:
            label = 'IGNORED ' + code
        else:
            reason = 'READY' if code in requested_codes else 'NOT REQUESTED'
            label = '%s %s %s %+.1f deg | %s' % (code, scanner.SHAPE_NAMES[code[0]], scanner.COLOR_NAMES[code[1]], scanner.target_bearing(cx, width, turn_deg), reason)
        cv2.putText(view, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if recorded else paint, 2)
    for item, point in marks:
        if point is None:
            continue
        paint = scanner.DISPLAY_COLORS[item['code'][1]]
        cv2.circle(view, point, 9, (0, 0, 0), 3)
        cv2.circle(view, point, 7, paint, -1)
        cv2.putText(view, 'SAVED ' + item['code'], (min(point[0] + 10, width - 110), max(point[1] - 10, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, paint, 2)
    center = (width // 2, height // 2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    if requested_codes is not None:
        accept_x = max(24, round(width * scanner.ACCEPT_HALF_WIDTH_FRACTION))
        accept_y = max(20, round(height * scanner.ACCEPT_HALF_HEIGHT_FRACTION))
        inner_x = max(8, round(width * scanner.INNER_HALF_WIDTH_FRACTION))
        inner_y = max(8, round(height * scanner.INNER_HALF_HEIGHT_FRACTION))
        cv2.rectangle(view, (center[0] - accept_x, center[1] - accept_y), (center[0] + accept_x, center[1] + accept_y), (0, 165, 255), 2)
        cv2.rectangle(view, (center[0] - inner_x, center[1] - inner_y), (center[0] + inner_x, center[1] + inner_y), (0, 255, 0), 1)
    if selected is not None:
        x, y, w, h = selected[1]
        cv2.rectangle(view, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.putText(view, 'FOCUS ' + selected[0], (x, min(y + h + 22, height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(view, '%s | %s %+d deg | Q stop | H reload HSV' % (state, direction, offset), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow(SCAN_CAMERA_WINDOW, view)
    if ranges is not None and target_map is not None:
        cv2.imshow(SCAN_MASK_WINDOW, scanner.draw_detection_mask(frame, ranges, found, marks, selected, state))
        cv2.imshow(SCAN_MAP_WINDOW, scanner.draw_map(target_map))
    key = cv2.waitKey(1) & 255
    if key in (ord('q'), ord('Q'), 27):
        raise SafetyStop('Target scan stopped from camera window')
    if key in (ord('h'), ord('H')) and ranges is not None:
        ranges.clear()
        ranges.update(scanner.load_ranges())
        LOG.info('TARGET SCAN HSV settings reloaded')
    if cv2.getWindowProperty(SCAN_CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        raise SafetyStop('Target scan camera window closed')

def _aim_scan_target(nav, scanner, codes, ranges, target_map, output_path, fired_bearings, direction, offset, code, bearing, all_codes):
    """Use map_scan's continuous target lock and image-error aiming."""
    started = last_seen = time.monotonic()
    try:
        while time.monotonic() - started < scanner.AIM_TIMEOUT_S:
            try:
                frame = nav.robot.camera.read_cv2_image(timeout=0.2, strategy='newest')
            except scanner.Empty:
                frame = None
            if frame is None:
                if time.monotonic() - last_seen > scanner.TARGET_LOST_S:
                    break
                continue
            height, width = frame.shape[:2]
            found = scanner.find_matches(frame, all_codes, ranges)
            angles, _ = nav.data.read()
            turn_deg = wrap(angles[3] - nav.main_yaw)
            chosen = scanner.match_locked_target(found, code, bearing, width, turn_deg)
            _show_scan_camera(scanner, frame, found, direction, offset, ranges, target_map, codes, selected=chosen, nav=nav, state='AIM')
            if chosen is None:
                nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                if time.monotonic() - last_seen > scanner.TARGET_LOST_S:
                    break
                continue
            last_seen = time.monotonic()
            _code, box, cx, cy = chosen
            ex, ey = (cx - width // 2, cy - height // 2)
            accept_x = max(24, round(width * scanner.ACCEPT_HALF_WIDTH_FRACTION))
            accept_y = max(20, round(height * scanner.ACCEPT_HALF_HEIGHT_FRACTION))
            if abs(ex) <= accept_x and abs(ey) <= accept_y:
                nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                chassis_yaw = wrap(angles[3] - angles[1])
                saved = target_map.observe(code, box, cx, cy, width, height, chassis_yaw, angles[1], angles[0], turn_deg)
                saved['scan_direction'] = direction
                saved['scan_offset_deg'] = offset
                scanner.record_and_fire(nav.robot, target_map, saved, output_path)
                fired_bearings.append((code, bearing))
                _show_scan_camera(scanner, frame, found, direction, offset, ranges, target_map, codes, selected=chosen, nav=nav, state='SAVED + IR')
                LOG.info('TARGET SCAN fired requested %s toward %s at bearing %.1f', code, direction, bearing)
                return True
            yaw_speed = limit(ex * scanner.YAW_GAIN, scanner.MAX_YAW_AIM_DPS)
            pitch_speed = limit(-ey * scanner.PITCH_GAIN, scanner.MAX_PITCH_AIM_DPS)
            if pitch_speed < 0:
                pitch_speed = max(pitch_speed, -max(0.0, angles[0] - scanner.GIMBAL_MIN_PITCH_DEG) * 2.0)
            if abs(angles[1]) > 245 and yaw_speed * angles[1] > 0:
                yaw_speed = 0.0
            if nav.robot.gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=nav.gimbal_yaw_command_sign * yaw_speed) is False:
                raise RuntimeError('Gimbal rejected target aiming command')
    finally:
        nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    LOG.info('TARGET SCAN aim lost/timed out: %s toward %s', code, direction)
    return False

def _scan_sweep_direction(nav, scanner, codes, ranges, target_map, output_path,
                          fired_bearings, direction, base, yaw_sign):
    """Process every camera frame while sweeping base -> +30 -> -30."""
    nav._align_point(yaw_sign * base)
    angles, _ = nav.data.read()
    action = nav.robot.gimbal.moveto(
        pitch=getattr(scanner, "HOME_PITCH_DEG", -3.0),
        yaw=nav.gimbal_yaw_command_sign * angles[1],
        pitch_speed=90, yaw_speed=180)
    if action is not None and (not action.wait_for_completed(timeout=5)
                               or not action.has_succeeded):
        raise RuntimeError("Target-scan pitch adjustment failed")
    all_codes = {shape + color for shape in scanner.SHAPE_NAMES
                 for color in scanner.COLOR_NAMES}
    skipped_bearings = []
    pitch_direction = -1
    last_frame_at = time.monotonic()
    try:
        for waypoint in (base + SCAN_SWEEP_OFFSET_DEG,
                         base - SCAN_SWEEP_OFFSET_DEG):
            deadline = time.monotonic() + SCAN_SWEEP_STAGE_TIMEOUT_S
            LOG.info("TARGET SCAN %s sweeping toward main_yaw %+.0f deg",
                     direction, waypoint)
            while time.monotonic() < deadline:
                angles, _ = nav.data.read()
                desired_ground = wrap(nav.main_yaw + yaw_sign * waypoint)
                error = wrap(desired_ground - angles[3])
                try:
                    frame = nav.robot.camera.read_cv2_image(
                        timeout=.2, strategy="newest")
                except scanner.Empty:
                    frame = None
                if frame is not None:
                    last_frame_at = time.monotonic()
                    height, width = frame.shape[:2]
                    found = scanner.find_matches(frame, all_codes, ranges)
                    actual_offset = round(wrap(angles[3] - nav.main_yaw)
                                          / yaw_sign)
                    _show_scan_camera(scanner, frame, found, direction,
                                      actual_offset, ranges, target_map, codes,
                                      nav=nav, state="SWEEP")
                    for code, _box, cx, _cy in found:
                        if code not in codes:
                            continue
                        bearing = scanner.target_bearing(
                            cx, width, wrap(angles[3] - nav.main_yaw))
                        if any(old_code == code and
                               abs(wrap(old_bearing - bearing)) < 8
                               for old_code, old_bearing in
                               fired_bearings + skipped_bearings):
                            continue
                        nav.robot.gimbal.drive_speed(pitch_speed=0,
                                                     yaw_speed=0)
                        if not _aim_scan_target(
                                nav, scanner, codes, ranges, target_map,
                                output_path, fired_bearings, direction,
                                actual_offset, code, bearing, all_codes):
                            skipped_bearings.append((code, bearing))
                        last_frame_at = time.monotonic()
                        # Aiming can turn the head; recompute the remaining
                        # sweep from fresh telemetry on the next iteration.
                        break
                elif time.monotonic() - last_frame_at > SCAN_CAMERA_GAP_S:
                    raise RuntimeError("Target-scan camera stopped producing frames")
                angles, _ = nav.data.read()
                error = wrap(desired_ground - angles[3])
                if abs(error) <= getattr(scanner, "STOP_TOLERANCE_DEG", 2.0):
                    nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    break
                speed = (SCAN_SWEEP_SLOW_DPS if abs(error) < 8.0
                         else SCAN_SWEEP_SPEED_DPS)
                yaw_speed = math.copysign(min(speed, max(4.0, 3.0 * abs(error))),
                                          error)
                pitch_speed, pitch_direction = scanner.scan_pitch_velocity(
                    angles[0], pitch_direction)
                if nav.robot.gimbal.drive_speed(
                        pitch_speed=pitch_speed,
                        yaw_speed=nav.gimbal_yaw_command_sign * yaw_speed) is False:
                    raise RuntimeError("Gimbal rejected sweep command")
                time.sleep(LOOP_PERIOD_S)
            else:
                raise RuntimeError("Target-scan yaw sweep timed out toward %+.0f" %
                                   waypoint)
    finally:
        nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        nav._align_point(yaw_sign * base)

def run_target_scan_four_sides(nav, scanner, requested_codes, output_path, right_yaw_sign):
    """Fixed front/right/left/back scan with a wheel-measured return each time."""
    nav.phase = 'target_scan'
    nav.stop_chassis()
    saved_main_yaw = nav.navigation_yaw()
    start_position = nav.data.read_position()
    ranges = scanner.load_ranges()
    target_map = scanner.TargetMap(saved_main_yaw)
    fired_bearings = []
    center_fallback_directions = []
    sides = (('front', 'x', -0.1, 0), ('right', 'y', -0.1, 90), ('left', 'y', +0.1, -90), ('back', 'x', +0.1, -180))
    streaming = False
    camera_window_open = False
    try:
        nav._align_point(0)
        if nav.robot.camera.start_video_stream(display=False) is False:
            raise RuntimeError('Target-scan camera stream failed')
        streaming = True
        if getattr(scanner, 'cv2', None) is not None:
            scanner.cv2.namedWindow(SCAN_CAMERA_WINDOW, scanner.cv2.WINDOW_NORMAL)
            scanner.cv2.namedWindow(SCAN_MASK_WINDOW, scanner.cv2.WINDOW_NORMAL)
            scanner.cv2.namedWindow(SCAN_MAP_WINDOW, scanner.cv2.WINDOW_NORMAL)
            camera_window_open = True
            preview_deadline = time.monotonic() + 3.0
            while time.monotonic() < preview_deadline:
                try:
                    preview = nav.robot.camera.read_cv2_image(timeout=0.2, strategy='newest')
                except scanner.Empty:
                    preview = None
                if preview is not None:
                    all_codes = {shape + color for shape in scanner.SHAPE_NAMES for color in scanner.COLOR_NAMES}
                    found = scanner.find_matches(preview, all_codes, ranges)
                    _show_scan_camera(scanner, preview, found, 'ready', 0, ranges, target_map, requested_codes, nav=nav, state='READY')
                    break
            else:
                raise RuntimeError('Target-scan camera did not provide an initial frame')
        for direction, axis, move, base in sides:
            nav._align_point(0)
            LOG.info('TARGET SCAN %s: move %+.2f m on %s; sweep base %+.0f -> +30 -> -30', direction, move, axis, base)
            try:
                distance = _scan_move_10cm(nav, axis, move)
            except ScanOffsetBlocked as exc:
                LOG.warning('TARGET SCAN %s offset blocked: %s; returning to scan center', direction, exc)
                if axis == 'x':
                    _scan_center_between_two_walls(nav)
                if exc.travelled >= 0.005:
                    _scan_move_10cm(nav, axis, -math.copysign(exc.travelled, move))
                _scan_center_between_two_walls(nav)
                LOG.info('TARGET SCAN %s scanning from center without offset', direction)
                _scan_sweep_direction(nav, scanner, requested_codes, ranges,
                                      target_map, Path(output_path),
                                      fired_bearings, direction, base,
                                      right_yaw_sign)
                center_fallback_directions.append(direction)
                continue
            try:
                _scan_sweep_direction(nav, scanner, requested_codes, ranges,
                                      target_map, Path(output_path),
                                      fired_bearings, direction, base,
                                      right_yaw_sign)
            finally:
                _scan_move_10cm(nav, axis, -math.copysign(distance, move))
                _scan_center_between_two_walls(nav)
        target_map.save(Path(output_path))
        if getattr(scanner, 'cv2', None) is not None:
            plot_path = Path(output_path).with_suffix('.png')
            if not scanner.cv2.imwrite(str(plot_path), scanner.draw_map(target_map)):
                LOG.warning('TARGET SCAN map plot could not be saved: %s', plot_path)
        diagnostics = Path(output_path).with_name(Path(output_path).stem + '_diagnostics.json')
        diagnostics.write_text(json.dumps({'completed': True, 'directions': [side[0] for side in sides], 'center_fallback_directions': center_fallback_directions, 'confirmed_ir': len(fired_bearings), 'requested': sorted(requested_codes)}, indent=2), encoding='utf-8')
        LOG.info('TARGET SCAN completed: %d requested target(s) fired; continue route', len(fired_bearings))
    finally:
        nav.stop_chassis()
        nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        if streaming:
            nav.robot.camera.stop_video_stream()
        if camera_window_open:
            scanner.cv2.destroyWindow(SCAN_CAMERA_WINDOW)
            scanner.cv2.destroyWindow(SCAN_MASK_WINDOW)
            scanner.cv2.destroyWindow(SCAN_MAP_WINDOW)
        nav.main_yaw = saved_main_yaw
        nav.target_yaw = saved_main_yaw
        nav.target_pitch = 0.0
        nav._align_point(0)
    end_position = nav.data.read_position()
    residual = math.hypot(end_position[0] - start_position[0], end_position[1] - start_position[1])
    if residual > 0.16:
        raise SafetyStop('Target scan left robot %.3f m from cell center' % residual)
    nav.settle(follow=True)
    nav.phase = 'at_cell'

class ScanNavigator:
    """Minimal stopped-robot adapter for the extracted scan functions."""
    def __init__(self, ep_robot, data, main_yaw, gimbal_yaw_sign=1):
        self.robot = ep_robot
        self.data = data
        self.main_yaw = wrap(main_yaw)
        self.target_yaw = self.main_yaw
        self.target_pitch = 0.0
        self.gimbal_yaw_command_sign = gimbal_yaw_sign
        self.phase = "target_scan"
        self.sharp = SharpMonitor(ep_robot)

    def navigation_yaw(self):
        return self.main_yaw

    def stop_chassis(self):
        # drive_speed may return False after the SDK has already sent it.
        self.robot.chassis.drive_speed(x=0, y=0, z=0)

    def _align_point(self, offset_yaw):
        self.stop_chassis()
        self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        desired = wrap(self.main_yaw + offset_yaw)
        for attempt in range(2):
            angles, _ = self.data.read()
            body_ground = wrap(angles[3] - angles[1])
            relative = offset_yaw + wrap(self.main_yaw - body_ground)
            command = self.gimbal_yaw_command_sign * relative
            if abs(command) > 250:
                raise RuntimeError("Gimbal yaw command exceeds 250 degrees")
            action = self.robot.gimbal.moveto(
                pitch=0, yaw=command, pitch_speed=180, yaw_speed=180)
            if action is not None and (not action.wait_for_completed(timeout=5)
                                       or not action.has_succeeded):
                raise RuntimeError("Gimbal motion failed at offset %+.1f" % offset_yaw)
            completed_at = time.monotonic()
            deadline = completed_at + 2.0
            while time.monotonic() < deadline:
                with self.data.lock:
                    fresh = self.data.gimbal_time > completed_at
                if fresh:
                    angles, _ = self.data.read()
                    if (abs(wrap(desired - angles[3])) <= 2.0
                            and abs(angles[2]) <= 1.0):
                        return
                time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("Gimbal did not settle at main_yaw offset %+.1f" % offset_yaw)

    def settle(self, follow=True):
        self.stop_chassis()
        self._align_point(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conn", choices=("ap", "sta"), default="ap")
    parser.add_argument("--map-scan", type=Path,
                        default=Path(__file__).resolve().with_name("map_scan.py"),
                        help="Original map_scan.py with detection, mask and firing helpers")
    parser.add_argument("--targets", help="Shape+color codes, e.g. og,pg,pr")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().with_name("map_scan2_target_map.json"))
    parser.add_argument("--right-yaw-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--gimbal-yaw-sign", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(Path(__file__).with_name(
                                      "map_scan2_run.log"), mode="w", encoding="utf-8")],
                        force=True)
    LOG.info("map_scan2 version %s", VERSION)
    scanner = load_target_scanner(args.map_scan)
    if args.targets is None:
        print("Shapes: u=square, o=circle, i=horizontal rectangle, p=vertical rectangle")
        print("Colors: r=red, y=yellow, g=green, b=blue")
        while True:
            try:
                args.targets = input("Targets to record + IR x1 (e.g. og,pg): ")
                codes = set(scanner.parse_targets(args.targets))
                break
            except ValueError as exc:
                print(exc)
    else:
        try:
            codes = set(scanner.parse_targets(args.targets))
        except ValueError as exc:
            parser.error(str(exc))
    from robomaster import robot
    bot = robot.Robot()
    initialized = False
    subscribed = []
    nav = None
    try:
        bot.initialize(conn_type=args.conn)
        initialized = True
        if not bot.set_robot_mode(mode=robot.FREE):
            raise RuntimeError("Cannot set FREE mode")
        data = Telemetry()
        for subscribe, unsubscribe in (
            (lambda: bot.gimbal.sub_angle(freq=20, callback=data.on_gimbal),
             bot.gimbal.unsub_angle),
            (lambda: bot.chassis.sub_position(cs=1, freq=20,
                                              callback=data.on_position),
             bot.chassis.unsub_position)):
            if not subscribe():
                raise RuntimeError("Scan telemetry subscription failed")
            subscribed.append(unsubscribe)
        data.wait_ready()
        bot.chassis.drive_speed(x=0, y=0, z=0)
        action = bot.gimbal.recenter(pitch_speed=60, yaw_speed=60)
        if action is not None and (not action.wait_for_completed(timeout=10)
                                   or not action.has_succeeded):
            raise RuntimeError("Gimbal recenter failed")
        completed_at = time.monotonic()
        deadline = completed_at + 3.0
        while time.monotonic() < deadline:
            with data.lock:
                fresh = data.gimbal_time > completed_at
            if fresh:
                break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError("No fresh gimbal yaw after recenter")
        main_yaw = wrap(data.read()[0][3])
        LOG.info("SCAN main_yaw=%.2f requested=%s", main_yaw, sorted(codes))
        nav = ScanNavigator(bot, data, main_yaw, args.gimbal_yaw_sign)
        nav.sharp.start()
        nav.sharp.wait_fresh()
        run_target_scan_four_sides(nav, scanner, codes, args.output,
                                   args.right_yaw_sign)
        LOG.info("SCAN complete: %s", args.output)
    finally:
        if initialized:
            for stop in (lambda: bot.chassis.drive_speed(x=0, y=0, z=0),
                         lambda: bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)):
                try:
                    stop()
                except Exception:
                    LOG.exception("Stop command failed")
            if nav is not None:
                nav.sharp.close()
            for unsubscribe in reversed(subscribed):
                try:
                    unsubscribe()
                except Exception:
                    LOG.exception("Unsubscribe failed")
        bot.close()


if __name__ == "__main__":
    main()
