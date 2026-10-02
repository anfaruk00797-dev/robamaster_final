"""Align RoboMaster chassis yaw to walls using a gimbal-mounted ToF.

Run: python alig.py (RoboMaster Python SDK required).
Assumes a level ToF, its optical axis aligned with gimbal yaw, and both
offset rays hitting the SAME flat wall. Four walls should be parallel or
perpendicular; aligning a later non-orthogonal wall can undo earlier alignment.
No chassis translation is commanded: every chassis call has x=0, y=0.
"""

import argparse
import logging
import math
import statistics
import threading
import time

# -------- Tune these values for the robot and wall --------
CONNECTION_TYPE = "ap"  # "sta" for a robot on the same network
CHECK_OFFSET_DEG = 10.0
DIST_TOLERANCE_M = 0.004  # 4 mm; ToF readings in the log resolve to whole mm.
DIST_TOLERANCE_MM = DIST_TOLERANCE_M * 1000
PID_KP = 0.25  # Small proportional corrections avoid crossing the wall normal.
PID_KI = 0.0
PID_KD = 0.0
PID_INTEGRAL_LIMIT_DEG_S = 30.0
MAX_ROTATION_STEP_DEG = 3.0
MAX_TOTAL_CORRECTION_DEG = 24.0  # per direction, cumulative absolute rotation
MAX_CORRECTIONS_PER_DIRECTION = 8
REQUIRED_GOOD_PAIRS = 1
SETTLE_TIME_S = 0.04
CHASSIS_SETTLE_TIME_S = 0.08
GIMBAL_SPEED_DEG_S = 540
CHASSIS_YAW_SPEED_DEG_S = 120
ACTION_TIMEOUT_S = 5
SCAN_DIRECTION_BUDGET_S = 2.0
TOF_FREQUENCY_HZ = 50
TOF_INDEX = 0  # select the connected sensor from the SDK distance tuple
SAMPLES_PER_READING = 3
READ_TIMEOUT_S = 2.0
MIN_DISTANCE_MM = 50
MAX_DISTANCE_MM = 3000
WALL_DETECTION_MAX_MM = 300
MAX_SAMPLE_SPREAD_MM = 60
# Distance from yaw pivot to ToF optical origin along the beam. Set from
# mounting geometry; zero assumes sensor origin at the pivot. Lateral offset
# is not modelled and must be calibrated/mechanically minimized.
TOF_RADIAL_OFFSET_MM = 0.0
# SDK yaw conventions: positive gimbal yaw points right; adjust these only
# if verified hardware coordinates/mounting differ.
RIGHT_GIMBAL_YAW_SIGN = 1
# Official SDK example: positive chassis z turns left, negative turns right.
CHASSIS_YAW_PER_GIMBAL_YAW_SIGN = -1
DIRECTIONS = (("front", 0.0), ("right", 90.0),
              ("back", 180.0), ("left", -90.0))
DIRECTION_ROTATION_SIGN = {"front": 1, "right": 1, "back": -1, "left": 1}

LOG = logging.getLogger("alignment")


class ToFReadingError(RuntimeError):
    """ToF callbacks arrived, but the readings could not be used."""


class YawPID:
    def __init__(self):
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None

    def update(self, error, now=None):
        now = time.monotonic() if now is None else now
        dt = 0.0 if self.previous_time is None else max(0.0, now - self.previous_time)
        derivative = 0.0
        crossed = self.previous_error is not None and error * self.previous_error <= 0
        if crossed:
            self.integral = 0.0
        elif dt > 0 and self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        candidate = max(-PID_INTEGRAL_LIMIT_DEG_S,
                        min(PID_INTEGRAL_LIMIT_DEG_S, self.integral + error * dt))
        output = PID_KP * error + PID_KI * candidate + PID_KD * derivative
        if abs(output) <= MAX_ROTATION_STEP_DEG or output * error < 0:
            self.integral = candidate
        else:
            output = PID_KP * error + PID_KI * self.integral + PID_KD * derivative
        if output * error <= 0 and error != 0:
            output = PID_KP * error
        self.previous_error = error
        self.previous_time = now
        return max(-MAX_ROTATION_STEP_DEG, min(MAX_ROTATION_STEP_DEG, output))


class DistanceReader:
    """Collect distinct callback samples; never reuse a pre-movement reading."""

    def __init__(self):
        self.condition = threading.Condition()
        self.sequence = 0
        self.samples = []
        self.last_raw = None
        self.last_callback_time = None

    def callback(self, distances):
        with self.condition:
            self.sequence += 1
            self.last_raw = distances
            self.last_callback_time = time.monotonic()
            try:
                value = float(distances[TOF_INDEX])
            except (IndexError, TypeError, ValueError):
                value = float("nan")
            self.samples.append((self.sequence, value))
            self.samples = self.samples[-200:]
            self.condition.notify_all()

    def read(self, scan=False, deadline=None):
        deadline = min(time.monotonic() + READ_TIMEOUT_S,
                       deadline if deadline is not None else float("inf"))
        with self.condition:
            start = self.sequence  # called AFTER motion completion and settling
            while True:
                values = [v for seq, v in self.samples if seq > start
                          and math.isfinite(v)
                          and MIN_DISTANCE_MM <= v <= (65534 if scan else MAX_DISTANCE_MM)]
                if len(values) >= SAMPLES_PER_READING:
                    values = values[:SAMPLES_PER_READING]
                    if max(values) - min(values) > MAX_SAMPLE_SPREAD_MM:
                        raise ToFReadingError("Unstable ToF samples; check wall/beam obstruction")
                    return statistics.median(values)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ToFReadingError(
                        "No fresh valid ToF data within timeout; callbacks=%d, "
                        "valid=%d/%d, raw=%r" %
                        (self.sequence - start, len(values),
                         SAMPLES_PER_READING, self.last_raw))
                self.condition.wait(remaining)


def correction_from_distances(plus_mm, minus_mm):
    """Return yaw toward wall normal in gimbal coordinates (degrees).

    For a plane, r+ cos(offset-alpha) = r- cos(offset+alpha).
    Thus tan(alpha) = (r- - r+) / ((r+ + r-) tan(offset)).
    A radial mounting offset is added to recover ray lengths from yaw pivot.
    Equal readings imply the base beam is perpendicular to the wall.
    """
    plus = plus_mm + TOF_RADIAL_OFFSET_MM
    minus = minus_mm + TOF_RADIAL_OFFSET_MM
    return math.degrees(math.atan2(
        minus - plus,
        (plus + minus) * math.tan(math.radians(CHECK_OFFSET_DEG))))


def wait_action(action, label, timeout=ACTION_TIMEOUT_S, settle=SETTLE_TIME_S):
    if not action.wait_for_completed(timeout=timeout) or not action.has_succeeded:
        raise RuntimeError(label + " failed or timed out")
    time.sleep(settle)


def point_gimbal(ep_robot, yaw, deadline=None):
    if not -250 <= yaw <= 250:
        raise ValueError("Requested gimbal yaw exceeds SDK limits")
    # moveto uses chassis-relative yaw (COORDINATE_YCPN in SDK).
    # Reissue after every chassis correction so the base follows the new body.
    timeout = ACTION_TIMEOUT_S
    if deadline is not None:
        timeout = deadline - time.monotonic() - SETTLE_TIME_S
        if timeout <= 0:
            raise RuntimeError("Gimbal scan exceeded the direction time budget")
    wait_action(ep_robot.gimbal.moveto(
        pitch=0, yaw=yaw, pitch_speed=GIMBAL_SPEED_DEG_S,
        yaw_speed=GIMBAL_SPEED_DEG_S), "Gimbal movement", timeout=timeout)


def rotate_yaw_only(ep_robot, degrees):
    # Sole rotation command: x/y are explicitly zero, never forward/strafe.
    wait_action(ep_robot.chassis.move(
        x=0, y=0, z=degrees, z_speed=CHASSIS_YAW_SPEED_DEG_S),
        "Chassis yaw rotation", settle=CHASSIS_SETTLE_TIME_S)


def align_direction(ep_robot, reader, name, base_yaw):
    pid = YawPID()
    total_rotation = 0.0
    corrections = 0
    for attempt in range(1, MAX_CORRECTIONS_PER_DIRECTION + 2):
        # Required sequence: base -> base+10 -> base -> base-10.
        # Also return to base before rotating chassis or proceeding onward.
        point_gimbal(ep_robot, base_yaw)
        point_gimbal(ep_robot, base_yaw + CHECK_OFFSET_DEG)
        plus = reader.read()
        point_gimbal(ep_robot, base_yaw)
        point_gimbal(ep_robot, base_yaw - CHECK_OFFSET_DEG)
        minus = reader.read()
        point_gimbal(ep_robot, base_yaw)
        error_deg = correction_from_distances(plus, minus)
        LOG.info("%s #%d: +%.1f mm, -%.1f mm, difference %.1f mm, yaw error %.2f deg",
                 name, attempt, plus, minus, abs(plus - minus), error_deg)
        if abs(plus - minus) <= DIST_TOLERANCE_MM:
            LOG.info("%s aligned", name)
            return True
        if corrections >= MAX_CORRECTIONS_PER_DIRECTION:
            LOG.warning("%s not aligned after %d corrections", name, corrections)
            return False
        signed_error = (DIRECTION_ROTATION_SIGN[name]
                        * CHASSIS_YAW_PER_GIMBAL_YAW_SIGN * error_deg)
        step = pid.update(signed_error)
        # SDK yaw resolution is 0.1 degree. Do not pretend smaller moves work.
        step = round(step, 1)
        if step == 0:
            raise RuntimeError(name + ": correction below SDK resolution; tune tolerances")
        total_rotation += abs(step)
        if total_rotation > MAX_TOTAL_CORRECTION_DEG:
            raise RuntimeError(name + ": correction budget exceeded; check wall/signs")
        LOG.info("%s correction %d/%d: %s %.1f deg", name,
                 corrections + 1, MAX_CORRECTIONS_PER_DIRECTION,
                 "counterclockwise" if step > 0 else "clockwise", abs(step))
        rotate_yaw_only(ep_robot, step)
        corrections += 1
    return False


def scan_walls(ep_robot, reader):
    walls = []
    for name, yaw in DIRECTIONS:
        deadline = time.monotonic() + SCAN_DIRECTION_BUDGET_S
        base = yaw * RIGHT_GIMBAL_YAW_SIGN
        point_gimbal(ep_robot, base, deadline=deadline)
        try:
            distance = reader.read(scan=True, deadline=deadline)
        except ToFReadingError as exc:
            LOG.warning("Scan %s: UNKNOWN (%s)", name, exc)
            continue
        if distance <= WALL_DETECTION_MAX_MM:
            walls.append((name, base))
            LOG.info("Scan %s: wall candidate at %.1f mm", name, distance)
    return walls


def align_detected_walls(ep_robot, reader, walls):
    results = {}
    for name, base in walls:
        try:
            results[name] = align_direction(ep_robot, reader, name, base)
        except ToFReadingError as exc:
            LOG.warning("%s: unusable offset ToF (%s)", name, exc)
            results[name] = False
        if results[name]:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tof-only", action="store_true")
    args = parser.parse_args()
    from robomaster import robot

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ep_robot = robot.Robot()
    initialized = subscribed = False
    try:
        ep_robot.initialize(conn_type=CONNECTION_TYPE)
        initialized = True
        # Free mode prevents automatic chassis following of gimbal motion.
        if not ep_robot.set_robot_mode(mode=robot.FREE):
            raise RuntimeError("Cannot select FREE mode")
        reader = DistanceReader()
        subscribed = ep_robot.sensor.sub_distance(
            freq=TOF_FREQUENCY_HZ, callback=reader.callback)
        if not subscribed:
            raise RuntimeError("Cannot subscribe to ToF")
        if args.tof_only:
            for _ in range(20):
                time.sleep(0.5)
                with reader.condition:
                    LOG.info("ToF callbacks=%d raw=%r", reader.sequence, reader.last_raw)
            return
        wait_action(ep_robot.gimbal.recenter(
            pitch_speed=GIMBAL_SPEED_DEG_S, yaw_speed=GIMBAL_SPEED_DEG_S),
            "Initial gimbal recenter")
        walls = scan_walls(ep_robot, reader)
        results = align_detected_walls(ep_robot, reader, walls)
        point_gimbal(ep_robot, 0)
        LOG.info("Two-pass alignment finished: %s", results)
    finally:
        if initialized:
            # Stop on success, sensor error, timeout, or Ctrl+C; no translation.
            for label, stop in (
                ("chassis", lambda: ep_robot.chassis.drive_speed(x=0, y=0, z=0)),
                ("gimbal", lambda: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0))):
                try:
                    stop()
                except Exception:
                    LOG.exception("Could not stop %s", label)
            if subscribed:
                try:
                    ep_robot.sensor.unsub_distance()
                except Exception:
                    LOG.exception("Could not unsubscribe ToF")
        ep_robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Alignment interrupted")
    except Exception as exc:
        logging.error("Alignment stopped: %s", exc)
        raise SystemExit(1)
