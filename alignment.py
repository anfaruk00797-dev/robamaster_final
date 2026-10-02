"""Align RoboMaster chassis yaw to walls using a gimbal-mounted ToF.

Run: python alignment.py (RoboMaster Python SDK required).
Assumes a level ToF, its optical axis aligned with gimbal yaw, and both
offset rays hitting the SAME flat wall. Four walls should be parallel or
perpendicular; aligning a later non-orthogonal wall can undo earlier alignment.
No chassis translation is commanded: every chassis call has x=0, y=0.
"""

import logging
import math
import statistics
import threading
import time

# -------- Tune these values for the robot and wall --------
CONNECTION_TYPE = "ap"  # "sta" for a robot on the same network
CHECK_OFFSET_DEG = 20.0
DIST_TOLERANCE_MM = 20.0
ANGLE_TOLERANCE_DEG = 3.0
ROTATION_GAIN = 0.6
MAX_ROTATION_STEP_DEG = 5.0
MAX_TOTAL_CORRECTION_DEG = 30.0  # per direction, cumulative absolute rotation
MAX_ITERATIONS = 15
REQUIRED_GOOD_PAIRS = 2
SETTLE_TIME_S = 0.35
GIMBAL_SPEED_DEG_S = 60
CHASSIS_YAW_SPEED_DEG_S = 20
ACTION_TIMEOUT_S = 15
TOF_FREQUENCY_HZ = 20
TOF_INDEX = 0  # select the connected sensor from the SDK distance tuple
SAMPLES_PER_READING = 5
READ_TIMEOUT_S = 2.0
MIN_DISTANCE_MM = 100
MAX_DISTANCE_MM = 3000
MAX_SAMPLE_SPREAD_MM = 60
# Distance from yaw pivot to ToF optical origin along the beam. Set from
# mounting geometry; zero assumes sensor origin at the pivot. Lateral offset
# is not modelled and must be calibrated/mechanically minimized.
TOF_RADIAL_OFFSET_MM = 0.0
# SDK yaw conventions: positive gimbal yaw points right; adjust these only
# if verified hardware coordinates/mounting differ.
RIGHT_GIMBAL_YAW_SIGN = 1
CHASSIS_YAW_PER_GIMBAL_YAW_SIGN = 1
DIRECTIONS = (("front", 0.0), ("right", 90.0),
              ("back", 180.0), ("left", -90.0))

LOG = logging.getLogger("alignment")


class DistanceReader:
    """Collect distinct callback samples; never reuse a pre-movement reading."""

    def __init__(self):
        self.condition = threading.Condition()
        self.sequence = 0
        self.samples = []

    def callback(self, distances):
        with self.condition:
            self.sequence += 1
            try:
                value = float(distances[TOF_INDEX])
            except (IndexError, TypeError, ValueError):
                value = float("nan")
            self.samples.append((self.sequence, value))
            self.samples = self.samples[-200:]
            self.condition.notify_all()

    def read(self):
        deadline = time.monotonic() + READ_TIMEOUT_S
        with self.condition:
            start = self.sequence  # called AFTER motion completion and settling
            while True:
                values = [v for seq, v in self.samples if seq > start
                          and math.isfinite(v)
                          and MIN_DISTANCE_MM <= v <= MAX_DISTANCE_MM]
                if len(values) >= SAMPLES_PER_READING:
                    values = values[:SAMPLES_PER_READING]
                    if max(values) - min(values) > MAX_SAMPLE_SPREAD_MM:
                        raise RuntimeError("Unstable ToF samples; check wall/beam obstruction")
                    return statistics.median(values)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("No fresh valid ToF data within timeout")
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


def wait_action(action, label):
    if not action.wait_for_completed(timeout=ACTION_TIMEOUT_S) or not action.has_succeeded:
        raise RuntimeError(label + " failed or timed out")
    time.sleep(SETTLE_TIME_S)


def point_gimbal(ep_robot, yaw):
    if not -250 <= yaw <= 250:
        raise ValueError("Requested gimbal yaw exceeds SDK limits")
    # moveto uses chassis-relative yaw (COORDINATE_YCPN in SDK).
    # Reissue after every chassis correction so the base follows the new body.
    wait_action(ep_robot.gimbal.moveto(
        pitch=0, yaw=yaw, pitch_speed=GIMBAL_SPEED_DEG_S,
        yaw_speed=GIMBAL_SPEED_DEG_S), "Gimbal movement")


def rotate_yaw_only(ep_robot, degrees):
    # Sole rotation command: x/y are explicitly zero, never forward/strafe.
    wait_action(ep_robot.chassis.move(
        x=0, y=0, z=degrees, z_speed=CHASSIS_YAW_SPEED_DEG_S),
        "Chassis yaw rotation")


def align_direction(ep_robot, reader, name, base_yaw):
    good_pairs = 0
    total_rotation = 0.0
    for attempt in range(1, MAX_ITERATIONS + 1):
        # Required sequence: base -> base+20 -> base -> base-20.
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
        if (abs(plus - minus) <= DIST_TOLERANCE_MM
                and abs(error_deg) <= ANGLE_TOLERANCE_DEG):
            good_pairs += 1
            if good_pairs >= REQUIRED_GOOD_PAIRS:
                LOG.info("%s aligned", name)
                return
            continue
        good_pairs = 0
        step = CHASSIS_YAW_PER_GIMBAL_YAW_SIGN * ROTATION_GAIN * error_deg
        step = max(-MAX_ROTATION_STEP_DEG, min(MAX_ROTATION_STEP_DEG, step))
        # SDK yaw resolution is 0.1 degree. Do not pretend smaller moves work.
        step = round(step, 1)
        if step == 0:
            raise RuntimeError(name + ": correction below SDK resolution; tune tolerances")
        total_rotation += abs(step)
        if total_rotation > MAX_TOTAL_CORRECTION_DEG:
            raise RuntimeError(name + ": correction budget exceeded; check wall/signs")
        rotate_yaw_only(ep_robot, step)
    raise RuntimeError(name + ": did not converge; no next direction was attempted")


def main():
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
        for name, yaw in DIRECTIONS:
            align_direction(ep_robot, reader, name, yaw * RIGHT_GIMBAL_YAW_SIGN)
        point_gimbal(ep_robot, 0)
        LOG.info("Four-direction alignment finished")
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
