"""Run: python work_gimball.py

Route: forward 3 x 0.60 m, gimbal right 90 degrees, chassis follows,
forward 2 x 0.60 m. Only RoboMaster SDK and the Python standard library
are required. No camera, shooting, or wall alignment is started.

FREE mode with two feedback loops: hold the gimbal's ground yaw and
rotate the chassis toward the gimbal's measured chassis-relative yaw.
Do not repeatedly moveto(yaw=0): that would follow a drifting chassis.
Distance is measured with wheel odometry, not elapsed driving time.
"""

import argparse
import logging
import math
from pathlib import Path
import sys
import threading
import time

CELL_SIZE_M = 0.60
CENTER_YAW_DEG = 0.0
CENTER_PITCH_DEG = -8.0
FORWARD_SPEED_MPS = 0.14
LOOP_PERIOD_S = 0.05
DATA_TIMEOUT_S = 0.75
ANGLE_TOLERANCE_DEG = 1.0
DISTANCE_TOLERANCE_M = 0.01
SETTLE_S = 0.50
GIMBAL_GAIN = 3.0
CHASSIS_GAIN = 2.0
MAX_GIMBAL_SPEED = 45.0
MAX_CHASSIS_SPEED = 30.0
LOG = logging.getLogger("gimbal_locked_route")
VERSION = "2026-09-29-r4"


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def limit(value, maximum):
    return max(-maximum, min(maximum, value))


class Telemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.gimbal = self.position = None
        self.gimbal_time = self.position_time = 0.0

    def on_gimbal(self, values):
        with self.lock:
            self.gimbal = tuple(float(v) for v in values)
            self.gimbal_time = time.monotonic()

    def on_position(self, values):
        with self.lock:
            self.position = tuple(float(v) for v in values)
            self.position_time = time.monotonic()

    def read(self):
        with self.lock:
            now = time.monotonic()
            if (self.gimbal is None or self.position is None
                    or now - self.gimbal_time > DATA_TIMEOUT_S
                    or now - self.position_time > DATA_TIMEOUT_S):
                raise RuntimeError("Missing/stale gimbal or position telemetry")
            if not all(math.isfinite(v) for v in self.gimbal + self.position):
                raise RuntimeError("Invalid telemetry")
            return self.gimbal, self.position

    def wait_ready(self):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                return self.read()
            except RuntimeError:
                time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("Telemetry did not become ready within 5 seconds")


class Route:
    def __init__(self, ep_robot, telemetry, cell_size):
        self.robot = ep_robot
        self.data = telemetry
        self.cell_size = cell_size
        self.target_yaw = None
        self.target_pitch = None
        self.last_status_time = -float("inf")
        self.logged_gimbal_result = False

    def stop_chassis(self):
        # ProtoChassisSpeedMode is PUSH (no ACK). The installed SDK's
        # _send_sync_proto nevertheless checks for an ACK and returns False
        # after send_sync_msg has already sent it. This return is unusable
        # as an acceptance test for drive_speed, including stop commands.
        self.robot.chassis.drive_speed(x=0, y=0, z=0)

    def control(self, forward=0.0, follow=True):
        angles, position = self.data.read()
        # SDK tuple: pitch_body, yaw_body, pitch_ground, yaw_ground.
        _, yaw_body, pitch_ground, yaw_ground = angles
        yaw_error = wrap(self.target_yaw - yaw_ground)
        pitch_error = self.target_pitch - pitch_ground
        if abs(yaw_body) > 150:
            raise RuntimeError("Gimbal approaching its mechanical yaw limit")
        # The installed SDK sends this asynchronously and normally returns None.
        # None is not a rejection/ack; only an explicit False indicates failure.
        # Ground-angle feedback and settle timeout verify the resulting motion.
        gimbal_result = self.robot.gimbal.drive_speed(
                pitch_speed=limit(3.0 * pitch_error, 20.0),
                yaw_speed=limit(GIMBAL_GAIN * yaw_error, MAX_GIMBAL_SPEED))
        if not self.logged_gimbal_result:
            LOG.info("Gimbal command return: %r (%s); None is normal for async SDK",
                     gimbal_result, type(gimbal_result).__name__)
            self.logged_gimbal_result = True
        if gimbal_result is False:
            raise RuntimeError("Gimbal speed command could not be sent")
        # Positive body-relative gimbal yaw and positive chassis speed are right.
        chassis_speed = limit(CHASSIS_GAIN * wrap(yaw_body), MAX_CHASSIS_SPEED)
        if not follow:
            chassis_speed = 0.0
        # Pause forward movement when alignment is lost; keep both loops running.
        if abs(yaw_body) > 3.0 or abs(yaw_error) > 3.0:
            forward = 0.0
        # This SDK returns False for its no-ACK chassis speed packets.
        # Verify motion through telemetry, progress timeout and settle timeout.
        self.robot.chassis.drive_speed(
                x=forward, y=0, z=chassis_speed, timeout=0.5)
        now = time.monotonic()
        if now - self.last_status_time >= 1.0:
            LOG.info("target=%.2f ground=%.2f yaw_body=%.2f error=%.2f "
                     "forward=%.3f turn=%.2f position=(%.3f, %.3f)",
                     self.target_yaw, yaw_ground, yaw_body, yaw_error,
                     forward, chassis_speed, position[0], position[1])
            self.last_status_time = now
        return yaw_error, wrap(yaw_body), position

    def settle(self, follow=True, timeout=15.0):
        deadline = time.monotonic() + timeout
        stable_since = None
        while time.monotonic() < deadline:
            yaw_error, body_error, _ = self.control(follow=follow)
            good = abs(yaw_error) <= ANGLE_TOLERANCE_DEG
            if follow:
                good = good and abs(body_error) <= ANGLE_TOLERANCE_DEG
            if good:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= SETTLE_S:
                    self.stop_chassis()
                    return
            else:
                stable_since = None
            time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("Alignment timed out: target_error=%.2f, "
                           "body_error=%.2f, follow=%s" %
                           (yaw_error, body_error, follow))

    def forward_cell(self, number):
        self.settle()
        _, start = self.data.read()
        deadline = time.monotonic() + self.cell_size / FORWARD_SPEED_MPS * 3 + 10
        last_progress = 0.0
        progress_time = time.monotonic()
        while time.monotonic() < deadline:
            _, position = self.data.read()
            # Net displacement from this cell's start in the fixed odometry frame.
            # Works after the turn without assuming IMU and odometry yaw origins match.
            progress = math.hypot(position[0] - start[0], position[1] - start[1])
            remaining = self.cell_size - progress
            if remaining <= DISTANCE_TOLERANCE_M:
                self.stop_chassis()
                self.settle()
                LOG.info("Cell %d done: %.3f m", number, progress)
                return
            if progress > last_progress + 0.01:
                last_progress = progress
                progress_time = time.monotonic()
            if time.monotonic() - progress_time > 5.0:
                raise RuntimeError("No distance progress for 5 seconds")
            speed = min(FORWARD_SPEED_MPS, max(0.025, 1.2 * remaining))
            self.control(forward=speed)
            time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("Forward movement timed out")

    def run(self):
        angles, _ = self.data.read()
        self.target_pitch = angles[2]
        self.target_yaw = angles[3]
        LOG.info("Initial gimbal ground heading: %.2f degrees", self.target_yaw)
        self.settle()
        for number in range(1, 4):
            self.forward_cell(number)
        self.stop_chassis()
        # Rotate gimbal first while chassis stays still. Use ground feedback,
        # because SDK move(yaw=90) is outside its +/-55 degree documented range.
        self.target_yaw = wrap(self.target_yaw + 90.0)
        LOG.info("Gimbal right 90 degrees; target %.2f", self.target_yaw)
        self.settle(follow=False)
        LOG.info("Chassis following gimbal to within %.1f degree", ANGLE_TOLERANCE_DEG)
        self.settle(follow=True)
        for number in range(4, 6):
            self.forward_cell(number)
        LOG.info("Route complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conn", choices=("ap", "sta"), default="ap")
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE_M, help="Meters per cell")
    parser.add_argument("--center-yaw", type=float, default=CENTER_YAW_DEG)
    parser.add_argument("--center-pitch", type=float, default=CENTER_PITCH_DEG)
    args = parser.parse_args()
    if not math.isfinite(args.cell_size) or not 0.05 <= args.cell_size <= 5.0:
        parser.error("--cell-size must be 0.05..5 meters")
    if not -45 <= args.center_yaw <= 45 or not -25 <= args.center_pitch <= 30:
        parser.error("Center yaw must be -45..45; pitch must be -25..30 degrees")
    log_path = Path(__file__).resolve().with_name("work_gimball_run.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, mode="w", encoding="utf-8")],
                        force=True)
    LOG.info("work_gimball version %s", VERSION)
    LOG.info("Running file: %s", Path(__file__).resolve())
    LOG.info("Python: %s", sys.executable)
    from robomaster import robot
    LOG.info("SDK robot module: %s", robot.__file__)
    ep_robot = robot.Robot()
    initialized = False
    subscriptions = []
    try:
        ep_robot.initialize(conn_type=args.conn)
        initialized = True
        if not ep_robot.set_robot_mode(mode=robot.FREE):
            raise RuntimeError("Cannot set FREE mode")
        data = Telemetry()
        for subscribe, unsubscribe in (
                (lambda: ep_robot.gimbal.sub_angle(freq=20, callback=data.on_gimbal),
                 ep_robot.gimbal.unsub_angle),
                (lambda: ep_robot.chassis.sub_position(cs=1, freq=20, callback=data.on_position),
                 ep_robot.chassis.unsub_position)):
            if not subscribe():
                raise RuntimeError("Telemetry subscription failed")
            subscriptions.append(unsubscribe)
        data.wait_ready()
        ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        action = ep_robot.gimbal.moveto(
            pitch=args.center_pitch, yaw=args.center_yaw,
            pitch_speed=60, yaw_speed=60)
        if not action.wait_for_completed(timeout=10) or not action.has_succeeded:
            raise RuntimeError("Initial gimbal centering failed")
        time.sleep(0.30)
        Route(ep_robot, data, args.cell_size).run()
    finally:
        if initialized:
            for stop in (
                    lambda: ep_robot.chassis.drive_speed(x=0, y=0, z=0),
                    lambda: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)):
                try:
                    stop()
                except Exception:
                    LOG.exception("Stop command failed")
            for unsubscribe in reversed(subscriptions):
                try:
                    unsubscribe()
                except Exception:
                    LOG.exception("Unsubscribe failed")
        ep_robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C")
    except Exception as exc:
        logging.exception("Route stopped: %s", exc)
        raise SystemExit(1)
