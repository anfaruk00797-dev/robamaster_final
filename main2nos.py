"""A* RoboMaster navigation led by main_yaw, without Sharp steering while driving.

Run from any directory: python main2nos.py --map path/to/map_config.json
The ToF must be mounted on the gimbal. Put the robot at the configured start
cell's center facing +y; +x is to its right. Test in a clear, guarded area.
"""

import argparse
import heapq
import importlib.util
import itertools
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass, replace
from statistics import median

CELL_SIZE_M = 0.60
FORWARD_SPEED_MPS = 0.30
LOOP_PERIOD_S = 0.05
DATA_TIMEOUT_S = 0.75
ANGLE_TOLERANCE_DEG = 0.5
DISTANCE_TOLERANCE_M = 0.01
# The camera/ToF projects forward of the wheel centre. At a mapped wall just
# beyond the destination cell, 170 mm ToF clearance can stop wheel odometry
# around 5-10 cm short of the nominal 60 cm cell centre.
FRONT_WALL_CELL_TOLERANCE_M = 0.10
FRONT_WALL_POST_BACKOFF_TOLERANCE_M = 0.15
MAX_FRONT_ADJUSTMENTS_PER_CELL = 3
SETTLE_S = 0.50
GIMBAL_GAIN = 3.0
CHASSIS_GAIN = 2.0
MAX_GIMBAL_SPEED = 45.0
MAX_CHASSIS_SPEED = 30.0
LOG = logging.getLogger("main2nos")
VERSION = "main2nos-gel-target-lock-12"
SHARP_NAV_ENABLED = False
MIN_FINISH_SPEED_MPS = 0.06
FINISH_RETRY_SPEED_MPS = 0.09

ALIGN_OFFSET_DEG = 10.0
ALIGN_WALL_MM = 300.0
ALIGN_TOLERANCE_MM = 7.0
ALIGN_KP = 0.25
ALIGN_MAX_STEP_DEG = 7.0
ALIGN_MAX_TOTAL_DEG = 30.0
ALIGN_MAX_CORRECTIONS = 3
FRONT_TARGET_MM = 170.0
FRONT_TOLERANCE_MM = 10.0
SHARP_WALL_CM = 26.0
SHARP_STOP_CM = 3.0
SHARP_ESCAPE_CM = 6.0
SHARP_ESCAPE_MAX_TRAVEL_M = 0.12
SHARP_ESCAPE_TIMEOUT_S = 4.0
SHARP_RELEASE_CM = 3.5
SHARP_CLEARANCE_CM = 7.0
SHARP_CLEARANCE_SPEED_MPS = 0.035
SHARP_CLEARANCE_MAX_TRAVEL_M = 0.15
SHARP_CENTER_TOLERANCE_CM = 1.0
SHARP_CENTER_GAIN = 0.004
SHARP_MAX_LATERAL_MPS = 0.06
SHARP_MAX_AGE_S = 1.5


class SafetyStop(RuntimeError):
    """A requested stop that ends navigation cleanly without an error traceback."""


class FrontObstacleStop(SafetyStop):
    """The front ToF required a backoff before a full grid step was completed."""


class FrontWallAdjusted(RuntimeError):
    """A front ToF correction completed; re-evaluate wheel progress."""


class ScanOffsetBlocked(RuntimeError):
    """A side scan cannot continue safely; return its measured partial offset."""

    def __init__(self, travelled, left_cm, right_cm):
        self.travelled = travelled
        super().__init__("Sharp blocked target-scan offset after %.3f m "
                         "(left=%s cm, right=%s cm)" %
                         (travelled, left_cm, right_cm))


SHARP_CALIBRATION = {
    "LEFT_SENSOR": {"a": 0.00019024, "b": -0.18564551, "c": 46.21151396,
                    "min_adc": 103, "max_adc": 635, "noise_threshold": 219},
    "RIGHT_SENSOR": {"a": 0.00022938, "b": -0.21975928, "c": 55.31746333,
                     "min_adc": 158, "max_adc": 584, "noise_threshold": 368},
}
# Matches alig.py's chassis.move yaw convention: positive is CCW.
ALIGN_DIRECTION_SIGN = {"front": 1, "right": 1, "back": -1, "left": 1}
ALIGN_CHASSIS_PER_GIMBAL_SIGN = -1


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def limit(value, maximum):
    return max(-maximum, min(maximum, value))


@dataclass(frozen=True)
class MapConfig:
    width: int
    height: int
    start: tuple
    targets: tuple
    walls: tuple
    cell_size_m: float


def load_map_config(path):
    """Read mock_map.py's JSON format; reject a mismatched or unsafe map."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    size = raw["grid_size"]
    width, height = int(size["w"]), int(size["h"])
    if not (1 <= width <= 50 and 1 <= height <= 50):
        raise ValueError("Map width and height must be 1..50")

    def cell(value):
        if isinstance(value, dict):
            value = (value["x"], value["y"])
        if len(value) != 2 or any(isinstance(v, bool) or int(v) != v for v in value):
            raise ValueError("Map cell must contain two integer coordinates")
        result = (int(value[0]), int(value[1]))
        if not (0 <= result[0] < width and 0 <= result[1] < height):
            raise ValueError("Map cell outside grid: %r" % (result,))
        return result

    # Older final_final_robot.py uses (2,0) if the JSON has no start field.
    # Keep that compatibility for existing hand-written map_config.json files.
    start = cell(raw.get("start", [2, 0]))
    if float(raw.get("initial_heading_deg", 0)) != 0:
        raise ValueError("Map initial_heading_deg must be 0 (+y)")
    cell_size_m = float(raw.get("cell_size_m", CELL_SIZE_M))
    if not math.isfinite(cell_size_m) or not .20 <= cell_size_m <= 2.0:
        raise ValueError("Map cell_size_m must be 0.20..2.0")
    targets = tuple(dict.fromkeys(cell(v) for v in raw["targets"]))
    walls = set()
    for item in raw.get("walls", []):
        a, b = cell(item[0]), cell(item[1])
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise ValueError("Wall endpoints must be adjacent: %r" % (item,))
        walls.add(tuple(sorted((a, b))))
    return MapConfig(width, height, start, targets, tuple(sorted(walls)), cell_size_m)


def parse_grid_targets(raw, width, height):
    """Parse temporary navigation goals: x,y; x,y (zero-based grid cells)."""
    targets = []
    for item in raw.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2 or any(not part.isdecimal() for part in parts):
            raise ValueError("Enter grid targets as x,y; x,y (example: 1,1; 2,3)")
        x, y = map(int, parts)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Grid target (%d,%d) is outside %dx%d map" %
                             (x, y, width, height))
        targets.append((x, y))
    return tuple(dict.fromkeys(targets))


def a_star(maze, start, goal):
    """A* on mapped walls. Unknown edges are estimates, never move permission."""
    queue = [(abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start)]
    best = {start: 0}
    parent = {}
    while queue:
        _, cost, here = heapq.heappop(queue)
        if cost != best[here]:
            continue
        if here == goal:
            path = [here]
            while here in parent:
                here = parent[here]
                path.append(here)
            return list(reversed(path))
        for direction in range(4):
            if maze.edge(here, direction) is True:
                continue
            there = maze.neighbor(here, direction)
            new_cost = cost + 1
            if new_cost < best.get(there, math.inf):
                best[there], parent[there] = new_cost, here
                priority = new_cost + abs(there[0]-goal[0]) + abs(there[1]-goal[1])
                heapq.heappush(queue, (priority, new_cost, there))
    return None




def optimal_target_order(maze, start, targets):
    """Same A*/permutation objective as final_final_robot.py."""
    remaining = tuple(sorted(set(targets) - {start}))
    if len(remaining) > 8:
        raise ValueError("At most 8 remaining targets supported by permutation planner")
    best = None
    best_length = math.inf
    nodes = (start,) + remaining
    distances = {}
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            path = a_star(maze, a, b)
            distances[tuple(sorted((a, b)))] = math.inf if path is None else len(path)-1
    for order in itertools.permutations(remaining):
        here, length = start, 0
        for target in order:
            length += distances[tuple(sorted((here, target)))]
            here = target
        if length < best_length:
            best, best_length = order, length
    return best


class Telemetry:
    def __init__(self):
        self.lock = threading.Lock()
        self.gimbal = self.position = None
        self.gimbal_time = self.position_time = 0.0
        self.chassis_attitude = None
        self.attitude_time = 0.0

    def on_gimbal(self, values):
        with self.lock:
            self.gimbal = tuple(float(v) for v in values)
            self.gimbal_time = time.monotonic()

    def on_position(self, values):
        with self.lock:
            self.position = tuple(float(v) for v in values)
            self.position_time = time.monotonic()

    def on_attitude(self, values):
        # RoboMaster SDK order is yaw, pitch, roll. This is display/log data;
        # steering remains referenced to the saved gimbal ground yaw.
        with self.lock:
            self.chassis_attitude = tuple(float(v) for v in values)
            self.attitude_time = time.monotonic()

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

    def read_position(self):
        """Wheel telemetry stays valid while map_scan owns gimbal angles."""
        with self.lock:
            if (self.position is None
                    or time.monotonic() - self.position_time > DATA_TIMEOUT_S
                    or not all(math.isfinite(v) for v in self.position)):
                raise RuntimeError("Missing/stale wheel-position telemetry")
            return self.position

    def read_cached(self):
        """Use the last sensor values only for a stopped/error map snapshot."""
        with self.lock:
            if (self.gimbal is None or self.position is None
                    or not all(math.isfinite(v) for v in self.gimbal + self.position)):
                raise RuntimeError("No valid cached telemetry for stopped map")
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

    def control(self, forward=0.0, follow=True, lateral=0.0):
        angles, position = self.data.read()
        # SDK tuple: pitch_body, yaw_body, pitch_ground, yaw_ground.
        _, yaw_body, pitch_ground, yaw_ground = angles
        yaw_error = wrap(self.target_yaw - yaw_ground)
        pitch_error = self.target_pitch - pitch_ground
        # The chassis aims at the STORED ground heading, never at a drifting
        # actual gimbal heading. ground - body_relative estimates body heading.
        chassis_error = wrap(yaw_body + yaw_error)
        walking = getattr(self, "phase", None) == "moving" and forward > 0
        gimbal_yaw_speed = limit(GIMBAL_GAIN * yaw_error, MAX_GIMBAL_SPEED)
        if walking:
            if abs(yaw_error) <= ANGLE_TOLERANCE_DEG:
                # FREE mode holds the inertial gimbal heading at zero commanded
                # yaw speed; do not steer the gimbal in response to chassis yaw.
                gimbal_yaw_speed = 0.0
            else:
                # Stop translation BEFORE commanding recovery of the saved yaw.
                self.stop_chassis()
                forward = 0.0
                lateral = 0.0
        if abs(yaw_body) > 150:
            raise RuntimeError("Gimbal approaching its mechanical yaw limit")
        # The installed SDK sends this asynchronously and normally returns None.
        # None is not a rejection/ack; only an explicit False indicates failure.
        # Ground-angle feedback and settle timeout verify the resulting motion.
        gimbal_result = self.robot.gimbal.drive_speed(
                pitch_speed=limit(3.0 * pitch_error, 20.0),
                yaw_speed=getattr(self, "gimbal_yaw_command_sign", 1) * gimbal_yaw_speed)
        if not self.logged_gimbal_result:
            LOG.info("Gimbal command return: %r (%s); None is normal for async SDK",
                     gimbal_result, type(gimbal_result).__name__)
            self.logged_gimbal_result = True
        if gimbal_result is False:
            raise RuntimeError("Gimbal speed command could not be sent")
        # Positive body-relative gimbal yaw and positive chassis speed are right.
        chassis_speed = limit(CHASSIS_GAIN * chassis_error, MAX_CHASSIS_SPEED)
        if not follow:
            chassis_speed = 0.0
        # Pause forward movement when alignment is lost; keep both loops running.
        if (abs(yaw_body) > ANGLE_TOLERANCE_DEG
                or abs(chassis_error) > ANGLE_TOLERANCE_DEG
                or abs(yaw_error) > ANGLE_TOLERANCE_DEG):
            forward = 0.0
            lateral = 0.0
        # This SDK returns False for its no-ACK chassis speed packets.
        # Verify motion through telemetry, progress timeout and settle timeout.
        self.robot.chassis.drive_speed(
                x=forward, y=lateral, z=chassis_speed, timeout=0.5)
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
            angles, _ = self.data.read()
            good = good and abs(self.target_pitch - angles[2]) <= ANGLE_TOLERANCE_DEG
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
        front_adjustments = 0
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
            if progress > last_progress + 0.002:
                last_progress = progress
                progress_time = time.monotonic()
            stagnant_s = time.monotonic() - progress_time
            if stagnant_s > 5.0:
                raise SafetyStop("Wheel progress stopped at %.3f/%.3f m despite finish-speed retry" %
                                 (progress, self.cell_size))
            speed = min(FORWARD_SPEED_MPS, max(MIN_FINISH_SPEED_MPS, 1.2 * remaining))
            if remaining <= .04 and stagnant_s > 1.5:
                speed = FINISH_RETRY_SPEED_MPS
                if stagnant_s < 1.5 + LOOP_PERIOD_S:
                    LOG.warning("Wheel progress slow near cell end: %.3f/%.3f m; "
                                "finish speed %.3f m/s", progress, self.cell_size, speed)
            try:
                self.control(forward=speed)
            except FrontWallAdjusted as exc:
                front_adjustments += 1
                next_cell = self.maze.neighbor(self.cell, self.heading)
                wall_beyond = self.maze.edge(next_cell, self.heading) is True
                _, settled_position = self.data.read()
                settled_progress = math.hypot(settled_position[0] - start[0],
                                              settled_position[1] - start[1])
                front_mm = self.tof.latest()
                if (wall_beyond
                        and progress >= self.cell_size - FRONT_WALL_CELL_TOLERANCE_M
                        and settled_progress >= self.cell_size - FRONT_WALL_POST_BACKOFF_TOLERANCE_M
                        and abs(front_mm - FRONT_TARGET_MM) <= FRONT_TOLERANCE_MM):
                    self.stop_chassis()
                    LOG.info("Cell %d reached beside mapped wall: wheel progress "
                             "before/after ToF backoff %.3f/%.3f of %.3f m, "
                             "front ToF %.1f mm; correction=%s",
                             number, progress, settled_progress, self.cell_size,
                             front_mm, exc)
                    return
                if front_adjustments >= MAX_FRONT_ADJUSTMENTS_PER_CELL:
                    raise FrontObstacleStop(
                        "Front ToF needed %d corrections before cell completion; "
                        "wheel progress %.3f/%.3f m" %
                        (front_adjustments, progress, self.cell_size)) from None
                LOG.info("Front ToF correction %d/%d complete; resuming cell %d "
                         "from wheel position", front_adjustments,
                         MAX_FRONT_ADJUSTMENTS_PER_CELL, number)
                progress_time = time.monotonic()
                continue
            except FrontObstacleStop as exc:
                raise FrontObstacleStop("%s; wheel progress before backoff %.3f/%.3f m" %
                                        (exc, progress, self.cell_size)) from None
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop("Wheel distance did not reach %.3f m within cell timeout" %
                         self.cell_size)



# Grid direction order is clockwise: +y, +x, -y, -x.
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIRECTION_NAMES = ("+y", "+x", "-y", "-x")


class Maze:
    def __init__(self, width, height, goal):
        self.width, self.height, self.goal = width, height, tuple(goal)
        self.edges = {}  # Canonical undirected edge: None unknown, False open, True wall.
        self.visited = set()

    def inside(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def neighbor(self, cell, direction):
        dx, dy = DIRECTIONS[direction]
        return cell[0] + dx, cell[1] + dy

    def key(self, a, b):
        return tuple(sorted((tuple(a), tuple(b))))

    def edge(self, cell, direction):
        other = self.neighbor(cell, direction)
        if not self.inside(other):
            return True  # Map boundary is always closed.
        return self.edges.get(self.key(cell, other))

    def observe(self, cell, direction, wall):
        other = self.neighbor(cell, direction)
        if not self.inside(other):
            if not wall:
                LOG.warning("Sensor reports open at map boundary %s %s; boundary remains closed",
                            cell, DIRECTION_NAMES[direction])
            return
        key = self.key(cell, other)
        previous = self.edges.get(key)
        if previous is not None and previous != wall:
            # Do not silently discard a known wall/open edge after noisy scans.
            raise RuntimeError("Conflicting wall readings at %s -> %s; check cell size/ToF" % (cell, other))
        self.edges[key] = bool(wall)

    def flood(self):
        from collections import deque
        costs = {self.goal: 0}
        queue = deque([self.goal])
        while queue:
            cell = queue.popleft()
            for direction in range(4):
                other = self.neighbor(cell, direction)
                # Optimistically allow unknown passages when estimating distance.
                # They must be measured OPEN before a physical move is allowed.
                if self.edge(cell, direction) is not True and other not in costs:
                    costs[other] = costs[cell] + 1
                    queue.append(other)
        return costs



class ToF:
    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.sequence = 0
        self.samples = []
        self.last_raw = None

    def callback(self, values):
        with self.lock:
            self.sequence += 1
            self.last_raw = tuple(values)
            try:
                value = float(values[self.index])
            except (IndexError, ValueError, TypeError):
                value = float("nan")
            self.samples.append((self.sequence, time.monotonic(), value))
            self.samples = self.samples[-100:]

    def mark(self):
        with self.lock:
            return self.sequence

    @staticmethod
    def valid(value):
        # Zero and 65535 are invalid, never evidence of an open passage.
        return math.isfinite(value) and 0 < value < 65535

    def after(self, mark):
        with self.lock:
            return [v for seq, timestamp, v in self.samples
                    if seq > mark and time.monotonic() - timestamp <= DATA_TIMEOUT_S
                    and self.valid(v)]

    def latest(self):
        with self.lock:
            if not self.samples:
                raise RuntimeError("No ToF data received")
            _, timestamp, value = self.samples[-1]
            if time.monotonic() - timestamp > DATA_TIMEOUT_S or not self.valid(value):
                raise RuntimeError("Stale/invalid ToF: index=%d raw=%r" % (self.index, self.last_raw))
            return value

    def fresh_median(self, mark, timeout=2.0, max_spread=60.0):
        """Alignment uses three new readings after each completed gimbal move."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            values = [v for v in self.after(mark) if 50 <= v <= 3000]
            if len(values) >= 3:
                window = values[:3]
                if max(window) - min(window) <= max_spread:
                    return median(window)
                raise RuntimeError("Unstable alignment ToF: %r" % (window,))
            time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("No fresh alignment ToF: index=%d raw=%r" %
                           (self.index, self.last_raw))


def sharp_distance_cm(adc, sensor):
    """Use the supplied quadratic calibration; a missing ADC is unknown."""
    if adc is None or not math.isfinite(float(adc)) or float(adc) <= 0:
        return None
    cfg = SHARP_CALIBRATION[sensor]
    adc = max(cfg["min_adc"], min(float(adc), cfg["max_adc"]))
    return max(0.0, cfg["a"] * adc * adc + cfg["b"] * adc + cfg["c"])


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
        if getattr(self.robot, "sensor_adaptor", None) is None:
            raise RuntimeError("Sharp sensor adaptor unavailable; cannot enable 3 cm brake")
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.sample_time > 0 and self.left_cm is not None and self.right_cm is not None
            if ready:
                return
            time.sleep(.05)
        raise RuntimeError("No valid readings from both Sharp sensors")

    def _poll(self):
        adaptor = self.robot.sensor_adaptor
        while not self.stop_event.is_set():
            try:
                left_adc = adaptor.get_adc(id=2, port=1)
                left = sharp_distance_cm(left_adc, "LEFT_SENSOR")
                time.sleep(.03)
                right_adc = adaptor.get_adc(id=1, port=1)
                right = sharp_distance_cm(right_adc, "RIGHT_SENSOR")
                close = any(value is not None and value <= SHARP_STOP_CM
                            for value in (left, right))
                with self.lock:
                    self.left_cm, self.right_cm = left, right
                    self.left_adc, self.right_adc = left_adc, right_adc
                    self.sample_time = time.monotonic()
                    was_braking = self.too_close
                    clear = (left is not None and right is not None
                             and left > SHARP_RELEASE_CM and right > SHARP_RELEASE_CM)
                    self.too_close = close or (was_braking and not clear)
                    braking = self.too_close
                    correction_active = self.correction_active
                if braking and not correction_active and SHARP_NAV_ENABLED:
                    self.robot.chassis.drive_speed(x=0, y=0, z=0)
                if braking and not was_braking:
                    LOG.warning("SHARP BRAKE left=%s cm (ADC=%s) right=%s cm (ADC=%s), limit=%.1f cm",
                                left, left_adc, right, right_adc, SHARP_STOP_CM)
                elif not braking and was_braking:
                    LOG.info("SHARP BRAKE released left=%.2f cm right=%.2f cm",
                             left, right)
            except Exception as exc:
                LOG.warning("Sharp read unavailable: %s", exc)
            self.stop_event.wait(.07)

    def read(self):
        with self.lock:
            if self.thread is None:
                return None, None
            if (time.monotonic() - self.sample_time > SHARP_MAX_AGE_S
                    or self.left_cm is None or self.right_cm is None):
                return None, None
            return self.left_cm, self.right_cm

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
            time.sleep(.10)

    def wait_fresh(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            readings = self.read()
            if readings[0] is not None and readings[1] is not None:
                return readings
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop("Sharp data unavailable for %.1f s: stopped" % timeout)

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class ScanSharpUnavailable:
    """Allow a target scan to use wheel return when side ADC is unavailable."""

    def read(self):
        return None, None

    def allow_correction(self, _enabled):
        pass

    def close(self):
        pass


def start_target_scan_sharp(robot):
    """Sharp is optional for target-scan centering; wheel return still works."""
    monitor = SharpMonitor(robot)
    try:
        monitor.start()
        monitor.wait_fresh()
        return monitor
    except (RuntimeError, SafetyStop) as exc:
        monitor.close()
        LOG.warning("TARGET SCAN Sharp unavailable (%s); scanning with wheel-measured return and no Sharp centering", exc)
        return ScanSharpUnavailable()


def sharp_center_speed(left_cm, right_cm):
    if (left_cm is None or right_cm is None
            or left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM):
        return 0.0
    difference = left_cm - right_cm
    if abs(difference) <= SHARP_CENTER_TOLERANCE_CM:
        return 0.0
    # +y is robot-right: when right range is smaller, slide left.
    return limit(-SHARP_CENTER_GAIN * difference, SHARP_MAX_LATERAL_MPS)


def sharp_clearance_speed(left_cm, right_cm, center_single_wall=False):
    """Robot-right is +y. Return zero only when side clearance is safe."""
    left_wall = left_cm <= SHARP_WALL_CM
    right_wall = right_cm <= SHARP_WALL_CM
    if left_wall and right_wall:
        # In a narrow passage 7 cm on BOTH sides may be impossible. The
        # two-wall rule is to center, while keeping above the brake-release
        # distance on each side; do not demand an impossible 14 cm width.
        if left_cm + right_cm < 2 * SHARP_RELEASE_CM:
            raise SafetyStop("Both side walls are too close to release Sharp brake; stopped")
        difference = left_cm - right_cm
        if (min(left_cm, right_cm) >= SHARP_RELEASE_CM
                and abs(difference) <= SHARP_CENTER_TOLERANCE_CM):
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
            raise SafetyStop("Sharp cannot center after target scan: no side wall detected")
        return 0.0
    return direction * SHARP_CLEARANCE_SPEED_MPS




class OdometryTrack:
    """Convert wheel coordinates into the configured map frame."""

    def sample(self, angles, position):
        self.current_angles = angles
        point = (position[0], position[1])
        if self.origin is None:
            self.origin = point
        if not self.raw_track or math.hypot(point[0]-self.raw_track[-1][0], point[1]-self.raw_track[-1][1]) >= .003:
            self.raw_track.append(point)
        # Odometry and gimbal ground yaw need not share the same origin.
        # Calibrate the graph axes from the first actual cell movement.
        if self.forward_axis is None and self.nav.phase == "moving":
            dx, dy = point[0]-self.origin[0], point[1]-self.origin[1]
            length = math.hypot(dx, dy)
            if length >= .06:
                theta = math.radians(self.nav.heading * 90)
                self.forward_axis = ((dx*math.cos(theta)+dy*math.sin(theta))/length,
                                     (-dx*math.sin(theta)+dy*math.cos(theta))/length)
        self.update()

    def grid_position(self, point):
        start_x, start_y = self.nav.path[0]
        if self.forward_axis is None:
            return start_x, start_y
        fx, fy = self.forward_axis
        dx, dy = point[0]-self.origin[0], point[1]-self.origin[1]
        return (start_x + (-dx*fy+dy*fx)/self.nav.cell_size,
                start_y + (dx*fx+dy*fy)/self.nav.cell_size)


class PygameMap(OdometryTrack):
    """Pygame view; the control loop pumps events about ten times a second."""

    def __init__(self, navigator):
        import pygame
        self.pg = pygame
        self.nav = navigator
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 820), pygame.RESIZABLE)
        pygame.display.set_caption("RoboMaster | main2 A* map")
        self.font = pygame.font.SysFont("Tahoma", 16)
        self.small = pygame.font.SysFont("Tahoma", 13)
        self.stop_requested = False
        self.finished = False
        self.last_update = -float("inf")
        self.origin = None
        self.forward_axis = None
        self.raw_track = []
        self.current_angles = None

    def update(self, force=False):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and
                    event.key in (self.pg.K_q, self.pg.K_ESCAPE)):
                self.stop_requested = True
        if self.stop_requested:
            raise KeyboardInterrupt("Map window stop requested")
        now = time.monotonic()
        if force or now - self.last_update >= .10:
            self.last_update = now
            self.draw()

    def draw(self):
        pg, nav = self.pg, self.nav
        screen = self.screen
        width, height = screen.get_size()
        screen.fill((25, 28, 36))
        scale = min((width-360)/nav.maze.width, (height-110)/nav.maze.height)
        scale = max(12, scale)
        left, top = 25, 45
        def center(cell):
            return (int(left+(cell[0]+.5)*scale),
                    int(top+(nav.maze.height-cell[1]-.5)*scale))
        costs = nav.maze.flood()
        for x in range(nav.maze.width):
            for y in range(nav.maze.height):
                px, py = center((x, y))
                rect = pg.Rect(px-scale/2, py-scale/2, scale, scale)
                pg.draw.rect(screen, (51, 63, 78) if (x,y) in nav.maze.visited else (41, 46, 57), rect)
                pg.draw.rect(screen, (91, 103, 118), rect, 1)
                if scale >= 28:
                    label = self.small.render(str(costs.get((x,y), "X")), True, (150, 160, 175))
                    screen.blit(label, (px-label.get_width()/2, py-label.get_height()/2))
        for (a,b), wall in nav.maze.edges.items():
            if wall is not True:
                continue
            ax, ay = center(a)
            bx, by = center(b)
            mx, my = (ax+bx)//2, (ay+by)//2
            if a[0] != b[0]:
                pg.draw.line(screen, (234, 86, 76), (mx,my-scale/2), (mx,my+scale/2), 5)
            else:
                pg.draw.line(screen, (234, 86, 76), (mx-scale/2,my), (mx+scale/2,my), 5)
        if len(nav.path) >= 2:
            pg.draw.lines(screen, (156, 171, 190), False, [center(c) for c in nav.path], 2)
        if self.forward_axis is not None and len(self.raw_track) >= 2:
            pg.draw.lines(screen, (54, 149, 250), False,
                          [center(self.grid_position(p)) for p in self.raw_track], 3)
        for cell, color in ((nav.path[0], (23, 173, 133)),):
            pg.draw.circle(screen, color, center(cell), 9)
        for goal in nav.targets:
            pg.draw.circle(screen, (245, 174, 47), center(goal), 9)
        robot_cell = self.grid_position(self.raw_track[-1]) if self.raw_track else nav.cell
        px, py = center(robot_cell)
        pg.draw.circle(screen, (53, 151, 239), (px, py), 10)
        if self.current_angles and nav.initial_yaw is not None:
            angle = math.radians(wrap(self.current_angles[3]-nav.initial_yaw))
            end = (int(px+math.sin(angle)*26), int(py-math.cos(angle)*26))
            pg.draw.line(screen, (53, 151, 239), (px,py), end, 3)
            pg.draw.circle(screen, (53, 151, 239), end, 4)
        panel_x = int(left+nav.maze.width*scale+22)
        status = ["MAIN2  |  A* route", "Cell: %s" % (nav.cell,),
                  "Heading: %s" % DIRECTION_NAMES[nav.heading],
                  "Phase: %s" % nav.phase, "Steps: %d" % nav.steps,
                  "Targets: %s" % (nav.targets,),
                  "Done: %s" % (sorted(nav.completed_targets),),
                  "Main yaw: %s" % ("%.1f" % nav.navigation_yaw() if nav.initial_yaw is not None else "--"),
                  "Chassis yaw: %s" % ("%.1f" % nav.data.chassis_attitude[0]
                                       if nav.data.chassis_attitude else "--"),
                  "ToF: %s" % ("%.0f mm" % nav.tof.last_raw[nav.tof.index]
                               if nav.tof.last_raw and len(nav.tof.last_raw) > nav.tof.index else "--"),
                  "Q / Esc / close: stop"]
        for index, line in enumerate(status):
            screen.blit(self.font.render(line, True, (230, 235, 240)), (panel_x, 50+index*28))
        pg.display.flip()

    def finish(self):
        self.finished = True
        while not self.stop_requested:
            try:
                self.update(force=True)
            except KeyboardInterrupt:
                break
            time.sleep(.10)
        self.close()

    def close(self):
        self.pg.quit()


class MainNavigator(Route):
    def __init__(self, ep_robot, data, tof, args, config):
        super().__init__(ep_robot, data, args.cell_size)
        self.tof = tof
        self.maze = Maze(config.width, config.height, config.targets[0])
        self.cell = config.start
        self.heading = 0
        self.targets = config.targets
        self.completed_targets = set()
        self.target_scan_outputs = {}
        self.planned_order = ()
        # map_config.json is the route authority. Every internal edge absent
        # from its wall list is open, so no FloodNavigator cell scan is needed.
        for x in range(config.width):
            for y in range(config.height):
                for direction in range(4):
                    cell = (x, y)
                    neighbor = self.maze.neighbor(cell, direction)
                    if self.maze.inside(neighbor):
                        self.maze.edges[self.maze.key(cell, neighbor)] = False
        for a, b in config.walls:
            self.maze.edges[self.maze.key(a, b)] = True
        self.configured_walls = set(config.walls)
        self.last_alignment_step = 0
        self.straight_cells = 0
        self.initial_yaw = None  # Immutable heading reference captured ONCE.
        self.main_yaw = None  # Updated after a deliberate turn or wall yaw correction.
        self.gimbal_yaw_command_sign = getattr(args, "gimbal_yaw_sign", 1)
        self.sharp = SharpMonitor(ep_robot)
        self.last_sharp_log_time = -float("inf")
        self.turn_close_samples = 0
        self.turn_close_sample_time = None
        self.clearance_mm = args.clearance_mm
        self.max_steps = args.max_steps or config.width * config.height * max(10, len(config.targets) * 4)
        self.path = [self.cell]
        self.steps = 0
        self.phase = "starting"
        self.arrived = False
        self.live_map = None
        self.output = Path(__file__).resolve().with_name("main2nos_map.json")

    def navigation_yaw(self):
        return self.main_yaw if self.main_yaw is not None else wrap(self.initial_yaw)

    def calibrate_gimbal_yaw_sign(self):
        """Measure whether a +10 degree SDK yaw command raises or lowers yaw_body."""
        self.stop_chassis()
        before, _ = self.data.read()
        if abs(before[1]) > 5:
            raise RuntimeError("Gimbal is not centered before yaw-sign calibration: %r" % (before,))
        action = self.robot.gimbal.moveto(pitch=0, yaw=10,
                                         pitch_speed=60, yaw_speed=120)
        if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
            raise RuntimeError("Gimbal yaw-sign calibration motion failed")
        completed_at = time.monotonic()
        deadline = completed_at + 2.0
        while time.monotonic() < deadline:
            with self.data.lock:
                if self.data.gimbal_time > completed_at:
                    break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError("No fresh gimbal angle during yaw-sign calibration")
        after, _ = self.data.read()
        delta = wrap(after[1] - before[1])
        LOG.info("GIMBAL YAW SIGN probe commanded=+10 body_before=%.2f body_after=%.2f "
                 "ground_before=%.2f ground_after=%.2f pitch_ground=%.2f",
                 before[1], after[1], before[3], after[3], after[2])
        if not 4 <= abs(delta) <= 20:
            raise RuntimeError("Cannot determine gimbal yaw command sign: delta=%.2f angles=%r" %
                               (delta, after))
        self.gimbal_yaw_command_sign = 1 if delta > 0 else -1
        action = self.robot.gimbal.recenter(pitch_speed=60, yaw_speed=120)
        if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
            raise RuntimeError("Could not recenter gimbal after yaw-sign calibration")
        completed_at = time.monotonic()
        deadline = completed_at + 2.0
        while time.monotonic() < deadline:
            with self.data.lock:
                if self.data.gimbal_time > completed_at:
                    break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError("No fresh angle after yaw-sign calibration recenter")
        centered, _ = self.data.read()
        LOG.info("GIMBAL YAW SIGN result=%+d centered_angles=%r",
                 self.gimbal_yaw_command_sign, centered)
        if abs(centered[1]) > 2 or abs(centered[2]) > 2:
            raise RuntimeError("Gimbal did not recenter after yaw-sign calibration: %r" % (centered,))

    def _align_point(self, offset_yaw):
        """Point to saved main_yaw + offset, converted to body coordinates."""
        self.stop_chassis()
        self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        angles, _ = self.data.read()
        body_ground_yaw = wrap(angles[3] - angles[1])
        desired_ground_yaw = wrap(self.navigation_yaw() + offset_yaw)
        # Preserve the chosen rear branch: -180 +/- 10 must remain -170/-190.
        # wrap(-190) == +170 points the same way, but makes the gimbal sweep
        # almost a full revolution when moving between the two samples.
        relative_yaw = offset_yaw + wrap(self.navigation_yaw() - body_ground_yaw)
        if self.gimbal_yaw_command_sign not in (-1, 1):
            raise RuntimeError("Gimbal yaw sign has not been calibrated")
        commanded_yaw = self.gimbal_yaw_command_sign * relative_yaw
        if abs(commanded_yaw) > 250:
            raise RuntimeError("Alignment gimbal yaw exceeds SDK limit: %.1f" % commanded_yaw)
        for attempt in range(2):
            action = self.robot.gimbal.moveto(pitch=0, yaw=commanded_yaw,
                                             pitch_speed=540, yaw_speed=540)
            if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
                raise RuntimeError("Alignment gimbal motion failed: command=%.1f offset=%+.1f" %
                                   (commanded_yaw, offset_yaw))
            completed_at = time.monotonic()
            deadline = completed_at + 1.5
            while time.monotonic() < deadline:
                with self.data.lock:
                    fresh = self.data.gimbal_time > completed_at
                if fresh:
                    angles, _ = self.data.read()
                    if (abs(wrap(desired_ground_yaw - angles[3])) <= 2.0
                            and abs(angles[2]) <= 1.0):
                        time.sleep(.04)
                        return self.tof.mark()
                time.sleep(LOOP_PERIOD_S)
            angles, _ = self.data.read()
            residual = wrap(desired_ground_yaw - angles[3])
            if attempt == 0 and abs(residual) <= 10.0 and abs(angles[2]) <= 3.0:
                # The body may coast a few degrees after a turn. Correct only
                # that measured residual; never sweep another direction.
                commanded_yaw = self.gimbal_yaw_command_sign * (angles[1] + residual)
                LOG.info("ALIGN gimbal trim offset=%+.1f residual=%+.2f command=%.2f",
                         offset_yaw, residual, commanded_yaw)
                continue
            raise RuntimeError("Gimbal did not settle: offset=%+.1f desired_ground=%.2f "
                               "command=%.2f measured=%r" %
                               (offset_yaw, desired_ground_yaw, commanded_yaw, angles))

    def _align_body_center(self, update_main=True):
        """After a chassis correction, establish the new walking main_yaw."""
        self.stop_chassis()
        self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        action = self.robot.gimbal.moveto(pitch=0, yaw=0,
                                         pitch_speed=540, yaw_speed=540)
        if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
            raise RuntimeError("Could not recenter gimbal after wall alignment")
        completed_at = time.monotonic()
        deadline = completed_at + 1.0
        while time.monotonic() < deadline:
            with self.data.lock:
                fresh = self.data.gimbal_time > completed_at
            if fresh:
                angles, _ = self.data.read()
                if abs(angles[2]) <= 2.0 and abs(angles[1]) <= 2.0:
                    break
            time.sleep(LOOP_PERIOD_S)
        angles, _ = self.data.read()
        if abs(angles[2]) > 2.0 or abs(angles[1]) > 6.0:
            raise RuntimeError("Gimbal did not return to ground pitch=0/body yaw=0 "
                               "after alignment: angles=%r" % (angles,))
        if abs(angles[1]) > 2.0:
            LOG.warning("ALIGN recenter residual body yaw %.2f deg; keeping chassis heading",
                        angles[1])
        if update_main:
            # An SDK recenter may finish with a small body-relative residual.
            # Save the actual chassis heading so the following settle removes
            # that residual without undoing the wall yaw correction.
            self.main_yaw = wrap(angles[3] - angles[1])
        self.target_yaw = self.main_yaw
        self.target_pitch = 0.0

    def _align_read(self, relative_yaw):
        mark = self._align_point(relative_yaw)
        return self.tof.fresh_median(mark)

    @staticmethod
    def alignment_step(name, plus, minus, used):
        """Convert millimetre asymmetry to a bounded yaw command in degrees.

        A 3 mm distance difference is not intrinsically 3 degrees: range and
        scan offset determine the angle. The 4 mm tolerance takes precedence.
        """
        if abs(plus - minus) <= ALIGN_TOLERANCE_MM:
            return 0.0
        error = math.degrees(math.atan2(
            minus - plus, (plus + minus) * math.tan(math.radians(ALIGN_OFFSET_DEG))))
        signed = ALIGN_DIRECTION_SIGN[name] * ALIGN_CHASSIS_PER_GIMBAL_SIGN * error
        remaining = ALIGN_MAX_TOTAL_DEG - used
        if remaining < .1:
            return 0.0
        step = round(limit(ALIGN_KP * signed, min(ALIGN_MAX_STEP_DEG, remaining)), 1)
        # A minimum SDK-sized correction is useful if P would round to zero.
        return step or math.copysign(.1, signed)

    def align_to_wall(self):
        """Find one wall, right before left before back, then align only it."""
        self.phase = "wall_alignment"
        self.stop_chassis()
        selected = None
        # -180 is the same rear heading as +180, and is closer to -90.
        for name, base in (("right",90), ("left",-90), ("back",-180)):
            try:
                distance = self._align_read(base)
                LOG.info("ALIGN scan %s %.1f mm", name, distance)
                if distance <= ALIGN_WALL_MM:
                    selected = (name, base)
                    break
            except RuntimeError as exc:
                LOG.warning("ALIGN scan %s unavailable: %s", name, exc)
        aligned = False
        try:
            if selected is not None:
                name, base = selected
                LOG.info("ALIGN selected %s (search order right, left, back)", name)
                used = 0.0
                for correction in range(ALIGN_MAX_CORRECTIONS + 1):
                    try:
                        self._align_point(base)
                        plus = self._align_read(base + ALIGN_OFFSET_DEG)
                        self._align_point(base)
                        minus = self._align_read(base - ALIGN_OFFSET_DEG)
                        self._align_point(base)
                    except RuntimeError as exc:
                        LOG.warning("ALIGN %s readings unavailable: %s", name, exc)
                        break
                    difference = abs(plus-minus)
                    LOG.info("ALIGN %s check %d: +%.1f -%.1f diff=%.1f mm",
                             name, correction+1, plus, minus, difference)
                    if difference <= ALIGN_TOLERANCE_MM:
                        aligned = True
                        break
                    if correction == ALIGN_MAX_CORRECTIONS:
                        break
                    step = self.alignment_step(name, plus, minus, used)
                    if abs(step) < .1:
                        break
                    action = self.robot.chassis.move(x=0, y=0, z=step, z_speed=120)
                    if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
                        raise RuntimeError("Alignment chassis yaw failed")
                    used += abs(step)
                    self.stop_chassis()
                    time.sleep(.08)
                    self._align_body_center()
                    LOG.info("ALIGN %s yaw %+.1f degrees (%d/%d, total %.1f)",
                             name, step, correction+1, ALIGN_MAX_CORRECTIONS, used)
        finally:
            # Restore the walking heading once, after all readings/corrections.
            self._align_body_center(update_main=False)
            self.settle()
            self.last_alignment_step = self.steps
            self.straight_cells = 0
        LOG.info("ALIGN result selected=%s aligned=%s main_yaw=%.2f",
                 selected[0] if selected else None,
                 aligned, self.main_yaw)
        if selected is not None and not aligned:
            LOG.warning("ALIGN %s outside %.1f mm after %d corrections; continuing",
                        selected[0], ALIGN_TOLERANCE_MM, ALIGN_MAX_CORRECTIONS)
        try:
            self.adjust_front_wall()
        except RuntimeError as exc:
            if (self.maze.edge(self.cell, self.heading) is True
                    or not str(exc).startswith(("No fresh alignment ToF", "Unstable alignment ToF"))):
                raise
            LOG.warning("Front ToF check unavailable on open mapped edge: %s", exc)
        return aligned

    def adjust_front_wall(self, emergency=False):
        """Stop and hold the configured front-wall distance, without changing grid pose."""
        self.stop_chassis()
        self.settle()
        mark = self.tof.mark()
        distance = self.tof.fresh_median(mark)
        if distance > ALIGN_WALL_MM:
            return False
        unexpected = self.maze.edge(self.cell, self.heading) is not True
        LOG.info("FRONT WALL adjusting %.1f toward %.0f mm", distance, FRONT_TARGET_MM)
        _, start = self.data.read()
        deadline = time.monotonic() + 6.0
        try:
            while time.monotonic() < deadline:
                if SHARP_NAV_ENABLED:
                    self._sharp_brake()
                    if self.sharp.read()[0] is None:
                        self.stop_chassis()
                        self.sharp.wait_fresh()
                distance = self.tof.latest()
                error = distance - FRONT_TARGET_MM
                if abs(error) <= FRONT_TOLERANCE_MM:
                    LOG.info("FRONT WALL settled at %.1f mm", distance)
                    if unexpected and not emergency:
                        raise RuntimeError("Front wall conflicts with mapped open path; stopped at %.0f mm" %
                                           FRONT_TARGET_MM)
                    return True
                _, position = self.data.read()
                if math.hypot(position[0]-start[0], position[1]-start[1]) > .25:
                    raise RuntimeError("Front adjustment exceeded 0.25 m")
                Route.control(self, forward=-0.05 if error < 0 else 0.05)
                time.sleep(LOOP_PERIOD_S)
            raise RuntimeError("Front wall did not reach %.0f mm within timeout" % FRONT_TARGET_MM)
        finally:
            self.stop_chassis()

    def _sharp_brake(self, force_center=False):
        left_cm, right_cm = self.sharp.read()
        needs_center = (force_center and left_cm is not None and right_cm is not None
                        and left_cm <= SHARP_WALL_CM and right_cm <= SHARP_WALL_CM
                        and abs(left_cm-right_cm) > SHARP_CENTER_TOLERANCE_CM)
        if (not needs_center and not self.sharp.braking()
                and all(value is None or value > SHARP_CLEARANCE_CM
                        for value in (left_cm, right_cm))):
            return
        previous_phase = self.phase
        self.phase = "sharp_clearance"
        self.stop_chassis()
        _, start = self.data.read()
        deadline = time.monotonic() + 6.0
        impossible_samples = 0
        impossible_since = None
        last_impossible_sample_time = None
        self.sharp.allow_correction(True)
        try:
            while time.monotonic() < deadline:
                left_cm, right_cm = self.sharp.read()
                if left_cm is None or right_cm is None:
                    self.stop_chassis()
                    self.sharp.wait_fresh(timeout=1.0)
                    continue
                try:
                    speed = sharp_clearance_speed(left_cm, right_cm)
                except SafetyStop as exc:
                    if not str(exc).startswith("Both side walls are too close"):
                        raise
                    # A rotating chassis can briefly point both Sharp beams
                    # at corners. Confirm the impossible width with distinct
                    # sensor updates before holding the navigation.
                    self.stop_chassis()
                    with self.sharp.lock:
                        sample_time = self.sharp.sample_time
                    if sample_time != last_impossible_sample_time:
                        if impossible_since is None:
                            impossible_since = time.monotonic()
                        impossible_samples += 1
                        last_impossible_sample_time = sample_time
                    if (impossible_samples >= 3 and
                            time.monotonic() - impossible_since >= .3):
                        raise SafetyStop(
                            "Both side walls remained too close across %d fresh Sharp samples; stopped" %
                            impossible_samples) from None
                    time.sleep(LOOP_PERIOD_S)
                    continue
                impossible_samples = 0
                impossible_since = None
                last_impossible_sample_time = None
                if speed == 0:
                    LOG.info("SHARP CLEARANCE complete left=%.1f right=%.1f cm; resuming %s",
                             left_cm, right_cm, previous_phase)
                    return
                _, position = self.data.read()
                if math.hypot(position[0]-start[0], position[1]-start[1]) > SHARP_CLEARANCE_MAX_TRAVEL_M:
                    if min(left_cm, right_cm) > SHARP_RELEASE_CM:
                        LOG.warning("SHARP CLEARANCE reached %.2f m travel limit; "
                                    "both sides clear of brake, resuming %s",
                                    SHARP_CLEARANCE_MAX_TRAVEL_M, previous_phase)
                        return
                    raise SafetyStop("Sharp clearance exceeded %.2f m while still within brake range" %
                                     SHARP_CLEARANCE_MAX_TRAVEL_M)
                self.robot.chassis.drive_speed(x=0, y=speed, z=0, timeout=.5)
                if self.live_map is not None:
                    self.live_map.update(force=True)
                time.sleep(LOOP_PERIOD_S)
            left_cm, right_cm = self.sharp.read()
            if (left_cm is not None and right_cm is not None
                    and min(left_cm, right_cm) > SHARP_RELEASE_CM):
                LOG.warning("SHARP CLEARANCE timed out before ideal 7 cm/center; "
                            "brake cleared, resuming %s", previous_phase)
                return
            raise SafetyStop("Sharp clearance timed out while still within brake range")
        finally:
            self.stop_chassis()
            self.sharp.allow_correction(False)
            self.phase = previous_phase

    def control(self, forward=0.0, follow=True, lateral=0.0):
        if SHARP_NAV_ENABLED and self.phase == "turning":
            # Side ranges change as the body rotates. Do not recenter from
            # those changing readings; only pause for a confirmed 3 cm risk.
            left_cm, right_cm = self.sharp.read()
            if left_cm is None or right_cm is None:
                self.stop_chassis()
                self.sharp.wait_fresh(timeout=1.0)
                return Route.control(self, forward=0.0, follow=False)
            if min(left_cm, right_cm) <= SHARP_STOP_CM:
                self.stop_chassis()
                with self.sharp.lock:
                    sample_time = self.sharp.sample_time
                if sample_time != self.turn_close_sample_time:
                    self.turn_close_samples += 1
                    self.turn_close_sample_time = sample_time
                if self.turn_close_samples >= 3:
                    LOG.warning("Sharp stayed within 3 cm during turn; correcting side clearance")
                    self.turn_close_samples = 0
                    self.turn_close_sample_time = None
                    self.sharp.allow_correction(False)
                    try:
                        self._sharp_brake()
                    finally:
                        self.sharp.allow_correction(True)
                return Route.control(self, forward=0.0, follow=False)
            self.turn_close_samples = 0
            self.turn_close_sample_time = None
        elif SHARP_NAV_ENABLED:
            self._sharp_brake()
        left_cm, right_cm = self.sharp.read() if SHARP_NAV_ENABLED else (None, None)
        if SHARP_NAV_ENABLED and forward > 0 and (left_cm is None or right_cm is None):
            self.stop_chassis()
            left_cm, right_cm = self.sharp.wait_fresh()
        lateral = (sharp_center_speed(left_cm, right_cm)
                   if SHARP_NAV_ENABLED and forward > 0 else 0.0)
        if lateral and time.monotonic() - self.last_sharp_log_time >= 1.0:
            LOG.info("SHARP CENTER left=%.1f cm right=%.1f cm lateral=%+.3f m/s",
                     left_cm, right_cm, lateral)
            self.last_sharp_log_time = time.monotonic()
        if self.phase == "moving":
            # Side-scan targets cannot leak into a driving command.
            self.target_yaw = self.navigation_yaw()
        if forward > 0:
            distance = self.tof.latest()
            if distance <= self.clearance_mm:
                self.stop_chassis()
                try:
                    self.adjust_front_wall(emergency=True)
                except SafetyStop:
                    raise
                except RuntimeError as exc:
                    raise FrontObstacleStop(
                        "Front ToF %.1f mm during movement; backoff unavailable: %s" %
                        (distance, exc)) from None
                raise FrontWallAdjusted(
                    "Front ToF %.1f mm during movement; backed toward %.0f mm" %
                    (distance, FRONT_TARGET_MM))
        result = super().control(forward=forward, follow=follow, lateral=lateral)
        if self.live_map is not None:
            angles, position = self.data.read()
            self.live_map.sample(angles, position)
        return result


    def turn_to(self, direction):
        delta = (direction - self.heading) % 4
        increments = {0: (), 1: (1,), 2: (1, 1), 3: (-1,)}[delta]
        if not increments:
            return
        if SHARP_NAV_ENABLED:
            # Only the Sharp-enabled variant corrects its lateral pose here.
            self.phase = "pre_turn_clearance"
            LOG.info("PRE-TURN Sharp check before %s turn", DIRECTION_NAMES[direction])
            self._sharp_brake(force_center=True)
        self.phase = "turning"
        self.turn_close_samples = 0
        self.turn_close_sample_time = None
        if SHARP_NAV_ENABLED:
            self.sharp.allow_correction(True)
        try:
            self._turn_increments(increments)
        finally:
            self.stop_chassis()
            if SHARP_NAV_ENABLED:
                self.sharp.allow_correction(False)

    def _turn_increments(self, increments):
        for increment in increments:
            previous_main = self.navigation_yaw()
            next_heading = (self.heading + increment) % 4
            self.target_yaw = wrap(previous_main + increment * 90.0)
            LOG.info("TURN previous_main=%.2f offset=%+d target=%.2f heading=%s",
                     previous_main, increment * 90, self.target_yaw, DIRECTION_NAMES[next_heading])
            # Gimbal moves first; chassis then catches up. A U-turn is two
            # 90-degree turns to avoid swinging the gimbal through its limit.
            self.settle(follow=False)
            self.settle(follow=True)
            angles, _ = self.data.read()
            if (abs(wrap(self.target_yaw-angles[3])) > ANGLE_TOLERANCE_DEG
                    or abs(wrap(angles[1])) > ANGLE_TOLERANCE_DEG):
                raise RuntimeError("Turn not aligned; previous main yaw retained")
            self.main_yaw = wrap(angles[3])
            self.heading = next_heading
            self.target_yaw = self.main_yaw
            LOG.info("MAIN YAW SAVED previous=%.2f new=%.2f body_offset=%.2f",
                     previous_main, self.main_yaw, angles[1])
            self.save()

    def save(self):
        costs = self.maze.flood()
        try:
            angles, position = self.data.read()
        except RuntimeError:
            if self.phase not in ("stopped", "safety_stop", "front_obstacle_stop"):
                raise
            angles, position = self.data.read_cached()
        with self.sharp.lock:
            sharp_state = dict(left_cm=self.sharp.left_cm,
                               right_cm=self.sharp.right_cm,
                               brake_active=self.sharp.too_close)
        payload = dict(width=self.maze.width, height=self.maze.height,
                       cell_size_m=self.cell_size, targets=self.targets,
                       completed_targets=sorted(self.completed_targets),
                       target_scan_outputs={"%d,%d" % cell: path
                                            for cell, path in self.target_scan_outputs.items()},
                       planned_order=self.planned_order,
                       configured_walls=sorted(self.configured_walls),
                       current_cell=self.cell, heading=DIRECTION_NAMES[self.heading],
                       initial_yaw=self.initial_yaw, main_yaw=self.main_yaw,
                       navigation_yaw=self.navigation_yaw(),
                       actual_gimbal_yaw=angles[3], odometry_position=position,
                       sharp=sharp_state,
                       chassis_attitude=self.data.chassis_attitude,
                       phase=self.phase, all_targets_reached=self.arrived, path=self.path,
                       visited=sorted(self.maze.visited),
                       flood_values=[[costs.get((x, y)) for x in range(self.maze.width)]
                                     for y in range(self.maze.height)],
                       edges=[dict(a=a, b=b, wall=wall) for (a, b), wall in sorted(self.maze.edges.items())],
                       notes="A* uses mapped walls and the saved main_yaw for navigation. Sharp stays out of normal walking and turning; during map_scan an under-3 cm side is moved away to 6 cm. ToF corrects front clearance toward 170 mm at 120 mm. map_scan runs after each target and fires gel rounds; rejected targets return by wheel distance.")
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.output)
        if self.live_map is not None:
            self.live_map.update(force=True)

    def run(self, on_target=None):
        angles, _ = self.data.read()
        self.initial_yaw = angles[3]
        self.main_yaw = wrap(angles[3])
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = angles[2]
        LOG.info("START cell=%s targets=%s map=%dx%d initial_yaw=%.2f",
                 self.cell, self.targets, self.maze.width, self.maze.height, self.initial_yaw)
        try:
            self.settle()
            if self.cell in self.targets:
                self.completed_targets.add(self.cell)
            self.save()
            if self.cell in self.completed_targets and on_target is not None:
                try:
                    on_target(self)
                except BaseException:
                    self.completed_targets.discard(self.cell)
                    raise
            while len(self.completed_targets) < len(self.targets):
                if self.steps >= self.max_steps:
                    raise RuntimeError("Step limit reached before goal")
                remaining = set(self.targets) - self.completed_targets
                order = optimal_target_order(self.maze, self.cell, remaining)
                if not order:
                    raise RuntimeError("Remaining targets unreachable through configured walls")
                self.planned_order = order
                self.maze.goal = order[0]  # Display flood distance to current A* target.
                route = a_star(self.maze, self.cell, order[0])
                if route is None or len(route) < 2:
                    raise RuntimeError("No route to next target")
                next_cell = route[1]
                direction = DIRECTIONS.index((next_cell[0]-self.cell[0], next_cell[1]-self.cell[1]))
                if self.maze.edge(self.cell, direction) is not False:
                    raise RuntimeError("A* selected a wall or unknown edge")
                LOG.info("A* order=%s path=%s choose=%s -> %s", order, route, self.cell, next_cell)
                old_heading = self.heading
                self.turn_to(direction)
                if self.heading != old_heading:
                    self.align_to_wall()
                # The configured map selects the passage; the current ToF
                # still guards against an unexpected near obstacle.
                if self.tof.latest() <= self.clearance_mm:
                    LOG.warning("Front ToF %.1f mm before moving; correcting toward %.0f mm",
                                self.tof.latest(), FRONT_TARGET_MM)
                    self.adjust_front_wall(emergency=True)
                    if self.tof.latest() <= self.clearance_mm:
                        raise FrontObstacleStop("Front ToF remains within %.0f mm after adjustment" %
                                                self.clearance_mm)
                self.phase = "moving"
                self.forward_cell(self.steps + 1)
                self.maze.observe(self.cell, direction, False)
                self.cell = next_cell
                self.steps += 1
                self.path.append(self.cell)
                self.maze.visited.add(self.cell)
                self.phase = "at_cell"
                self.straight_cells += 1
                new_target = self.cell in self.targets and self.cell not in self.completed_targets
                if new_target:
                    self.completed_targets.add(self.cell)
                    LOG.info("TARGET reached %s (%d/%d)", self.cell,
                             len(self.completed_targets), len(self.targets))
                if self.straight_cells >= 3:
                    self.align_to_wall()
                    self.phase = "at_cell"
                self.save()
                if new_target and on_target is not None:
                    try:
                        on_target(self)
                    except BaseException:
                        self.completed_targets.discard(self.cell)
                        raise
            self.stop_chassis()
            self.arrived = True
            self.phase = "all_targets_reached"
            self.save()
            LOG.info("ALL TARGETS REACHED cell=%s completed_moves=%d", self.cell, self.steps)
        except SafetyStop as exc:
            self.stop_chassis()
            self.phase = ("front_obstacle_stop" if isinstance(exc, FrontObstacleStop)
                          else "safety_stop")
            LOG.warning("NAVIGATION HELD: %s; cell=%s; target not marked reached",
                        exc, self.cell)
            try:
                self.save()
            except Exception:
                LOG.exception("Could not save safety stop map")
            return
        except BaseException:
            self.stop_chassis()
            self.phase = "stopped"
            try:
                self.save()
            except Exception:
                LOG.exception("Could not save final map")
            raise



class _SharpGuardedChassis:
    def __init__(self, chassis, sharp):
        self.chassis = chassis
        self.sharp = sharp

    def __getattr__(self, name):
        return getattr(self.chassis, name)

    def move(self, **kwargs):
        if self.sharp.braking():
            self.chassis.drive_speed(x=0, y=0, z=0)
            self.sharp.wait_clear()
        if self.sharp.read()[0] is None:
            self.chassis.drive_speed(x=0, y=0, z=0)
            self.sharp.wait_fresh()
        return self.chassis.move(**kwargs)

    def drive_speed(self, **kwargs):
        if any(kwargs.get(axis, 0) for axis in ("x", "y", "z")):
            if self.sharp.braking():
                self.chassis.drive_speed(x=0, y=0, z=0)
                self.sharp.wait_clear()
            if self.sharp.read()[0] is None:
                self.chassis.drive_speed(x=0, y=0, z=0)
                self.sharp.wait_fresh()
        return self.chassis.drive_speed(**kwargs)


class _SharedRobotForMapScan:
    """Keep main2 as owner of the SDK connection used by map_scan.py."""

    def __init__(self, connected_robot, sharp=None):
        self.connected_robot = connected_robot
        self.chassis = (_SharpGuardedChassis(connected_robot.chassis, sharp)
                        if sharp is not None else connected_robot.chassis)

    def __getattr__(self, name):
        return getattr(self.connected_robot, name)

    def initialize(self, **_kwargs):
        return True

    def close(self):
        return None


def load_target_scanner(path):
    """Load map_scan.py and its adjacent helper modules only when needed."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("map_scan.py not found: %s" % path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("main2_map_scan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load map_scan.py: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_yaw_scan_tracker(original_tracker, main_yaw):
    """Expose an unwrapped ground yaw offset from the saved navigation yaw."""
    class MainYawScanTracker(original_tracker):
        def __init__(self):
            super().__init__()
            self._last_ground_yaw = None

        def callback(self, angles):
            ground_yaw = float(angles[3])
            with self.lock:
                self.pitch_angle = float(angles[0])
                if self._last_ground_yaw is None:
                    self.yaw = wrap(ground_yaw - main_yaw)
                else:
                    self.yaw += wrap(ground_yaw - self._last_ground_yaw)
                self._last_ground_yaw = ground_yaw
                self.updated_at = time.monotonic()

    return MainYawScanTracker


def record_gel_round(bot, scanner, target_map, saved, output_path):
    """Persist a gel-shot attempt before sending it; never retry a lost ACK."""
    saved["fire_type"] = "gel"
    saved["gel_status"] = "pending"
    target_map.save(output_path)
    try:
        result = bot.blaster.fire(fire_type=scanner.blaster.WATER_FIRE, times=1)
    except Exception:
        saved["gel_status"] = "unknown"
        target_map.save(output_path)
        raise
    saved["gel_status"] = "confirmed" if result is True else "unconfirmed"
    target_map.save(output_path)
    if result is not True:
        raise RuntimeError("Gel shot not confirmed; no automatic retry")


def run_target_scan(nav, scanner, requested_codes, output_path, right_yaw_sign):
    """Pause walking, run map_scan.py's full detector/aim/fire cycle, then resume.

    map_scan.py usually creates/closes its own robot and repeats alignment.
    Give it this stopped robot, and keep main2's alignment as the sole wall
    alignment controller. Angle subscriptions are handed over and restored.
    """
    nav.phase = "target_scan"
    nav.stop_chassis()
    nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    saved_main_yaw = nav.navigation_yaw()
    original_tracker = getattr(scanner, "GimbalTracker", None)
    original_reset = getattr(scanner, "reset_gimbal", None)
    original_target_map = getattr(scanner, "TargetMap", None)
    if original_tracker is not None and original_reset is not None:
        # Do this while main2 still owns fresh ground-angle telemetry.
        nav._align_point(0)
        angles, _ = nav.data.read()
        LOG.info("TARGET SCAN yaw anchor main=%.2f ground=%.2f offset=%+.2f",
                 saved_main_yaw, angles[3], wrap(angles[3]-saved_main_yaw))
    start_position = nav.data.read_position()
    robot_module = scanner.robot
    original_factory = robot_module.Robot
    original_alignment = scanner.align_before_scan
    original_offset_move = getattr(scanner, "OffsetMove", None)
    original_record_and_fire = getattr(scanner, "record_and_fire", None)
    original_selection_reason = getattr(scanner, "selection_reason", None)
    original_tracking_bearing = getattr(scanner, "TRACK_BEARING_DEG", None)
    original_target_lost = getattr(scanner, "TARGET_LOST_S", None)
    original_wait_key = getattr(getattr(scanner, "cv2", None), "waitKey", None)
    fire_count = [0]
    paused_angle = paused_attitude = False
    if original_tracker is not None and original_reset is not None:
        scanner.GimbalTracker = main_yaw_scan_tracker(original_tracker, saved_main_yaw)

        if original_target_map is not None:
            class MainYawTargetMap(original_target_map):
                def __init__(self, _chassis_yaw):
                    super().__init__(saved_main_yaw)

            scanner.TargetMap = MainYawTargetMap

        def reset_pitch_at_main_yaw(gimbal):
            # map_scan normally recenters yaw to the body. Keep the saved
            # ground yaw; only apply its customary camera pitch offset.
            pitch = getattr(scanner, "HOME_PITCH_DEG", -3.0)
            action = gimbal.move(pitch=pitch, yaw=0)
            if action is not None:
                if not action.wait_for_completed(timeout=5):
                    raise RuntimeError("Target-scan pitch adjustment did not complete")
                if getattr(action, "has_succeeded", True) is False:
                    raise RuntimeError("Target-scan pitch adjustment failed")

        scanner.reset_gimbal = reset_pitch_at_main_yaw
    if original_offset_move is not None:
        class CenteredOffsetMove(original_offset_move):
            """Move the scan offset by wheel distance, without SDK move ACKs."""

            def begin(self, bearing_deg=None, returning=False):
                self.returning = returning
                self.started = time.monotonic()
                self.completed_at = None
                self._move_start = nav.data.read_position()
                self._centering = False
                self._escape_side = None
                self._escape_started = None
                self._escape_position = None
                self._offset_done_m = 0.0
                if not returning:
                    self._fire_count_at_departure = fire_count[0]
                    sector = int(((bearing_deg % 360) + 45) // 90) % 4
                    self.offset = ((-.10, 0.0), (0.0, -.10),
                                   (.10, 0.0), (0.0, .10))[sector]
                    self._axis = "x" if self.offset[0] else "y"
                    self._direction = 1 if (self.offset[0] or self.offset[1]) > 0 else -1
                    self._target_m = .10
                    LOG.info("TARGET SCAN step away x=%+.2f y=%+.2f m (wheel measured)",
                             *self.offset)
                else:
                    # Always restore the measured outward offset with wheels.
                    # Sharp centering starts only after this return, when both
                    # side walls are present.
                    travelled = getattr(self, "_away_travel_m", None)
                    if travelled is None or not .03 <= travelled <= .14:
                        raise SafetyStop("Target-scan outward wheel distance unavailable/out of bounds")
                    self._axis = "x" if self.offset[0] else "y"
                    self._direction = -1 if (self.offset[0] or self.offset[1]) > 0 else 1
                    self._target_m = travelled
                    LOG.info("TARGET SCAN return %s %.3f m by wheel position%s",
                             "fore/aft" if self._axis == "x" else "side", travelled,
                             " (target not confirmed)" if fire_count[0] == self._fire_count_at_departure
                             else "")
                self._center_after_return = fire_count[0] > self._fire_count_at_departure

            def poll(self):
                if not self._centering:
                    position = nav.data.read_position()
                    left_cm, right_cm = nav.sharp.read()
                    if left_cm is not None and right_cm is not None:
                        if self._escape_side is None:
                            if left_cm <= SHARP_STOP_CM or right_cm <= SHARP_STOP_CM:
                                # Pause the scan offset. Its wheel progress is
                                # retained separately from this lateral escape.
                                self._offset_done_m += math.hypot(
                                    position[0]-self._move_start[0],
                                    position[1]-self._move_start[1])
                                self.bot.chassis.drive_speed(x=0, y=0, z=0)
                                self._escape_side = ("left" if left_cm <= right_cm
                                                     else "right")
                                self._escape_position = position
                                self._escape_started = time.monotonic()
                                LOG.warning("TARGET SCAN Sharp %s %.1f cm <= %.1f; "
                                            "sliding away until %.1f cm",
                                            self._escape_side,
                                            min(left_cm, right_cm), SHARP_STOP_CM,
                                            SHARP_ESCAPE_CM)
                        if self._escape_side is not None:
                            near = left_cm if self._escape_side == "left" else right_cm
                            opposite = right_cm if self._escape_side == "left" else left_cm
                            if near >= SHARP_ESCAPE_CM:
                                self.bot.chassis.drive_speed(x=0, y=0, z=0)
                                self._move_start = position
                                self.started = time.monotonic()
                                LOG.info("TARGET SCAN Sharp %s clearance %.1f cm; "
                                         "resuming offset", self._escape_side, near)
                                self._escape_side = None
                                return False
                            if opposite <= SHARP_STOP_CM:
                                self.bot.chassis.drive_speed(x=0, y=0, z=0)
                                raise SafetyStop("Both Sharp sides are within 3 cm; "
                                                 "cannot slide safely to 6 cm")
                            escape_travel = math.hypot(
                                position[0]-self._escape_position[0],
                                position[1]-self._escape_position[1])
                            if (time.monotonic()-self._escape_started > SHARP_ESCAPE_TIMEOUT_S
                                    or escape_travel > SHARP_ESCAPE_MAX_TRAVEL_M):
                                self.bot.chassis.drive_speed(x=0, y=0, z=0)
                                raise SafetyStop("Sharp %s remained below 6 cm "
                                                 "after bounded lateral escape" %
                                                 self._escape_side)
                            direction = 1 if self._escape_side == "left" else -1
                            self.bot.chassis.drive_speed(
                                x=0, y=direction * SHARP_CLEARANCE_SPEED_MPS,
                                z=0, timeout=.5)
                            return False
                    elif self._escape_side is not None:
                        self.bot.chassis.drive_speed(x=0, y=0, z=0)
                        raise SafetyStop("Sharp reading lost during 6 cm escape")
                    travelled = self._offset_done_m + math.hypot(
                        position[0]-self._move_start[0],
                        position[1]-self._move_start[1])
                    if self._axis is not None and travelled > self._target_m + .03:
                        self.bot.chassis.drive_speed(x=0, y=0, z=0)
                        raise SafetyStop("Target-scan wheel offset overshot: %.3f/%.3f m" %
                                         (travelled, self._target_m))
                    if self._axis is not None and travelled < self._target_m - .005:
                        if time.monotonic() - self.started > 5.0:
                            self.bot.chassis.drive_speed(x=0, y=0, z=0)
                            raise SafetyStop("Target-scan wheel offset stalled: "
                                             "%.3f/%.3f m" % (travelled, self._target_m))
                        remaining = self._target_m - travelled
                        speed = min(.12, max(.035, 1.5 * remaining)) * self._direction
                        self.bot.chassis.drive_speed(
                            x=speed if self._axis == "x" else 0,
                            y=speed if self._axis == "y" else 0,
                            z=0, timeout=.5)
                        return False
                    self.bot.chassis.drive_speed(x=0, y=0, z=0)
                    if self.completed_at is None:
                        self.completed_at = time.monotonic()
                    if time.monotonic() - self.completed_at < .20:
                        return False
                    if not self.returning:
                        self._away_travel_m = travelled
                        self.away = True
                        LOG.info("TARGET SCAN step away complete %.3f m by wheel position",
                                 travelled)
                        return True
                    if not self._center_after_return:
                        self.away = False
                        LOG.info("TARGET SCAN rejected target; scan pose restored by wheels")
                        return True
                    left_cm, right_cm = nav.sharp.read()
                    if (left_cm is None or right_cm is None or
                            left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM):
                        self.away = False
                        LOG.info("TARGET SCAN return restored by wheels; Sharp centering skipped without two side walls")
                        return True
                    self._centering = True
                    self._center_started = time.monotonic()
                    self._center_position = nav.data.read_position()
                    self.away = True
                left_cm, right_cm = nav.sharp.read()
                if left_cm is None or right_cm is None:
                    self.bot.chassis.drive_speed(x=0, y=0, z=0)
                    self.away = False
                    LOG.warning("TARGET SCAN Sharp became unavailable after wheel return; using wheel-restored pose")
                    return True
                if left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM:
                    self.bot.chassis.drive_speed(x=0, y=0, z=0)
                    self.away = False
                    LOG.info("TARGET SCAN Sharp centering ended: two side walls no longer present")
                    return True
                speed = sharp_clearance_speed(left_cm, right_cm)
                if speed == 0:
                    self.bot.chassis.drive_speed(x=0, y=0, z=0)
                    self.away = False
                    LOG.info("TARGET SCAN return centered by Sharp left=%.1f right=%.1f cm",
                             left_cm, right_cm)
                    return True
                position = nav.data.read_position()
                if (time.monotonic() - self._center_started > 3.0
                        or math.hypot(position[0]-self._center_position[0],
                                      position[1]-self._center_position[1]) > .05):
                    self.bot.chassis.drive_speed(x=0, y=0, z=0)
                    self.away = False
                    if min(left_cm, right_cm) <= SHARP_STOP_CM:
                        raise SafetyStop("Sharp remains within brake distance after wheel return; stopped")
                    LOG.warning("TARGET SCAN Sharp centering reached limit after wheel return; continuing from measured pose left=%.1f right=%.1f cm", left_cm, right_cm)
                    return True
                self.bot.chassis.drive_speed(x=0, y=speed, z=0, timeout=.5)
                return False

    try:
        if nav.robot.gimbal.unsub_angle() is False:
            raise RuntimeError("Could not pause navigation gimbal subscription")
        paused_angle = True
        if nav.robot.chassis.unsub_attitude() is False:
            raise RuntimeError("Could not pause navigation chassis subscription")
        paused_attitude = True
        # The camera sweep and firing use their own motions. Sharp intervenes
        # only for a critical side escape or return centering during map_scan.
        nav.sharp.allow_correction(True)
        robot_module.Robot = lambda: _SharedRobotForMapScan(nav.robot)
        scanner.align_before_scan = lambda _bot: {"main2_alignment": True}
        if original_offset_move is not None:
            scanner.OffsetMove = CenteredOffsetMove
        if original_selection_reason is not None:
            def requested_shape_and_color(item, codes, *selection_args):
                # map_scan's original selection checks color but can start an
                # aim at a different shape of that color.
                if item[0] not in codes:
                    return "NOT REQUESTED"
                return original_selection_reason(item, codes, *selection_args)

            scanner.selection_reason = requested_shape_and_color
        if original_tracking_bearing is not None:
            # A 10 cm camera offset changes bearing through parallax.
            scanner.TRACK_BEARING_DEG = max(original_tracking_bearing, 18.0)
        if original_target_lost is not None:
            scanner.TARGET_LOST_S = max(original_target_lost, 1.2)
        LOG.info("TARGET SCAN requested codes=%s; target lock %.1f deg, "
                 "lost timeout %.1f s; gel fire",
                 sorted(requested_codes),
                 getattr(scanner, "TRACK_BEARING_DEG", float("nan")),
                 getattr(scanner, "TARGET_LOST_S", float("nan")))
        if original_record_and_fire is not None:
            def record_and_fire_gel(bot, target_map, saved, output_path):
                fire_count[0] += 1
                record_gel_round(bot, scanner, target_map, saved, output_path)
                LOG.info("TARGET SCAN gel round fired at cell=%s", nav.cell)

            scanner.record_and_fire = record_and_fire_gel
        if original_wait_key is not None:
            # map_scan.py normally leaves its final map open with waitKey(0).
            # This integrated run must continue to the next mapped goal.
            scanner.cv2.waitKey = lambda delay: original_wait_key(1 if delay == 0 else delay)
        scanner.run(Path(output_path), right_yaw_sign, requested_codes)
        diagnostics = Path(output_path).with_name(Path(output_path).stem + "_diagnostics.json")
        if not diagnostics.is_file() or not json.loads(diagnostics.read_text(encoding="utf-8")).get("completed"):
            raise RuntimeError("map_scan.py ended before its camera sweep completed")
    finally:
        nav.stop_chassis()
        nav.sharp.allow_correction(False)
        robot_module.Robot = original_factory
        scanner.align_before_scan = original_alignment
        if original_tracker is not None and original_reset is not None:
            scanner.GimbalTracker = original_tracker
            scanner.reset_gimbal = original_reset
            if original_target_map is not None:
                scanner.TargetMap = original_target_map
        if original_offset_move is not None:
            scanner.OffsetMove = original_offset_move
        if original_record_and_fire is not None:
            scanner.record_and_fire = original_record_and_fire
        if original_selection_reason is not None:
            scanner.selection_reason = original_selection_reason
        if original_tracking_bearing is not None:
            scanner.TRACK_BEARING_DEG = original_tracking_bearing
        if original_target_lost is not None:
            scanner.TARGET_LOST_S = original_target_lost
        if original_wait_key is not None:
            scanner.cv2.waitKey = original_wait_key
        if paused_angle:
            if not nav.robot.gimbal.sub_angle(freq=20, callback=nav.data.on_gimbal):
                raise RuntimeError("Could not restore gimbal telemetry after target scan")
        if paused_attitude:
            if not nav.robot.chassis.sub_attitude(freq=20, callback=nav.data.on_attitude):
                raise RuntimeError("Could not restore chassis telemetry after target scan")
    # The camera sweep may end at either rear limit. Restore the saved ground
    # main_yaw directly instead of recentering relative to the chassis.
    nav.main_yaw = saved_main_yaw
    nav.target_yaw = saved_main_yaw
    nav.target_pitch = 0.0
    completed_at = time.monotonic()
    deadline = completed_at + 3.0
    while time.monotonic() < deadline:
        with nav.data.lock:
            if nav.data.gimbal_time > completed_at:
                break
        time.sleep(LOOP_PERIOD_S)
    else:
        raise RuntimeError("No fresh gimbal angle after target scan")
    nav._align_point(0)
    end_position = nav.data.read_position()
    residual = math.hypot(end_position[0]-start_position[0],
                          end_position[1]-start_position[1])
    if residual > .16:
        raise SafetyStop("Target scan left robot %.3f m from cell center; stopped" % residual)
    nav.settle(follow=True)
    nav.phase = "at_cell"
    LOG.info("TARGET SCAN complete cell=%s result=%s, residual=%.3f m, main_yaw=%.2f",
             nav.cell, output_path, residual, nav.main_yaw)


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
                # Sliding away from the close side is the safe return motion.
                moving_away = (axis == "y" and
                               ((metres > 0 and left_cm is not None and left_cm <= SHARP_STOP_CM
                                 and (right_cm is None or right_cm > SHARP_STOP_CM)) or
                                (metres < 0 and right_cm is not None and right_cm <= SHARP_STOP_CM
                                 and (left_cm is None or left_cm > SHARP_STOP_CM))))
                if not moving_away:
                    nav.stop_chassis()
                    # One noisy Sharp sample must not cancel a target scan.
                    clear_by = time.monotonic() + .35
                    while nav.sharp.braking() and time.monotonic() < clear_by:
                        time.sleep(LOOP_PERIOD_S)
                    # The chassis can coast while the brake command settles.
                    # Return by the final wheel distance, not its earlier pose.
                    position = nav.data.read_position()
                    distance = math.hypot(position[0] - start[0],
                                          position[1] - start[1])
                    if nav.sharp.braking():
                        left_cm, right_cm = nav.sharp.read()
                        raise ScanOffsetBlocked(distance, left_cm, right_cm)
            if distance >= requested - .005:
                if distance > requested + .03:
                    raise SafetyStop("Target-scan offset overshot %.3f/%.3f m" %
                                     (distance, requested))
                return distance
            if time.monotonic() - started > 6.0:
                raise SafetyStop("Target-scan offset stalled %.3f/%.3f m" %
                                 (distance, requested))
            speed = math.copysign(min(.12, max(.04, 1.5 * (requested - distance))), metres)
            nav.robot.chassis.drive_speed(x=speed if axis == "x" else 0,
                                          y=speed if axis == "y" else 0,
                                          z=0, timeout=.5)
            time.sleep(LOOP_PERIOD_S)
    finally:
        nav.stop_chassis()


def _scan_center_between_two_walls(nav):
    """Sharp is consulted only after returning, and only for two side walls."""
    left_cm, right_cm = nav.sharp.read()
    if (left_cm is None or right_cm is None or
            left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM):
        LOG.info("TARGET SCAN Sharp centering skipped: two side walls not present")
        return
    start = nav.data.read_position()
    deadline = time.monotonic() + 6.0
    try:
        while time.monotonic() < deadline:
            left_cm, right_cm = nav.sharp.read()
            if left_cm is None or right_cm is None:
                raise SafetyStop("Sharp data lost while centering after target scan")
            if left_cm > SHARP_WALL_CM or right_cm > SHARP_WALL_CM:
                LOG.info("TARGET SCAN Sharp centering ended: two walls no longer detected")
                return
            speed = sharp_clearance_speed(left_cm, right_cm)
            if speed == 0:
                LOG.info("TARGET SCAN Sharp centered left=%.1f right=%.1f cm",
                         left_cm, right_cm)
                return
            position = nav.data.read_position()
            if math.hypot(position[0]-start[0], position[1]-start[1]) > .15:
                raise SafetyStop("Sharp scan centering exceeded 15 cm")
            nav.robot.chassis.drive_speed(x=0, y=speed, z=0, timeout=.5)
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop("Sharp scan centering timed out")
    finally:
        nav.stop_chassis()


SCAN_CAMERA_WINDOW = "RoboMaster scan"
SCAN_MASK_WINDOW = "Detection mask"
SCAN_MAP_WINDOW = "1x1 target map"


def _show_scan_camera(scanner, frame, found, direction, offset, ranges=None,
                      target_map=None, requested_codes=None, selected=None,
                      nav=None, state="SCAN"):
    """Render map_scan's three views and saved-target overlays."""
    cv2 = getattr(scanner, "cv2", None)
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
        marks = [(item, scanner.project_saved_mark(item, width, height,
                                                   turn_deg, pitch_deg))
                 for item in target_map.targets
                 if item.get("scan_direction") == direction]
    for code, (x, y, w, h), cx, _cy in found:
        recorded = any(scanner.mark_inside_box(point, (x, y, w, h),
                                               scanner.MARK_MATCH_MARGIN_PX)
                       for _item, point in marks)
        paint = (0, 0, 0) if recorded else scanner.DISPLAY_COLORS[code[1]]
        cv2.rectangle(view, (x, y), (x+w, y+h), paint,
                      -1 if recorded else 2)
        if requested_codes is None:
            label = code
        elif recorded:
            label = "IGNORED " + code
        else:
            reason = "READY" if code in requested_codes else "NOT REQUESTED"
            label = ("%s %s %s %+.1f deg | %s" %
                     (code, scanner.SHAPE_NAMES[code[0]],
                      scanner.COLOR_NAMES[code[1]],
                      scanner.target_bearing(cx, width, turn_deg), reason))
        cv2.putText(view, label, (x, max(18, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (255, 255, 255) if recorded else paint, 2)
    for item, point in marks:
        if point is None:
            continue
        paint = scanner.DISPLAY_COLORS[item["code"][1]]
        cv2.circle(view, point, 9, (0, 0, 0), 3)
        cv2.circle(view, point, 7, paint, -1)
        cv2.putText(view, "SAVED " + item["code"],
                    (min(point[0]+10, width-110), max(point[1]-10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, paint, 2)
    center = (width // 2, height // 2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    if requested_codes is not None:
        accept_x = max(24, round(width * scanner.ACCEPT_HALF_WIDTH_FRACTION))
        accept_y = max(20, round(height * scanner.ACCEPT_HALF_HEIGHT_FRACTION))
        inner_x = max(8, round(width * scanner.INNER_HALF_WIDTH_FRACTION))
        inner_y = max(8, round(height * scanner.INNER_HALF_HEIGHT_FRACTION))
        cv2.rectangle(view, (center[0]-accept_x, center[1]-accept_y),
                      (center[0]+accept_x, center[1]+accept_y), (0, 165, 255), 2)
        cv2.rectangle(view, (center[0]-inner_x, center[1]-inner_y),
                      (center[0]+inner_x, center[1]+inner_y), (0, 255, 0), 1)
    if selected is not None:
        x, y, w, h = selected[1]
        cv2.rectangle(view, (x, y), (x+w, y+h), (255, 0, 255), 3)
        cv2.putText(view, "FOCUS " + selected[0],
                    (x, min(y+h+22, height-5)), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 0, 255), 2)
    cv2.putText(view, "%s | %s %+d deg | Q stop | H reload HSV" %
                (state, direction, offset),
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    cv2.imshow(SCAN_CAMERA_WINDOW, view)
    if ranges is not None and target_map is not None:
        cv2.imshow(SCAN_MASK_WINDOW, scanner.draw_detection_mask(
            frame, ranges, found, marks, selected, state))
        cv2.imshow(SCAN_MAP_WINDOW, scanner.draw_map(target_map))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        raise SafetyStop("Target scan stopped from camera window")
    if key in (ord("h"), ord("H")) and ranges is not None:
        ranges.clear()
        ranges.update(scanner.load_ranges())
        LOG.info("TARGET SCAN HSV settings reloaded")
    if cv2.getWindowProperty(SCAN_CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        raise SafetyStop("Target scan camera window closed")


def _aim_scan_target(nav, scanner, codes, ranges, target_map, output_path,
                     fired_bearings, direction, offset, code, bearing,
                     all_codes):
    """Use map_scan's continuous target lock and image-error aiming."""
    started = last_seen = time.monotonic()
    try:
        while time.monotonic() - started < scanner.AIM_TIMEOUT_S:
            try:
                frame = nav.robot.camera.read_cv2_image(timeout=.2,
                                                         strategy="newest")
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
            chosen = scanner.match_locked_target(found, code, bearing,
                                                 width, turn_deg)
            _show_scan_camera(scanner, frame, found, direction, offset,
                              ranges, target_map, codes, selected=chosen,
                              nav=nav, state="AIM")
            if chosen is None:
                nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                if time.monotonic() - last_seen > scanner.TARGET_LOST_S:
                    break
                continue
            last_seen = time.monotonic()
            _code, box, cx, cy = chosen
            ex, ey = cx-width//2, cy-height//2
            accept_x = max(24, round(width * scanner.ACCEPT_HALF_WIDTH_FRACTION))
            accept_y = max(20, round(height * scanner.ACCEPT_HALF_HEIGHT_FRACTION))
            if abs(ex) <= accept_x and abs(ey) <= accept_y:
                nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                chassis_yaw = wrap(angles[3]-angles[1])
                saved = target_map.observe(code, box, cx, cy, width, height,
                                           chassis_yaw, angles[1], angles[0],
                                           turn_deg)
                saved["scan_direction"] = direction
                saved["scan_offset_deg"] = offset
                record_gel_round(nav.robot, scanner, target_map, saved,
                                 output_path)
                fired_bearings.append((code, bearing))
                _show_scan_camera(scanner, frame, found, direction, offset,
                                  ranges, target_map, codes, selected=chosen,
                                  nav=nav, state="SAVED + GEL")
                LOG.info("TARGET SCAN fired requested %s toward %s at bearing %.1f",
                         code, direction, bearing)
                return True
            yaw_speed = limit(ex * scanner.YAW_GAIN, scanner.MAX_YAW_AIM_DPS)
            pitch_speed = limit(-ey * scanner.PITCH_GAIN,
                                scanner.MAX_PITCH_AIM_DPS)
            if pitch_speed < 0:
                pitch_speed = max(pitch_speed,
                                  -max(0.0, angles[0] -
                                       scanner.GIMBAL_MIN_PITCH_DEG) * 2.0)
            if abs(angles[1]) > 245 and yaw_speed * angles[1] > 0:
                yaw_speed = 0.0
            if nav.robot.gimbal.drive_speed(
                    pitch_speed=pitch_speed,
                    yaw_speed=nav.gimbal_yaw_command_sign * yaw_speed) is False:
                raise RuntimeError("Gimbal rejected target aiming command")
    finally:
        nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    LOG.info("TARGET SCAN aim lost/timed out: %s toward %s", code, direction)
    return False


def _scan_one_heading(nav, scanner, codes, ranges, target_map, output_path,
                      fired_bearings, direction, offset, yaw_sign):
    """Look at one main_yaw-relative heading and fire only exact requested codes."""
    nav._align_point(yaw_sign * offset)
    # A small downward pitch matches the camera setting used by map_scan.py.
    angles, _ = nav.data.read()
    action = nav.robot.gimbal.moveto(pitch=getattr(scanner, "HOME_PITCH_DEG", -3.0),
                                    yaw=nav.gimbal_yaw_command_sign * angles[1],
                                    pitch_speed=90, yaw_speed=180)
    if action is not None and not action.wait_for_completed(timeout=5):
        raise RuntimeError("Target-scan pitch adjustment timed out")
    deadline = time.monotonic() + 1.2
    frames_seen = 0
    pitch_direction = -1
    skipped_bearings = []
    while time.monotonic() < deadline:
        try:
            frame = nav.robot.camera.read_cv2_image(timeout=.2, strategy="newest")
        except scanner.Empty:
            continue
        if frame is None:
            continue
        frames_seen += 1
        height, width = frame.shape[:2]
        all_codes = {shape + color for shape in scanner.SHAPE_NAMES
                     for color in scanner.COLOR_NAMES}
        found = scanner.find_matches(frame, all_codes, ranges)
        _show_scan_camera(scanner, frame, found, direction, offset,
                          ranges, target_map, codes, nav=nav)
        for code, box, cx, cy in found:
            if code not in codes:
                continue
            angles, _ = nav.data.read()
            bearing = scanner.target_bearing(cx, width,
                                             wrap(angles[3] - nav.main_yaw))
            if any(item[0] == code and abs(wrap(item[1] - bearing)) < 8
                   for item in fired_bearings):
                continue
            if any(old_code == code and abs(wrap(old_bearing-bearing)) < 8
                   for old_code, old_bearing in skipped_bearings):
                continue
            if _aim_scan_target(nav, scanner, codes, ranges, target_map,
                                output_path, fired_bearings, direction, offset,
                                code, bearing, all_codes):
                return
            skipped_bearings.append((code, bearing))
            nav._align_point(yaw_sign * offset)
        # Keep map_scan's vertical search while yaw stays at this fixed
        # main_yaw-relative waypoint. A failed or already-fired candidate
        # must not freeze the search at one pitch.
        angles, _ = nav.data.read()
        pitch_speed, pitch_direction = scanner.scan_pitch_velocity(
            angles[0], pitch_direction)
        nav.robot.gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=0)
    nav.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    if frames_seen == 0:
        raise RuntimeError("Target-scan camera produced no frames at %s %+d degrees" %
                           (direction, offset))


def run_target_scan_four_sides(nav, scanner, requested_codes, output_path,
                               right_yaw_sign):
    """Fixed front/right/left/back scan with a wheel-measured return each time."""
    nav.phase = "target_scan"
    nav.stop_chassis()
    saved_main_yaw = nav.navigation_yaw()
    start_position = nav.data.read_position()
    ranges = scanner.load_ranges()
    target_map = scanner.TargetMap(saved_main_yaw)
    fired_bearings = []
    skipped_directions = []
    # x is forward, y is right in RoboMaster chassis.drive_speed.
    sides = (("front", "x", -.10, 0),
             ("right", "y", -.10, 90),
             ("left", "y", +.10, -90),
             ("back", "x", +.10, -180))
    streaming = False
    camera_window_open = False
    try:
        nav._align_point(0)
        if nav.robot.camera.start_video_stream(display=False) is False:
            raise RuntimeError("Target-scan camera stream failed")
        streaming = True
        if getattr(scanner, "cv2", None) is not None:
            scanner.cv2.namedWindow(SCAN_CAMERA_WINDOW, scanner.cv2.WINDOW_NORMAL)
            scanner.cv2.namedWindow(SCAN_MASK_WINDOW, scanner.cv2.WINDOW_NORMAL)
            scanner.cv2.namedWindow(SCAN_MAP_WINDOW, scanner.cv2.WINDOW_NORMAL)
            camera_window_open = True
            preview_deadline = time.monotonic() + 3.0
            while time.monotonic() < preview_deadline:
                try:
                    preview = nav.robot.camera.read_cv2_image(
                        timeout=.2, strategy="newest")
                except scanner.Empty:
                    preview = None
                if preview is not None:
                    all_codes = {shape + color for shape in scanner.SHAPE_NAMES
                                 for color in scanner.COLOR_NAMES}
                    found = scanner.find_matches(preview, all_codes, ranges)
                    _show_scan_camera(scanner, preview, found, "ready", 0,
                                      ranges, target_map, requested_codes,
                                      nav=nav, state="READY")
                    break
            else:
                raise RuntimeError("Target-scan camera did not provide an initial frame")
        for direction, axis, move, base in sides:
            nav._align_point(0)
            LOG.info("TARGET SCAN %s: move %+.2f m on %s; head %+.0f,+45,base,-45",
                     direction, move, axis, base)
            try:
                distance = _scan_move_10cm(nav, axis, move)
            except ScanOffsetBlocked as exc:
                LOG.warning("TARGET SCAN %s skipped: %s; returning to scan center",
                            direction, exc)
                # A close side wall can block both the outward and the
                # fore/aft return. Recenter between two walls before trying
                # that return; the lateral return already moves away from
                # the side that caused the brake.
                if axis == "x":
                    _scan_center_between_two_walls(nav)
                if exc.travelled >= .005:
                    _scan_move_10cm(nav, axis,
                                    -math.copysign(exc.travelled, move))
                _scan_center_between_two_walls(nav)
                skipped_directions.append(direction)
                continue
            try:
                for offset in (base, base+45, base, base-45):
                    _scan_one_heading(nav, scanner, requested_codes, ranges,
                                      target_map, Path(output_path), fired_bearings,
                                      direction, offset, right_yaw_sign)
                nav._align_point(right_yaw_sign * base)
            finally:
                _scan_move_10cm(nav, axis, -math.copysign(distance, move))
                _scan_center_between_two_walls(nav)
        target_map.save(Path(output_path))
        if getattr(scanner, "cv2", None) is not None:
            plot_path = Path(output_path).with_suffix(".png")
            if not scanner.cv2.imwrite(str(plot_path), scanner.draw_map(target_map)):
                LOG.warning("TARGET SCAN map plot could not be saved: %s", plot_path)
        diagnostics = Path(output_path).with_name(Path(output_path).stem + "_diagnostics.json")
        diagnostics.write_text(json.dumps({"completed": True,
                                           "directions": [side[0] for side in sides],
                                           "skipped_directions": skipped_directions,
                                           "confirmed_ir": len(fired_bearings),
                                           "requested": sorted(requested_codes)},
                                          indent=2), encoding="utf-8")
        LOG.info("TARGET SCAN completed: %d requested target(s) fired; continue route",
                 len(fired_bearings))
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
    residual = math.hypot(end_position[0]-start_position[0],
                          end_position[1]-start_position[1])
    if residual > .16:
        raise SafetyStop("Target scan left robot %.3f m from cell center" % residual)
    nav.settle(follow=True)
    nav.phase = "at_cell"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conn", choices=("ap", "sta"), default="ap")
    parser.add_argument("--map", type=Path,
                        default=Path(__file__).resolve().with_name("map_config.json"),
                        help="map_config.json created by mock_map.py")
    parser.add_argument("--cell-size", type=float,
                        help="Override map_config.json cell_size_m")
    parser.add_argument("--clearance-mm", type=float, default=120.0)
    parser.add_argument("--map-scan", type=Path,
                        help="map_scan.py; defaults beside map_config.json")
    parser.add_argument("--scan-targets",
                        help="shape+color codes for every target, e.g. pg,ir; prompts before connecting if omitted")
    parser.add_argument("--scan-right-yaw-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--gimbal-yaw-sign", type=int, choices=(-1, 1), default=1,
                        help="SDK gimbal yaw command sign; use -1 if head turns opposite to command")
    parser.add_argument("--calibrate-gimbal-sign", action="store_true",
                        help="Probe the sign once with a 10-degree head motion before navigation")
    parser.add_argument("--tof-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-graph", action="store_true", help="Disable Pygame live map")
    args = parser.parse_args()
    try:
        config = load_map_config(args.map)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error("Invalid map %s: %s" % (args.map, exc))
    if not config.targets:
        print("map_config.json has no navigation targets (targets: []).")
        print("Set targets in mock_map.py and save, or enter temporary grid cells here.")
        while True:
            try:
                raw = input("Navigation target grid x,y; x,y (example: 1,1): ")
                config = replace(config, targets=parse_grid_targets(
                    raw, config.width, config.height))
                break
            except ValueError as exc:
                print(exc)
            except EOFError:
                parser.error("Map has no targets; set them in mock_map.py or enter grid coordinates")
    if args.cell_size is None:
        args.cell_size = config.cell_size_m
    if not math.isfinite(args.cell_size) or not .20 <= args.cell_size <= 2.0:
        parser.error("--cell-size must be 0.20..2.0 m")
    if args.tof_index not in range(4) or args.max_steps < 0:
        parser.error("--tof-index 0..3; --max-steps >= 0")
    if not (0 < args.clearance_mm < args.cell_size * 1000):
        parser.error("Require 0 < clearance-mm < cell size in mm")
    if len(config.targets) > 8:
        parser.error("At most 8 targets supported by the A* permutation planner")
    if args.map_scan is None:
        args.map_scan = args.map.resolve().with_name("map_scan.py")
    if not args.map_scan.is_file():
        parser.error("map_scan.py not found: %s" % args.map_scan)
    def valid_scan_codes(raw):
        requested = [part.strip().lower() for part in raw.split(",")]
        return (bool(requested) and all(len(code) == 2 and code[0] in "uoip"
                                        and code[1] in "rygb" for code in requested))

    if args.scan_targets is None:
        print("map_scan shapes: u=square, o=circle, i=horizontal rectangle, p=vertical rectangle")
        print("map_scan colors: r=red, y=yellow, g=green, b=blue")
        while True:
            try:
                args.scan_targets = input("Shapes/colors to scan and fire at goals (e.g. pg,ir): ").strip().lower()
            except EOFError:
                parser.error("Enter shape+color codes before starting, e.g. pg,ir")
            if valid_scan_codes(args.scan_targets):
                break
            print("Enter shape+color pairs such as pg,ir")
    elif not valid_scan_codes(args.scan_targets):
        parser.error("--scan-targets must use shape+color codes such as pg,ir")
    scanner = None  # Import and execute map_scan.py only after reaching a target.
    log_path = Path(__file__).resolve().with_name("main2nos_run.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, mode="w", encoding="utf-8")], force=True)
    LOG.info("main2nos version %s", VERSION)
    LOG.info("Running file: %s", Path(__file__).resolve())
    LOG.info("Map: %s; start=%s facing +y; targets=%s; configured walls=%d",
             args.map.resolve(), config.start, config.targets, len(config.walls))
    with args.map.open(encoding="utf-8") as stream:
        if "start" not in json.load(stream):
            LOG.warning("Map has no start field; using final_final_robot.py default (2,0)")
    LOG.info("Cell %.2f m; map_config walls=%d; alignment wall threshold %.0f mm; "
             "clearance %.0f mm; ToF channel %d; target scanner=%s",
             args.cell_size, len(config.walls), ALIGN_WALL_MM, args.clearance_mm,
             args.tof_index, args.map_scan)
    from robomaster import robot
    LOG.info("Python: %s; SDK: %s", sys.executable, robot.__file__)
    ep_robot = robot.Robot()
    initialized = False
    subscriptions = []
    live_map = None
    navigator = None
    completed = False
    try:
        ep_robot.initialize(conn_type=args.conn)
        initialized = True
        if not ep_robot.set_robot_mode(mode=robot.FREE):
            raise RuntimeError("Cannot set FREE mode")
        data, tof = Telemetry(), ToF(args.tof_index)
        for subscribe, unsubscribe in (
                (lambda: ep_robot.gimbal.sub_angle(freq=20, callback=data.on_gimbal), ep_robot.gimbal.unsub_angle),
                (lambda: ep_robot.chassis.sub_position(cs=1, freq=20, callback=data.on_position), ep_robot.chassis.unsub_position),
                (lambda: ep_robot.chassis.sub_attitude(freq=20, callback=data.on_attitude), ep_robot.chassis.unsub_attitude),
                (lambda: ep_robot.sensor.sub_distance(freq=20, callback=tof.callback), ep_robot.sensor.unsub_distance)):
            if not subscribe():
                raise RuntimeError("Telemetry subscription failed")
            subscriptions.append(unsubscribe)
        data.wait_ready()
        ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        LOG.info("RECENTER before recording the first main yaw")
        action = ep_robot.gimbal.recenter(pitch_speed=60, yaw_speed=60)
        if not action.wait_for_completed(timeout=10) or not action.has_succeeded:
            raise RuntimeError("Gimbal recenter failed; initial yaw was not recorded")
        # Require angle feedback NEWER than action completion, not a stale
        # pre-recenter yaw. Callbacks only update their timestamp on new data.
        completed_at = time.monotonic()
        deadline = completed_at + 3.0
        while time.monotonic() < deadline:
            with data.lock:
                fresh = data.gimbal_time > completed_at
            if fresh:
                break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError("No fresh gimbal angle after recenter; initial yaw was not recorded")
        time.sleep(.30)
        data.read()
        navigator = MainNavigator(ep_robot, data, tof, args, config)
        # Sharp is intentionally dormant during route planning and movement.
        # It is started only while map_scan may need gel-scan clearance.
        if args.calibrate_gimbal_sign:
            navigator.calibrate_gimbal_yaw_sign()
        if not args.no_graph:
            live_map = PygameMap(navigator)
            navigator.live_map = live_map

        def on_target(nav):
            nonlocal live_map, scanner
            # Only called after a completed cell move (or if start is a target).
            nav.phase = "target_scan"
            nav.stop_chassis()
            nav.save()
            # OpenCV owns the screen during map_scan; restore the Pygame map
            # afterward without losing the wheel trajectory drawn so far.
            saved_view = None
            if live_map is not None:
                saved_view = (live_map.origin, live_map.forward_axis,
                              list(live_map.raw_track), live_map.current_angles)
                live_map.close()
                live_map = None
                nav.live_map = None
            if scanner is None:
                scanner = load_target_scanner(args.map_scan)
            codes = set(scanner.parse_targets(args.scan_targets))
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output = Path(__file__).resolve().with_name(
                "target_map_nos_%d_%d_%s.json" % (nav.cell[0], nav.cell[1], stamp))
            LOG.info("TARGET SCAN starting cell=%s requested=%s output=%s",
                     nav.cell, sorted(codes), output)
            nav.sharp = start_target_scan_sharp(nav.robot)
            try:
                run_target_scan(nav, scanner, codes, output,
                                args.scan_right_yaw_sign)
            finally:
                nav.sharp.close()
            nav.target_scan_outputs[nav.cell] = str(output)
            if saved_view is not None:
                live_map = PygameMap(nav)
                (live_map.origin, live_map.forward_axis,
                 live_map.raw_track, live_map.current_angles) = saved_view
                nav.live_map = live_map
                live_map.update(force=True)
            nav.save()

        navigator.run(on_target=on_target)
        completed = True
    finally:
        if initialized:
            for stop in (lambda: ep_robot.chassis.drive_speed(x=0, y=0, z=0),
                         lambda: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)):
                try:
                    stop()
                except Exception:
                    LOG.exception("Stop command failed")
            if navigator is not None:
                navigator.sharp.close()
            for unsubscribe in reversed(subscriptions):
                try:
                    unsubscribe()
                except Exception:
                    LOG.exception("Unsubscribe failed")
        ep_robot.close()
        if live_map is not None and not completed:
            live_map.close()
    if live_map is not None:
        live_map.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C")
    except Exception as exc:
        logging.exception("Navigation stopped: %s", exc)
        raise SystemExit(1)
