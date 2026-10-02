"""Explore every reachable RoboMaster grid cell and scan its four sides.

Movement, gimbal-following and wheel odometry come from main2nos.py.  The
camera stream follows run_round1.py; color/shape detection follows
test_firing_mock.py.  This is grid-map exploration, with wheel/ToF pose
estimation, rather than camera-feature loop-closure SLAM.
"""
import argparse
from collections import deque
import importlib.util
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time

import main2nos as base

LOG = logging.getLogger("SLAM")
VERSION = "slam-main-yaw-2"
WALL_THRESHOLD_MM = 520.0
WALL_MARGIN_MM = 30.0
WALL_SAMPLES = 3
WALL_REQUIRED = 3
WALL_SAMPLE_TIMEOUT_S = 2.0
TARGET_WALL_MAX_MM = 1100.0
TARGET_CONFIRM_FRAMES = 10
TARGET_REQUIRED_FRAMES = 7
TARGET_AIM_TIMEOUT_S = 3.5
TARGET_AIM_TOLERANCE_PX = 18
TARGET_AIM_Y_OFFSET_PX = 32
TARGET_SCAN_PITCH_DEG = -22.0
TARGET_SWEEP_OFFSETS_DEG = (0.0, 30.0, -30.0)
CAMERA_FOCAL_PX = 650.0
DIRECTIONS = base.DIRECTIONS
DIRECTION_NAMES = ("front", "right", "back", "left")
SHAPES = {"u": "SQUARE", "o": "CIRCLE", "i": "RECT_H", "p": "RECT_V"}
COLORS = {"r": "red", "y": "yellow", "g": "green", "b": "blue"}


def parse_target_codes(raw):
    """Accept the familiar map_scan two-letter color/shape codes."""
    rules = {color: [] for color in COLORS.values()}
    for code in raw.replace(" ", "").split(","):
        if not code:
            continue
        if len(code) != 2 or code[0] not in SHAPES or code[1] not in COLORS:
            raise ValueError("Target codes must be like pg,ir (shape then color)")
        shape, color = SHAPES[code[0]], COLORS[code[1]]
        if shape not in rules[color]:
            rules[color].append(shape)
    if not any(rules.values()):
        raise ValueError("Select at least one target code")
    return rules


def load_detector(path):
    """Load only reusable vision functions; never run its standalone robot."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Target detector not found: %s" % path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("slam_target_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shortest_known_path(maze, start, visited, heading=0):
    """BFS to the nearest unvisited cell, crossing only measured open edges."""
    queue = deque([start])
    parents = {start: None}
    order = (heading, (heading + 1) % 4, (heading + 3) % 4,
             (heading + 2) % 4)
    while queue:
        cell = queue.popleft()
        if cell not in visited:
            path = [cell]
            while parents[cell] is not None:
                cell = parents[cell]
                path.append(cell)
            return list(reversed(path))
        for direction in order:
            if maze.edge(cell, direction) is not False:
                continue
            neighbor = maze.neighbor(cell, direction)
            if neighbor not in parents:
                parents[neighbor] = cell
                queue.append(neighbor)
    return None


def remaining_unknown_edges(maze, visited):
    return [(cell, direction) for cell in sorted(visited)
            for direction in range(4)
            if maze.inside(maze.neighbor(cell, direction))
            and maze.edge(cell, direction) is None]


def classify_wall(readings, threshold=WALL_THRESHOLD_MM):
    """Near/far votes ignore the 52 +/- 3 cm uncertain band."""
    near = sum(50 <= value <= threshold - WALL_MARGIN_MM
               for value in readings)
    far = sum(value >= threshold + WALL_MARGIN_MM
              for value in readings)
    if near >= WALL_REQUIRED and far == 0:
        return True
    if far >= WALL_REQUIRED and near == 0:
        return False
    return None


class CameraFrames:
    """One continuous camera stream, cached by a worker as in run_round1."""

    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None
        self.frame = None
        self.sequence = 0
        self.error = None

    def start(self):
        if self.camera.start_video_stream(display=False) is False:
            raise RuntimeError("Camera stream could not start")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.after(0, timeout=5.0)

    def _worker(self):
        while not self.stopping.is_set():
            try:
                frame = self.camera.read_cv2_image(timeout=.2,
                                                   strategy="newest")
                if frame is not None:
                    with self.lock:
                        self.frame = frame.copy()
                        self.sequence += 1
            except Exception as exc:
                self.error = exc
                self.stopping.wait(.05)

    def mark(self):
        with self.lock:
            return self.sequence

    def after(self, mark, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.sequence > mark and self.frame is not None:
                    return self.sequence, self.frame.copy()
            time.sleep(.02)
        raise RuntimeError("Camera did not provide a fresh image: %s" %
                           self.error)

    def close(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.camera.stop_video_stream()


class GridExplorer(base.MainNavigator):
    """Use main2nos's pose/motion, replacing its known-map goal policy."""

    def __init__(self, robot, data, tof, args, camera, detector, rules):
        # MainNavigator expects a goal, though exploration has none.
        config = base.MapConfig(args.width, args.height, args.start,
                                (args.start,), (), args.cell_size)
        super().__init__(robot, data, tof, args, config)
        self.maze = base.Maze(args.width, args.height, args.start)
        self.heading = {"N": 0, "E": 1, "S": 2, "W": 3}[args.initial_direction]
        self.initial_heading_deg = (0, 90, 180, -90)[self.heading]
        self.targets = ()
        self.visited_cells = set()
        self.edge_observations = {}
        self.detector = detector
        self.rules = rules
        self.camera_frames = camera
        self.burst = args.burst
        self.detected_targets = []
        self.shot_keys = set()
        self.window_enabled = not args.no_camera_window
        self.output = args.output
        self.max_steps = args.max_steps or args.width * args.height * 4
        self.completed_targets = set()
        self.configured_walls = set()
        self.wheel_origin = tuple(data.read_position()[:2])
        # cs=1 position is fixed to the chassis axes at subscription:
        # x forward, y right. Rotate by the physical initial map heading.
        self.pose_grid = tuple(float(v) for v in args.start)
        self.deadline = time.monotonic() + args.time_limit
        self.stop_reason = None

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise base.SafetyStop("570-second exploration time limit reached")

    def wheel_grid(self):
        position = self.data.read_position()
        dx, dy = position[0]-self.wheel_origin[0], position[1]-self.wheel_origin[1]
        theta = math.radians(self.initial_heading_deg)
        self.pose_grid = (self.path[0][0] + (dx*math.sin(theta)+dy*math.cos(theta))/self.cell_size,
                          self.path[0][1] + (dx*math.cos(theta)-dy*math.sin(theta))/self.cell_size)
        return self.pose_grid

    def control(self, forward=0.0, follow=True, lateral=0.0):
        self.check_deadline()
        with self.data.lock:
            if (self.data.chassis_attitude is None or
                    time.monotonic()-self.data.attitude_time > base.DATA_TIMEOUT_S or
                    not all(math.isfinite(v) for v in self.data.chassis_attitude)):
                raise base.SafetyStop("Missing/stale chassis attitude")
        if forward or lateral:
            self.target_yaw = self.navigation_yaw()
            if forward > 0 and self.tof.latest() <= self.clearance_mm:
                self.stop_chassis()
                raise base.SafetyStop("Front wall too close during motion")
        result = base.Route.control(self, forward, follow, lateral)
        self.wheel_grid()
        if self.live_map is not None:
            angles, position = self.data.read()
            self.live_map.sample(angles, position)
        return result

    def center_cell(self):
        self.phase = "centering"
        self._align_point(0)
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = 0.0
        self.settle()
        start = self.data.read_position()
        end = time.monotonic()+4.0
        try:
            while time.monotonic() < end:
                self.check_deadline()
                left, right = self.sharp.wait_fresh(timeout=1.0)
                if left > base.SHARP_WALL_CM and right > base.SHARP_WALL_CM:
                    return  # No side wall: hold wheel pose without correction.
                speed = base.sharp_clearance_speed(left, right, center_single_wall=True)
                if not speed:
                    return
                pos = self.data.read_position()
                if math.hypot(pos[0]-start[0], pos[1]-start[1]) >= .15:
                    return
                self.control(lateral=speed)
                time.sleep(base.LOOP_PERIOD_S)
        finally:
            self.stop_chassis()

    def align_to_wall(self):
        # Keep the two-angle geometry from main2nos, but execute corrections
        # using the same ground-yaw feedback loop as every other chassis turn.
        self.phase = "wall_alignment"
        try:
            for name, offset in (("right",90), ("left",-90), ("back",-180)):
                try:
                    if self._align_read(offset) > base.ALIGN_WALL_MM:
                        continue
                    used = 0.0
                    for _ in range(base.ALIGN_MAX_CORRECTIONS):
                        plus = self._align_read(offset+base.ALIGN_OFFSET_DEG)
                        minus = self._align_read(offset-base.ALIGN_OFFSET_DEG)
                        step = self.alignment_step(name, plus, minus, used)
                        if not step:
                            break
                        self._align_point(0)
                        # chassis.move positive z is CCW; ground yaw is CW.
                        desired = base.wrap(self.navigation_yaw()-step)
                        self.target_yaw, self.target_pitch = desired, 0.0
                        self.settle(follow=False)
                        self.settle(follow=True)
                        self.main_yaw = desired
                        used += abs(step)
                    break
                except RuntimeError as exc:
                    if isinstance(exc, base.SafetyStop):
                        raise
                    LOG.warning("Wall alignment unavailable: %s", exc)
        finally:
            self._align_point(0)
            self.target_yaw, self.target_pitch = self.navigation_yaw(), 0.0
            self.settle()


    def save(self):
        try:
            angles, position = self.data.read()
        except RuntimeError:
            angles, position = self.data.read_cached()
        edges = [dict(a=a, b=b, wall=wall)
                 for (a, b), wall in sorted(self.maze.edges.items())]
        payload = dict(version=VERSION, width=self.maze.width,
                       height=self.maze.height, cell_size_m=self.cell_size,
                       start=self.path[0], wheel_origin=self.wheel_origin,
                       pose_grid=self.pose_grid, stop_reason=self.stop_reason,
                       current_cell=self.cell, heading=self.heading,
                       main_yaw=self.main_yaw, initial_yaw=self.initial_yaw,
                       gimbal_ground_yaw=angles[3], wheel_position=position,
                       path=self.path, visited=sorted(self.visited_cells),
                       edges=edges, observations=self.edge_observations,
                       targets=self.detected_targets,
                       unknown_edges=remaining_unknown_edges(self.maze,
                                                             self.visited_cells),
                       phase=self.phase, complete=self.arrived)
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(self.output)
        if self.live_map is not None:
            self.live_map.update(force=True)

    def _tof_after_pointing(self, offset):
        mark = self._align_point(offset)
        deadline = time.monotonic() + WALL_SAMPLE_TIMEOUT_S
        readings = []
        while time.monotonic() < deadline and len(readings) < WALL_SAMPLES:
            self.check_deadline()
            readings = [value for value in self.tof.after(mark)
                        if 50 <= value < 65535]
            time.sleep(base.LOOP_PERIOD_S)
        if len(readings) < WALL_SAMPLES:
            return None, readings
        return classify_wall(readings[:WALL_SAMPLES], self.cell_size*1000-80), readings[:WALL_SAMPLES]

    def _point_pitch(self, offset, pitch):
        self._align_point(offset)
        angles, _ = self.data.read()
        action = self.robot.gimbal.moveto(
            pitch=pitch,
            yaw=self.gimbal_yaw_command_sign * angles[1],
            pitch_speed=90, yaw_speed=180)
        if action is not None and (not action.wait_for_completed(timeout=5)
                                   or getattr(action, "has_succeeded", True) is False):
            raise RuntimeError("Gimbal pitch did not settle for target scan")
        time.sleep(.12)

    def _show_camera(self, image, targets, label):
        if not self.window_enabled:
            return
        cv2 = self.detector.cv2
        view = image.copy()
        for target in targets:
            x, y, w, h = target["bbox"]
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, "%s %s" % (target["color"], target["shape"]),
                        (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (0, 255, 0), 2)
        cv2.putText(view, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (255, 255, 255), 2)
        cv2.imshow("SLAM target scan", view)
        if cv2.waitKey(1) & 255 in (ord("q"), ord("Q"), 27):
            raise base.SafetyStop("SLAM camera stop requested")

    def _detect(self, image, distance_mm, yaw):
        return self.detector.process_frame_multi_targets(
            image, distance_mm / 10.0, target_rules_input=self.rules,
            current_yaw=yaw, fired_yaws=None)

    def _confirm_target(self, color, shape, distance_mm):
        hits = 0
        mark = self.camera_frames.mark()
        for _ in range(TARGET_CONFIRM_FRAMES):
            self.check_deadline()
            mark, image = self.camera_frames.after(mark)
            yaw = self.data.read()[0][3]
            detections = self._detect(image, distance_mm, yaw)
            self._show_camera(image, detections, "VERIFY %s %s" % (color, shape))
            if any(t["color"] == color and t["shape"] == shape and
                   abs(t["cx"] - image.shape[1] / 2) <= TARGET_AIM_TOLERANCE_PX * 2
                   for t in detections):
                hits += 1
        return hits >= TARGET_REQUIRED_FRAMES

    def _aim_target(self, color, shape, distance_mm):
        deadline = time.monotonic() + TARGET_AIM_TIMEOUT_S
        stable = 0
        mark = self.camera_frames.mark()
        try:
            while time.monotonic() < deadline:
                self.check_deadline()
                mark, image = self.camera_frames.after(mark)
                yaw = self.data.read()[0][3]
                targets = [t for t in self._detect(image, distance_mm, yaw)
                           if t["color"] == color and t["shape"] == shape]
                self._show_camera(image, targets, "AIM %s %s" % (color, shape))
                if not targets:
                    stable = 0
                    continue
                center_x = image.shape[1] / 2
                center_y = image.shape[0] / 2 + TARGET_AIM_Y_OFFSET_PX
                chosen = min(targets, key=lambda t: abs(t["cx"] - center_x)
                             + abs(t["cy"] - center_y))
                ex, ey = chosen["cx"] - center_x, chosen["cy"] - center_y
                if (abs(ex) <= TARGET_AIM_TOLERANCE_PX and
                        abs(ey) <= TARGET_AIM_TOLERANCE_PX):
                    stable += 1
                    self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    if stable >= 3:
                        return True
                    continue
                stable = 0
                self.robot.gimbal.drive_speed(
                    yaw_speed=self.gimbal_yaw_command_sign *
                    base.limit(.10 * ex, 35),
                    pitch_speed=base.limit(-.10 * ey, 28))
            return False
        finally:
            self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

    def _target_key(self, offset, target, distance_mm, image_width):
        bearing = base.wrap(self.initial_heading_deg + self.navigation_yaw()
                            + offset - self.initial_yaw
                            + math.degrees(math.atan((target["cx"] - image_width / 2)
                                                / CAMERA_FOCAL_PX)))
        distance_m = distance_mm / 1000.0
        pose = self.wheel_grid()
        gx = (pose[0] + .5) * self.cell_size + distance_m * math.sin(math.radians(bearing))
        gy = (pose[1] + .5) * self.cell_size + distance_m * math.cos(math.radians(bearing))
        return (target["color"], target["shape"], round(gx / .15), round(gy / .15))

    def _scan_targets(self, offset, distance_mm):
        if distance_mm > TARGET_WALL_MAX_MM:
            return
        for sweep in TARGET_SWEEP_OFFSETS_DEG:
            look_offset = offset + sweep
            if abs(look_offset) > 250:
                continue  # Gimbal mechanical range, especially behind robot.
            self._point_pitch(look_offset, TARGET_SCAN_PITCH_DEG)
            mark = self.camera_frames.mark()
            mark, image = self.camera_frames.after(mark)
            yaw = self.data.read()[0][3]
            targets = self._detect(image, distance_mm, yaw)
            self._show_camera(image, targets, "SCAN cell %s offset %+.0f" %
                              (self.cell, look_offset))
            for target in targets:
                key = self._target_key(look_offset, target, distance_mm, image.shape[1])
                if key in self.shot_keys:
                    continue
                color, shape = target["color"], target["shape"]
                camera_offset = math.degrees(math.atan(
                    (target["cx"] - image.shape[1] / 2) / CAMERA_FOCAL_PX))
                self._point_pitch(look_offset + camera_offset, TARGET_SCAN_PITCH_DEG)
                if not self._aim_target(color, shape, distance_mm):
                    LOG.info("Target %s %s aim not stable at %s", color, shape,
                             self.cell)
                    continue
                if not self._confirm_target(color, shape, distance_mm):
                    LOG.info("Target %s %s failed %d/%d frame confirmation",
                             color, shape, TARGET_REQUIRED_FRAMES,
                             TARGET_CONFIRM_FRAMES)
                    continue
                record = dict(cell=self.cell, estimated_position_m=[key[2]*.15, key[3]*.15], offset_deg=look_offset,
                              color=color, shape=shape, key=key,
                              distance_mm=distance_mm, status="pending")
                self.detected_targets.append(record)
                self.shot_keys.add(key)  # Never repeat an uncertain IR command.
                self.save()
                # Confirmation is followed by another lock and a fresh range
                # at the actual aimed bearing, never the earlier wall scan.
                if not self._aim_target(color, shape, distance_mm):
                    record["status"] = "confirmed_unlocked"
                    self.shot_keys.discard(key)
                    self.save()
                    continue
                fresh_range = self.tof.fresh_median(self.tof.mark())
                if not self.clearance_mm < fresh_range <= TARGET_WALL_MAX_MM:
                    record["status"] = "confirmed_out_of_range"
                    self.shot_keys.discard(key)
                    self.save()
                    continue
                record["distance_mm"] = fresh_range
                try:
                    from robomaster import blaster
                    for _ in range(self.burst):
                        self.check_deadline()
                        result = self.robot.blaster.fire(
                            fire_type=blaster.INFRARED_FIRE, times=1)
                        if result is False:
                            raise RuntimeError("IR command rejected")
                        time.sleep(.15)
                except base.SafetyStop:
                    raise
                except Exception as exc:
                    record["status"] = "uncertain"
                    record["error"] = str(exc)
                    self.save()
                    LOG.warning("IR status uncertain at %s: %s", self.cell, exc)
                else:
                    record["status"] = "sent"
                    self.save()
                    LOG.info("IR fired %dx at %s: %s %s", self.burst, self.cell,
                             color, shape)
            self._point_pitch(offset, TARGET_SCAN_PITCH_DEG)

    def scan_cell(self):
        """Stop, inspect all four map edges, then scan targets on near walls."""
        self.phase = "scanning"
        self.stop_chassis()
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = 0.0
        self.settle()
        try:
            for relative in (0, 1, 2, 3):
                self.check_deadline()
                absolute = (self.heading + relative) % 4
                offset = (0, 90, -180, -90)[relative]
                neighbor = self.maze.neighbor(self.cell, absolute)
                wall, readings = self._tof_after_pointing(offset)
                key = "%d,%d:%d" % (self.cell[0], self.cell[1], absolute)
                self.edge_observations[key] = readings
                if not self.maze.inside(neighbor):
                    LOG.info("SCAN %s %s map boundary ToF=%s", self.cell,
                             DIRECTION_NAMES[relative], readings)
                elif wall is None:
                    LOG.warning("SCAN %s %s undecided ToF=%s", self.cell,
                                DIRECTION_NAMES[relative], readings)
                else:
                    edge_key = self.maze.key(self.cell, neighbor)
                    previous = self.maze.edges.get(edge_key)
                    if previous is False and wall:
                        # A passage already crossed by the wheels cannot become a wall.
                        if self.cell in self.visited_cells and neighbor in self.visited_cells:
                            LOG.warning("SCAN conflict at travelled edge %s; keeping open", edge_key)
                        else:
                            self.maze.edges[edge_key] = True
                    elif previous is True and not wall:
                        LOG.warning("SCAN conflict at %s; keeping known wall", edge_key)
                    else:
                        self.maze.edges[edge_key] = wall
                LOG.info("SCAN %s %s wall=%s ToF=%s", self.cell,
                         DIRECTION_NAMES[relative], wall, readings)
                if len(readings) == WALL_SAMPLES and sorted(readings)[len(readings)//2] <= TARGET_WALL_MAX_MM:
                    try:
                        self._scan_targets(offset, sorted(readings)[len(readings)//2])
                    except base.SafetyStop:
                        raise
                    except RuntimeError as exc:
                        LOG.warning("Target scan unavailable at %s %s: %s",
                                    self.cell, DIRECTION_NAMES[relative], exc)
        finally:
            self._align_point(0)
            self.target_yaw = self.navigation_yaw()
            self.target_pitch = 0.0
            self.settle()
            self.save()

    def run_exploration(self):
        angles, _ = self.data.read()
        self.initial_yaw = angles[3]
        self.main_yaw = base.wrap(angles[3])
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = angles[2]
        self.maze.visited.add(self.cell)
        self.visited_cells.add(self.cell)
        self.save()
        try:
            while True:
                self.check_deadline()
                self.scan_cell()
                self.center_cell()
                self.align_to_wall()
                path = shortest_known_path(self.maze, self.cell,
                                           self.visited_cells, self.heading)
                if path is None:
                    unknown = remaining_unknown_edges(self.maze,
                                                      self.visited_cells)
                    self.phase = ("scan_incomplete" if unknown else
                                  "all_reachable_cells_visited")
                    self.arrived = not unknown
                    LOG.info("Exploration ended: %s; visited %d/%d; unknown edges=%d",
                             self.phase, len(self.visited_cells),
                             self.maze.width * self.maze.height, len(unknown))
                    break
                if self.steps >= self.max_steps:
                    raise base.SafetyStop("Exploration step limit reached")
                next_cell = path[1]
                direction = DIRECTIONS.index((next_cell[0] - self.cell[0],
                                              next_cell[1] - self.cell[1]))
                self.turn_to(direction)
                # Scan again after turning; never rely on a stale side reading
                # to authorize a forward move.
                front_wall, fresh = self._tof_after_pointing(0)
                if front_wall is None:
                    raise base.SafetyStop("Forward ToF verification inconclusive; "
                                          "position held, map edge remains unchanged")
                if front_wall:
                    self.stop_chassis()
                    self.maze.edges[self.maze.key(self.cell, next_cell)] = True
                    LOG.warning("Forward passage %s -> %s now blocked; replanning",
                                self.cell, next_cell)
                    self.save()
                    continue
                self.phase = "moving"
                self.forward_cell(self.steps + 1)
                actual = self.wheel_grid()
                measured = tuple(math.floor(v+.5) for v in actual)
                if measured != next_cell or math.hypot(actual[0]-next_cell[0], actual[1]-next_cell[1]) > .30:
                    raise base.SafetyStop("Wheel/grid mismatch: expected %s, actual %s" % (next_cell, actual))
                self.cell = next_cell
                self.steps += 1
                self.path.append(self.cell)
                self.visited_cells.add(self.cell)
                self.maze.visited.add(self.cell)
                self.save()
        except BaseException as exc:
            self.stop_chassis()
            self.phase = "stopped"
            self.stop_reason = str(exc)
            LOG.error("Exploration stopped at %s: %s", self.cell, exc)
            self.save()
            if not isinstance(exc, base.SafetyStop):
                raise
        finally:
            self.stop_chassis()
            self.save()


class SlamMap(base.PygameMap):
    def sample(self, angles, position):
        self.current_angles = angles
        self.raw_track.append(tuple(position[:2]))
        self.raw_track = self.raw_track[-10000:]
        self.forward_axis = (1, 0)
        self.update()

    def grid_position(self, point):
        dx, dy = point[0]-self.nav.wheel_origin[0], point[1]-self.nav.wheel_origin[1]
        theta = math.radians(self.nav.initial_heading_deg)
        return (self.nav.path[0][0]+(dx*math.sin(theta)+dy*math.cos(theta))/self.nav.cell_size,
                self.nav.path[0][1]+(dx*math.cos(theta)-dy*math.sin(theta))/self.nav.cell_size)


def parse_cell(raw, width, height):
    try:
        x, y = (int(part.strip()) for part in raw.split(","))
    except (ValueError, TypeError):
        raise ValueError("Start must be x,y")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Start outside grid")
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conn", choices=("ap", "sta"), default="ap")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--start", help="0-based x,y; default 0,0")
    parser.add_argument("--initial-direction", choices=("N", "E", "S", "W"),
                        default=None, help="Physical starting heading, default N (+y)")
    parser.add_argument("--targets", help="Target codes, e.g. pg,ir")
    parser.add_argument("--burst", type=int)
    parser.add_argument("--time-limit", type=float, default=570.0)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--cell-size", type=float, default=.60)
    parser.add_argument("--clearance-mm", type=float, default=120.0)
    parser.add_argument("--tof-index", type=int, default=0)
    parser.add_argument("--gimbal-yaw-sign", type=int, choices=(-1, 1),
                        default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-camera-window", action="store_true")
    parser.add_argument("--detector", type=Path,
                        default=Path(__file__).with_name("test_firing_mock.py"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("SLAM_main_yaw_map.json"))
    args = parser.parse_args()
    if args.width is None:
        args.width = int(input("Field width (grids): "))
    if args.height is None:
        args.height = int(input("Field height (grids): "))
    if not (1 <= args.width <= 50 and 1 <= args.height <= 50):
        parser.error("Grid size must be 1..50")
    args.start = parse_cell(args.start or input("Start x,y [0,0]: ").strip()
                            or "0,0", args.width, args.height)
    try:
        args.rules = parse_target_codes(args.targets or
                                        input("Target codes (e.g. pg,ir): "))
    except ValueError as exc:
        parser.error(str(exc))
    if args.initial_direction is None:
        args.initial_direction = input("Initial heading N/E/S/W [N]: ").strip().upper() or "N"
    if args.initial_direction not in ("N", "E", "S", "W"):
        parser.error("Initial heading must be N/E/S/W")
    if args.burst is None:
        args.burst = int(input("IR shots per target [2]: ").strip() or "2")
    if not math.isfinite(args.time_limit) or not 0 < args.time_limit <= 570 or args.max_steps < 0:
        parser.error("time-limit must be 0..570 seconds; max-steps >= 0")
    if args.burst < 1 or args.burst > 5 or args.tof_index not in range(4):
        parser.error("burst must be 1..5; ToF index must be 0..3")
    if not math.isfinite(args.cell_size) or not .2 <= args.cell_size <= 2.0 or not math.isfinite(args.clearance_mm) or not 0 < args.clearance_mm < args.cell_size * 1000:
        parser.error("Invalid cell size or front clearance")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            Path(__file__).with_name("SLAM_main_yaw_run.log"), mode="w",
                            encoding="utf-8")], force=True)
    LOG.info("SLAM version %s grid=%dx%d start=%s target_rules=%s",
             VERSION, args.width, args.height, args.start, args.rules)
    detector = load_detector(args.detector)
    from robomaster import robot
    bot = robot.Robot()
    subscriptions = []
    camera = None
    initialized = False
    nav = None
    try:
        bot.initialize(conn_type=args.conn)
        initialized = True
        if not bot.set_robot_mode(mode=robot.FREE):
            raise RuntimeError("Cannot set FREE mode")
        data, tof = base.Telemetry(), base.ToF(args.tof_index)
        for subscribe, unsubscribe in (
                (lambda: bot.gimbal.sub_angle(freq=20, callback=data.on_gimbal),
                 bot.gimbal.unsub_angle),
                (lambda: bot.chassis.sub_position(cs=1, freq=20,
                                                  callback=data.on_position),
                 bot.chassis.unsub_position),
                (lambda: bot.chassis.sub_attitude(freq=20,
                                                  callback=data.on_attitude),
                 bot.chassis.unsub_attitude),
                (lambda: bot.sensor.sub_distance(freq=20,
                                                 callback=tof.callback),
                 bot.sensor.unsub_distance)):
            if not subscribe():
                raise RuntimeError("Telemetry subscription failed")
            subscriptions.append(unsubscribe)
        data.wait_ready()
        bot.chassis.drive_speed(x=0, y=0, z=0)
        with data.lock:
            before_recenter = data.gimbal_time
        action = bot.gimbal.recenter(pitch_speed=60, yaw_speed=60)
        if action is not None and (not action.wait_for_completed(timeout=10)
                                   or not action.has_succeeded):
            raise RuntimeError("Initial gimbal recenter failed")
        before_recenter = time.monotonic()
        deadline = before_recenter + 3.0
        while time.monotonic() < deadline:
            with data.lock:
                fresh = data.gimbal_time > before_recenter
            if fresh and abs(data.read()[0][1]) <= 1.0:
                break
            time.sleep(base.LOOP_PERIOD_S)
        else:
            raise RuntimeError("No fresh centered gimbal yaw after recenter")
        camera = CameraFrames(bot.camera)
        camera.start()
        nav = GridExplorer(bot, data, tof, args, camera, detector, args.rules)
        nav.sharp.start()
        nav.sharp.wait_fresh()
        if not args.no_graph:
            nav.live_map = SlamMap(nav)
        nav.run_exploration()
    finally:
        if initialized:
            try:
                bot.chassis.drive_speed(x=0, y=0, z=0)
                bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                LOG.exception("Stop command failed")
            if nav is not None:
                nav.sharp.close()
                if nav.live_map is not None:
                    nav.live_map.close()
            if camera is not None:
                camera.close()
            for unsubscribe in reversed(subscriptions):
                try:
                    unsubscribe()
                except Exception:
                    LOG.exception("Unsubscribe failed")
            bot.close()
        if not args.no_camera_window:
            detector.cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
