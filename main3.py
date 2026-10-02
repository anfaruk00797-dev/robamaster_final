"""Navigation-only RoboMaster route through every map_config.json target. A* selects paths; main_yaw and wheel odometry guide motion. ToF guards the front and aligns walls. No camera scan, firing, or Sharp."""
import argparse
import heapq
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
CELL_SIZE_M = 0.6
FORWARD_SPEED_MPS = 0.3
LOOP_PERIOD_S = 0.05
DATA_TIMEOUT_S = 0.75
ANGLE_TOLERANCE_DEG = 0.5
DISTANCE_TOLERANCE_M = 0.01
FRONT_WALL_CELL_TOLERANCE_M = 0.1
FRONT_WALL_POST_BACKOFF_TOLERANCE_M = 0.15
MAX_FRONT_ADJUSTMENTS_PER_CELL = 3
SETTLE_S = 0.5
GIMBAL_GAIN = 3.0
CHASSIS_GAIN = 2.0
MAX_GIMBAL_SPEED = 45.0
MAX_CHASSIS_SPEED = 30.0
LOG = logging.getLogger('main3')
VERSION = 'main3-fixed-main-yaw-2'
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

class SafetyStop(RuntimeError):
    """A requested stop that ends navigation cleanly without an error traceback."""

class FrontObstacleStop(SafetyStop):
    """The front ToF required a backoff before a full grid step was completed."""

class FrontWallAdjusted(RuntimeError):
    """A front ToF correction completed; re-evaluate wheel progress."""
# SDK chassis z>0 is counterclockwise. On the back wall, -20 farther
# than +20 must give z>0; +20 farther must give z<0.
ALIGN_DIRECTION_SIGN = {'front': 1, 'right': 1, 'back': -1, 'left': 1}
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
    with Path(path).open(encoding='utf-8') as stream:
        raw = json.load(stream)
    size = raw['grid_size']
    width, height = (int(size['w']), int(size['h']))
    if not (1 <= width <= 50 and 1 <= height <= 50):
        raise ValueError('Map width and height must be 1..50')

    def cell(value):
        if isinstance(value, dict):
            value = (value['x'], value['y'])
        if len(value) != 2 or any((isinstance(v, bool) or int(v) != v for v in value)):
            raise ValueError('Map cell must contain two integer coordinates')
        result = (int(value[0]), int(value[1]))
        if not (0 <= result[0] < width and 0 <= result[1] < height):
            raise ValueError('Map cell outside grid: %r' % (result,))
        return result
    start = cell(raw.get('start', [2, 0]))
    if float(raw.get('initial_heading_deg', 0)) != 0:
        raise ValueError('Map initial_heading_deg must be 0 (+y)')
    cell_size_m = float(raw.get('cell_size_m', CELL_SIZE_M))
    if not math.isfinite(cell_size_m) or not 0.2 <= cell_size_m <= 2.0:
        raise ValueError('Map cell_size_m must be 0.20..2.0')
    targets = tuple(dict.fromkeys((cell(v) for v in raw['targets'])))
    walls = set()
    for item in raw.get('walls', []):
        a, b = (cell(item[0]), cell(item[1]))
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise ValueError('Wall endpoints must be adjacent: %r' % (item,))
        walls.add(tuple(sorted((a, b))))
    return MapConfig(width, height, start, targets, tuple(sorted(walls)), cell_size_m)

def parse_grid_targets(raw, width, height):
    """Parse temporary navigation goals: x,y; x,y (zero-based grid cells)."""
    targets = []
    for item in raw.split(';'):
        parts = [part.strip() for part in item.split(',')]
        if len(parts) != 2 or any((not part.isdecimal() for part in parts)):
            raise ValueError('Enter grid targets as x,y; x,y (example: 1,1; 2,3)')
        x, y = map(int, parts)
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError('Grid target (%d,%d) is outside %dx%d map' % (x, y, width, height))
        targets.append((x, y))
    return tuple(dict.fromkeys(targets))

def a_star(maze, start, goal):
    """A* on mapped walls. Unknown edges are estimates, never move permission."""
    queue = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
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
                best[there], parent[there] = (new_cost, here)
                priority = new_cost + abs(there[0] - goal[0]) + abs(there[1] - goal[1])
                heapq.heappush(queue, (priority, new_cost, there))
    return None

def optimal_target_order(maze, start, targets):
    """Same A*/permutation objective as final_final_robot.py."""
    remaining = tuple(sorted(set(targets) - {start}))
    if len(remaining) > 8:
        raise ValueError('At most 8 remaining targets supported by permutation planner')
    best = None
    best_length = math.inf
    nodes = (start,) + remaining
    distances = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            path = a_star(maze, a, b)
            distances[tuple(sorted((a, b)))] = math.inf if path is None else len(path) - 1
    for order in itertools.permutations(remaining):
        here, length = (start, 0)
        for target in order:
            length += distances[tuple(sorted((here, target)))]
            here = target
        if length < best_length:
            best, best_length = (order, length)
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
        """Return fresh wheel position for grid-distance control."""
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

class Route:

    def __init__(self, ep_robot, telemetry, cell_size):
        self.robot = ep_robot
        self.data = telemetry
        self.cell_size = cell_size
        self.target_yaw = None
        self.target_pitch = None
        self.last_status_time = -float('inf')
        self.logged_gimbal_result = False

    def stop_chassis(self):
        self.robot.chassis.drive_speed(x=0, y=0, z=0)

    def control(self, forward=0.0, follow=True, lateral=0.0):
        angles, position = self.data.read()
        _, yaw_body, pitch_ground, yaw_ground = angles
        yaw_error = wrap(self.target_yaw - yaw_ground)
        pitch_error = self.target_pitch - pitch_ground
        chassis_error = wrap(yaw_body + yaw_error)
        walking = getattr(self, 'phase', None) == 'moving' and forward > 0
        gimbal_yaw_speed = limit(GIMBAL_GAIN * yaw_error, MAX_GIMBAL_SPEED)
        if walking:
            if abs(yaw_error) <= ANGLE_TOLERANCE_DEG:
                gimbal_yaw_speed = 0.0
            else:
                self.stop_chassis()
                forward = 0.0
                lateral = 0.0
        if abs(yaw_body) > 150:
            raise RuntimeError('Gimbal approaching its mechanical yaw limit')
        gimbal_result = self.robot.gimbal.drive_speed(pitch_speed=limit(3.0 * pitch_error, 20.0), yaw_speed=getattr(self, 'gimbal_yaw_command_sign', 1) * gimbal_yaw_speed)
        if not self.logged_gimbal_result:
            LOG.info('Gimbal command return: %r (%s); None is normal for async SDK', gimbal_result, type(gimbal_result).__name__)
            self.logged_gimbal_result = True
        if gimbal_result is False:
            raise RuntimeError('Gimbal speed command could not be sent')
        chassis_speed = limit(CHASSIS_GAIN * chassis_error, MAX_CHASSIS_SPEED)
        if not follow:
            chassis_speed = 0.0
        if abs(yaw_body) > ANGLE_TOLERANCE_DEG or abs(chassis_error) > ANGLE_TOLERANCE_DEG or abs(yaw_error) > ANGLE_TOLERANCE_DEG:
            forward = 0.0
            lateral = 0.0
        self.robot.chassis.drive_speed(x=forward, y=lateral, z=chassis_speed, timeout=0.5)
        now = time.monotonic()
        if now - self.last_status_time >= 1.0:
            LOG.info('target=%.2f ground=%.2f yaw_body=%.2f error=%.2f forward=%.3f turn=%.2f position=(%.3f, %.3f)', self.target_yaw, yaw_ground, yaw_body, yaw_error, forward, chassis_speed, position[0], position[1])
            self.last_status_time = now
        return (yaw_error, wrap(yaw_body), position)

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
        raise RuntimeError('Alignment timed out: target_error=%.2f, body_error=%.2f, follow=%s' % (yaw_error, body_error, follow))

    def forward_cell(self, number):
        self.settle()
        _, start = self.data.read()
        deadline = time.monotonic() + self.cell_size / FORWARD_SPEED_MPS * 3 + 10
        last_progress = 0.0
        progress_time = time.monotonic()
        front_adjustments = 0
        while time.monotonic() < deadline:
            _, position = self.data.read()
            progress = math.hypot(position[0] - start[0], position[1] - start[1])
            remaining = self.cell_size - progress
            if remaining <= DISTANCE_TOLERANCE_M:
                self.stop_chassis()
                self.settle()
                LOG.info('Cell %d done: %.3f m', number, progress)
                return
            if progress > last_progress + 0.002:
                last_progress = progress
                progress_time = time.monotonic()
            stagnant_s = time.monotonic() - progress_time
            if stagnant_s > 5.0:
                raise SafetyStop('Wheel progress stopped at %.3f/%.3f m despite finish-speed retry' % (progress, self.cell_size))
            speed = min(FORWARD_SPEED_MPS, max(MIN_FINISH_SPEED_MPS, 1.2 * remaining))
            if remaining <= 0.04 and stagnant_s > 1.5:
                speed = FINISH_RETRY_SPEED_MPS
                if stagnant_s < 1.5 + LOOP_PERIOD_S:
                    LOG.warning('Wheel progress slow near cell end: %.3f/%.3f m; finish speed %.3f m/s', progress, self.cell_size, speed)
            try:
                self.control(forward=speed)
            except FrontWallAdjusted as exc:
                front_adjustments += 1
                next_cell = self.maze.neighbor(self.cell, self.heading)
                wall_beyond = self.maze.edge(next_cell, self.heading) is True
                _, settled_position = self.data.read()
                settled_progress = math.hypot(settled_position[0] - start[0], settled_position[1] - start[1])
                front_mm = self.tof.latest()
                if wall_beyond and progress >= self.cell_size - FRONT_WALL_CELL_TOLERANCE_M and (settled_progress >= self.cell_size - FRONT_WALL_POST_BACKOFF_TOLERANCE_M) and (abs(front_mm - FRONT_TARGET_MM) <= FRONT_TOLERANCE_MM):
                    self.stop_chassis()
                    LOG.info('Cell %d reached beside mapped wall: wheel progress before/after ToF backoff %.3f/%.3f of %.3f m, front ToF %.1f mm; correction=%s', number, progress, settled_progress, self.cell_size, front_mm, exc)
                    return
                if front_adjustments >= MAX_FRONT_ADJUSTMENTS_PER_CELL:
                    raise FrontObstacleStop('Front ToF needed %d corrections before cell completion; wheel progress %.3f/%.3f m' % (front_adjustments, progress, self.cell_size)) from None
                LOG.info('Front ToF correction %d/%d complete; resuming cell %d from wheel position', front_adjustments, MAX_FRONT_ADJUSTMENTS_PER_CELL, number)
                progress_time = time.monotonic()
                continue
            except FrontObstacleStop as exc:
                raise FrontObstacleStop('%s; wheel progress before backoff %.3f/%.3f m' % (exc, progress, self.cell_size)) from None
            time.sleep(LOOP_PERIOD_S)
        raise SafetyStop('Wheel distance did not reach %.3f m within cell timeout' % self.cell_size)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIRECTION_NAMES = ('+y', '+x', '-y', '-x')

class Maze:

    def __init__(self, width, height, goal):
        self.width, self.height, self.goal = (width, height, tuple(goal))
        self.edges = {}
        self.visited = set()

    def inside(self, cell):
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def neighbor(self, cell, direction):
        dx, dy = DIRECTIONS[direction]
        return (cell[0] + dx, cell[1] + dy)

    def key(self, a, b):
        return tuple(sorted((tuple(a), tuple(b))))

    def edge(self, cell, direction):
        other = self.neighbor(cell, direction)
        if not self.inside(other):
            return True
        return self.edges.get(self.key(cell, other))

    def observe(self, cell, direction, wall):
        other = self.neighbor(cell, direction)
        if not self.inside(other):
            if not wall:
                LOG.warning('Sensor reports open at map boundary %s %s; boundary remains closed', cell, DIRECTION_NAMES[direction])
            return
        key = self.key(cell, other)
        previous = self.edges.get(key)
        if previous is not None and previous != wall:
            raise RuntimeError('Conflicting wall readings at %s -> %s; check cell size/ToF' % (cell, other))
        self.edges[key] = bool(wall)

    def flood(self):
        from collections import deque
        costs = {self.goal: 0}
        queue = deque([self.goal])
        while queue:
            cell = queue.popleft()
            for direction in range(4):
                other = self.neighbor(cell, direction)
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
                value = float('nan')
            self.samples.append((self.sequence, time.monotonic(), value))
            self.samples = self.samples[-100:]

    def mark(self):
        with self.lock:
            return self.sequence

    @staticmethod
    def valid(value):
        return math.isfinite(value) and 0 < value < 65535

    def after(self, mark):
        with self.lock:
            return [v for seq, timestamp, v in self.samples if seq > mark and time.monotonic() - timestamp <= DATA_TIMEOUT_S and self.valid(v)]

    def latest(self):
        with self.lock:
            if not self.samples:
                raise RuntimeError('No ToF data received')
            _, timestamp, value = self.samples[-1]
            if time.monotonic() - timestamp > DATA_TIMEOUT_S or not self.valid(value):
                raise RuntimeError('Stale/invalid ToF: index=%d raw=%r' % (self.index, self.last_raw))
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
                raise RuntimeError('Unstable alignment ToF: %r' % (window,))
            time.sleep(LOOP_PERIOD_S)
        raise RuntimeError('No fresh alignment ToF: index=%d raw=%r' % (self.index, self.last_raw))

class OdometryTrack:
    """Convert wheel coordinates into the configured map frame."""

    def sample(self, angles, position):
        self.current_angles = angles
        point = (position[0], position[1])
        if self.origin is None:
            self.origin = point
        if not self.raw_track or math.hypot(point[0] - self.raw_track[-1][0], point[1] - self.raw_track[-1][1]) >= 0.003:
            self.raw_track.append(point)
        if self.forward_axis is None and self.nav.phase == 'moving':
            dx, dy = (point[0] - self.origin[0], point[1] - self.origin[1])
            length = math.hypot(dx, dy)
            if length >= 0.06:
                theta = math.radians(self.nav.heading * 90)
                self.forward_axis = ((dx * math.cos(theta) + dy * math.sin(theta)) / length, (-dx * math.sin(theta) + dy * math.cos(theta)) / length)
        self.update()

    def grid_position(self, point):
        start_x, start_y = self.nav.path[0]
        if self.forward_axis is None:
            return (start_x, start_y)
        fx, fy = self.forward_axis
        dx, dy = (point[0] - self.origin[0], point[1] - self.origin[1])
        return (start_x + (-dx * fy + dy * fx) / self.nav.cell_size, start_y + (dx * fx + dy * fy) / self.nav.cell_size)

class PygameMap(OdometryTrack):
    """Pygame view; the control loop pumps events about ten times a second."""

    def __init__(self, navigator):
        import pygame
        self.pg = pygame
        self.nav = navigator
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 820), pygame.RESIZABLE)
        pygame.display.set_caption('RoboMaster | main2 A* map')
        self.font = pygame.font.SysFont('Tahoma', 16)
        self.small = pygame.font.SysFont('Tahoma', 13)
        self.stop_requested = False
        self.finished = False
        self.last_update = -float('inf')
        self.origin = None
        self.forward_axis = None
        self.raw_track = []
        self.current_angles = None

    def update(self, force=False):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT or (event.type == self.pg.KEYDOWN and event.key in (self.pg.K_q, self.pg.K_ESCAPE)):
                self.stop_requested = True
        if self.stop_requested:
            raise KeyboardInterrupt('Map window stop requested')
        now = time.monotonic()
        if force or now - self.last_update >= 0.1:
            self.last_update = now
            self.draw()

    def draw(self):
        pg, nav = (self.pg, self.nav)
        screen = self.screen
        width, height = screen.get_size()
        screen.fill((25, 28, 36))
        scale = min((width - 360) / nav.maze.width, (height - 110) / nav.maze.height)
        scale = max(12, scale)
        left, top = (25, 45)

        def center(cell):
            return (int(left + (cell[0] + 0.5) * scale), int(top + (nav.maze.height - cell[1] - 0.5) * scale))
        costs = nav.maze.flood()
        for x in range(nav.maze.width):
            for y in range(nav.maze.height):
                px, py = center((x, y))
                rect = pg.Rect(px - scale / 2, py - scale / 2, scale, scale)
                pg.draw.rect(screen, (51, 63, 78) if (x, y) in nav.maze.visited else (41, 46, 57), rect)
                pg.draw.rect(screen, (91, 103, 118), rect, 1)
                if scale >= 28:
                    label = self.small.render(str(costs.get((x, y), 'X')), True, (150, 160, 175))
                    screen.blit(label, (px - label.get_width() / 2, py - label.get_height() / 2))
        for (a, b), wall in nav.maze.edges.items():
            if wall is not True:
                continue
            ax, ay = center(a)
            bx, by = center(b)
            mx, my = ((ax + bx) // 2, (ay + by) // 2)
            if a[0] != b[0]:
                pg.draw.line(screen, (234, 86, 76), (mx, my - scale / 2), (mx, my + scale / 2), 5)
            else:
                pg.draw.line(screen, (234, 86, 76), (mx - scale / 2, my), (mx + scale / 2, my), 5)
        if len(nav.path) >= 2:
            pg.draw.lines(screen, (156, 171, 190), False, [center(c) for c in nav.path], 2)
        if self.forward_axis is not None and len(self.raw_track) >= 2:
            pg.draw.lines(screen, (54, 149, 250), False, [center(self.grid_position(p)) for p in self.raw_track], 3)
        for cell, color in ((nav.path[0], (23, 173, 133)),):
            pg.draw.circle(screen, color, center(cell), 9)
        for goal in nav.targets:
            pg.draw.circle(screen, (245, 174, 47), center(goal), 9)
        robot_cell = self.grid_position(self.raw_track[-1]) if self.raw_track else nav.cell
        px, py = center(robot_cell)
        pg.draw.circle(screen, (53, 151, 239), (px, py), 10)
        if self.current_angles and nav.initial_yaw is not None:
            angle = math.radians(wrap(self.current_angles[3] - nav.initial_yaw))
            end = (int(px + math.sin(angle) * 26), int(py - math.cos(angle) * 26))
            pg.draw.line(screen, (53, 151, 239), (px, py), end, 3)
            pg.draw.circle(screen, (53, 151, 239), end, 4)
        panel_x = int(left + nav.maze.width * scale + 22)
        status = ['MAIN3  |  A* route', 'Cell: %s' % (nav.cell,), 'Heading: %s' % DIRECTION_NAMES[nav.heading], 'Phase: %s' % nav.phase, 'Steps: %d' % nav.steps, 'Targets: %s' % (nav.targets,), 'Done: %s' % (sorted(nav.completed_targets),), 'Fixed main yaw: %s' % ('%.1f' % nav.main_yaw if nav.main_yaw is not None else '--'), 'Walking yaw: %s' % ('%.1f' % nav.navigation_yaw() if nav.initial_yaw is not None else '--'), 'Chassis yaw: %s' % ('%.1f' % nav.data.chassis_attitude[0] if nav.data.chassis_attitude else '--'), 'ToF: %s' % ('%.0f mm' % nav.tof.last_raw[nav.tof.index] if nav.tof.last_raw and len(nav.tof.last_raw) > nav.tof.index else '--'), 'Q / Esc / close: stop']
        for index, line in enumerate(status):
            screen.blit(self.font.render(line, True, (230, 235, 240)), (panel_x, 50 + index * 28))
        pg.display.flip()

    def finish(self):
        self.finished = True
        while not self.stop_requested:
            try:
                self.update(force=True)
            except KeyboardInterrupt:
                break
            time.sleep(0.1)
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
        self.planned_order = ()
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
        self.initial_yaw = None
        self.main_yaw = None
        # Corrections change the walking heading, never the recenter anchor.
        self.heading_trim_deg = 0.0
        self.gimbal_yaw_command_sign = getattr(args, 'gimbal_yaw_sign', 1)
        self.clearance_mm = args.clearance_mm
        self.max_steps = args.max_steps or config.width * config.height * max(10, len(config.targets) * 4)
        self.path = [self.cell]
        self.steps = 0
        self.phase = 'starting'
        self.arrived = False
        self.live_map = None
        self.output = Path(__file__).resolve().with_name('main3_map.json')

    def navigation_yaw(self):
        anchor = self.main_yaw if self.main_yaw is not None else self.initial_yaw
        return wrap(anchor + self.heading * 90.0 + self.heading_trim_deg)

    def calibrate_gimbal_yaw_sign(self):
        """Measure whether a +10 degree SDK yaw command raises or lowers yaw_body."""
        self.stop_chassis()
        before, _ = self.data.read()
        if abs(before[1]) > 5:
            raise RuntimeError('Gimbal is not centered before yaw-sign calibration: %r' % (before,))
        action = self.robot.gimbal.moveto(pitch=0, yaw=10, pitch_speed=60, yaw_speed=120)
        if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
            raise RuntimeError('Gimbal yaw-sign calibration motion failed')
        completed_at = time.monotonic()
        deadline = completed_at + 2.0
        while time.monotonic() < deadline:
            with self.data.lock:
                if self.data.gimbal_time > completed_at:
                    break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError('No fresh gimbal angle during yaw-sign calibration')
        after, _ = self.data.read()
        delta = wrap(after[1] - before[1])
        LOG.info('GIMBAL YAW SIGN probe commanded=+10 body_before=%.2f body_after=%.2f ground_before=%.2f ground_after=%.2f pitch_ground=%.2f', before[1], after[1], before[3], after[3], after[2])
        if not 4 <= abs(delta) <= 20:
            raise RuntimeError('Cannot determine gimbal yaw command sign: delta=%.2f angles=%r' % (delta, after))
        self.gimbal_yaw_command_sign = 1 if delta > 0 else -1
        action = self.robot.gimbal.recenter(pitch_speed=60, yaw_speed=120)
        if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
            raise RuntimeError('Could not recenter gimbal after yaw-sign calibration')
        completed_at = time.monotonic()
        deadline = completed_at + 2.0
        while time.monotonic() < deadline:
            with self.data.lock:
                if self.data.gimbal_time > completed_at:
                    break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError('No fresh angle after yaw-sign calibration recenter')
        centered, _ = self.data.read()
        LOG.info('GIMBAL YAW SIGN result=%+d centered_angles=%r', self.gimbal_yaw_command_sign, centered)
        if abs(centered[1]) > 2 or abs(centered[2]) > 2:
            raise RuntimeError('Gimbal did not recenter after yaw-sign calibration: %r' % (centered,))

    def _align_point(self, offset_yaw):
        """Point to saved main_yaw + offset, converted to body coordinates."""
        self.stop_chassis()
        self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        angles, _ = self.data.read()
        body_ground_yaw = wrap(angles[3] - angles[1])
        desired_ground_yaw = wrap(self.navigation_yaw() + offset_yaw)
        relative_yaw = offset_yaw + wrap(self.navigation_yaw() - body_ground_yaw)
        if self.gimbal_yaw_command_sign not in (-1, 1):
            raise RuntimeError('Gimbal yaw sign has not been calibrated')
        commanded_yaw = self.gimbal_yaw_command_sign * relative_yaw
        if abs(commanded_yaw) > 250:
            raise RuntimeError('Alignment gimbal yaw exceeds SDK limit: %.1f' % commanded_yaw)
        for attempt in range(2):
            action = self.robot.gimbal.moveto(pitch=0, yaw=commanded_yaw, pitch_speed=540, yaw_speed=540)
            if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
                raise RuntimeError('Alignment gimbal motion failed: command=%.1f offset=%+.1f' % (commanded_yaw, offset_yaw))
            completed_at = time.monotonic()
            deadline = completed_at + 1.5
            while time.monotonic() < deadline:
                with self.data.lock:
                    fresh = self.data.gimbal_time > completed_at
                if fresh:
                    angles, _ = self.data.read()
                    if abs(wrap(desired_ground_yaw - angles[3])) <= 2.0 and abs(angles[2]) <= 1.0:
                        time.sleep(0.04)
                        return self.tof.mark()
                time.sleep(LOOP_PERIOD_S)
            angles, _ = self.data.read()
            residual = wrap(desired_ground_yaw - angles[3])
            if attempt == 0 and abs(residual) <= 10.0 and (abs(angles[2]) <= 3.0):
                commanded_yaw = self.gimbal_yaw_command_sign * (angles[1] + residual)
                LOG.info('ALIGN gimbal trim offset=%+.1f residual=%+.2f command=%.2f', offset_yaw, residual, commanded_yaw)
                continue
            raise RuntimeError('Gimbal did not settle: offset=%+.1f desired_ground=%.2f command=%.2f measured=%r' % (offset_yaw, desired_ground_yaw, commanded_yaw, angles))

    def _align_body_center(self, update_main=False):
        """Return the head to the walking bearing without changing main_yaw."""
        self.stop_chassis()
        self.robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        self._align_point(0)
        self.target_yaw = self.navigation_yaw()
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
        error = math.degrees(math.atan2(minus - plus, (plus + minus) * math.tan(math.radians(ALIGN_OFFSET_DEG))))
        signed = ALIGN_DIRECTION_SIGN[name] * ALIGN_CHASSIS_PER_GIMBAL_SIGN * error
        remaining = ALIGN_MAX_TOTAL_DEG - used
        if remaining < 0.1:
            return 0.0
        step = round(limit(ALIGN_KP * signed, min(ALIGN_MAX_STEP_DEG, remaining)), 1)
        return step or math.copysign(0.1, signed)

    def align_to_wall(self):
        """Find one wall, right before left before back, then align only it."""
        self.phase = 'wall_alignment'
        self.stop_chassis()
        selected = None
        for name, base in (('right', 90), ('left', -90), ('back', -180)):
            try:
                distance = self._align_read(base)
                LOG.info('ALIGN scan %s %.1f mm', name, distance)
                if distance <= ALIGN_WALL_MM:
                    selected = (name, base)
                    break
            except RuntimeError as exc:
                LOG.warning('ALIGN scan %s unavailable: %s', name, exc)
        aligned = False
        try:
            if selected is not None:
                name, base = selected
                LOG.info('ALIGN selected %s (search order right, left, back)', name)
                used = 0.0
                for correction in range(ALIGN_MAX_CORRECTIONS + 1):
                    try:
                        self._align_point(base)
                        plus = self._align_read(base + ALIGN_OFFSET_DEG)
                        self._align_point(base)
                        minus = self._align_read(base - ALIGN_OFFSET_DEG)
                        self._align_point(base)
                    except RuntimeError as exc:
                        LOG.warning('ALIGN %s readings unavailable: %s', name, exc)
                        break
                    difference = abs(plus - minus)
                    LOG.info('ALIGN %s check %d: +%.1f -%.1f diff=%.1f mm', name, correction + 1, plus, minus, difference)
                    if difference <= ALIGN_TOLERANCE_MM:
                        aligned = True
                        break
                    if correction == ALIGN_MAX_CORRECTIONS:
                        break
                    step = self.alignment_step(name, plus, minus, used)
                    if abs(step) < 0.1:
                        break
                    action = self.robot.chassis.move(x=0, y=0, z=step, z_speed=120)
                    if not action.wait_for_completed(timeout=5) or not action.has_succeeded:
                        raise RuntimeError('Alignment chassis yaw failed')
                    used += abs(step)
                    self.heading_trim_deg = wrap(self.heading_trim_deg + step)
                    self.stop_chassis()
                    time.sleep(0.08)
                    self._align_body_center()
                    LOG.info('ALIGN %s yaw %+.1f degrees (%d/%d, total %.1f)', name, step, correction + 1, ALIGN_MAX_CORRECTIONS, used)
        finally:
            self._align_body_center()
            self.settle()
            self.last_alignment_step = self.steps
            self.straight_cells = 0
        LOG.info('ALIGN result selected=%s aligned=%s fixed_main_yaw=%.2f walking_yaw=%.2f', selected[0] if selected else None, aligned, self.main_yaw, self.navigation_yaw())
        if selected is not None and (not aligned):
            LOG.warning('ALIGN %s outside %.1f mm after %d corrections; continuing', selected[0], ALIGN_TOLERANCE_MM, ALIGN_MAX_CORRECTIONS)
        try:
            self.adjust_front_wall()
        except RuntimeError as exc:
            if self.maze.edge(self.cell, self.heading) is True or not str(exc).startswith(('No fresh alignment ToF', 'Unstable alignment ToF')):
                raise
            LOG.warning('Front ToF check unavailable on open mapped edge: %s', exc)
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
        LOG.info('FRONT WALL adjusting %.1f toward %.0f mm', distance, FRONT_TARGET_MM)
        _, start = self.data.read()
        deadline = time.monotonic() + 6.0
        try:
            while time.monotonic() < deadline:
                distance = self.tof.latest()
                error = distance - FRONT_TARGET_MM
                if abs(error) <= FRONT_TOLERANCE_MM:
                    LOG.info('FRONT WALL settled at %.1f mm', distance)
                    if unexpected and (not emergency):
                        raise RuntimeError('Front wall conflicts with mapped open path; stopped at %.0f mm' % FRONT_TARGET_MM)
                    return True
                _, position = self.data.read()
                if math.hypot(position[0] - start[0], position[1] - start[1]) > 0.25:
                    raise RuntimeError('Front adjustment exceeded 0.25 m')
                Route.control(self, forward=-0.05 if error < 0 else 0.05)
                time.sleep(LOOP_PERIOD_S)
            raise RuntimeError('Front wall did not reach %.0f mm within timeout' % FRONT_TARGET_MM)
        finally:
            self.stop_chassis()

    def control(self, forward=0.0, follow=True, lateral=0.0):
        if self.phase == 'moving':
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
                    raise FrontObstacleStop('Front ToF %.1f mm during movement; backoff unavailable: %s' % (distance, exc)) from None
                raise FrontWallAdjusted('Front ToF %.1f mm during movement; backed toward %.0f mm' % (distance, FRONT_TARGET_MM))
        result = super().control(forward=forward, follow=follow, lateral=0.0)
        if self.live_map is not None:
            angles, position = self.data.read()
            self.live_map.sample(angles, position)
        return result

    def turn_to(self, direction):
        delta = (direction - self.heading) % 4
        increments = {0: (), 1: (1,), 2: (1, 1), 3: (-1,)}[delta]
        if not increments:
            return
        self.phase = 'turning'
        try:
            self._turn_increments(increments)
        finally:
            self.stop_chassis()

    def _turn_increments(self, increments):
        for increment in increments:
            previous_yaw = self.navigation_yaw()
            next_heading = (self.heading + increment) % 4
            self.target_yaw = wrap(self.main_yaw + next_heading * 90.0 + self.heading_trim_deg)
            LOG.info('TURN fixed_main=%.2f previous_yaw=%.2f offset=%+d target=%.2f heading=%s', self.main_yaw, previous_yaw, increment * 90, self.target_yaw, DIRECTION_NAMES[next_heading])
            self.settle(follow=False)
            self.settle(follow=True)
            angles, _ = self.data.read()
            if abs(wrap(self.target_yaw - angles[3])) > ANGLE_TOLERANCE_DEG or abs(wrap(angles[1])) > ANGLE_TOLERANCE_DEG:
                raise RuntimeError('Turn not aligned; previous main yaw retained')
            self.heading = next_heading
            self.target_yaw = self.navigation_yaw()
            LOG.info('TURN complete fixed_main=%.2f walking_yaw=%.2f body_offset=%.2f', self.main_yaw, self.target_yaw, angles[1])
            self.save()

    def save(self):
        costs = self.maze.flood()
        try:
            angles, position = self.data.read()
        except RuntimeError:
            if self.phase not in ('stopped', 'safety_stop', 'front_obstacle_stop'):
                raise
            angles, position = self.data.read_cached()
        payload = dict(width=self.maze.width, height=self.maze.height, cell_size_m=self.cell_size, targets=self.targets, completed_targets=sorted(self.completed_targets), planned_order=self.planned_order, configured_walls=sorted(self.configured_walls), current_cell=self.cell, heading=DIRECTION_NAMES[self.heading], initial_yaw=self.initial_yaw, main_yaw=self.main_yaw, heading_trim_deg=self.heading_trim_deg, navigation_yaw=self.navigation_yaw(), actual_gimbal_yaw=angles[3], odometry_position=position, chassis_attitude=self.data.chassis_attitude, phase=self.phase, all_targets_reached=self.arrived, path=self.path, visited=sorted(self.maze.visited), flood_values=[[costs.get((x, y)) for x in range(self.maze.width)] for y in range(self.maze.height)], edges=[dict(a=a, b=b, wall=wall) for (a, b), wall in sorted(self.maze.edges.items())], notes='A* routes through configured map walls; fixed main_yaw plus heading and alignment trim guides walking. ToF handles front clearance and wall alignment. All mapped targets are visited without camera scanning or Sharp.')
        temporary = self.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(self.output)
        if self.live_map is not None:
            self.live_map.update(force=True)

    def run(self):
        angles, _ = self.data.read()
        self.initial_yaw = angles[3]
        self.main_yaw = wrap(angles[3])
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = angles[2]
        LOG.info('START cell=%s targets=%s map=%dx%d initial_yaw=%.2f', self.cell, self.targets, self.maze.width, self.maze.height, self.initial_yaw)
        try:
            self.settle()
            if self.cell in self.targets:
                self.completed_targets.add(self.cell)
            self.save()
            while len(self.completed_targets) < len(self.targets):
                if self.steps >= self.max_steps:
                    raise RuntimeError('Step limit reached before goal')
                remaining = set(self.targets) - self.completed_targets
                order = optimal_target_order(self.maze, self.cell, remaining)
                if not order:
                    raise RuntimeError('Remaining targets unreachable through configured walls')
                self.planned_order = order
                self.maze.goal = order[0]
                route = a_star(self.maze, self.cell, order[0])
                if route is None or len(route) < 2:
                    raise RuntimeError('No route to next target')
                next_cell = route[1]
                direction = DIRECTIONS.index((next_cell[0] - self.cell[0], next_cell[1] - self.cell[1]))
                if self.maze.edge(self.cell, direction) is not False:
                    raise RuntimeError('A* selected a wall or unknown edge')
                LOG.info('A* order=%s path=%s choose=%s -> %s', order, route, self.cell, next_cell)
                old_heading = self.heading
                self.turn_to(direction)
                if self.heading != old_heading:
                    self.align_to_wall()
                if self.tof.latest() <= self.clearance_mm:
                    LOG.warning('Front ToF %.1f mm before moving; correcting toward %.0f mm', self.tof.latest(), FRONT_TARGET_MM)
                    self.adjust_front_wall(emergency=True)
                    if self.tof.latest() <= self.clearance_mm:
                        raise FrontObstacleStop('Front ToF remains within %.0f mm after adjustment' % self.clearance_mm)
                self.phase = 'moving'
                self.forward_cell(self.steps + 1)
                self.maze.observe(self.cell, direction, False)
                self.cell = next_cell
                self.steps += 1
                self.path.append(self.cell)
                self.maze.visited.add(self.cell)
                self.phase = 'at_cell'
                self.straight_cells += 1
                new_target = self.cell in self.targets and self.cell not in self.completed_targets
                if new_target:
                    self.completed_targets.add(self.cell)
                    LOG.info('TARGET reached %s (%d/%d)', self.cell, len(self.completed_targets), len(self.targets))
                if self.straight_cells >= 3:
                    self.align_to_wall()
                    self.phase = 'at_cell'
                self.save()
            self.stop_chassis()
            self.arrived = True
            self.phase = 'all_targets_reached'
            self.save()
            LOG.info('ALL TARGETS REACHED cell=%s completed_moves=%d', self.cell, self.steps)
        except SafetyStop as exc:
            self.stop_chassis()
            self.phase = 'front_obstacle_stop' if isinstance(exc, FrontObstacleStop) else 'safety_stop'
            LOG.warning('NAVIGATION HELD: %s; cell=%s; target not marked reached', exc, self.cell)
            try:
                self.save()
            except Exception:
                LOG.exception('Could not save safety stop map')
            return
        except BaseException:
            self.stop_chassis()
            self.phase = 'stopped'
            try:
                self.save()
            except Exception:
                LOG.exception('Could not save final map')
            raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conn', choices=('ap', 'sta'), default='ap')
    parser.add_argument('--map', type=Path, default=Path(__file__).resolve().with_name('map_config.json'))
    parser.add_argument('--cell-size', type=float, help='Override map_config.json cell_size_m')
    parser.add_argument('--clearance-mm', type=float, default=120.0)
    parser.add_argument('--gimbal-yaw-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--calibrate-gimbal-sign', action='store_true')
    parser.add_argument('--tof-index', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--no-graph', action='store_true')
    args = parser.parse_args()
    try:
        config = load_map_config(args.map)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error('Invalid map %s: %s' % (args.map, exc))
    if not config.targets:
        print('map_config.json has no navigation targets (targets: []).')
        while True:
            try:
                raw = input('Navigation target grid x,y; x,y (example: 1,1): ')
                config = replace(config, targets=parse_grid_targets(raw, config.width, config.height))
                break
            except ValueError as exc:
                print(exc)
            except EOFError:
                parser.error('Map has no targets; set them in map_config.json')
    if args.cell_size is None:
        args.cell_size = config.cell_size_m
    if not math.isfinite(args.cell_size) or not 0.2 <= args.cell_size <= 2.0:
        parser.error('--cell-size must be 0.20..2.0 m')
    if args.tof_index not in range(4) or args.max_steps < 0:
        parser.error('--tof-index 0..3; --max-steps >= 0')
    if not 0 < args.clearance_mm < args.cell_size * 1000:
        parser.error('Require 0 < clearance-mm < cell size in mm')
    if len(config.targets) > 8:
        parser.error('At most 8 targets supported by the A* permutation planner')
    log_path = Path(__file__).resolve().with_name('main3_run.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode='w', encoding='utf-8')], force=True)
    LOG.info('main3 version %s', VERSION)
    LOG.info('Running file: %s', Path(__file__).resolve())
    LOG.info('Map: %s; start=%s facing +y; targets=%s; walls=%d', args.map.resolve(), config.start, config.targets, len(config.walls))
    from robomaster import robot
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
            raise RuntimeError('Cannot set FREE mode')
        data, tof = (Telemetry(), ToF(args.tof_index))
        for subscribe, unsubscribe in ((lambda: ep_robot.gimbal.sub_angle(freq=20, callback=data.on_gimbal), ep_robot.gimbal.unsub_angle), (lambda: ep_robot.chassis.sub_position(cs=1, freq=20, callback=data.on_position), ep_robot.chassis.unsub_position), (lambda: ep_robot.chassis.sub_attitude(freq=20, callback=data.on_attitude), ep_robot.chassis.unsub_attitude), (lambda: ep_robot.sensor.sub_distance(freq=20, callback=tof.callback), ep_robot.sensor.unsub_distance)):
            if not subscribe():
                raise RuntimeError('Telemetry subscription failed')
            subscriptions.append(unsubscribe)
        data.wait_ready()
        ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        LOG.info('RECENTER before recording the first main yaw')
        action = ep_robot.gimbal.recenter(pitch_speed=60, yaw_speed=60)
        if action is not None and (not action.wait_for_completed(timeout=10) or not action.has_succeeded):
            raise RuntimeError('Gimbal recenter failed')
        completed_at = time.monotonic()
        deadline = completed_at + 3.0
        while time.monotonic() < deadline:
            with data.lock:
                fresh = data.gimbal_time > completed_at
            if fresh:
                break
            time.sleep(LOOP_PERIOD_S)
        else:
            raise RuntimeError('No fresh gimbal angle after recenter')
        time.sleep(0.3)
        data.read()
        navigator = MainNavigator(ep_robot, data, tof, args, config)
        if args.calibrate_gimbal_sign:
            navigator.calibrate_gimbal_yaw_sign()
        if not args.no_graph:
            live_map = PygameMap(navigator)
            navigator.live_map = live_map
        navigator.run()
        completed = True
    finally:
        if initialized:
            for stop in (lambda: ep_robot.chassis.drive_speed(x=0, y=0, z=0), lambda: ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)):
                try:
                    stop()
                except Exception:
                    LOG.exception('Stop command failed')
            for unsubscribe in reversed(subscriptions):
                try:
                    unsubscribe()
                except Exception:
                    LOG.exception('Unsubscribe failed')
        ep_robot.close()
        if live_map is not None and (not completed):
            live_map.close()
    if live_map is not None:
        live_map.finish()
if __name__ == '__main__':
    main()
