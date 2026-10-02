"""Run: python work_gimball_floodfill.py

Interactive map width,height and goal x,y; origin is (0,0), facing +y.
Example: python work_gimball_floodfill.py --size 4,5 --goal 2,3
Standalone: RoboMaster SDK + Python standard library only.
Requires a ToF sensor mounted on the gimbal, beam aligned with gimbal yaw.
Place the robot at the CENTER of the starting cell; use a horizontal beam.
Unknown passages are allowed for flood estimates, but must be scanned open
before moving. Equal flood costs prefer front, right, left, then back.
The saved main ground yaw stays fixed during travel and side scans. After
each completed turn, measured gimbal yaw becomes the next saved main yaw.
Side scans NEVER rotate the chassis. Default cell size is 0.60 m.
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
ANGLE_TOLERANCE_DEG = 0.5
DISTANCE_TOLERANCE_M = 0.01
SETTLE_S = 0.50
GIMBAL_GAIN = 3.0
CHASSIS_GAIN = 2.0
MAX_GIMBAL_SPEED = 45.0
MAX_CHASSIS_SPEED = 30.0
LOG = logging.getLogger("gimbal_locked_route")
VERSION = "2026-09-29-floodfill-r6-recenter"


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
        if abs(yaw_body) > 150:
            raise RuntimeError("Gimbal approaching its mechanical yaw limit")
        # The installed SDK sends this asynchronously and normally returns None.
        # None is not a rejection/ack; only an explicit False indicates failure.
        # Ground-angle feedback and settle timeout verify the resulting motion.
        gimbal_result = self.robot.gimbal.drive_speed(
                pitch_speed=limit(3.0 * pitch_error, 20.0),
                yaw_speed=gimbal_yaw_speed)
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

    def choose(self, cell, heading):
        costs = self.flood()
        if cell not in costs:
            raise RuntimeError("Goal unreachable through discovered walls")
        priority = (heading, (heading + 1) % 4, (heading - 1) % 4, (heading + 2) % 4)
        candidates = [d for d in priority
                      if self.edge(cell, d) is False
                      and self.neighbor(cell, d) in costs]
        if not candidates:
            raise RuntimeError("No confirmed open passage from %s" % (cell,))
        # Flood distance first; tie-break front, right, left, then back.
        return min(candidates, key=lambda d: costs[self.neighbor(cell, d)])


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


def classify_wall_samples(samples, wall_mm, margin_mm=30.0):
    """Return representative distance only when wall/open votes agree.

    Absolute distance spread is irrelevant when every reading is far beyond
    the adjacent-wall boundary. Near-threshold or conflicting readings need
    a new window; never turn missing/uncertain data into an open passage.
    """
    if len(samples) < 9:
        return None
    window = samples[-9:]
    close = [v for v in window if v <= wall_mm - margin_mm]
    far = [v for v in window if v >= wall_mm + margin_mm]
    if len(close) >= 8 and not far:
        return sorted(close)[len(close) // 2]
    if len(far) >= 8 and not close:
        return sorted(far)[len(far) // 2]
    return None


class LiveMap:
    """Tk window updated by the same thread that controls the robot."""

    def __init__(self, navigator):
        import tkinter as tk
        self.nav = navigator
        self.root = tk.Tk()
        self.root.title("RoboMaster - Live Flood Fill Map")
        self.canvas = tk.Canvas(self.root, width=940, height=820, bg="#f7f9fc", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.minsize(700, 650)
        self.stop_requested = False
        self.finished = False
        self.last_update = -float("inf")
        self.origin = None
        self.forward_axis = None
        self.raw_track = []
        self.current_angles = None
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self.root.bind("<KeyPress-q>", lambda event: self.request_close())
        self.root.bind("<Escape>", lambda event: self.request_close())
        self.root.update()

    def request_close(self):
        if self.finished:
            self.root.destroy()
        else:
            self.stop_requested = True

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

    def update(self, force=False):
        if self.stop_requested:
            raise KeyboardInterrupt("Live map stop requested")
        now = time.monotonic()
        if not force and now-self.last_update < .10:
            return
        self.last_update = now
        self.draw()
        self.root.update_idletasks()
        self.root.update()
        if self.stop_requested:
            raise KeyboardInterrupt("Live map stop requested")

    def draw(self):
        c, nav = self.canvas, self.nav
        c.delete("all")
        width = max(700, c.winfo_width())
        height = max(650, c.winfo_height())
        size = min((width-120)/nav.maze.width, (height-260)/nav.maze.height)
        left = (width-size*nav.maze.width)/2
        top = 155
        def point(x, y):
            return left+(x+.5)*size, top+(nav.maze.height-y-.5)*size
        def text(x, y, value, color="#172c48", font_size=12):
            c.create_text(x, y, text=value, fill=color, font=("Segoe UI", font_size), anchor="w")
        text(28, 28, "RoboMaster | Live Flood Fill", font_size=20)
        text(28, 65, "Blue: wheel odometry   Grey dashed: completed cell path   Red: detected wall", font_size=11)
        text(28, 91, "Green: start   Orange: goal   Cell numbers: flood distance   Q / Esc / close: stop", font_size=11)
        text(28, 120, "cell=%s  goal=%s  heading=%s  phase=%s  moves=%d" %
             (nav.cell, nav.maze.goal, DIRECTION_NAMES[nav.heading], nav.phase, nav.steps), font_size=12)
        costs = nav.maze.flood()
        for x in range(nav.maze.width):
            for y in range(nav.maze.height):
                px, py = point(x, y)
                fill = "#edf5ff" if (x,y) in nav.maze.visited else "white"
                c.create_rectangle(px-size/2, py-size/2, px+size/2, py+size/2, fill=fill, outline="#dfe5ee")
                if size >= 26:
                    c.create_text(px, py, text=str(costs.get((x,y), "X")), fill="#b2bdcd", font=("Segoe UI", min(14,int(size/4))))
        # Thin grid lines represent unknown passages. Walls are drawn once.
        c.create_rectangle(left, top, left+size*nav.maze.width, top+size*nav.maze.height, outline="#d74b4b", width=3)
        for (a,b), wall in nav.maze.edges.items():
            if not wall:
                continue
            ax, ay = point(*a)
            bx, by = point(*b)
            mx, my = (ax+bx)/2, (ay+by)/2
            if a[0] != b[0]:
                c.create_line(mx,my-size/2,mx,my+size/2,fill="#d74b4b",width=4)
            else:
                c.create_line(mx-size/2,my,mx+size/2,my,fill="#d74b4b",width=4)
        if len(nav.path)>1:
            coords=[v for cell in nav.path for v in point(*cell)]
            c.create_line(*coords,fill="#adb8c8",width=2,dash=(5,4))
        if self.forward_axis is not None and len(self.raw_track)>1:
            coords=[v for p in self.raw_track for v in point(*self.grid_position(p))]
            c.create_line(*coords,fill="#1765d1",width=3)
        for cell, color, label in ((nav.path[0],"#169e79","S"),(nav.maze.goal,"#ed9419","G")):
            x,y=point(*cell)
            radius=min(12,size*.2)
            c.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,outline="white",width=2)
            c.create_text(x,y,text=label,fill="white",font=("Segoe UI",10,"bold"))
        if self.raw_track:
            x,y=point(*self.grid_position(self.raw_track[-1]))
            radius=min(8,size*.12)
            c.create_oval(x-radius,y-radius,x+radius,y+radius,fill="#1765d1",outline="white",width=2)
            if self.current_angles and nav.initial_yaw is not None:
                theta=math.radians(wrap(self.current_angles[3]-nav.initial_yaw))
                c.create_line(x,y,x+math.sin(theta)*size*.32,y-math.cos(theta)*size*.32,
                              fill="#1765d1",width=3,arrow="last")
        bottom=top+size*nav.maze.height
        for x in range(nav.maze.width):
            if size>=26 or x%5==0:
                c.create_text(point(x,0)[0],bottom+18,text=str(x),fill="#65758d")
        for y in range(nav.maze.height):
            if size>=26 or y%5==0:
                c.create_text(left-18,point(0,y)[1],text=str(y),fill="#65758d")
        text(left+size*nav.maze.width/2-55,bottom+44,"x -> right   y -> up",font_size=11)
        note=("Wheel odometry; %.2f m/cell; refreshed at 10 Hz" % nav.cell_size
              if self.forward_axis is not None else "Calibrating odometry direction during first movement...")
        text(28,height-47,note,font_size=11)
        if self.current_angles and nav.initial_yaw is not None:
            text(28,height-22,"Main yaw %.1f | Gimbal yaw %.1f | Body offset %.1f deg" %
                 (nav.navigation_yaw(),self.current_angles[3],self.current_angles[1]),font_size=11)

    def finish(self):
        self.finished = True
        self.update(force=True)
        self.root.mainloop()  # Robot connection has already been closed.

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


class FloodNavigator(Route):
    def __init__(self, ep_robot, data, tof, args):
        super().__init__(ep_robot, data, args.cell_size)
        self.tof = tof
        self.maze = Maze(args.width, args.height, args.goal)
        self.cell = tuple(args.start)
        self.heading = 0
        self.initial_yaw = None  # Immutable heading reference captured ONCE.
        self.main_yaw = None  # Updated ONLY after a completed deliberate turn.
        self.wall_mm = args.wall_mm
        self.clearance_mm = args.clearance_mm
        self.max_steps = args.max_steps or args.width * args.height * 10
        self.path = [self.cell]
        self.steps = 0
        self.phase = "starting"
        self.arrived = False
        self.live_map = None
        self.output = Path(__file__).resolve().with_name("work_gimball_floodfill_map.json")

    def navigation_yaw(self):
        return self.main_yaw if self.main_yaw is not None else wrap(self.initial_yaw)

    def control(self, forward=0.0, follow=True):
        if self.phase == "moving":
            # Side-scan targets cannot leak into a driving command.
            self.target_yaw = self.navigation_yaw()
        if forward > 0:
            distance = self.tof.latest()
            if distance <= self.clearance_mm:
                self.stop_chassis()
                raise RuntimeError("Obstacle too close: %.1f mm; stopped mid-cell, grid pose not advanced" % distance)
        result = super().control(forward=forward, follow=follow)
        if self.live_map is not None:
            angles, position = self.data.read()
            self.live_map.sample(angles, position)
        return result

    def point_and_read(self, scan_target):
        self.target_yaw = wrap(scan_target)
        # No chassis following while looking sideways.
        self.settle(follow=False)
        # Let the sensor pipeline clear while actively holding the scan angle.
        flush_until = time.monotonic() + 0.25
        while time.monotonic() < flush_until:
            self.control(follow=False)
            time.sleep(LOOP_PERIOD_S)
        mark = self.tof.mark()  # Discard every sample received before settling.
        deadline = time.monotonic() + 6.0
        samples = []
        while time.monotonic() < deadline:
            yaw_error, _, _ = self.control(follow=False)
            angles, _ = self.data.read()
            if (abs(yaw_error) > ANGLE_TOLERANCE_DEG
                    or abs(self.target_pitch - angles[2]) > ANGLE_TOLERANCE_DEG):
                mark = self.tof.mark()
                time.sleep(LOOP_PERIOD_S)
                continue
            samples = self.tof.after(mark)
            distance = classify_wall_samples(samples, self.wall_mm,
                                             min(30.0, self.cell_size * 50.0))
            if distance is not None:
                window = samples[-9:]
                LOG.info("TOF consensus=%s samples=%r spread=%.1f mm",
                         "WALL" if distance <= self.wall_mm else "OPEN",
                         window, max(window) - min(window))
                return distance
            time.sleep(LOOP_PERIOD_S)
        raise RuntimeError("ToF wall/open decision remained uncertain for 6 seconds: "
                           "threshold=%.1f mm index=%d recent_valid=%r raw=%r" %
                           (self.wall_mm, self.tof.index, samples[-9:], self.tof.last_raw))

    def scan_cell(self):
        self.phase = "scanning"
        self.stop_chassis()
        self.target_yaw = self.navigation_yaw()
        self.settle()
        # This main heading never becomes the previous side-scan heading.
        main_yaw = self.navigation_yaw()
        navigation_pitch = self.target_pitch
        self.target_pitch = 0.0  # Horizontal ToF ray for all wall scans.
        observations = []
        try:
            for label, offset, direction in (
                    ("front", 0, self.heading),
                    ("right", 90, (self.heading + 1) % 4),
                    ("left", -90, (self.heading - 1) % 4)):
                distance = self.point_and_read(main_yaw + offset)
                wall = distance <= self.wall_mm
                observations.append((direction, wall))
                LOG.info("SCAN cell=%s %s main=%.2f offset=%+d target=%.2f distance=%.1f mm %s",
                         self.cell, label, main_yaw, offset, self.target_yaw,
                         distance, "WALL" if wall else "OPEN")
        finally:
            # Restore the saved navigation heading, even when a scan fails.
            self.target_yaw = main_yaw
            self.target_pitch = navigation_pitch
            self.settle(follow=False)
        for direction, wall in observations:
            self.maze.observe(self.cell, direction, wall)
        # Usually the rear edge is the known passage just traversed, or the
        # start boundary. A custom interior start may have an unknown rear edge.
        rear = (self.heading + 2) % 4
        if self.maze.edge(self.cell, rear) is None:
            original_heading = self.heading
            self.turn_to(rear)
            distance = self.point_and_read(self.navigation_yaw())
            self.maze.observe(self.cell, rear, distance <= self.wall_mm)
            self.turn_to(original_heading)
        self.maze.visited.add(self.cell)
        self.settle(follow=True)
        self.save()

    def turn_to(self, direction):
        self.phase = "turning"
        delta = (direction - self.heading) % 4
        increments = {0: (), 1: (1,), 2: (1, 1), 3: (-1,)}[delta]
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
        import json
        costs = self.maze.flood()
        angles, position = self.data.read()
        payload = dict(width=self.maze.width, height=self.maze.height,
                       cell_size_m=self.cell_size, goal=self.maze.goal,
                       current_cell=self.cell, heading=DIRECTION_NAMES[self.heading],
                       initial_yaw=self.initial_yaw, main_yaw=self.main_yaw,
                       navigation_yaw=self.navigation_yaw(),
                       actual_gimbal_yaw=angles[3], odometry_position=position,
                       phase=self.phase, goal_reached=self.arrived, path=self.path,
                       visited=sorted(self.maze.visited),
                       flood_values=[[costs.get((x, y)) for x in range(self.maze.width)]
                                     for y in range(self.maze.height)],
                       edges=[dict(a=a, b=b, wall=wall) for (a, b), wall in sorted(self.maze.edges.items())],
                       notes="x right, y forward, zero-based. Unknown internal edges omitted; map boundaries closed. Grid pose changes only after a completed cell move.")
        temporary = self.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.output)
        if self.live_map is not None:
            self.live_map.update(force=True)

    def run(self):
        angles, _ = self.data.read()
        self.initial_yaw = angles[3]
        self.main_yaw = wrap(angles[3])
        self.target_yaw = self.navigation_yaw()
        self.target_pitch = angles[2]
        LOG.info("START cell=%s goal=%s map=%dx%d initial_yaw=%.2f",
                 self.cell, self.maze.goal, self.maze.width, self.maze.height, self.initial_yaw)
        self.settle()
        self.save()
        try:
            while self.cell != self.maze.goal:
                if self.steps >= self.max_steps:
                    raise RuntimeError("Step limit reached before goal")
                self.scan_cell()
                direction = self.maze.choose(self.cell, self.heading)
                next_cell = self.maze.neighbor(self.cell, direction)
                costs = self.maze.flood()
                options = {}
                for label, candidate in (("front", self.heading),
                                         ("right", (self.heading + 1) % 4),
                                         ("left", (self.heading - 1) % 4),
                                         ("back", (self.heading + 2) % 4)):
                    options[label] = (costs.get(self.maze.neighbor(self.cell, candidate))
                                      if self.maze.edge(self.cell, candidate) is False else "blocked/unknown")
                LOG.info("FLOOD choices=%s; choose lowest distance, priority only for equal distances", options)
                LOG.info("PLAN %s -> %s heading=%s flood=%d", self.cell, next_cell,
                         DIRECTION_NAMES[direction], self.maze.flood()[next_cell])
                self.turn_to(direction)
                # Recheck forward after turning; only fresh readings authorize a move.
                distance = self.point_and_read(self.navigation_yaw())
                if distance <= self.wall_mm:
                    raise RuntimeError("Previously open passage now blocked; robot stays at %s" % (self.cell,))
                self.phase = "moving"
                self.forward_cell(self.steps + 1)
                self.maze.observe(self.cell, direction, False)
                self.cell = next_cell
                self.steps += 1
                self.path.append(self.cell)
                self.maze.visited.add(self.cell)
                self.phase = "at_cell"
                self.save()
            self.stop_chassis()
            self.arrived = True
            self.phase = "goal_reached"
            self.save()
            LOG.info("GOAL REACHED cell=%s completed_moves=%d", self.cell, self.steps)
        except BaseException:
            self.stop_chassis()
            self.phase = "stopped"
            try:
                self.save()
            except Exception:
                LOG.exception("Could not save final map")
            raise


def pair(value):
    try:
        values = tuple(int(v.strip()) for v in value.split(","))
        if len(values) != 2:
            raise ValueError
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("Use two integers: x,y")


def input_pair(prompt, predicate):
    while True:
        try:
            values = pair(input(prompt))
            if predicate(values):
                return values
        except (ValueError, argparse.ArgumentTypeError):
            pass
        print("Invalid value. Enter x,y within the allowed range.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conn", choices=("ap", "sta"), default="ap")
    parser.add_argument("--size", type=pair, help="Map width,height; prompts if omitted")
    parser.add_argument("--goal", type=pair, help="Goal x,y, zero-based; prompts if omitted")
    parser.add_argument("--start", type=pair, default=(0, 0))
    parser.add_argument("--cell-size", type=float, default=0.60)
    parser.add_argument("--center-yaw", type=float, default=0.0)
    parser.add_argument("--center-pitch", type=float, default=0.0,
                        help="Use horizontal ToF; this flood-fill file defaults to pitch 0")
    parser.add_argument("--wall-mm", type=float, help="Wall threshold; default 0.7 x cell size in mm")
    parser.add_argument("--clearance-mm", type=float, default=120.0)
    parser.add_argument("--tof-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-graph", action="store_true", help="Disable the default live map window")
    args = parser.parse_args()
    if not math.isfinite(args.cell_size) or not .20 <= args.cell_size <= 2.0:
        parser.error("--cell-size must be 0.20..2.0 m")
    if not -45 <= args.center_yaw <= 45 or not -25 <= args.center_pitch <= 30:
        parser.error("Center yaw -45..45; pitch -25..30")
    if args.tof_index not in range(4) or args.max_steps < 0:
        parser.error("--tof-index 0..3; --max-steps >= 0")
    args.wall_mm = args.wall_mm if args.wall_mm is not None else args.cell_size * 700
    if not (0 < args.clearance_mm < args.wall_mm < args.cell_size * 1000):
        parser.error("Require 0 < clearance-mm < wall-mm < cell size in mm")
    if args.size is None:
        args.size = input_pair("Map width,height (e.g. 4,5): ", lambda v: 1 <= v[0] <= 50 and 1 <= v[1] <= 50)
    args.width, args.height = args.size
    if not 1 <= args.width <= 50 or not 1 <= args.height <= 50:
        parser.error("Map dimensions must be 1..50")
    within = lambda v: 0 <= v[0] < args.width and 0 <= v[1] < args.height
    if not within(args.start):
        parser.error("Start must be inside map")
    if args.goal is None:
        args.goal = input_pair("Goal x,y (e.g. 2,3): ", within)
    if not within(args.goal):
        parser.error("Goal must be inside map; coordinates begin at 0")
    log_path = Path(__file__).resolve().with_name("work_gimball_floodfill_run.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_path, mode="w", encoding="utf-8")], force=True)
    LOG.info("work_gimball_floodfill version %s", VERSION)
    LOG.info("Running file: %s", Path(__file__).resolve())
    LOG.info("Coordinates: start=%s facing +y, +x is right; goal=%s", args.start, args.goal)
    LOG.info("Cell %.2f m; wall threshold %.0f mm; clearance %.0f mm; ToF channel %d",
             args.cell_size, args.wall_mm, args.clearance_mm, args.tof_index)
    from robomaster import robot
    LOG.info("Python: %s; SDK: %s", sys.executable, robot.__file__)
    ep_robot = robot.Robot()
    initialized = False
    subscriptions = []
    live_map = None
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
        # Defaults record the actual recentered heading directly. Preserve
        # explicitly requested center offsets only AFTER successful recenter.
        if args.center_yaw != 0 or args.center_pitch != 0:
            LOG.info("Applying requested center offset after recenter: yaw=%.2f pitch=%.2f",
                     args.center_yaw, args.center_pitch)
            action = ep_robot.gimbal.moveto(pitch=args.center_pitch, yaw=args.center_yaw,
                                           pitch_speed=60, yaw_speed=60)
            if not action.wait_for_completed(timeout=10) or not action.has_succeeded:
                raise RuntimeError("Requested center offset failed; initial yaw was not recorded")
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
        navigator = FloodNavigator(ep_robot, data, tof, args)
        if not args.no_graph:
            live_map = LiveMap(navigator)
            navigator.live_map = live_map
        navigator.run()
        completed = True
    finally:
        if initialized:
            for stop in (lambda: ep_robot.chassis.drive_speed(x=0, y=0, z=0),
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
