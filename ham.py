import threading
import time
import logging
import sys
import heapq
import itertools
import math
import numpy as np
import pygame
from robomaster import robot
import json
import os
import cv2

# ==========================================
# 1. SETUP & FILE LOGGING CONFIGURATION
# ==========================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

file_handler = logging.FileHandler("navigation_log.txt", mode="w", encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[stream_handler, file_handler],
)


def log_event(msg):
    logging.info(msg)
    file_handler.flush()


def normalize_angle(deg):
    return (deg + 180.0) % 360.0 - 180.0


def snap_to_cardinal_yaw(raw_yaw):
    norm_yaw = normalize_angle(raw_yaw)
    snapped = round(norm_yaw / 90.0) * 90.0
    return normalize_angle(snapped)


GIMBAL_FAST_SPEED = 320
CELL_SIZE_CM = 60.0
move_target_step_m = 0.60
target_speed_mps = 0.14
DISTANCE_CALIB_FACTOR = 1.0
SHOW_CAMERA_DEBUG = True

# ถ้าหุ่นสไลด์ผิดฝั่ง ให้เปลี่ยนเป็น -1.0
STRAFE_SIGN = 1.0

speed_key = f"{target_speed_mps:.2f}"
if os.path.exists("distance_calib_params.json"):
    try:
        with open("distance_calib_params.json", "r", encoding="utf-8") as f:
            dist_calib_db = json.load(f)
            if speed_key in dist_calib_db:
                DISTANCE_CALIB_FACTOR = float(dist_calib_db[speed_key])
    except Exception:
        pass


class MiniPID:
    def __init__(self, kp, ki, kd, out_limit, i_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit if i_limit is not None else out_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = None

    def update(self, error, now=None):
        now = time.time() if now is None else now
        if self.prev_t is None:
            dt = 0.02
        else:
            dt = max(0.005, min(0.08, now - self.prev_t))
        self.prev_t = now

        self.integral = float(np.clip(self.integral + error * dt, -self.i_limit, self.i_limit))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(out, -self.out_limit, self.out_limit))


# ==========================================
# 2. NAVIGATION HUB
# ==========================================
class NavigationHub:
    def __init__(self, filename="map_config.json"):
        self.lock = threading.Lock()
        self.running = True
        self.grid_w = 4
        self.grid_h = 5
        self.grid_x = 0
        self.grid_y = 0

        self.pose = {"chassis_yaw": 0.0}
        self.yaw_offset_deg = 0.0
        self.latest_raw_pos = [None, None]
        self.latest_raw_yaw = None
        self.is_calibrated = False

        self.walls = set()
        self.target_cells = []
        self.status_msg = "Initializing..."
        self.wall_offset_px = 0.0
        self.wall_confidence = 0.0

        self.load_map_config(filename)

    def load_map_config(self, filename="map_config.json"):
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    gs = data.get("grid_size", {"w": 4, "h": 5})
                    self.grid_w = gs.get("w", 4)
                    self.grid_h = gs.get("h", 5)

                    self.target_cells = [(t["x"], t["y"]) for t in data.get("targets", [])]

                    raw_walls = data.get("walls", [])
                    for w in raw_walls:
                        c1 = tuple(w[0])
                        c2 = tuple(w[1])
                        self.walls.add((c1, c2))

                    log_event(
                        f"[MAP LOADED] โหลดสำเร็จ: ขนาด {self.grid_w}x{self.grid_h} | เป้าหมาย {len(self.target_cells)} จุด"
                    )
            except Exception as e:
                log_event(f"[MAP LOAD ERROR] อ่านไฟล์แผนที่ไม่ได้: {e}")
        else:
            log_event("[MAP WARNING] ไม่พบไฟล์ map_config.json ใช้ค่าเริ่มต้น")
            self.target_cells = [(1, 1), (2, 4)]

    def set_grid_pos(self, gx, gy):
        with self.lock:
            self.grid_x = gx
            self.grid_y = gy

    def set_status(self, msg):
        with self.lock:
            self.status_msg = msg

    def has_wall_between(self, cell1, cell2):
        return (cell1, cell2) in self.walls or (cell2, cell1) in self.walls

    def side_walls_of_move(self, from_cell, to_cell):
        """บอกว่าตอนเดินจาก from->to มีกำแพงซ้าย/ขวาของช่องหรือไม่ (ในกริด)"""
        dx = to_cell[0] - from_cell[0]
        dy = to_cell[1] - from_cell[1]
        # ซ้าย/ขวาเทียบทิศการเดิน
        if dx == 0 and dy == 1:  # เหนือ
            left_n, right_n = (-1, 0), (1, 0)
        elif dx == 0 and dy == -1:  # ใต้
            left_n, right_n = (1, 0), (-1, 0)
        elif dx == 1 and dy == 0:  # ออก
            left_n, right_n = (0, 1), (0, -1)
        elif dx == -1 and dy == 0:  # ตก
            left_n, right_n = (0, -1), (0, 1)
        else:
            return False, False

        def blocked(cell, nxy):
            nx, ny = cell[0] + nxy[0], cell[1] + nxy[1]
            if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
                return True
            return self.has_wall_between(cell, (nx, ny))

        left = blocked(from_cell, left_n) or blocked(to_cell, left_n)
        right = blocked(from_cell, right_n) or blocked(to_cell, right_n)
        return left, right

    def calibrate_origin(self):
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            with self.lock:
                if self.latest_raw_pos[0] is not None and self.latest_raw_yaw is not None:
                    self.pose["chassis_yaw"] = 0.0
                    raw_yaw_deg = self.latest_raw_yaw
                    self.yaw_offset_deg = (0.0 - raw_yaw_deg) % 360.0
                    self.is_calibrated = True
                    log_event("[CALIBRATION] Calibrated Initial Heading to North (0 deg)")
                    return True
            time.sleep(0.02)
        return False


hub = NavigationHub("map_config.json")


# ==========================================
# 3. SENSOR & ODOMETRY CALLBACKS
# ==========================================
def chassis_pos_handler(sub_info):
    if sub_info and len(sub_info) >= 2 and sub_info[0] is not None and sub_info[1] is not None:
        with hub.lock:
            hub.latest_raw_pos = [float(sub_info[0]), float(sub_info[1])]


def chassis_att_handler(sub_info):
    if sub_info and len(sub_info) >= 2 and sub_info[1] is not None:
        with hub.lock:
            hub.latest_raw_yaw = float(sub_info[1])
            hub.pose["chassis_yaw"] = normalize_angle(hub.latest_raw_yaw + hub.yaw_offset_deg)


# ==========================================
# 4. MOVEMENT CONTROL + CORRIDOR CENTERING
# ==========================================
def stop_chassis_active_brake(ep_chassis, brake_duration=0.12):
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(brake_duration)
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    except Exception:
        pass


def reset_gimbal_to_center_fast(ep_gimbal, pitch=-8):
    """เงยกล้องเกือบตรง จะได้เห็นกำแพงซ้าย-ขวา ไม่ก้มพื้นจนจับเส้นพื้น"""
    try:
        ep_gimbal.moveto(
            pitch=pitch,
            yaw=0,
            pitch_speed=GIMBAL_FAST_SPEED,
            yaw_speed=GIMBAL_FAST_SPEED,
        ).wait_for_completed(timeout=0.8)
    except Exception:
        pass


def get_pose_snapshot():
    with hub.lock:
        px = hub.latest_raw_pos[0]
        py = hub.latest_raw_pos[1]
        yaw = hub.pose["chassis_yaw"]
    return px, py, yaw


def project_progress_and_lateral(start_xy, curr_xy, heading_deg):
    """
    แยกระยะตามแนวเดิน (progress) กับระยะเบี้ยวข้าง (lateral, เมตร)
    ในเฟรมโลกของ odometry ตอนบูต
    lateral > 0 แปลว่าหุ่นเลยไปทางซ้ายของเส้นทาง
    """
    if start_xy[0] is None or curr_xy[0] is None:
        return 0.0, 0.0
    dx = curr_xy[0] - start_xy[0]
    dy = curr_xy[1] - start_xy[1]
    rad = math.radians(heading_deg)
    fwd_x, fwd_y = math.cos(rad), math.sin(rad)
    left_x, left_y = -math.sin(rad), math.cos(rad)
    progress = dx * fwd_x + dy * fwd_y
    lateral = dx * left_x + dy * left_y
    return progress, lateral


def detect_corridor_offset(img, expect_left=True, expect_right=True):
    """
    หาขอบกำแพงแนวตั้งซ้าย/ขวาจากกล้องหน้า
    คืนค่า:
      offset_norm ∈ [-1, 1]   บวก = หุ่นอยู่ขวากลางช่อง (ควรสไลด์ซ้าย)
      confidence ∈ [0, 1]
      debug_img
    """
    if img is None:
        return 0.0, 0.0, None

    h, w, _ = img.shape
    y1, y2 = int(h * 0.28), int(h * 0.72)
    x1, x2 = int(w * 0.02), int(w * 0.98)
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)

    # คงแต่เส้นแนวตั้ง ลดเส้นพื้น / เงา
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))
    vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vert_kernel)
    vert = cv2.dilate(vert, np.ones((3, 3), np.uint8), iterations=1)

    col = np.sum(vert > 0, axis=0).astype(np.float32)
    if col.size >= 9:
        col = np.convolve(col, np.ones(9) / 9.0, mode="same")

    min_hits = max(6.0, 0.10 * rh)

    def first_peak(arr, start, end, step):
        best_i, best_v = None, min_hits
        for i in range(start, end, step):
            if arr[i] > best_v:
                best_i, best_v = i, arr[i]
                # ยืนยันว่าเป็นขอบกำแพงแรกจริง ไม่ใช่ noise กลางภาพ
                if arr[i] >= min_hits * 1.4:
                    return i
        return best_i

    left_i = first_peak(col, 0, int(rw * 0.48), 1) if expect_left else None
    right_i = first_peak(col, rw - 1, int(rw * 0.52), -1) if expect_right else None

    offset_norm = 0.0
    confidence = 0.0

    # เมื่อเห็นสองฝั่ง: กลางช่อง = กลางระหว่างกำแพง
    if left_i is not None and right_i is not None and right_i > left_i + 20:
        corridor_mid = 0.5 * (left_i + right_i)
        offset_px = corridor_mid - (rw * 0.5)
        offset_norm = float(np.clip(offset_px / (rw * 0.35), -1.0, 1.0))
        confidence = 0.95
    elif left_i is not None and expect_left:
        # ถ้ามีแต่ซ้าย ให้กำแพงซ้ายอยู่ประมาณ 18% ของภาพเมื่ออยู่กลาง
        desired = rw * 0.18
        offset_norm = float(np.clip((left_i - desired) / (rw * 0.28), -1.0, 1.0))
        confidence = 0.55
    elif right_i is not None and expect_right:
        desired = rw * 0.82
        offset_norm = float(np.clip((right_i - desired) / (rw * 0.28), -1.0, 1.0))
        confidence = 0.55

    debug = None
    if SHOW_CAMERA_DEBUG:
        debug = img.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if left_i is not None:
            cv2.line(debug, (x1 + left_i, y1), (x1 + left_i, y2), (255, 80, 80), 2)
        if right_i is not None:
            cv2.line(debug, (x1 + right_i, y1), (x1 + right_i, y2), (80, 80, 255), 2)
        mid = x1 + rw // 2
        cv2.line(debug, (mid, y1), (mid, y2), (0, 255, 255), 1)
        cv2.putText(
            debug,
            f"off={offset_norm:+.2f} conf={confidence:.2f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return offset_norm, confidence, debug


def turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw_deg):
    target_yaw_deg = normalize_angle(target_yaw_deg)
    reset_gimbal_to_center_fast(ep_gimbal)
    time.sleep(0.05)

    curr_yaw = hub.pose["chassis_yaw"]
    diff = normalize_angle(target_yaw_deg - curr_yaw)
    if abs(diff) > 0.8:
        abs_diff = abs(diff)
        exec_z_speed = 45 if abs_diff <= 15.0 else (65 if abs_diff <= 45.0 else 85)
        try:
            ep_chassis.move(x=0, y=0, z=-diff, z_speed=exec_z_speed).wait_for_completed(timeout=3.2)
        except Exception:
            pass
        stop_chassis_active_brake(ep_chassis, 0.08)

    # ล็อก heading จริงจาก IMU ไม่ snap บังคับ เพราะ snap แล้วหุ่นยังเอียงอยู่แต่โปรแกรมคิดว่าตรง
    fine_pid = MiniPID(kp=2.4, ki=0.15, kd=0.35, out_limit=35.0, i_limit=8.0)
    t0 = time.time()
    while time.time() - t0 < 0.9:
        curr_yaw = hub.pose["chassis_yaw"]
        err = normalize_angle(target_yaw_deg - curr_yaw)
        if abs(err) <= 0.7:
            break
        z_cmd = fine_pid.update(err)
        ep_chassis.drive_speed(x=0, y=0, z=-z_cmd)
        time.sleep(0.02)
    stop_chassis_active_brake(ep_chassis, 0.08)

    reset_gimbal_to_center_fast(ep_gimbal)
    time.sleep(0.08)

    with hub.lock:
        if hub.latest_raw_yaw is not None:
            # เก็บ offset ให้ pose ตรงกับมุมที่สั่ง แต่ไม่ทับค่า IMU ดิบ
            hub.yaw_offset_deg = (target_yaw_deg - hub.latest_raw_yaw) % 360.0
            hub.pose["chassis_yaw"] = normalize_angle(hub.latest_raw_yaw + hub.yaw_offset_deg)
    return True


def hold_heading_and_center(
    ep_chassis,
    ep_camera,
    target_yaw,
    expect_left=True,
    expect_right=True,
    duration=0.7,
    allow_forward=False,
):
    """หยุดหน้าแล้วยืนกลางช่อง: หมุนแก้เฉพาะ heading, สไลด์แก้เฉพาะข้าง"""
    yaw_pid = MiniPID(kp=2.2, ki=0.08, kd=0.28, out_limit=28.0, i_limit=6.0)
    lat_pid = MiniPID(kp=0.16, ki=0.01, kd=0.04, out_limit=0.12, i_limit=0.2)
    offset_filt = 0.0
    t0 = time.time()
    stable = 0
    while time.time() - t0 < duration:
        img = ep_camera.read_cv2_image(strategy="newest")
        offset_norm, conf, debug = detect_corridor_offset(img, expect_left, expect_right)
        if debug is not None:
            cv2.imshow("Original Camera Feed", debug)
            cv2.waitKey(1)

        offset_filt = 0.65 * offset_filt + 0.35 * offset_norm
        yaw_err = normalize_angle(target_yaw - hub.pose["chassis_yaw"])
        z_cmd = yaw_pid.update(yaw_err)

        y_cmd = 0.0
        if conf >= 0.45 and abs(offset_filt) > 0.06:
            # offset_norm > 0 = หุ่นอยู่ขวาของกลางช่อง → สไลด์ซ้าย (y บวกใน RM)
            y_cmd = STRAFE_SIGN * lat_pid.update(offset_filt)
        else:
            lat_pid.integral *= 0.8

        x_cmd = 0.04 if allow_forward else 0.0
        ep_chassis.drive_speed(x=x_cmd, y=y_cmd, z=-z_cmd)

        if abs(yaw_err) < 1.2 and (conf < 0.45 or abs(offset_filt) < 0.08):
            stable += 1
            if stable >= 8:
                break
        else:
            stable = 0
        time.sleep(0.02)
    stop_chassis_active_brake(ep_chassis, 0.08)


def move_forward_60cm_straight(ep_chassis, ep_camera, from_cell, to_cell):
    calibrated_target_m = move_target_step_m * DISTANCE_CALIB_FACTOR
    start_x, start_y, start_yaw = get_pose_snapshot()
    if start_x is None:
        start_x, start_y = 0.0, 0.0
    target_yaw = hub.pose["chassis_yaw"]
    start_xy = (start_x, start_y)

    expect_left, expect_right = hub.side_walls_of_move(from_cell, to_cell)

    yaw_pid = MiniPID(kp=1.8, ki=0.06, kd=0.22, out_limit=22.0, i_limit=5.0)
    vision_pid = MiniPID(kp=0.14, ki=0.008, kd=0.03, out_limit=0.10, i_limit=0.15)
    odom_pid = MiniPID(kp=0.55, ki=0.02, kd=0.08, out_limit=0.08, i_limit=0.12)

    offset_filt = 0.0
    max_timeout = (calibrated_target_m / max(0.08, target_speed_mps)) + 2.5
    start_time = time.time()
    loop_count = 0
    progress = 0.0

    log_event(
        f"[STEP START] {from_cell} -> {to_cell} | Dist {calibrated_target_m:.2f}m | walls L/R={expect_left}/{expect_right}"
    )

    # จัดกลางก่อนออกตัว กันสะสมความเอียงจากรอบที่แล้ว
    hold_heading_and_center(
        ep_chassis,
        ep_camera,
        target_yaw,
        expect_left,
        expect_right,
        duration=0.55,
    )

    while progress < calibrated_target_m and (time.time() - start_time < max_timeout):
        curr_x, curr_y, curr_yaw = get_pose_snapshot()
        if curr_x is None:
            curr_x, curr_y = start_x, start_y

        progress, odom_lateral = project_progress_and_lateral(
            start_xy, (curr_x, curr_y), target_yaw
        )

        yaw_err = normalize_angle(target_yaw - curr_yaw)
        z_cmd = yaw_pid.update(yaw_err)

        img = ep_camera.read_cv2_image(strategy="newest")
        offset_norm, conf, debug = detect_corridor_offset(img, expect_left, expect_right)
        if debug is not None:
            edge_vis = debug
            cv2.imshow("Original Camera Feed", edge_vis)
            cv2.waitKey(1)

        offset_filt = 0.72 * offset_filt + 0.28 * offset_norm
        with hub.lock:
            hub.wall_offset_px = offset_filt
            hub.wall_confidence = conf

        # --- แกน Y = สไลด์ให้อยู่กลางช่อง, ห้ามไปยุ่งแกน Z ---
        y_cmd = 0.0
        remain = calibrated_target_m - progress

        # odometry กันเดินเฉในระยะสั้น (ช่วยแม้กล้องไม่เจอกำแพง)
        if abs(odom_lateral) > 0.012 and remain > 0.08:
            y_cmd += STRAFE_SIGN * odom_pid.update(-odom_lateral)
        else:
            odom_pid.integral *= 0.85

        # กล้องแม่นกว่าเมื่อเห็นกำแพงชัด
        if conf >= 0.50 and abs(offset_filt) > 0.05:
            y_cmd += STRAFE_SIGN * vision_pid.update(offset_filt)
        else:
            vision_pid.integral *= 0.85

        y_cmd = float(np.clip(y_cmd, -0.12, 0.12))

        # ชะลอปลายทาง ไม่ให้เบรกลั่นแล้วเฉ
        if remain < 0.12:
            x_cmd = 0.06
        elif remain < 0.22:
            x_cmd = 0.09
        else:
            x_cmd = target_speed_mps

        # ถ้าเอียงมาก ให้ชะลอแล้วสไลด์ก่อน อย่าดันชนกำแพง
        if abs(offset_filt) > 0.45 and conf >= 0.55:
            x_cmd = min(x_cmd, 0.07)

        ep_chassis.drive_speed(x=x_cmd, y=y_cmd, z=-z_cmd)

        loop_count += 1
        if loop_count % 40 == 0:
            log_event(
                f"[TELEMETRY] {from_cell}->{to_cell} | prog={progress:.3f}m "
                f"yaw={curr_yaw:.1f} ye={yaw_err:+.1f} "
                f"odom_lat={odom_lateral:+.3f} vis={offset_filt:+.2f} c={conf:.2f} y={y_cmd:+.3f}"
            )
        time.sleep(0.012)

    stop_chassis_active_brake(ep_chassis, brake_duration=0.12)

    # ยืนจัดกลางอีกครั้งท้ายช่อง กัน error สะสมเข้าช่องถัดไป
    hold_heading_and_center(
        ep_chassis,
        ep_camera,
        target_yaw,
        expect_left,
        expect_right,
        duration=0.65,
    )

    end_x, end_y, end_yaw = get_pose_snapshot()
    final_prog, final_lat = project_progress_and_lateral(start_xy, (end_x, end_y), target_yaw)
    log_event(
        f"[STEP COMPLETE] {to_cell} | prog={final_prog:.3f}m lat={final_lat:+.3f}m yaw={end_yaw:.1f}"
    )
    return True


# ==========================================
# 5. PATH PLANNING (A* & GLOBAL TSP)
# ==========================================
def a_star_search(start, target, grid_w, grid_h):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                if not hub.has_wall_between(current, (nx, ny)):
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f_score[(nx, ny)] = tentative_g + heuristic((nx, ny), target)
                        heapq.heappush(open_set, (f_score[(nx, ny)], (nx, ny)))
    return None


def calculate_optimal_route(start_pos, target_cells):
    best_route = None
    min_len = float("inf")

    for perm in itertools.permutations(target_cells):
        current = start_pos
        total_len = 0
        valid = True
        for dest in perm:
            path = a_star_search(current, dest, hub.grid_w, hub.grid_h)
            if path is None:
                valid = False
                break
            total_len += len(path)
            current = dest
        if valid and total_len < min_len:
            min_len = total_len
            best_route = perm

    return best_route


# ==========================================
# 6. PYGAME REAL-TIME MAP GUI THREAD
# ==========================================
def run_pygame_gui():
    pygame.init()
    pygame.font.init()

    font_large = pygame.font.SysFont("Tahoma", 18, bold=True)
    font_small = pygame.font.SysFont("Tahoma", 14)

    CELL_PIXELS = 60
    MARGIN = 20
    PANEL_WIDTH = 260
    WALL_THICKNESS = 4

    map_px_w = hub.grid_w * CELL_PIXELS
    map_px_h = hub.grid_h * CELL_PIXELS
    win_w = map_px_w + (MARGIN * 3) + PANEL_WIDTH
    win_h = max(map_px_h + (MARGIN * 2), 400)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("RoboMaster EP - Target Navigation Map")
    clock = pygame.time.Clock()
    while hub.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                hub.running = False
                return
        screen.fill((25, 28, 36))
        map_origin_x = MARGIN
        map_origin_y = MARGIN
        pygame.draw.rect(screen, (40, 44, 52), (MARGIN, MARGIN, map_px_w, map_px_h))
        for gy in range(hub.grid_h):
            for gx in range(hub.grid_w):
                px = map_origin_x + (gx * CELL_PIXELS)
                py = map_origin_y + ((hub.grid_h - 1 - gy) * CELL_PIXELS)
                cell_rect = (px + 1, py + 1, CELL_PIXELS - 2, CELL_PIXELS - 2)

                if (gx, gy) in hub.target_cells:
                    color = (241, 196, 15)
                else:
                    color = (70, 75, 85)

                pygame.draw.rect(screen, color, cell_rect)
                lbl = font_small.render(f"{gx},{gy}", True, (200, 200, 200))
                screen.blit(lbl, (px + 6, py + 6))
        for gx in range(hub.grid_w + 1):
            x = map_origin_x + (gx * CELL_PIXELS)
            pygame.draw.line(screen, (60, 64, 72), (x, MARGIN), (x, MARGIN + map_px_h))
        for gy in range(hub.grid_h + 1):
            y = map_origin_y + (gy * CELL_PIXELS)
            pygame.draw.line(screen, (60, 64, 72), (MARGIN, y), (MARGIN + map_px_w, y))
        for (c1, c2) in hub.walls:
            x1, y1 = c1
            x2, y2 = c2
            if x1 == x2:
                max_y = max(y1, y2)
                py = map_origin_y + ((hub.grid_h - max_y) * CELL_PIXELS)
                px1 = map_origin_x + (x1 * CELL_PIXELS)
                px2 = px1 + CELL_PIXELS
                pygame.draw.line(screen, (231, 76, 60), (px1, py), (px2, py), WALL_THICKNESS)
            else:
                max_x = max(x1, x2)
                px = map_origin_x + (max_x * CELL_PIXELS)
                py1 = map_origin_y + ((hub.grid_h - 1 - y1) * CELL_PIXELS)
                py2 = py1 + CELL_PIXELS
                pygame.draw.line(screen, (231, 76, 60), (px, py1), (px, py2), WALL_THICKNESS)
        with hub.lock:
            rx_px = map_origin_x + ((hub.grid_x + 0.5) * CELL_PIXELS)
            ry_px = map_origin_y + ((hub.grid_h - 1 - hub.grid_y) * CELL_PIXELS) + CELL_PIXELS // 2
            current_yaw = hub.pose["chassis_yaw"]
            status_text = hub.status_msg
        pygame.draw.circle(screen, (46, 204, 113), (int(rx_px), int(ry_px)), 14)
        panel_x = MARGIN + map_px_w + MARGIN
        pygame.draw.rect(screen, (33, 37, 43), (panel_x, MARGIN, PANEL_WIDTH, win_h - (MARGIN * 2)), border_radius=8)
        py_y = MARGIN + 15
        screen.blit(font_large.render("MISSION STATUS", True, (241, 196, 15)), (panel_x + 15, py_y))
        py_y += 35
        lines = [
            f"Status: {status_text}",
            f"Grid X, Y: ({hub.grid_x}, {hub.grid_y})",
            f"Heading: {current_yaw:.1f} deg",
            f"Targets: {len(hub.target_cells)} points",
        ]
        for line in lines:
            screen.blit(font_small.render(line, True, (236, 240, 241)), (panel_x + 15, py_y))
            py_y += 25
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


# ==========================================
# 7. MAIN NAVIGATION WORKER
# ==========================================
def navigation_main_worker(ep_chassis, ep_gimbal, ep_camera):
    time.sleep(1.0)
    if not hub.target_cells:
        log_event("❌ ไม่มีเป้าหมายให้เดินทางไป")
        hub.set_status("No Targets!")
        return
    start_pos = (hub.grid_x, hub.grid_y)
    route_order = calculate_optimal_route(start_pos, hub.target_cells)

    if not route_order:
        log_event("❌ ไม่สามารถคำนวณเส้นทางไปเก็บเป้าหมายได้")
        hub.set_status("Route Error!")
        return
    log_event(f"🚀 ลำดับการเดินทางสั้นที่สุด: {route_order}")
    dir_offsets = {0: (0, 1), 90: (1, 0), 180: (0, -1), -180: (0, -1), -90: (-1, 0)}
    current_pos = start_pos
    for target_cell in route_order:
        path = a_star_search(current_pos, target_cell, hub.grid_w, hub.grid_h)
        if path and path[0] == current_pos:
            path.pop(0)

        for next_cell in path:
            dx = next_cell[0] - current_pos[0]
            dy = next_cell[1] - current_pos[1]
            target_yaw = next((angle for angle, offset in dir_offsets.items() if offset == (dx, dy)), 0.0)

            hub.set_status(f"Turning to {target_yaw}°")
            log_event(f"[ACTION] Turning from {current_pos} to heading {target_yaw}°")
            turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw)

            hub.set_status(f"Moving to {next_cell}")
            move_forward_60cm_straight(ep_chassis, ep_camera, current_pos, next_cell)

            hub.set_grid_pos(next_cell[0], next_cell[1])
            current_pos = next_cell
            time.sleep(0.05)
        hub.set_status(f"At Target {target_cell}")
        log_event(f"🎯 [ARRIVED] หุ่นถึงพิกัด Grid {target_cell} แล้ว!")
        time.sleep(1.5)
    log_event("=== 🎉 ภารกิจการเดินทางสำเร็จลุล่วงทุกจุด! ===")
    hub.set_status("Mission Complete!")
    time.sleep(2.0)
    hub.running = False
    cv2.destroyAllWindows()


# ==========================================
# 8. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_camera = ep_robot.camera

    ep_chassis.sub_position(freq=20, callback=chassis_pos_handler)
    ep_chassis.sub_attitude(freq=20, callback=chassis_att_handler)

    reset_gimbal_to_center_fast(ep_gimbal)
    ep_camera.start_video_stream(display=False)
    time.sleep(1.0)

    log_event("Calibrating Initial Heading...")
    if hub.calibrate_origin():
        log_event("Calibration Complete!")
    else:
        log_event("Calibration timeout, proceeding with default yaw.")
    gui_thread = threading.Thread(target=run_pygame_gui, daemon=True)
    gui_thread.start()

    log_event("Starting Target Navigation Module with Real-time Map & Camera...")
    try:
        navigation_main_worker(ep_chassis, ep_gimbal, ep_camera)
    except KeyboardInterrupt:
        log_event("🛑 [KEYBOARD INTERRUPT] Stopped safely by user. Saving logs...")
    finally:
        hub.running = False
        try:
            ep_chassis.unsub_position()
            ep_chassis.unsub_attitude()
            ep_camera.stop_video_stream()
            cv2.destroyAllWindows()
            ep_robot.close()
        except Exception:
            pass
        log_event("Robot connection closed cleanly. Log saved to 'navigation_log.txt'.")