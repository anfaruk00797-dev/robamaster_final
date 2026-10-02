import csv
from collections import Counter, deque
import heapq
import json
import logging
import math
import os
import queue
import sys
import threading
import time
import traceback
import cv2
import numpy as np
import pygame
from robomaster import robot, blaster

# ==========================================
# 1. LOGGING & SYSTEM CONFIG
# ==========================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_FILENAME = "exploration_log.txt"
file_handler = logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[stream_handler, file_handler],
)

recent_logs = deque(maxlen=6)

def log_event(msg):
    logging.info(msg)
    recent_logs.append(msg)
    file_handler.flush()

TARGET_IMG_DIR = "detected_targets"
if not os.path.exists(TARGET_IMG_DIR):
    os.makedirs(TARGET_IMG_DIR, exist_ok=True)

CSV_FILE = "exploration_telemetry.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Step", "Grid_X", "Grid_Y", "Heading_Deg",
            "ToF_Ahead_cm", "ToF_Right_cm", "ToF_Back_cm", "ToF_Left_cm",
            "Wall_N", "Wall_E", "Wall_S", "Wall_W",
        ])

def log_telemetry_csv(step, gx, gy, heading, tof_dict, wall_dict):
    try:
        timestamp = time.strftime("%H:%M:%S")
        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, step, gx, gy, f"{heading:.1f}",
                f"{tof_dict.get('AHEAD', -1):.1f}", f"{tof_dict.get('RIGHT', -1):.1f}",
                f"{tof_dict.get('BACK', -1):.1f}", f"{tof_dict.get('LEFT', -1):.1f}",
                wall_dict.get("N", False), wall_dict.get("E", False),
                wall_dict.get("S", False), wall_dict.get("W", False),
            ])
            f.flush()
    except Exception as e:
        log_event(f"[CSV ERROR] Telemetry logging failed: {e}")

# ==========================================
# 2. PHYSICAL CONSTANTS & ASSET LOADERS
# ==========================================
CELL_SIZE_CM = 60.0
WALL_THRESHOLD_CM = 52.0
SAFE_MIN_TOF_CM = 16.0
MAX_FIRING_DIST_CM = 80.0
CAMERA_FOCAL_LENGTH_PX = 650.0
GIMBAL_FAST_SPEED = 320
MIN_PIXEL_FILL_RATIO = 35.0

# ระยะอุดมคติกำแพงข้าง
IDEAL_SIDE_WALL_DIST_CM = 18.0
MAX_SHARP_WALL_DETECT_CM = 26.0

sdk_lock = threading.Lock()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_asset_path(filename):
    p_script = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(p_script):
        return p_script
    return filename

# --- Load ToF Calibration ---
TOF_A, TOF_B, TOF_C = 0.0, 1.0, 0.0
TOF_SLOPE, TOF_INTERCEPT = 1.0, 0.0
MAX_CALIB_RANGE_CM = 180.0

tof_json_path = get_asset_path("tof_calib_params.json")
if os.path.exists(tof_json_path):
    try:
        with open(tof_json_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            TOF_A = cdata.get("a", 0.0)
            TOF_B = cdata.get("b", 1.0)
            TOF_C = cdata.get("c", 0.0)
            TOF_SLOPE = cdata.get("slope", TOF_B)
            TOF_INTERCEPT = cdata.get("intercept", TOF_C)
            MAX_CALIB_RANGE_CM = cdata.get("max_calib_range_cm", 180.0)
            log_event(f"[TOF CALIB LOADED] Loaded '{tof_json_path}' (Max Range: {MAX_CALIB_RANGE_CM} cm)")
    except Exception as e:
        log_event(f"[TOF CALIB ERROR] {e}")

def apply_tof_calibration(raw_dist_cm):
    if raw_dist_cm <= 0 or raw_dist_cm >= 400.0:
        return 999.0
    if raw_dist_cm <= MAX_CALIB_RANGE_CM:
        calib = (TOF_A * (raw_dist_cm ** 2)) + (TOF_B * raw_dist_cm) + TOF_C
    else:
        calib = (TOF_SLOPE * raw_dist_cm) + TOF_INTERCEPT
    return round(max(0.0, calib), 2)

# --- Load Sharp Poly Calibration ---
SHARP_CONFIG = None
sharp_json_path = get_asset_path("calibration_sharp_poly.json")

if os.path.exists(sharp_json_path):
    try:
        with open(sharp_json_path, "r", encoding="utf-8") as f:
            SHARP_CONFIG = json.load(f)
            log_event(f"[SHARP CALIB LOADED] Loaded '{sharp_json_path}' successfully")
    except Exception as e:
        log_event(f"[SHARP CALIB ERROR] Failed to load JSON: {e}")
else:
    log_event(f"⚠️ [SHARP CALIB MISSING] Cannot find '{sharp_json_path}'!")

def get_sharp_distance_cm(sensor_adaptor, s_id, s_port, sensor_key):
    if sensor_adaptor is None or SHARP_CONFIG is None:
        return 999.0
    try:
        with sdk_lock:
            adc = sensor_adaptor.get_adc(id=s_id, port=s_port)
        if adc is None or adc <= 0:
            return 999.0
        
        cfg = SHARP_CONFIG.get(sensor_key, {})
        a = cfg.get("a", 0.0)
        b = cfg.get("b", 0.0)
        c = cfg.get("c", 0.0)
        min_adc = cfg.get("min_adc", 100)
        max_adc = cfg.get("max_adc", 650)
        
        adc_clamped = max(min_adc, min(adc, max_adc))
        dist_cm = (a * (adc_clamped ** 2)) + (b * adc_clamped) + c
        return round(max(0.0, dist_cm), 2)
    except Exception:
        return 999.0

# --- Load ROI Mask ---
roi_mask_path = get_asset_path("roi_mask.png")
ROI_MASK_IMG = None
if os.path.exists(roi_mask_path):
    ROI_MASK_IMG = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    log_event(f"[ROI MASK] Loaded '{roi_mask_path}' successfully")

TOF_OFFSET_HORIZ_CM = 7.5
MAX_GROUND_DIST_CM = 120.0

ALL_COLORS = ["red", "yellow", "green", "blue"]
VALID_SHAPES = ["RECT_V", "RECT_H", "SQUARE", "CIRCLE"]

COLOR_MAIN_STAR = (156, 39, 176)
COLOR_SUB_STAR = (3, 169, 244)

DEFAULT_HSV_RANGES = {
    "red": [
        {"lower": [0, 100, 80], "upper": [10, 255, 255]},
        {"lower": [165, 100, 80], "upper": [180, 255, 255]},
    ],
    "yellow": [{"lower": [15, 100, 100], "upper": [35, 255, 255]}],
    "green": [{"lower": [38, 80, 50], "upper": [75, 255, 255]}],
    "blue": [{"lower": [95, 90, 50], "upper": [135, 255, 255]}],
}

def load_hsv_ranges(json_filename="hsv_config.json"):
    data = DEFAULT_HSV_RANGES
    json_path = get_asset_path(json_filename)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception: pass
    return {
        c: [(np.array(r["lower"]), np.array(r["upper"])) for r in ranges]
        for c, ranges in data.items()
    }

COLOR_RANGES = load_hsv_ranges("hsv_config.json")

def normalize_angle(deg):
    return (deg + 180.0) % 360.0 - 180.0

def snap_to_cardinal_yaw(raw_yaw):
    norm_yaw = normalize_angle(raw_yaw)
    return normalize_angle(round(norm_yaw / 90.0) * 90.0)

def calculate_ground_distance(tof_raw_cm):
    return tof_raw_cm + TOF_OFFSET_HORIZ_CM

def get_screen_quadrant(cx, cy, frame_w, frame_h):
    half_w, half_h = frame_w / 2.0, frame_h / 2.0
    col = "LEFT" if cx < half_w - 40 else ("RIGHT" if cx > half_w + 40 else "CENTER-X")
    row = "TOP" if cy < half_h - 40 else ("BOTTOM" if cy > half_h + 40 else "CENTER-Y")
    if col.startswith("CENTER") and row.startswith("CENTER"):
        return "CENTER"
    return f"{row}-{col}"

# ==========================================
# 3. SETUP USER MENU
# ==========================================
def setup_user_menu():
    print("\n==================================================")
    print("ROBOMASTER SLAM ROUND 1 - SETUP MENU")
    print("==================================================")
    grid_w = int(input("Field Width (Grid X count): "))
    grid_h = int(input("Field Height (Grid Y count): "))
    start_x = int(input("Start X (grid index 0-based): "))
    start_y = int(input("Start Y (grid index 0-based): "))
    init_dir = input("Initial Direction (N, E, S, W): ").strip().upper()

    print("\nTarget Color:")
    print("  1. Red | 2. Yellow | 3. Green | 4. Blue | 5. ALL_COLORS")
    c_choice = input("Select color [1-5] (Default=1 Red): ").strip()
    color_map = {"1": "red", "2": "yellow", "3": "green", "4": "blue", "5": "ALL_COLORS"}
    target_color = color_map.get(c_choice, "red")

    print("\nTarget Shape:")
    print("  1. RECT_V (6x9 cm) | 2. RECT_H (9x6 cm) | 3. SQUARE (7x7 cm) | 4. CIRCLE (7 cm) | 5. ALL_SHAPES")
    s_choice = input("Select shape [1-5] (Default=5 ALL_SHAPES): ").strip()
    shape_map = {"1": "RECT_V", "2": "RECT_H", "3": "SQUARE", "4": "CIRCLE", "5": "ALL_SHAPES"}
    target_shape = shape_map.get(s_choice, "ALL_SHAPES")

    burst_count_str = input("Burst count per target (Default=2): ").strip()
    try:
        burst_count = max(1, min(int(burst_count_str), 5))
    except ValueError:
        burst_count = 2

    return grid_w, grid_h, start_x, start_y, init_dir, target_color, target_shape, burst_count

GRID_W, GRID_H, START_X, START_Y, INIT_DIR, TARGET_COLOR, TARGET_SHAPE, BURST_FIRE_COUNT = setup_user_menu()

DIR_MAP = {"N": 0.0, "E": 90.0, "S": -180.0, "W": -90.0}
INITIAL_YAW_DEG = DIR_MAP.get(INIT_DIR, 0.0)

target_speed_mps = 0.24
move_target_step_m = 0.60
DISTANCE_CALIB_FACTOR = 1.0

ep_blaster_ref = None
ep_robot_ref = None

def play_fire_sound():
    global ep_robot_ref
    if ep_robot_ref is not None:
        try: ep_robot_ref.play_sound(robot.SOUND_ID_ATTACK)
        except Exception: pass

def fire_target_burst(count=2):
    global ep_blaster_ref
    if ep_blaster_ref is not None:
        try:
            log_event(f"[BLASTER FIRE] Firing {count} rounds burst")
            with sdk_lock:
                ep_blaster_ref.fire(fire_type=blaster.WATER_FIRE, times=count)
            time.sleep(0.15 * count)
        except Exception as e:
            log_event(f"[FIRE FAILED] Error during fire: {e}")

# ==========================================
# 4. SHARED HUB & SLAM MAPPING
# ==========================================
class SharedHub:

    def __init__(self, w, h, sx, sy):
        self.lock = threading.Lock()
        self.w = w
        self.h = h
        self.visited = np.zeros((h, w), dtype=int)
        self.has_target_grid = np.zeros((h, w), dtype=bool)

        self.N_wall_logodds = np.full((h, w), -2.0)
        self.E_wall_logodds = np.full((h, w), -2.0)
        self.S_outer_wall_logodds = np.full(w, 4.0)
        self.W_outer_wall_logodds = np.full(h, 4.0)
        self.N_wall_logodds[h - 1, :] = 4.0
        self.E_wall_logodds[:, w - 1] = 4.0

        self.N_wall_locked = np.zeros((h, w), dtype=bool)
        self.E_wall_locked = np.zeros((h, w), dtype=bool)

        self.grid_x = sx
        self.grid_y = sy
        self.pose = {
            "x_cm": (sx + 0.5) * CELL_SIZE_CM,
            "y_cm": (sy + 0.5) * CELL_SIZE_CM,
            "chassis_yaw": INITIAL_YAW_DEG,
        }

        self.latest_raw_pos = [None, None]
        self.latest_raw_yaw = None
        self.yaw_offset_deg = 0.0
        self.is_calibrated = False

        self.tof_raw = 0.0
        self.tof_filtered = 0.0
        self.tof_last_update_time = 0.0
        self.tof_buffer = deque(maxlen=5)

        self.sharp_left_dist = 999.0
        self.sharp_right_dist = 999.0
        self.sharp_alpha = 0.65

        self.gimbal_yaw = 0.0
        self.latest_frame = None
        self.active_detections = []
        self.targets_found = []
        self.target_id_counter = 1

        self.robot_path_history = [(sx, sy)]

        self.is_aligning = False
        self.is_scanning = False
        self.is_rotating = False
        self.emergency_stop = False
        self.running = True
        self.status_msg = "Initializing..."
        self.visited[sy][sx] = 1

    def get_frame_safe(self, retries=3, delay=0.02):
        for _ in range(retries):
            with self.lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            time.sleep(delay)
        return None

    def append_tof(self, val):
        with self.lock:
            self.tof_raw = val
            self.tof_last_update_time = time.time()
            if val < 800.0:
                self.tof_buffer.append(val)
            valid_samples = [v for v in self.tof_buffer if v < 800.0]
            self.tof_filtered = float(np.median(valid_samples)) if valid_samples else val

    def clear_tof_buffer(self):
        with self.lock:
            self.tof_buffer.clear()

    def update_sharp_distances(self, left_cm, right_cm):
        with self.lock:
            if left_cm < 900.0:
                if self.sharp_left_dist >= 900.0:
                    self.sharp_left_dist = left_cm
                else:
                    self.sharp_left_dist = round((self.sharp_alpha * left_cm) + ((1.0 - self.sharp_alpha) * self.sharp_left_dist), 2)
            
            if right_cm < 900.0:
                if self.sharp_right_dist >= 900.0:
                    self.sharp_right_dist = right_cm
                else:
                    self.sharp_right_dist = round((self.sharp_alpha * right_cm) + ((1.0 - self.sharp_alpha) * self.sharp_right_dist), 2)

    def set_aligning(self, state: bool):
        with self.lock: self.is_aligning = state

    def set_scanning(self, state: bool):
        with self.lock: self.is_scanning = state

    def get_robot_world_pos(self):
        return (self.grid_x + 0.5) * CELL_SIZE_CM, (self.grid_y + 0.5) * CELL_SIZE_CM

    def set_active_detections(self, detections_list):
        with self.lock:
            self.active_detections = detections_list.copy()

    def lock_passage(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        with self.lock:
            if x2 == x1 and y2 == y1 + 1:
                self.N_wall_logodds[y1, x1] = -4.0
                self.N_wall_locked[y1, x1] = True
            elif x2 == x1 + 1 and y2 == y1:
                self.E_wall_logodds[y1, x1] = -4.0
                self.E_wall_locked[y1, x1] = True
            elif x2 == x1 and y2 == y1 - 1 and y2 >= 0:
                self.N_wall_logodds[y2, x2] = -4.0
                self.N_wall_locked[y2, x2] = True
            elif x2 == x1 - 1 and y2 == y1 and x2 >= 0:
                self.E_wall_logodds[y2, x2] = -4.0
                self.E_wall_locked[y2, x2] = True

    def update_wall_prob(self, gx, gy, direction, is_wall, is_direct_adjacent=True):
        with self.lock:
            if not (0 <= gx < self.w and 0 <= gy < self.h): return
            
            L_OCC = 2.5 if is_direct_adjacent else 1.2
            L_FREE = -1.8 if is_direct_adjacent else -0.8

            if direction == "N":
                if self.N_wall_locked[gy, gx]: return
                self.N_wall_logodds[gy, gx] = np.clip(self.N_wall_logodds[gy, gx] + (L_OCC if is_wall else L_FREE), -4.0, 4.0)
            elif direction == "E":
                if self.E_wall_locked[gy, gx]: return
                self.E_wall_logodds[gy, gx] = np.clip(self.E_wall_logodds[gy, gx] + (L_OCC if is_wall else L_FREE), -4.0, 4.0)
            elif direction == "S":
                if gy > 0:
                    if self.N_wall_locked[gy - 1, gx]: return
                    self.N_wall_logodds[gy - 1, gx] = np.clip(self.N_wall_logodds[gy - 1, gx] + (L_OCC if is_wall else L_FREE), -4.0, 4.0)
                else:
                    self.S_outer_wall_logodds[gx] = np.clip(self.S_outer_wall_logodds[gx] + (L_OCC if is_wall else -4.0), -4.0, 4.0)
            elif direction == "W":
                if gx > 0:
                    if self.E_wall_locked[gy, gx - 1]: return
                    self.E_wall_logodds[gy, gx - 1] = np.clip(self.E_wall_logodds[gy, gx - 1] + (L_OCC if is_wall else L_FREE), -4.0, 4.0)
                else:
                    self.W_outer_wall_logodds[gy] = np.clip(self.W_outer_wall_logodds[gy] + (L_OCC if is_wall else -4.0), -4.0, 4.0)

    def has_wall_between(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        if x2 == x1 and y2 == y1 + 1: return self.N_wall_logodds[y1, x1] > 0.0 if y1 < self.h else True
        elif x2 == x1 + 1 and y2 == y1: return self.E_wall_logodds[y1, x1] > 0.0 if x1 < self.w else True
        elif x2 == x1 and y2 == y1 - 1: return self.N_wall_logodds[y2, x2] > 0.0 if y2 >= 0 else True
        elif x2 == x1 - 1 and y2 == y1: return self.E_wall_logodds[y2, x2] > 0.0 if x2 >= 0 else True
        return True

    def register_target_spatial(self, abs_yaw_deg, tof_dist_cm, color, shape, is_main_target=True):
        with self.lock:
            ground_dist_cm = calculate_ground_distance(tof_dist_cm)
            rx_cm, ry_cm = self.get_robot_world_pos()
            rad = math.radians(abs_yaw_deg)

            target_world_x = rx_cm + (ground_dist_cm * math.sin(rad))
            target_world_y = ry_cm + (ground_dist_cm * math.cos(rad))

            for t in self.targets_found:
                prev_x = t["world_pos_cm"]["x"]
                prev_y = t["world_pos_cm"]["y"]
                dist = math.sqrt((target_world_x - prev_x) ** 2 + (target_world_y - prev_y) ** 2)
                if dist <= 20.0 and t["color"] == color:
                    log_event(f"[TARGET REJECTED] Duplicate target detected at distance {dist:.1f} cm")
                    return False

            norm_dir = normalize_angle(abs_yaw_deg)
            cardinal_dir = "N"
            target_gx, target_gy = self.grid_x, self.grid_y

            if -45.0 <= norm_dir <= 45.0: target_gy += 1; cardinal_dir = "N"
            elif 45.0 < norm_dir <= 135.0: target_gx += 1; cardinal_dir = "E"
            elif norm_dir > 135.0 or norm_dir < -135.0: target_gy -= 1; cardinal_dir = "S"
            elif -135.0 <= norm_dir < -45.0: target_gx -= 1; cardinal_dir = "W"

            target_gx = int(np.clip(target_gx, 0, self.w - 1))
            target_gy = int(np.clip(target_gy, 0, self.h - 1))

            if self.has_wall_between((self.grid_x, self.grid_y), (target_gx, target_gy)):
                log_event(f"[TARGET BLOCKED] Wall between ({self.grid_x},{self.grid_y}) and ({target_gx},{target_gy})")
                return False

            if ground_dist_cm <= MAX_GROUND_DIST_CM:
                self.has_target_grid[target_gy, target_gx] = True
                log_event(f"[TARGET MARKED] Target marked at Grid({target_gx}, {target_gy})")

            record = {
                "target_id": self.target_id_counter,
                "color": color,
                "shape": shape,
                "is_main_target": is_main_target,
                "target_grid": {"x": target_gx, "y": target_gy},
                "standing_grid": {"x": self.grid_x, "y": self.grid_y},
                "world_pos_cm": {"x": round(target_world_x, 1), "y": round(target_world_y, 1)},
                "firing_direction": cardinal_dir,
                "firing_yaw_deg": round(norm_dir, 1),
                "is_knocked_down": False,
            }
            self.targets_found.append(record)
            log_event(f"[TARGET PASSED] ID#{self.target_id_counter}: {color.upper()} ({shape}) at World ({target_world_x:.1f}, {target_world_y:.1f})")
            self.target_id_counter += 1
            return True

    def export_map_data(self):
        data = {
            "grid_size": {"w": self.w, "h": self.h},
            "start_grid": {"x": START_X, "y": START_Y},
            "end_grid": {"x": self.grid_x, "y": self.grid_y},
            "targets_found": self.targets_found,
            "visited_matrix": self.visited.tolist(),
            "robot_path_history": self.robot_path_history,
            "N_wall_logodds": self.N_wall_logodds.tolist(),
            "E_wall_logodds": self.E_wall_logodds.tolist(),
            "S_outer_wall_logodds": self.S_outer_wall_logodds.tolist(),
            "W_outer_wall_logodds": self.W_outer_wall_logodds.tolist(),
            "N_wall_matrix": (self.N_wall_logodds > 0.0).tolist(),
            "E_wall_matrix": (self.E_wall_logodds > 0.0).tolist(),
        }
        with open("Map_Data_Round1.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log_event("[EXPORT SUCCESS] Map_Data_Round1.json exported successfully")
        self.export_round2_plan()

    def export_round2_plan(self):
        plan = {
            "start_grid": {"x": START_X, "y": START_Y},
            "targets_to_shoot": []
        }
        for t in self.targets_found:
            if t.get("is_main_target", True):
                plan["targets_to_shoot"].append({
                    "target_id": t["target_id"],
                    "color": t["color"],
                    "shape": t["shape"],
                    "standing_grid": t["standing_grid"],
                    "firing_yaw_deg": t["firing_yaw_deg"],
                    "firing_direction": t["firing_direction"]
                })
        
        with open("Round2_Target_Plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        log_event("[EXPORT SUCCESS] Round2_Target_Plan.json exported for Round 2 optimization")

    def set_grid_pos(self, gx, gy):
        with self.lock:
            self.grid_x = gx
            self.grid_y = gy
            self.pose["x_cm"] = (gx + 0.5) * CELL_SIZE_CM
            self.pose["y_cm"] = (gy + 0.5) * CELL_SIZE_CM
            if 0 <= gy < self.h and 0 <= gx < self.w:
                self.visited[gy][gx] = 1
            if not self.robot_path_history or self.robot_path_history[-1] != (gx, gy):
                self.robot_path_history.append((gx, gy))

    def set_status_msg(self, msg):
        with self.lock: self.status_msg = msg

    def set_raw_data(self, pos_xy, yaw_deg):
        with self.lock:
            if pos_xy[0] is not None: self.latest_raw_pos[0] = pos_xy[0]
            if pos_xy[1] is not None: self.latest_raw_pos[1] = pos_xy[1]
            if yaw_deg is not None:
                self.latest_raw_yaw = yaw_deg
                self.pose["chassis_yaw"] = normalize_angle(yaw_deg + self.yaw_offset_deg)

    def calibrate_origin(self):
        timeout = 5.0
        start_t = time.time()
        while time.time() - start_t < timeout:
            with self.lock:
                if self.latest_raw_pos[0] is not None and self.latest_raw_yaw is not None:
                    self.pose["chassis_yaw"] = INITIAL_YAW_DEG
                    self.yaw_offset_deg = (INITIAL_YAW_DEG - self.latest_raw_yaw) % 360.0
                    self.is_calibrated = True
                    return True
            time.sleep(0.02)
        return False

    def get_state(self):
        with self.lock:
            return {
                "visited": self.visited.copy(),
                "has_target_grid": self.has_target_grid.copy(),
                "N_confirmed": self.N_wall_logodds > 2.0,
                "N_predicted": (self.N_wall_logodds > 0.0) & (self.N_wall_logodds <= 2.0),
                "E_confirmed": self.E_wall_logodds > 2.0,
                "E_predicted": (self.E_wall_logodds > 0.0) & (self.E_wall_logodds <= 2.0),
                "S_outer_confirmed": self.S_outer_wall_logodds > 2.0,
                "S_outer_predicted": (self.S_outer_wall_logodds > 0.0) & (self.S_outer_wall_logodds <= 2.0),
                "W_outer_confirmed": self.W_outer_wall_logodds > 2.0,
                "W_outer_predicted": (self.W_outer_wall_logodds > 0.0) & (self.W_outer_wall_logodds <= 2.0),
                "pose": self.pose.copy(),
                "grid_x": self.grid_x, "grid_y": self.grid_y,
                "tof_raw": self.tof_raw, "filtered_tof": self.tof_filtered,
                "sharp_left": self.sharp_left_dist, "sharp_right": self.sharp_right_dist,
                "gimbal_yaw": self.gimbal_yaw, "is_aligning": self.is_aligning,
                "is_scanning": self.is_scanning, "is_rotating": self.is_rotating,
                "is_calibrated": self.is_calibrated, "emergency_stop": self.emergency_stop,
                "running": self.running, "status_msg": self.status_msg,
                "targets_found": list(self.targets_found),
                "active_detections": list(self.active_detections),
                "robot_path_history": list(self.robot_path_history),
            }

    def stop_workers(self):
        with self.lock: self.running = False

hub = SharedHub(GRID_W, GRID_H, START_X, START_Y)

# ==========================================
# 5. WORKERS & HANDLERS
# ==========================================
def safe_camera_worker(ep_camera):
    try:
        ep_camera.start_video_stream(display=False)
    except Exception as e:
        log_event(f"[CAMERA ERROR] Stream start failed: {e}")

    while hub.get_state()["running"]:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest")
            if frame is not None:
                with hub.lock:
                    hub.latest_frame = frame.copy()
        except Exception: pass
        time.sleep(0.04)

    try: ep_camera.stop_video_stream()
    except Exception: pass

def sharp_sensor_worker(ep_robot):
    sensor_adaptor = getattr(ep_robot, 'sensor_adaptor', None)
    if sensor_adaptor is None:
        log_event("⚠️ [SHARP ERROR] ep_robot.sensor_adaptor is None!")
        return
    
    log_event("[SHARP WORKER] Sharp Sensor Thread Started.")
    
    while hub.get_state()["running"]:
        try:
            l_cm = get_sharp_distance_cm(sensor_adaptor, s_id=2, s_port=1, sensor_key="LEFT_SENSOR")
            time.sleep(0.03)
            r_cm = get_sharp_distance_cm(sensor_adaptor, s_id=1, s_port=1, sensor_key="RIGHT_SENSOR")
            
            hub.update_sharp_distances(l_cm, r_cm)
        except Exception: pass
        time.sleep(0.08)

def tof_data_handler(sub_info):
    try:
        if sub_info and len(sub_info) > 0 and sub_info[0] is not None:
            raw_dist_cm = float(sub_info[0]) / 10.0
            dist_cm = apply_tof_calibration(raw_dist_cm)
        else: dist_cm = 999.0
    except (ValueError, TypeError): dist_cm = 999.0

    hub.append_tof(dist_cm)

def gimbal_data_handler(sub_info):
    if sub_info and len(sub_info) > 1 and sub_info[1] is not None:
        try:
            with hub.lock: hub.gimbal_yaw = float(sub_info[1])
        except (ValueError, TypeError): pass

def chassis_pos_handler(sub_info):
    if sub_info and len(sub_info) >= 2 and sub_info[0] is not None and sub_info[1] is not None:
        try:
            current_yaw = hub.latest_raw_yaw if hub.latest_raw_yaw is not None else 0.0
            hub.set_raw_data([float(sub_info[0]), float(sub_info[1])], current_yaw)
        except (ValueError, TypeError): pass

def chassis_att_handler(sub_info):
    if sub_info and len(sub_info) >= 1 and sub_info[0] is not None:
        try:
            pos = hub.latest_raw_pos if hub.latest_raw_pos[0] is not None else [0.0, 0.0]
            hub.set_raw_data(pos, float(sub_info[0]))
        except (ValueError, TypeError): pass

# ==========================================
# 6. OPENCV CONTOUR INSPECTION & SHAPE CLASSIFIER
# ==========================================
def save_target_inspection_image(img, target_info, bbox, tof_dist_cm, status="DETECTED"):
    if img is None: return
    try:
        annotated_img = img.copy()
        rx, ry, bw, bh = bbox
        color_name = target_info.get("color", "unknown")
        shape_name = target_info.get("shape", "unknown")
        t_id = target_info.get("target_id", 0)

        cv2.rectangle(annotated_img, (rx, ry), (rx + bw, ry + bh), (0, 255, 0), 2)
        label = f"ID#{t_id} {color_name.upper()} {shape_name} | {tof_dist_cm:.1f}cm | {status}"
        cv2.putText(annotated_img, label, (rx, max(20, ry - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"Target_ID{t_id}_{color_name}_{shape_name}_{status}_{timestamp_str}.png"
        filepath = os.path.join(TARGET_IMG_DIR, filename)
        
        cv2.imwrite(filepath, annotated_img)
        log_event(f"📸 [TARGET INSPECTION SAVED] Image saved: '{filepath}'")
    except Exception as e:
        log_event(f"⚠️ [TARGET IMG SAVE ERROR] {e}")

def get_color_mask(hsv_img, target_color):
    ranges = COLOR_RANGES.get(target_color, [])
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_img, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask

def process_frame_multi_targets(img, distance_cm, target_color="red", target_shape_type="ALL_SHAPES", verbose_log=False):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0: return []
    detected_list = []
    try:
        h, w, _ = img.shape
        if ROI_MASK_IMG is not None:
            resized_mask = cv2.resize(ROI_MASK_IMG, (w, h))
            img_masked = cv2.bitwise_and(img, img, mask=resized_mask)
        else: img_masked = img

        hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)
        mask = get_color_mask(hsv, target_color)
        if mask is None: return []

        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            cx_temp, cy_temp = 0, 0
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_temp = int(M["m10"] / M["m00"])
                cy_temp = int(M["m01"] / M["m00"])

            quadrant = get_screen_quadrant(cx_temp, cy_temp, w, h)
            if area < 120: continue

            rx, ry, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0: continue

            bbox_mask = mask_clean[ry : ry + bh, rx : rx + bw]
            fill_ratio = (cv2.countNonZero(bbox_mask) / float(bw * bh)) * 100.0
            if fill_ratio < MIN_PIXEL_FILL_RATIO: continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue

            # 🌟 ปรับลด Epsilon เพื่อไม่ให้ลดทอนจุดขอบภาพวงกลมมากเกินไป
            approx = cv2.approxPolyDP(cnt, 0.015 * perimeter, True)
            num_vertices = len(approx)

            circularity = (4.0 * math.pi * area) / (perimeter ** 2)
            extent = float(area) / (bw * bh)
            aspect_ratio = float(bw) / float(bh)

            # 🌟 ปรับปรุงการจำแนกวงกลม (CIRCLE) ป้องกันการแยกผิดเป็น RECT_V เมื่อมองจากมุมเฉียง
            subtype = None
            if circularity >= 0.68 or (0.80 <= aspect_ratio <= 1.25 and num_vertices > 6):
                subtype = "CIRCLE"
            elif 0.78 <= aspect_ratio <= 1.25 and extent >= 0.65:
                subtype = "SQUARE"
            elif aspect_ratio > 1.25:
                subtype = "RECT_H"
            elif aspect_ratio < 0.78:
                subtype = "RECT_V"

            if subtype not in VALID_SHAPES: continue

            cx = rx + (bw // 2)
            cy = ry + (bh // 2)
            deg_offset = ((cx - (w / 2.0)) / CAMERA_FOCAL_LENGTH_PX) * (180.0 / math.pi)

            item = {
                "color": target_color, "shape": subtype,
                "bbox": (rx, ry, bw, bh), "cx": cx, "cy": cy,
                "fill_ratio": fill_ratio, "circularity": circularity,
                "quadrant": quadrant, "deg_offset": deg_offset,
            }
            if target_shape_type == "ALL_SHAPES" or subtype == target_shape_type:
                detected_list.append(item)

        return detected_list
    except Exception as e:
        log_event(f"[DETECT ERROR] {e}")
    return []

def process_frame_scan_all_colors(img, distance_cm, main_color="red", main_shape="ALL_SHAPES"):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0: return []
    colors_to_scan = ALL_COLORS if main_color == "ALL_COLORS" else [main_color]
    found_list = []

    for c in colors_to_scan:
        detected = process_frame_multi_targets(img, distance_cm, target_color=c, target_shape_type=main_shape, verbose_log=False)
        for item in detected:
            is_main = (main_color == "ALL_COLORS" or c == main_color) and (main_shape == "ALL_SHAPES" or item["shape"] == main_shape)
            item["is_main"] = is_main
            found_list.append(item)

    hub.set_active_detections(found_list)
    return found_list

# ==========================================
# 7. GIMBAL SCANNING & SAMPLING
# ==========================================
def reset_gimbal_to_center_fast(ep_gimbal):
    try:
        with sdk_lock:
            ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=0.8)
    except Exception: pass

def deg_to_cardinal_exact(abs_deg):
    norm_deg = normalize_angle(abs_deg)
    if -45.0 <= norm_deg <= 45.0: return "N"
    elif 45.0 < norm_deg <= 135.0: return "E"
    elif norm_deg > 135.0 or norm_deg < -135.0: return "S"
    elif -135.0 <= norm_deg < -45.0: return "W"
    return "N"

def get_fresh_tof_distance(ep_gimbal, target_yaw, pitch=0):
    try:
        with sdk_lock:
            ep_gimbal.moveto(pitch=pitch, yaw=target_yaw, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=1.0)
    except Exception: pass

    time.sleep(0.08)
    hub.clear_tof_buffer()
    fresh_samples = []
    last_sample_t = time.time()
    start_t = time.time()
    
    while len(fresh_samples) < 5 and (time.time() - start_t < 0.30):
        with hub.lock:
            if hub.tof_last_update_time > last_sample_t:
                val = hub.tof_raw
                if 2.0 <= val <= 300.0:
                    fresh_samples.append(val)
                    last_sample_t = hub.tof_last_update_time
        time.sleep(0.02)

    return float(np.median(fresh_samples)) if fresh_samples else 999.0

def scan_and_process_targets(ep_gimbal, snap_yaw, current_gx, current_gy, target_color="red", target_shape="ALL_SHAPES"):
    scan_seq = [("AHEAD", 0), ("RIGHT", 90), ("BACK", 180), ("LEFT", -90)]
    tof_results = {}
    scan_queue = queue.Queue()

    hub.set_scanning(True)
    hub.set_status_msg("Scanning 4 Directions...")

    for dir_name, rel_y in scan_seq:
        if not hub.running: break
        dist_cm = get_fresh_tof_distance(ep_gimbal, target_yaw=rel_y, pitch=0)
        tof_results[dir_name] = dist_cm

        abs_dir_deg = normalize_angle(snap_yaw + rel_y)
        w_dir = deg_to_cardinal_exact(abs_dir_deg)

        grid_steps = int(round(dist_cm / CELL_SIZE_CM))
        grid_steps = max(1, min(grid_steps, 3))

        for step in range(1, grid_steps):
            target_gx, target_gy = current_gx, current_gy
            if w_dir == "N": target_gy += step
            elif w_dir == "E": target_gx += step
            elif w_dir == "S": target_gy -= step
            elif w_dir == "W": target_gx -= step
            
            target_gx = int(np.clip(target_gx, 0, hub.w - 1))
            target_gy = int(np.clip(target_gy, 0, hub.h - 1))
            hub.update_wall_prob(target_gx, target_gy, w_dir, is_wall=False, is_direct_adjacent=(step == 1))

        final_gx, final_gy = current_gx, current_gy
        if w_dir == "N": final_gy += (grid_steps - 1)
        elif w_dir == "E": final_gx += (grid_steps - 1)
        elif w_dir == "S": final_gy -= (grid_steps - 1)
        elif w_dir == "W": final_gx -= (grid_steps - 1)

        final_gx = int(np.clip(final_gx, 0, hub.w - 1))
        final_gy = int(np.clip(final_gy, 0, hub.h - 1))

        is_wall_detected = dist_cm < (grid_steps * CELL_SIZE_CM - 8.0)
        is_direct = (grid_steps == 1)
        hub.update_wall_prob(final_gx, final_gy, w_dir, is_wall=is_wall_detected, is_direct_adjacent=is_direct)

        img = hub.get_frame_safe(retries=2, delay=0.01)
        if img is not None:
            triggers = process_frame_scan_all_colors(img, dist_cm, main_color=target_color, main_shape=target_shape)
            if triggers:
                for item in triggers:
                    item_yaw = rel_y + item["deg_offset"]
                    scan_queue.put({
                        "dir_name": dir_name, "yaw": item_yaw,
                        "color": item["color"], "is_main": item["is_main"],
                    })

    if not scan_queue.empty():
        hub.set_status_msg("Target Lock & Sampling...")
        while not scan_queue.empty() and hub.running:
            item = scan_queue.get()
            target_yaw, t_color, is_main = item["yaw"], item["color"], item["is_main"]

            with sdk_lock:
                ep_gimbal.moveto(pitch=0, yaw=target_yaw, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=0.6)

            log_event(f"[TARGET LOCK] Warming up camera for target confirmation ({t_color.upper()} {target_shape})")
            time.sleep(0.30)

            log_event("[SAMPLING] Sampling 10 frames for target confirmation")
            valid_samples, tof_samples, last_bbox_sample, last_img_sample = [], [], None, None
            REQUIRED_MATCH_FRAMES = 6

            for frame_idx in range(1, 11):
                time.sleep(0.04)
                s_img = hub.get_frame_safe(retries=2, delay=0.01)
                with hub.lock: s_tof = hub.tof_filtered

                if s_img is not None:
                    s_targets = process_frame_multi_targets(s_img, s_tof, t_color, target_shape, verbose_log=False)
                    if s_targets:
                        s_res = s_targets[0]
                        valid_samples.append((s_res["color"], s_res["shape"]))
                        tof_samples.append(s_tof)
                        last_bbox_sample = s_res["bbox"]
                        last_img_sample = s_img.copy()
                        log_event(f"   Frame {frame_idx:02d}/10: PASS -> {s_res['color'].upper()} ({s_res['shape']}) | Quad: {s_res['quadrant']} | Circularity: {s_res['circularity']:.2f}")
                    else: log_event(f"   Frame {frame_idx:02d}/10: FAIL (Criteria not met)")
                else: log_event(f"   Frame {frame_idx:02d}/10: FAIL (No frame from camera)")

            success_count = len(valid_samples)
            log_event(f"[SAMPLING SUMMARY] Passed {success_count}/10 frames (Threshold >= {REQUIRED_MATCH_FRAMES})")

            if success_count >= REQUIRED_MATCH_FRAMES:
                most_common = Counter(valid_samples).most_common(1)[0][0]
                conf_color, conf_shape = most_common
                tof_final = float(np.median(tof_samples)) if tof_samples else hub.tof_filtered

                with hub.lock:
                    real_chassis_yaw = hub.pose["chassis_yaw"]
                
                actual_world_yaw = normalize_angle(real_chassis_yaw + target_yaw)
                is_registered = hub.register_target_spatial(
                    abs_yaw_deg=actual_world_yaw, tof_dist_cm=tof_final,
                    color=conf_color, shape=conf_shape, is_main_target=is_main,
                )

                if last_img_sample is not None and last_bbox_sample is not None:
                    t_info = {"target_id": hub.target_id_counter - 1, "color": conf_color, "shape": conf_shape}
                    save_target_inspection_image(last_img_sample, t_info, last_bbox_sample, tof_final, status="PASSED" if is_registered else "REJECTED_DUP")

                if is_registered and is_main:
                    if tof_final <= MAX_FIRING_DIST_CM:
                        log_event(f"[TARGET LOCKED & CONFIRMED] Initial fire burst at distance: {tof_final:.1f} cm")
                        play_fire_sound()
                        time.sleep(0.1)
                        fire_target_burst(count=BURST_FIRE_COUNT)

                        time.sleep(0.4)
                        recheck_img = hub.get_frame_safe(retries=2, delay=0.01)
                        if recheck_img is not None:
                            recheck_targets = process_frame_multi_targets(recheck_img, tof_final, conf_color, conf_shape, verbose_log=False)
                            if recheck_targets:
                                log_event("[RE-FIRE] Target still standing! Executing re-fire burst...")
                                play_fire_sound()
                                fire_target_burst(count=BURST_FIRE_COUNT)
                                time.sleep(0.3)

                        with hub.lock:
                            if hub.targets_found:
                                hub.targets_found[-1]["is_knocked_down"] = True
                    else:
                        log_event(f"[TARGET OUT OF RANGE] Target marked at {tof_final:.1f} cm (Exceeds max fire dist {MAX_FIRING_DIST_CM} cm). Shooting skipped until closer.")
            else:
                log_event(f"[TARGET REJECTED] Image confirmation failed ({success_count}/10)")

            scan_queue.task_done()

    reset_gimbal_to_center_fast(ep_gimbal)
    hub.set_active_detections([])
    hub.set_scanning(False)
    return tof_results

# ==========================================
# 8. CHASSIS MOVEMENT & SMOOTH SHARP IR
# ==========================================
def stop_chassis_active_brake(ep_chassis, brake_duration=0.10):
    try:
        with sdk_lock:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(brake_duration)
    except Exception: pass

def check_and_recenter(ep_chassis, ep_gimbal, tof_dict):
    if not hub.running: return False
    hub.set_aligning(True)
    max_detect_dist = 45.0

    dist_ahead, dist_right = tof_dict.get('AHEAD', 999.0), tof_dict.get('RIGHT', 999.0)
    dist_back, dist_left = tof_dict.get('BACK', 999.0), tof_dict.get('LEFT', 999.0)

    shift_x, shift_y = 0.0, 0.0
    if dist_right < max_detect_dist and dist_left < max_detect_dist:
        shift_y = np.clip(((dist_right - dist_left) / 2.0) / 100.0, -0.05, 0.05)
    elif dist_right < max_detect_dist: shift_y = np.clip((dist_right - IDEAL_SIDE_WALL_DIST_CM) / 100.0, -0.05, 0.05)
    elif dist_left < max_detect_dist: shift_y = np.clip(-(dist_left - IDEAL_SIDE_WALL_DIST_CM) / 100.0, -0.05, 0.05)

    if dist_ahead < max_detect_dist and dist_back < max_detect_dist:
        shift_x = np.clip(((dist_ahead - dist_back) / 2.0) / 100.0, -0.05, 0.05)
    elif dist_ahead < max_detect_dist: shift_x = np.clip((dist_ahead - 20.0) / 100.0, -0.05, 0.05)
    elif dist_back < max_detect_dist: shift_x = np.clip(-(dist_back - 20.0) / 100.0, -0.05, 0.05)

    has_moved = False
    if abs(shift_x) > 0.015 or abs(shift_y) > 0.015:
        try:
            with sdk_lock:
                ep_chassis.move(x=shift_x, y=shift_y, z=0, xy_speed=0.15).wait_for_completed(timeout=2.0)
            stop_chassis_active_brake(ep_chassis, brake_duration=0.08)
            has_moved = True
        except Exception: pass
        reset_gimbal_to_center_fast(ep_gimbal)

    hub.set_aligning(False)
    return has_moved

def turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw_deg):
    if not hub.running: return False

    with hub.lock: current_tof = hub.tof_filtered
    if 2.0 <= current_tof < 18.0:
        log_event(f"[AUTO-BACKUP] Obstacle too close ({current_tof:.1f} cm < 18cm) -> Backing up 10cm before turn")
        try:
            with sdk_lock:
                ep_chassis.move(x=-0.10, y=0, z=0, xy_speed=0.15).wait_for_completed(timeout=1.2)
            stop_chassis_active_brake(ep_chassis, brake_duration=0.1)
        except Exception: pass

    with hub.lock:
        hub.is_rotating = True
        hub.emergency_stop = False
        start_raw_yaw = hub.latest_raw_yaw

    target_yaw_deg = normalize_angle(target_yaw_deg)
    reset_gimbal_to_center_fast(ep_gimbal)
    time.sleep(0.05)

    with hub.lock:
        curr_yaw = normalize_angle(hub.latest_raw_yaw + hub.yaw_offset_deg) if hub.latest_raw_yaw is not None else hub.pose["chassis_yaw"]

    diff = normalize_angle(target_yaw_deg - curr_yaw)
    if abs(diff) <= 1.2:
        with hub.lock: hub.is_rotating = False
        return True

    abs_diff = abs(diff)
    exec_z_speed = 55 if abs_diff <= 15.0 else (75 if abs_diff <= 45.0 else 95)
    log_event(f"[TURN START] Relative {diff:.1f} deg to Target: {target_yaw_deg:.0f} deg (Speed: {exec_z_speed})")

    try:
        with sdk_lock:
            ep_chassis.move(x=0, y=0, z=-diff, z_speed=exec_z_speed).wait_for_completed(timeout=3.5)
    except Exception as e:
        log_event(f"[TURN TIMEOUT/ERROR] {e}")

    stop_chassis_active_brake(ep_chassis, brake_duration=0.1)
    reset_gimbal_to_center_fast(ep_gimbal)
    time.sleep(0.12)

    with hub.lock:
        if hub.latest_raw_yaw is not None and start_raw_yaw is not None:
            actual_turned = normalize_angle(hub.latest_raw_yaw - start_raw_yaw)
            if (diff * actual_turned < 0) or (abs(diff) > 20.0 and abs(actual_turned) < (abs(diff) * 0.4)):
                log_event(f"[TURN FAILED] Turn blocked! (Target turn: {diff:.1f} deg, Actual turn: {actual_turned:.1f} deg)")
                hub.is_rotating = False
                return False

            snapped_yaw = snap_to_cardinal_yaw(target_yaw_deg)
            hub.pose["chassis_yaw"] = snapped_yaw
            hub.yaw_offset_deg = (snapped_yaw - hub.latest_raw_yaw) % 360.0
            
        hub.is_rotating = False

    log_event(f"[TURN COMPLETED] Settled Heading: {hub.pose['chassis_yaw']:.1f} deg")
    return True

def move_forward_60cm_straight(ep_chassis, target_step_m=move_target_step_m, target_speed=target_speed_mps):
    calibrated_target_m = target_step_m * DISTANCE_CALIB_FACTOR
    with hub.lock:
        start_pos_x = hub.latest_raw_pos[0] if hub.latest_raw_pos[0] is not None else 0.0
        start_pos_y = hub.latest_raw_pos[1] if hub.latest_raw_pos[1] is not None else 0.0
        target_yaw = hub.pose["chassis_yaw"]

    effective_distance_traveled = 0.0
    max_timeout = (calibrated_target_m / target_speed) + 2.0
    start_time = time.time()

    prev_x, prev_y = start_pos_x, start_pos_y

    while effective_distance_traveled < calibrated_target_m and (time.time() - start_time < max_timeout):
        state = hub.get_state()
        if state["emergency_stop"]: break

        l_sharp, r_sharp = state["sharp_left"], state["sharp_right"]
        tof_val = state["filtered_tof"]

        # 1. Safety Brake ด้านหน้า
        if tof_val <= SAFE_MIN_TOF_CM:
            log_event(f"[SAFETY BRAKE] Front wall detected by ToF! ({tof_val:.1f} cm <= {SAFE_MIN_TOF_CM} cm)")
            stop_chassis_active_brake(ep_chassis, brake_duration=0.15)
            return False

        with hub.lock:
            curr_x = hub.latest_raw_pos[0] if hub.latest_raw_pos[0] is not None else prev_x
            curr_y = hub.latest_raw_pos[1] if hub.latest_raw_pos[1] is not None else prev_y

        step_dist = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        prev_x, prev_y = curr_x, curr_y

        # 2. Heading Lock ด้วย IMU (บังคับหัวหุ่นชี้ตรงทิศหลักเสมอ)
        curr_yaw = state["pose"]["chassis_yaw"]
        error_yaw = normalize_angle(target_yaw - curr_yaw)
        angular_z = float(np.clip(error_yaw * 0.08, -1.5, 1.5))

        cos_correction = math.cos(math.radians(error_yaw))
        effective_distance_traveled += step_dist * cos_correction

        # 3. Sharp IR Active Adjustment (ปรับช่วง Deadband 16-20 cm + Base Speed สไลด์จริง)
        speed_y = 0.0

        # เช็กกำแพงฝั่งขวา
        if r_sharp < MAX_SHARP_WALL_DETECT_CM:
            if r_sharp < 16.0:
                # ชิดขวาเกินไป (< 16 cm) -> สไลด์ออกซ้าย (Base 0.06 m/s)
                speed_y = -float(np.clip(0.06 + (16.0 - r_sharp) * 0.008, 0.06, 0.12))
            elif r_sharp > 20.0:
                # ห่างขวาเกินไป (20 - 26 cm) -> สไลด์เข้าขวา (Base 0.06 m/s)
                speed_y = float(np.clip(0.06 + (r_sharp - 20.0) * 0.008, 0.06, 0.12))

        # เช็กกำแพงฝั่งซ้าย (กรณีขวาไม่มีกำแพง)
        elif l_sharp < MAX_SHARP_WALL_DETECT_CM:
            if l_sharp < 16.0:
                # ชิดซ้ายเกินไป (< 16 cm) -> สไลด์ออกขวา (Base 0.06 m/s)
                speed_y = float(np.clip(0.06 + (16.0 - l_sharp) * 0.008, 0.06, 0.12))
            elif l_sharp > 20.0:
                # ห่างซ้ายเกินไป (20 - 26 cm) -> สไลด์เข้าซ้าย (Base 0.06 m/s)
                speed_y = -float(np.clip(0.06 + (l_sharp - 20.0) * 0.008, 0.06, 0.12))

        