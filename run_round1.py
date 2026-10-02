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
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
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
SAFE_MIN_TOF_CM = 15.0               
MAX_TARGET_DETECT_DIST_CM = 130.0    
MAX_FIRING_DIST_CM = 110.0            
CAMERA_FOCAL_LENGTH_PX = 650.0
GIMBAL_FAST_SPEED = 540
MIN_PIXEL_FILL_RATIO = 35.0          
MIN_CONTOUR_AREA = 220               

# พารามิเตอร์การเดิน
KP_YAW_HOLD = 0.8          # Proportional Yaw Holding Gain
DECEL_ZONE_M = 0.18        # ระยะชะลอก่อนถึงเป้าหมาย
MIN_V = 0.08               # ความเร็วขั้นต่ำช่วงชะลอตัว

# 🌟 ปรับระบบ Sharp IR เป็น Guardrail Safety (ผลักออกห่างอย่างเดียว)
SHARP_PUSH_DIST_CM = 11.0  # ระยะชิดเกินไปที่จะสั่งสไลด์ออกห่าง
STRAFE_SAFETY_V = 0.10     # ความเร็วสไลด์หลบกำแพงฉุกเฉิน

# 🌟 รายการวัตถุที่ต้องการสั่งให้ IGNORE (ระบุ Bounding Box Ratio หรือพิกัด)
KNOWN_IGNORED_OBJECTS = [
    # ตัวอย่าง: {"grid": (1, 2), "color": "green"}
]

TOF_FREQ_HZ = 50                     
TOF_FILTER_SIZE = 5                  
TOF_SAMPLE_PERIOD_SEC = 1.0 / TOF_FREQ_HZ
GIMBAL_LATENCY_FLUSH_SEC = 0.15

IDEAL_SIDE_WALL_DIST_CM = 17.5       
MAX_SHARP_WALL_DETECT_CM = 25.0       

MAX_ROUND1_TIME_SEC = 570.0  

sdk_lock = threading.Lock()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_asset_path(filename):
    p_script = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(p_script):
        return p_script
    return filename

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

SHARP_CONFIG = None
sharp_json_path = get_asset_path("calibration_sharp_poly.json")

if os.path.exists(sharp_json_path):
    try:
        with open(sharp_json_path, "r", encoding="utf-8") as f:
            SHARP_CONFIG = json.load(f)
            log_event(f"[SHARP CALIB LOADED] Loaded '{sharp_json_path}' successfully")
    except Exception as e:
        log_event(f"[SHARP CALIB ERROR] Failed to load JSON: {e}")

def get_sharp_distance_cm(sensor_adaptor, s_id, s_port, sensor_key):
    if sensor_adaptor is None or SHARP_CONFIG is None:
        return 999.0
    try:
        adc = sensor_adaptor.get_adc(id=s_id, port=s_port)
        if adc is None or adc <= 0:
            return 999.0
        
        cfg = SHARP_CONFIG.get(sensor_key, {})
        noise_th = cfg.get("noise_threshold", 0)

        if noise_th > 0 and adc <= noise_th:
            return 999.0

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

roi_mask_path = get_asset_path("roi_mask.png")
ROI_MASK_IMG = None
if os.path.exists(roi_mask_path):
    ROI_MASK_IMG = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    log_event(f"[ROI MASK] Loaded '{roi_mask_path}' successfully")

TOF_OFFSET_HORIZ_CM = 7.5
MAX_GROUND_DIST_CM = 180.0

ALL_COLORS = ["red", "yellow", "green", "blue"]
VALID_SHAPES = ["RECT_V", "RECT_H", "SQUARE", "CIRCLE"]

COLOR_MAIN_STAR = (156, 39, 176)
COLOR_SUB_STAR = (3, 169, 244)

DEFAULT_HSV_RANGES = {
    "red": [
        {"lower": [0, 150, 120], "upper": [10, 255, 255]},
        {"lower": [170, 150, 120], "upper": [180, 255, 255]},
    ],
    "yellow": [{"lower": [20, 150, 140], "upper": [34, 255, 255]}],
    "green": [{"lower": [40, 120, 80], "upper": [80, 255, 255]}],
    "blue": [{"lower": [100, 140, 90], "upper": [130, 255, 255]}],
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
    if tof_raw_cm <= 0 or tof_raw_cm >= 800.0:
        return 999.0
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

target_speed_mps = 0.32              
move_target_step_m = 0.60
DISTANCE_CALIB_FACTOR = 1.0

ep_blaster_ref = None
ep_robot_ref = None

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
        self.tof_fast_ema = 0.0
        self.tof_last_update_time = 0.0
        self.tof_buffer = deque(maxlen=TOF_FILTER_SIZE)

        self.sharp_left_dist = 999.0
        self.sharp_right_dist = 999.0
        self.sharp_alpha = 0.65

        self.gimbal_yaw = 0.0
        self.gimbal_pitch = 0.0
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

        self.save_map_flag = False
        self.pending_map_save_filename = "final_map.png"

    def request_map_save(self, filename="final_map.png"):
        with self.lock:
            self.pending_map_save_filename = filename
            self.save_map_flag = True

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
                if self.tof_fast_ema <= 0.0 or self.tof_fast_ema >= 800.0:
                    self.tof_fast_ema = val
                else:
                    self.tof_fast_ema = (0.85 * val) + (0.15 * self.tof_fast_ema)

                self.tof_buffer.append(val)
            else:
                self.tof_fast_ema = 999.0

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
            else:
                self.sharp_left_dist = 999.0
            
            if right_cm < 900.0:
                if self.sharp_right_dist >= 900.0:
                    self.sharp_right_dist = right_cm
                else:
                    self.sharp_right_dist = round((self.sharp_alpha * right_cm) + ((1.0 - self.sharp_alpha) * self.sharp_right_dist), 2)
            else:
                self.sharp_right_dist = 999.0

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

    def reset_predicted_walls_for_unvisited(self):
        with self.lock:
            for gy in range(self.h):
                for gx in range(self.w):
                    if not self.N_wall_locked[gy, gx] and self.N_wall_logodds[gy, gx] <= 2.0:
                        self.N_wall_logodds[gy, gx] = min(self.N_wall_logodds[gy, gx], 0.0)
                    if not self.E_wall_locked[gy, gx] and self.E_wall_logodds[gy, gx] <= 2.0:
                        self.E_wall_logodds[gy, gx] = min(self.E_wall_logodds[gy, gx], 0.0)

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
                if dist <= 25.0 and t["color"] == color:
                    log_event(f"[TARGET REJECTED] Duplicate target detected at distance {dist:.1f} cm")
                    return False

            norm_dir = normalize_angle(abs_yaw_deg)
            cardinal_dir = "N"
            target_gx, target_gy = self.grid_x, self.grid_y

            steps = int(round(ground_dist_cm / CELL_SIZE_CM))
            steps = max(1, steps)

            if -45.0 <= norm_dir <= 45.0: target_gy += steps; cardinal_dir = "N"
            elif 45.0 < norm_dir <= 135.0: target_gx += steps; cardinal_dir = "E"
            elif norm_dir > 135.0 or norm_dir < -135.0: target_gy -= steps; cardinal_dir = "S"
            elif -135.0 <= norm_dir < -45.0: target_gx -= steps; cardinal_dir = "W"

            target_gx = int(np.clip(target_gx, 0, self.w - 1))
            target_gy = int(np.clip(target_gy, 0, self.h - 1))

            if ground_dist_cm <= MAX_GROUND_DIST_CM:
                self.has_target_grid[target_gy, target_gx] = True
                log_event(f"[TARGET MARKED] Target marked at Grid({target_gx}, {target_gy}) from distance {ground_dist_cm:.1f} cm")

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
                    "firing_direction": t["firing_direction"],
                    "is_knocked_down": t.get("is_knocked_down", False)
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
                "tof_fast_ema": self.tof_fast_ema,
                "sharp_left": self.sharp_left_dist, "sharp_right": self.sharp_right_dist,
                "gimbal_yaw": self.gimbal_yaw, "gimbal_pitch": self.gimbal_pitch,
                "is_aligning": self.is_aligning, "is_scanning": self.is_scanning, 
                "is_rotating": self.is_rotating, "is_calibrated": self.is_calibrated, 
                "emergency_stop": self.emergency_stop, "running": self.running, 
                "status_msg": self.status_msg, "targets_found": list(self.targets_found),
                "active_detections": list(self.active_detections),
                "robot_path_history": list(self.robot_path_history),
            }

    def stop_workers(self):
        with self.lock:
            self.running = False
            self.emergency_stop = True

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
        log_event("⚠ [SHARP ERROR] ep_robot.sensor_adaptor is None!")
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
    if sub_info and len(sub_info) > 1 and sub_info[0] is not None and sub_info[1] is not None:
        try:
            with hub.lock: 
                hub.gimbal_pitch = float(sub_info[0])
                hub.gimbal_yaw = float(sub_info[1])
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
# 6. OPENCV CONTOUR INSPECTION & MULTI-CANDIDATE SHAPE EVALUATOR
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

def evaluate_multi_shape_candidates(bw, bh, area, perimeter, num_vertices, circularity, extent, is_far=False):
    """🌟 อัลกอริทึมประเมินคะแนนของทุกรูปทรงพร้อมกัน (Adaptive System)"""
    if bh == 0 or bw == 0 or perimeter == 0:
        return None, 0.0

    aspect_ratio = float(bw) / float(bh)

    if 0.75 <= aspect_ratio <= 1.30:
        if extent >= 0.80 and circularity < 0.84:
            score_square = (1.0 - abs(aspect_ratio - 1.0)) * extent
            return "SQUARE", round(score_square, 3)
        elif circularity >= 0.78 and extent <= 0.82:
            score_circle = circularity
            return "CIRCLE", round(score_circle, 3)
        elif extent >= 0.78 and circularity < 0.80:
            score_square = (1.0 - abs(aspect_ratio - 1.0)) * extent
            return "SQUARE", round(score_square, 3)

    if 0.45 <= aspect_ratio <= 0.78 and extent >= 0.75:
        score_rect_v = (1.0 - abs(aspect_ratio - 0.667) / 0.4) * extent
        return "RECT_V", round(score_rect_v, 3)

    if 1.22 <= aspect_ratio <= 2.20 and extent >= 0.75:
        score_rect_h = (1.0 - abs(aspect_ratio - 1.500) / 0.5) * extent
        return "RECT_H", round(score_rect_h, 3)

    return None, 0.0

def process_frame_multi_targets(img, distance_cm, target_color="red", target_shape_type="ALL_SHAPES", verbose_log=False):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0: return []
    
    ground_dist = calculate_ground_distance(distance_cm) if distance_cm < 800.0 else 999.0

    if ground_dist > (MAX_TARGET_DETECT_DIST_CM + TOF_OFFSET_HORIZ_CM):
        return []

    detected_list = []
    try:
        h, w, _ = img.shape

        if ground_dist < 800.0 and ground_dist > 0.0:
            crop_ratio = float(np.clip(0.12 + (ground_dist - 30.0) * 0.0040, 0.10, 0.48))
        else:
            crop_ratio = 0.35
            
        is_far = ground_dist > 65.0
            
        crop_top_pixels = int(h * crop_ratio)
        img_masked = img.copy()

        if crop_top_pixels > 0:
            top_left = (0, 0)
            top_right = (w, 0)
            bottom_right = (int(w * 0.88), crop_top_pixels)
            bottom_left = (int(w * 0.12), crop_top_pixels)
            trap_poly = np.array([[top_left, top_right, bottom_right, bottom_left]], dtype=np.int32)
            cv2.fillPoly(img_masked, trap_poly, (0, 0, 0))

        # Removed floor reflection filter to allow seeing lower targets in dynamic ROI

        if ROI_MASK_IMG is not None:
            resized_mask = cv2.resize(ROI_MASK_IMG, (w, h))
            img_masked = cv2.bitwise_and(img_masked, img_masked, mask=resized_mask)

        hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)
        mask = get_color_mask(hsv, target_color)
        if mask is None: return []

        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA: continue

            rx, ry, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0: continue

            cx = rx + (bw // 2)
            cy = ry + (bh // 2)

            deg_offset_x = ((cx - (w / 2.0)) / CAMERA_FOCAL_LENGTH_PX) * (180.0 / math.pi)
            deg_offset_y = ((cy - (h / 2.0)) / CAMERA_FOCAL_LENGTH_PX) * (180.0 / math.pi)

            if ground_dist < 800.0:
                real_w_cm = (bw * ground_dist) / CAMERA_FOCAL_LENGTH_PX
                real_h_cm = (bh * ground_dist) / CAMERA_FOCAL_LENGTH_PX

                if real_w_cm < 4.5 or real_w_cm > 11.5 or real_h_cm < 4.5 or real_h_cm > 11.5:
                    continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0

            if solidity < 0.88: continue

            # กรองตำแหน่งมุมก้มต่ำเกินไป
            if cy > int(h * 0.95) or deg_offset_y < -25.0: continue

            bbox_mask = mask_clean[ry : ry + bh, rx : rx + bw]
            fill_ratio = (cv2.countNonZero(bbox_mask) / float(bw * bh)) * 100.0
            if fill_ratio < MIN_PIXEL_FILL_RATIO: continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue

            approx = cv2.approxPolyDP(cnt, 0.015 * perimeter, True)
            num_vertices = len(approx)

            circularity = (4.0 * math.pi * area) / (perimeter ** 2)
            extent = float(area) / (bw * bh)

            # 🌟 เรียกใช้อัลกอริทึม Multi-Candidate Shape Evaluator
            subtype, score = evaluate_multi_shape_candidates(bw, bh, area, perimeter, num_vertices, circularity, extent, is_far=is_far)
            if subtype is None or subtype not in VALID_SHAPES: continue

            quadrant = get_screen_quadrant(cx, cy, w, h)

            item = {
                "color": target_color, "shape": subtype, "confidence": score,
                "bbox": (rx, ry, bw, bh), "cx": cx, "cy": cy,
                "fill_ratio": fill_ratio, "circularity": circularity,
                "quadrant": quadrant, "deg_offset": deg_offset_x,
                "deg_offset_y": deg_offset_y,
            }
            if target_shape_type == "ALL_SHAPES" or subtype == target_shape_type:
                detected_list.append(item)

        return detected_list
    except Exception as e:
        log_event(f"[DETECT ERROR] {e}")
    return []

def process_frame_scan_all_colors(img, distance_cm, main_color="red", main_shape="ALL_SHAPES"):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0: return []
    
    ground_dist = calculate_ground_distance(distance_cm) if distance_cm < 800.0 else 999.0
    if ground_dist > (MAX_TARGET_DETECT_DIST_CM + TOF_OFFSET_HORIZ_CM):
        return []

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
# 7. HARDWARE GIMBAL RECENTERING & ACTIVE PARALLEL ALIGNMENT
# ==========================================
def recenter_gimbal_hardware(ep_gimbal):
    try:
        with sdk_lock:
            ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=0.5)
    except Exception: pass

    start_t = time.time()
    while time.time() - start_t < 0.25:
        with hub.lock:
            g_yaw = abs(hub.gimbal_yaw)
        if g_yaw <= 1.0:
            break
        time.sleep(0.01)

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
            ep_gimbal.moveto(pitch=pitch, yaw=target_yaw, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=0.8)
    except Exception: pass

    start_wait_gimbal = time.time()
    while time.time() - start_wait_gimbal < 0.40:
        with hub.lock:
            curr_g_yaw = hub.gimbal_yaw
        if abs(normalize_angle(curr_g_yaw - target_yaw)) <= 2.5:
            break
        time.sleep(0.015)

    time.sleep(GIMBAL_LATENCY_FLUSH_SEC)

    hub.clear_tof_buffer()
    fresh_samples = []
    settle_start_time = time.time()
    last_sample_t = settle_start_time

    while len(fresh_samples) < TOF_FILTER_SIZE and (time.time() - settle_start_time < 0.45):
        with hub.lock:
            if hub.tof_last_update_time > last_sample_t:
                val = hub.tof_raw
                if 2.0 <= val <= 300.0:
                    fresh_samples.append(val)
                    last_sample_t = hub.tof_last_update_time
        time.sleep(TOF_SAMPLE_PERIOD_SEC / 2.0)

    if len(fresh_samples) > 0:
        return float(np.median(fresh_samples))
    else:
        with hub.lock:
            return hub.tof_filtered if hub.tof_filtered > 0 else 999.0

def align_wall_tof_gimbal(ep_chassis, ep_gimbal, tof_dict=None):
    """🌟 อัลกอริทึมจัดหุ่นยนต์ให้ตรงขนานกำแพง (Parallel Yaw Alignment) ทำงานหลังจากการสแกน ToF ครบแล้ว"""
    if not hub.running: return

    target_base_yaw = None
    if tof_dict is not None:
        d_ahead = tof_dict.get("AHEAD", 999.0)
        d_right = tof_dict.get("RIGHT", 999.0)
        d_left  = tof_dict.get("LEFT", 999.0)

        if d_ahead <= 40.0: target_base_yaw = 0
        elif d_right <= 35.0: target_base_yaw = 90
        elif d_left <= 35.0: target_base_yaw = -90
    else: target_base_yaw = 0

    if target_base_yaw is None: return

    # ยิง ToF สองมุมเฉียงซ้าย-ขวาเพื่อเปรียบเทียบระยะ ปรับหุ่นให้ตรงขนานกำแพง
    d_l = get_fresh_tof_distance(ep_gimbal, target_yaw=target_base_yaw - 14, pitch=0)
    d_r = get_fresh_tof_distance(ep_gimbal, target_yaw=target_base_yaw + 14, pitch=0)

    if d_l < 42.0 and d_r < 42.0 and abs(d_l - d_r) < 12.0:
        diff_cm = d_l - d_r
        if abs(diff_cm) > 0.6:
            z_adjust_deg = float(np.clip(diff_cm * 2.0, -8.0, 8.0))
            dir_label = "AHEAD" if target_base_yaw == 0 else ("RIGHT" if target_base_yaw == 90 else "LEFT")
            log_event(f"[WALL ALIGNMENT] Parallel adjust on ({dir_label}) L:{d_l:.1f} cm, R:{d_r:.1f} cm | Rotating Yaw: {z_adjust_deg:.1f}°")
            
            with sdk_lock:
                ep_chassis.move(x=0, y=0, z=z_adjust_deg, z_speed=30).wait_for_completed(timeout=0.5)

    recenter_gimbal_hardware(ep_gimbal)

def aim_and_fire_target_close_range(ep_gimbal, target_yaw_rel, target_pitch_rel, color, shape):
    log_event(f"[CLOSE FIRING] Target detected within {MAX_FIRING_DIST_CM} cm -> Fine alignment & Firing...")
    try:
        with sdk_lock:
            ep_gimbal.moveto(pitch=target_pitch_rel, yaw=target_yaw_rel, pitch_speed=300, yaw_speed=300).wait_for_completed(timeout=0.5)
    except Exception: pass

    time.sleep(0.15)
    ACCEPT_PX = 25.0  # ลดระยะยอมรับลงจาก 35 เป็น 25 เพื่อให้เล็งกลางเป้าได้แม่นยำขึ้น
    aim_start_time = time.time()
    target_locked = False
    locked_start = 0

    while time.time() - aim_start_time < 3.0: # ให้เวลาเล็งนานขึ้นเป็น 3 วิ
        if not hub.running: break
        img = hub.get_frame_safe(retries=2, delay=0.01)
        if img is None: break
        
        with hub.lock: tof_dist = hub.tof_filtered
        found_targets = process_frame_multi_targets(img, tof_dist, target_color=color, target_shape_type=shape)

        if found_targets:
            t = found_targets[0]
            cx, cy = t["cx"], t["cy"]
            h_frame, w_frame, _ = img.shape
            center_x, center_y = w_frame / 2.0, h_frame / 2.0
            
            ex = cx - center_x
            ey = cy - (center_y + 32)  # AIM_Y_OFFSET_PX = 32

            accept_y_px = 32.0 if tof_dist <= 45.0 else 18.0

            if abs(ex) <= 18.0 and abs(ey) <= accept_y_px:
                if locked_start == 0:
                    locked_start = time.time()
                elif time.time() - locked_start > 0.4:
                    target_locked = True
                    break
            else:
                locked_start = 0
                dist_err = math.hypot(ex, ey)
                if dist_err > 80.0:
                    max_speed_yaw, max_speed_pitch = 35.0, 28.0
                elif dist_err > 30.0:
                    max_speed_yaw, max_speed_pitch = 18.0, 14.0
                else:
                    max_speed_yaw, max_speed_pitch = 8.0, 6.0

                yaw_corr = float(np.clip(ex * 0.10, -max_speed_yaw, max_speed_yaw))
                pitch_corr = float(np.clip(-ey * 0.10, -max_speed_pitch, max_speed_pitch))
            
            with sdk_lock:
                ep_gimbal.drive_speed(pitch_speed=pitch_corr, yaw_speed=yaw_corr)
        else:
            # ถ้ามองไม่เห็นเป้า ให้หยุดกิมบอลไว้ก่อน
            locked_start = 0
            with sdk_lock:
                ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                
        time.sleep(0.03)

    try:
        with sdk_lock: ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    except Exception: pass

    if ep_blaster_ref is not None and target_locked and hub.running:
        log_event(f"🔥 [BLASTER FIRING] Shooting IR Laser at target ({color.upper()} {shape})!")
        try:
            for _ in range(BURST_FIRE_COUNT):
                with sdk_lock:
                    ep_blaster_ref.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                time.sleep(0.15)
            log_event("[BLASTER FIRING] IR Firing completed successfully.")
            return True
        except Exception as e:
            log_event(f"[BLASTER ERROR] Firing failed: {e}")
            return False
    return False

def check_potential_color(img, target_color):
    if img is None: return False
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, target_color)
    if mask is None: return False
    if ROI_MASK_IMG is not None:
        resized_mask = cv2.resize(ROI_MASK_IMG, (w, h))
        mask = cv2.bitwise_and(mask, mask, mask=resized_mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            return True
    return False

def active_target_search(ep_chassis, ep_gimbal, target_yaw, dir_name, color, shape):
    global SHARP_PUSH_DIST_CM
    original_sharp = SHARP_PUSH_DIST_CM
    SHARP_PUSH_DIST_CM = 6.0  
    current_pitch = -10.0
    moved_y, moved_x = 0.0, 0.0
    found_target = None
    
    try:
        with sdk_lock:
            ep_gimbal.moveto(pitch=current_pitch, yaw=target_yaw, pitch_speed=300, yaw_speed=300).wait_for_completed(timeout=0.5)
        for step in range(8):
            if not hub.running: break
            time.sleep(0.15)
            img = hub.get_frame_safe(retries=2)
            if img is None: continue
            
            h, w, _ = img.shape
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = get_color_mask(hsv, color)
            if mask is None: continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) < 150: continue
            
            rx, ry, bw, bh = cv2.boundingRect(largest_cnt)
            touching_bottom = (ry + bh >= h - 15)
            
            with hub.lock: tof_dist = hub.tof_filtered
            targets = process_frame_multi_targets(img, tof_dist, color, shape)
            if targets and not touching_bottom:
                found_target = targets[0]
                break
                
            if touching_bottom:
                if current_pitch > -25.0:
                    current_pitch -= 5.0
                    with sdk_lock:
                        ep_gimbal.moveto(pitch=current_pitch, yaw=target_yaw, pitch_speed=150, yaw_speed=150).wait_for_completed(timeout=0.3)
                else:
                    strafe_step = 0.05
                    if dir_name == "AHEAD":
                        with sdk_lock: ep_chassis.move(x=-strafe_step, y=0, z=0, xy_speed=0.2).wait_for_completed(timeout=0.5)
                        moved_x -= strafe_step
                    elif dir_name == "RIGHT":
                        with sdk_lock: ep_chassis.move(x=0, y=strafe_step, z=0, xy_speed=0.2).wait_for_completed(timeout=0.5)
                        moved_y += strafe_step
                    elif dir_name == "LEFT":
                        with sdk_lock: ep_chassis.move(x=0, y=-strafe_step, z=0, xy_speed=0.2).wait_for_completed(timeout=0.5)
                        moved_y -= strafe_step
                    elif dir_name == "BACK":
                        with sdk_lock: ep_chassis.move(x=strafe_step, y=0, z=0, xy_speed=0.2).wait_for_completed(timeout=0.5)
                        moved_x += strafe_step
            elif len(targets) == 0:
                break
    finally:
        SHARP_PUSH_DIST_CM = original_sharp
    return found_target, moved_x, moved_y, current_pitch

def scan_and_process_targets(ep_gimbal, snap_yaw, current_gx, current_gy, target_color="red", target_shape="ALL_SHAPES"):
    scan_seq = [("AHEAD", 0), ("RIGHT", 90), ("BACK", 180), ("LEFT", -90)]
    tof_results = {}
    scan_queue = queue.Queue()

    hub.set_scanning(True)
    hub.set_status_msg("Scanning 4 Directions with Accurate ToF...")

    for dir_name, rel_y in scan_seq:
        if not hub.running: break
        raw_tof_cm = get_fresh_tof_distance(ep_gimbal, target_yaw=rel_y, pitch=0)
        
        dist_cm = calculate_ground_distance(raw_tof_cm)
        tof_results[dir_name] = dist_cm

        abs_dir_deg = normalize_angle(snap_yaw + rel_y)
        w_dir = deg_to_cardinal_exact(abs_dir_deg)

        grid_steps = int(round(dist_cm / CELL_SIZE_CM))
        grid_steps = max(1, min(grid_steps, 3))

        for step in range(0, grid_steps - 1):
            target_gx, target_gy = current_gx, current_gy
            if w_dir == "N": target_gy += step
            elif w_dir == "E": target_gx += step
            elif w_dir == "S": target_gy -= step
            elif w_dir == "W": target_gx -= step
            
            target_gx = int(np.clip(target_gx, 0, hub.w - 1))
            target_gy = int(np.clip(target_gy, 0, hub.h - 1))
            hub.update_wall_prob(target_gx, target_gy, w_dir, is_wall=False, is_direct_adjacent=(step == 0))

        final_gx, final_gy = current_gx, current_gy
        if w_dir == "N": final_gy += (grid_steps - 1)
        elif w_dir == "E": final_gx += (grid_steps - 1)
        elif w_dir == "S": final_gy -= (grid_steps - 1)
        elif w_dir == "W": final_gx -= (grid_steps - 1)

        final_gx = int(np.clip(final_gx, 0, hub.w - 1))
        final_gy = int(np.clip(final_gy, 0, hub.h - 1))

        if grid_steps == 1:
            is_wall_detected = dist_cm <= WALL_THRESHOLD_CM
        else:
            is_wall_detected = dist_cm < (grid_steps * CELL_SIZE_CM - 10.0)

        is_direct = (grid_steps == 1)
        hub.update_wall_prob(final_gx, final_gy, w_dir, is_wall=is_wall_detected, is_direct_adjacent=is_direct)

        if dist_cm <= (MAX_TARGET_DETECT_DIST_CM + TOF_OFFSET_HORIZ_CM):
            img = hub.get_frame_safe(retries=2, delay=0.01)
            if img is not None:
                triggers = process_frame_scan_all_colors(img, raw_tof_cm, main_color=target_color, main_shape=target_shape)
                if triggers:
                    for item in triggers:
                        item_yaw = rel_y + item["deg_offset"]
                        item_pitch = -item.get("deg_offset_y", 0.0)
                        scan_queue.put({
                            "dir_name": dir_name, "yaw": item_yaw, "pitch": item_pitch,
                            "color": item["color"], "is_main": item["is_main"],
                            "dist_cm": raw_tof_cm, "is_partial": False
                        })
                elif is_wall_detected:
                    colors_to_check = [target_color] if target_color != "ALL_COLORS" else ALL_COLORS
                    for c in colors_to_check:
                        if check_potential_color(img, c):
                            log_event(f"[PARTIAL COLOR] Detected potential target on wall ({dir_name})")
                            scan_queue.put({
                                "dir_name": dir_name, "yaw": rel_y, "pitch": -10.0,
                                "color": c, "is_main": True,
                                "dist_cm": raw_tof_cm, "is_partial": True
                            })
                            break

    if not scan_queue.empty() and hub.running:
        hub.set_status_msg("Target Lock & Long-Range Marking...")
        while not scan_queue.empty() and hub.running:
            item = scan_queue.get()
            target_yaw, target_pitch = item["yaw"], item["pitch"]
            t_color, is_main, t_dist = item["color"], item["is_main"], item["dist_cm"]
            is_partial = item.get("is_partial", False)
            dir_name = item.get("dir_name", "UNKNOWN")
            moved_x, moved_y = 0.0, 0.0
            
            if is_partial:
                log_event(f"[ACTIVE SEARCH] Engaging active search for partial target {t_color}")
                found_tgt, moved_x, moved_y, target_pitch = active_target_search(ep_chassis, ep_gimbal, target_yaw, dir_name, t_color, target_shape)
                if not found_tgt:
                    log_event("[ACTIVE SEARCH] Failed to confirm full target.")
                    if moved_x != 0.0 or moved_y != 0.0:
                        with sdk_lock: ep_chassis.move(x=-moved_x, y=-moved_y, z=0, xy_speed=0.2).wait_for_completed(timeout=1.0)
                    scan_queue.task_done()
                    continue

            with sdk_lock:
                ep_gimbal.moveto(pitch=target_pitch, yaw=target_yaw, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED).wait_for_completed(timeout=0.5)

            log_event(f"[TARGET LOCK] Verifying target ({t_color.upper()} {target_shape}) at Pitch:{target_pitch:.1f}°")
            time.sleep(0.15)

            valid_samples, tof_samples, last_bbox_sample, last_img_sample = [], [], None, None
            REQUIRED_MATCH_FRAMES, TOTAL_CHECK_FRAMES = 7, 10

            for frame_idx in range(TOTAL_CHECK_FRAMES):
                if not hub.running: break
                time.sleep(0.025)
                s_img = hub.get_frame_safe(retries=2, delay=0.01)
                
                with hub.lock: s_tof = hub.tof_filtered

                if s_img is not None:
                    s_targets = process_frame_multi_targets(s_img, s_tof, t_color, target_shape, verbose_log=False)
                    if s_targets:
                        s_res = s_targets[0]
                        valid_samples.append((s_res["color"], s_res["shape"]))
                        if 10.0 <= s_tof <= MAX_TARGET_DETECT_DIST_CM:
                            tof_samples.append(s_tof)
                        last_bbox_sample = s_res["bbox"]
                        last_img_sample = s_img.copy()

            success_count = len(valid_samples)
            if success_count >= REQUIRED_MATCH_FRAMES and hub.running:
                most_common = Counter(valid_samples).most_common(1)[0][0]
                conf_color, conf_shape = most_common
                
                raw_tof_final = float(np.median(tof_samples)) if tof_samples else hub.tof_filtered
                ground_tof_final = calculate_ground_distance(raw_tof_final)

                with hub.lock:
                    real_chassis_yaw = hub.pose["chassis_yaw"]
                
                actual_world_yaw = normalize_angle(real_chassis_yaw + target_yaw)
                is_registered = hub.register_target_spatial(
                    abs_yaw_deg=actual_world_yaw, tof_dist_cm=raw_tof_final,
                    color=conf_color, shape=conf_shape, is_main_target=is_main,
                )

                if last_img_sample is not None and last_bbox_sample is not None:
                    t_info = {"target_id": hub.target_id_counter - 1, "color": conf_color, "shape": conf_shape}
                    save_target_inspection_image(last_img_sample, t_info, last_bbox_sample, ground_tof_final, status="PASSED" if is_registered else "REJECTED_DUP")

                if ground_tof_final <= (MAX_FIRING_DIST_CM + TOF_OFFSET_HORIZ_CM) and is_main:
                    shot_success = aim_and_fire_target_close_range(ep_gimbal, target_yaw, target_pitch, conf_color, conf_shape)
                    if shot_success:
                        with hub.lock:
                            if is_registered and hub.targets_found:
                                hub.targets_found[-1]["is_knocked_down"] = True
                            elif not is_registered and hub.targets_found:
                                for t in reversed(hub.targets_found):
                                    if t["color"] == conf_color:
                                        t["is_knocked_down"] = True
                                        break

                log_event(f"[TARGET MARKED FROM DISTANCE] Target registered at {ground_tof_final:.1f} cm from center")

            if is_partial and (moved_x != 0.0 or moved_y != 0.0):
                log_event("[ACTIVE SEARCH] Restoring original position...")
                with sdk_lock: ep_chassis.move(x=-moved_x, y=-moved_y, z=0, xy_speed=0.2).wait_for_completed(timeout=1.5)

            scan_queue.task_done()

    recenter_gimbal_hardware(ep_gimbal)
    hub.set_active_detections([])
    hub.set_scanning(False)
    return tof_results

# ==========================================
# 8. PURE IMU CLOSED-LOOP TURNING & GUARDRAIL MOVEMENT
# ==========================================
def stop_chassis_active_brake(ep_chassis, brake_duration=0.10):
    try:
        with sdk_lock:
            ep_chassis.drive_speed(x=-0.12, y=0, z=0)
        time.sleep(0.05)
        with sdk_lock:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(brake_duration)
        with sdk_lock:
            ep_chassis.drive_speed(x=0, y=0, z=0)
    except Exception: pass

def align_center_in_cell(ep_chassis):
    """🌟 ปรับตำแหน่งหุ่นยนต์ให้อยู่กึ่งกลางช่องหลังจากการสแกน ToF เสร็จสิ้นแล้ว"""
    if not hub.running: return
    state = hub.get_state()
    l_sharp, r_sharp = state["sharp_left"], state["sharp_right"]

    if l_sharp > MAX_SHARP_WALL_DETECT_CM and r_sharp > MAX_SHARP_WALL_DETECT_CM:
        return

    shift_cm = 0.0
    if l_sharp < MAX_SHARP_WALL_DETECT_CM and r_sharp < MAX_SHARP_WALL_DETECT_CM:
        error = r_sharp - l_sharp
        if abs(error) > 3.0:
            shift_cm = error * 0.35
    elif l_sharp < 12.0:
        shift_cm = (13.5 - l_sharp)
    elif r_sharp < 12.0:
        shift_cm = -(13.5 - r_sharp)

    if abs(shift_cm) <= 1.0:
        return

    shift_cm = float(np.clip(shift_cm, -4.5, 4.5))
    shift_m = shift_cm / 100.0
    
    speed_y = 0.15 if shift_m > 0 else -0.15   
    duration = abs(shift_m) / 0.15
    
    log_event(f"[CENTERING MICRO] Sharp IR (L:{l_sharp:.1f}, R:{r_sharp:.1f}) | Gentle Shift: {shift_cm:.1f} cm")
    
    with sdk_lock:
        ep_chassis.drive_speed(x=0, y=speed_y, z=0)
    time.sleep(duration)
    stop_chassis_active_brake(ep_chassis, brake_duration=0.10)

def turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw_deg):
    if not hub.running: return False

    target_yaw_deg = snap_to_cardinal_yaw(target_yaw_deg)
    recenter_gimbal_hardware(ep_gimbal)
    
    with hub.lock:
        hub.is_rotating = True
        hub.emergency_stop = False
        start_yaw = hub.pose["chassis_yaw"]

    diff = normalize_angle(target_yaw_deg - start_yaw)
    abs_diff = abs(diff)

    if abs_diff <= 1.5:
        with hub.lock: hub.is_rotating = False
        return True

    log_event(f"[TURNING START] Target: {target_yaw_deg:.0f}° | Current: {start_yaw:.1f}° | Diff: {diff:.1f}°")

    start_t = time.time()
    timeout_limit = 4.0 if abs_diff > 120.0 else 2.5

    while time.sleep(0.015) is None and (time.time() - start_t < timeout_limit):
        state = hub.get_state()
        if not state["running"] or state["emergency_stop"]:
            stop_chassis_active_brake(ep_chassis, brake_duration=0.1)
            return False
            
        curr_yaw = state["pose"]["chassis_yaw"]
        err = normalize_angle(target_yaw_deg - curr_yaw)
        abs_err = abs(err)
        
        if abs_err <= 1.5:
            break

        z_speed = float(np.clip(err * 1.8, -80.0, 80.0))
        min_z = 8.0 if abs_err <= 5.0 else (14.0 if abs_err <= 15.0 else 22.0)
        
        if abs(z_speed) < min_z:
            z_speed = min_z if z_speed > 0 else -min_z

        with sdk_lock:
            ep_chassis.drive_speed(x=0, y=0, z=z_speed)

    stop_chassis_active_brake(ep_chassis, brake_duration=0.15)
    recenter_gimbal_hardware(ep_gimbal)

    with hub.lock:
        hub.is_rotating = False

    log_event(f"[TURNING COMPLETED] Settled Heading: {hub.pose['chassis_yaw']:.1f}°")
    return True

def move_forward_60cm_straight(ep_chassis, target_step_m=move_target_step_m, target_speed=target_speed_mps):
    """🌟 ใช้หลัก Sharp IR ผลักออกห่างอย่างเดียว (Guardrail Only) ช่วยแก้ปัญหาหุ่นยนต์เดินแกว่งไปแกว่งมา"""
    calibrated_target_m = target_step_m * DISTANCE_CALIB_FACTOR
    with hub.lock:
        start_pos_x = hub.latest_raw_pos[0] if hub.latest_raw_pos[0] is not None else 0.0
        start_pos_y = hub.latest_raw_pos[1] if hub.latest_raw_pos[1] is not None else 0.0
        # ล็อกมุมเป้าหมายให้อ้างอิงจากมุมปัจจุบัน (ที่อาจจะถูกจัดให้ขนานกำแพงมาแล้ว) เพื่อไม่ให้ IMU ตีกับ Wall Alignment
        target_yaw = hub.pose["chassis_yaw"]

    effective_distance_traveled = 0.0
    max_timeout = (calibrated_target_m / target_speed) + 0.8
    start_time = time.time()

    prev_x, prev_y = start_pos_x, start_pos_y

    while effective_distance_traveled < calibrated_target_m and (time.time() - start_time < max_timeout):
        state = hub.get_state()
        if not state["running"] or state["emergency_stop"]:
            stop_chassis_active_brake(ep_chassis, brake_duration=0.15)
            return False

        l_sharp, r_sharp = state["sharp_left"], state["sharp_right"]
        tof_fast = state["tof_fast_ema"]
        ground_tof_fast = calculate_ground_distance(tof_fast)

        with hub.lock:
            curr_x = hub.latest_raw_pos[0] if hub.latest_raw_pos[0] is not None else prev_x
            curr_y = hub.latest_raw_pos[1] if hub.latest_raw_pos[1] is not None else prev_y

        step_dist = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        prev_x, prev_y = curr_x, curr_y

        curr_yaw = state["pose"]["chassis_yaw"]
        error_yaw = normalize_angle(target_yaw - curr_yaw)
        cos_correction = math.cos(math.radians(error_yaw))
        effective_distance_traveled += step_dist * cos_correction

        SAFE_STOP_GROUND_CM = 38.0
        if ground_tof_fast <= SAFE_STOP_GROUND_CM:
            log_event(f"[FAST SAFETY BRAKE] Front wall detected! Ground dist: {ground_tof_fast:.1f} cm <= {SAFE_STOP_GROUND_CM} cm | Traveled: {effective_distance_traveled*100:.1f} cm")
            stop_chassis_active_brake(ep_chassis, brake_duration=0.15)
            return effective_distance_traveled >= (calibrated_target_m * 0.60)

        # 1. Proportional Yaw Holding (รักษามุม IMU เป็นหลัก)
        z_val = max(min(error_yaw * KP_YAW_HOLD, 30.0), -30.0)

        # 2. Deceleration Zone (ชะลอความเร็วช่วง 18 cm สุดท้าย)
        remaining_m = calibrated_target_m - effective_distance_traveled
        v_forward = target_speed
        if remaining_m < DECEL_ZONE_M:
            v_forward = max(MIN_V, target_speed * (remaining_m / DECEL_ZONE_M))

        # 🌟 3. Sharp IR Guardrail Safety (ผลักออกห่างอย่างเดียวเฉพาะเมื่อชิดเกินไป < 11.0 cm)
        y_speed = 0.0
        # เปลี่ยนเป็น Proportional Control (P-Control) สำหรับสไลด์หลบ เพื่อลดอาการส่าย (Ping-Pong)
        if l_sharp < SHARP_PUSH_DIST_CM and l_sharp <= r_sharp:
            error_dist = SHARP_PUSH_DIST_CM - l_sharp
            y_speed = float(np.clip(error_dist * 0.025 + 0.02, 0.03, 0.15))
        elif r_sharp < SHARP_PUSH_DIST_CM:
            error_dist = SHARP_PUSH_DIST_CM - r_sharp
            y_speed = -float(np.clip(error_dist * 0.025 + 0.02, 0.03, 0.15))

        with sdk_lock:
            ep_chassis.drive_speed(x=v_forward, y=y_speed, z=z_val)
        time.sleep(0.05)

    stop_chassis_active_brake(ep_chassis, brake_duration=0.10)
    return effective_distance_traveled >= (calibrated_target_m * 0.65)

# ==========================================
# 9. A* PATHFINDING
# ==========================================
def a_star_search(start, target, grid_w, grid_h):
    def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
            neighbor = (nx, ny)
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                if not hub.has_wall_between(current, neighbor):
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float("inf")):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, target)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def find_reachable_frontier(start_x, start_y, grid_w, grid_h):
    state = hub.get_state()
    visited_map = state["visited"]

    visited = np.zeros_like(visited_map, dtype=bool)
    queue_nodes = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue_nodes:
        cx, cy = queue_nodes.popleft()
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h and not visited[ny][nx]:
                if not hub.has_wall_between((cx, cy), (nx, ny)):
                    visited[ny][nx] = True
                    if visited_map[ny][nx] == 0:
                        target_frontier = (nx, ny)
                        path = a_star_search((start_x, start_y), target_frontier, grid_w, grid_h)
                        if path: return target_frontier, path
                    elif visited_map[ny][nx] == 1:
                        queue_nodes.append((nx, ny))
    return None, None

# ==========================================
# 10. WORKERS & EXPLORATION PIPELINE
# ==========================================
def save_map_image(filename="final_map.png"):
    hub.request_map_save(filename)

def mapping_worker(ep_sensor, ep_gimbal, ep_chassis):
    ep_sensor.sub_distance(freq=TOF_FREQ_HZ, callback=tof_data_handler)
    ep_gimbal.sub_angle(freq=20, callback=gimbal_data_handler)
    ep_chassis.sub_position(freq=20, callback=chassis_pos_handler)
    ep_chassis.sub_attitude(freq=20, callback=chassis_att_handler)

    while hub.get_state()["running"]: time.sleep(0.01)

def exploration_worker(ep_chassis, ep_gimbal):
    while not hub.get_state()["is_calibrated"]: time.sleep(0.05)

    dir_offsets = {0: (0, 1), 90: (1, 0), -180: (0, -1), 180: (0, -1), -90: (-1, 0)}
    wall_dir_map = {0: "N", 90: "E", -180: "S", 180: "S", -90: "W"}
    step_count = 0
    start_exploration_time = time.time()

    while hub.get_state()["running"]:
        try:
            elapsed_time = time.time() - start_exploration_time
            if elapsed_time > MAX_ROUND1_TIME_SEC:
                log_event(f"⏱️ [TIME LIMIT REACHED] Approaching 10-minute limit ({elapsed_time:.1f}s). Completing Round 1 safely...")
                hub.set_status_msg("TIME LIMIT REACHED (9.5 Min) - Finalizing Round 1")
                save_map_image("final_map.png")
                hub.export_map_data()
                break

            state = hub.get_state()
            current_gx, current_gy = state["grid_x"], state["grid_y"]
            p = state["pose"]

            raw_yaw = p["chassis_yaw"]
            snap_yaw = int(normalize_angle(round(raw_yaw / 90.0) * 90))

            step_count += 1
            log_event(f"--- STEP {step_count} | Grid({current_gx}, {current_gy}) | Time: {elapsed_time:.1f}s ---")

            # 🌟 STEP A: สแกน ToF 4 ทิศทางและตรวจจับเป้าหมายด้วยกล้อง
            tof_dict = scan_and_process_targets(
                ep_gimbal, snap_yaw, current_gx, current_gy,
                target_color=TARGET_COLOR, target_shape=TARGET_SHAPE,
            )

            # 🌟 STEP B: ปรับตำแหน่งตัวรถให้อยู่กึ่งกลางช่อง (Centering)
            align_center_in_cell(ep_chassis)

            # 🌟 STEP C: ปรับองศาเอียงตัวรถให้ตรงขนานกำแพงด้วย ToF (Parallel Alignment)
            align_wall_tof_gimbal(ep_chassis, ep_gimbal, tof_dict)

            wall_results = {
                "N": state["N_confirmed"][current_gy, current_gx] or state["N_predicted"][current_gy, current_gx],
                "E": state["E_confirmed"][current_gy, current_gx] or state["E_predicted"][current_gy, current_gx],
                "S": (hub.N_wall_logodds[current_gy - 1, current_gx] > 0.0) if current_gy > 0 else (hub.S_outer_wall_logodds[current_gx] > 0.0),
                "W": (hub.E_wall_logodds[current_gy, current_gx - 1] > 0.0) if current_gx > 0 else (hub.W_outer_wall_logodds[current_gy] > 0.0),
            }
            log_telemetry_csv(step_count, current_gx, current_gy, raw_yaw, tof_dict, wall_results)

            target_frontier, path = find_reachable_frontier(current_gx, current_gy, GRID_W, GRID_H)
            
            if path and len(path) > 0 and path[0] == (current_gx, current_gy):
                path.pop(0)

            if not target_frontier or not path or len(path) == 0:
                log_event("[FRONTIER CHECK] Unlocking predicted walls to reach remaining unvisited cells...")
                hub.reset_predicted_walls_for_unvisited()
                target_frontier, path = find_reachable_frontier(current_gx, current_gy, GRID_W, GRID_H)
                if path and len(path) > 0 and path[0] == (current_gx, current_gy):
                    path.pop(0)

            if not target_frontier or not path or len(path) == 0:
                log_event("[FRONTIER FALLBACK] Searching backup path to remaining unvisited cells...")
                curr_state = hub.get_state()
                unvisited_cells = np.argwhere(curr_state["visited"] == 0)

                for gy, gx in unvisited_cells:
                    fallback_path = a_star_search((current_gx, current_gy), (gx, gy), GRID_W, GRID_H)
                    if fallback_path and len(fallback_path) > 0:
                        if fallback_path[0] == (current_gx, current_gy):
                            fallback_path.pop(0)
                        if len(fallback_path) > 0:
                            path = fallback_path
                            target_frontier = (gx, gy)
                            log_event(f"[FALLBACK PATH FOUND] Pathing to backup unvisited grid ({gx}, {gy})")
                            break

            if not target_frontier or not path or len(path) == 0:
                log_event("[EXPLORATION COMPLETE] All reachable frontiers visited. Stopping at current position.")
                hub.set_status_msg("EXPLORATION COMPLETE - Map Exported Successfully")
                save_map_image("final_map.png")
                hub.export_map_data()
                break

            next_cell = path[0]
            dx, dy = next_cell[0] - current_gx, next_cell[1] - current_gy

            target_yaw = None
            for angle, offset in dir_offsets.items():
                if offset == (dx, dy):
                    target_yaw = angle
                    break

            if target_yaw is not None and hub.running:
                target_snap_yaw = int(normalize_angle(round(target_yaw / 90.0) * 90))
                
                if snap_yaw != target_snap_yaw:
                    turn_success = turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw)
                    
                    if not turn_success and hub.running:
                        log_event("[TURN STUCK RECOVERY] Robot stuck during turn -> Backing up 12 cm")
                        try:
                            with sdk_lock:
                                ep_chassis.move(x=-0.12, y=0, z=0, xy_speed=0.15).wait_for_completed(timeout=1.5)
                            stop_chassis_active_brake(ep_chassis, brake_duration=0.1)
                        except Exception: pass

                        turn_success = turn_to_orientation_imu(ep_chassis, ep_gimbal, target_yaw)
                        
                        if not turn_success:
                            log_event(f"[MOVE SKIPPED] Turn failed after recovery -> Updating wall prob and retrying")
                            if target_yaw in wall_dir_map:
                                hub.update_wall_prob(current_gx, current_gy, wall_dir_map[target_yaw], is_wall=True, is_direct_adjacent=False)
                            continue

                if hub.running:
                    move_success = move_forward_60cm_straight(ep_chassis)

                    if move_success:
                        hub.lock_passage((current_gx, current_gy), next_cell)
                        hub.set_grid_pos(next_cell[0], next_cell[1])
                    else:
                        log_event(f"[MOVE BLOCKED] Failed to move into cell {next_cell} -> Updating wall logodds softly")
                        if target_yaw in wall_dir_map:
                            hub.update_wall_prob(current_gx, current_gy, wall_dir_map[target_yaw], is_wall=True, is_direct_adjacent=False)

            time.sleep(0.01)

        except Exception as e:
            err_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            log_event(f"💥 [EXPLORATION THREAD CRASH] Exception in exploration loop:\n{err_msg}")
            hub.set_status_msg("EXPLORATION CRASHED - See log")
            break

# ==========================================
# 11. PYGAME HD GUI SYSTEM (DYNAMIC SCALING 1280 x 720)
# ==========================================
COLOR_BG = (20, 24, 30)
COLOR_GRID_BG = (35, 40, 50)
COLOR_GRID_LINE = (65, 72, 85)
COLOR_UNVISITED = (60, 66, 78)
COLOR_VISITED = (46, 204, 113)
COLOR_HAS_TARGET_GRID = (231, 76, 60)
COLOR_WALL_CONFIRMED = (231, 76, 60)
COLOR_WALL_PREDICTED = (52, 152, 219)
COLOR_ROBOT = (241, 196, 15)
COLOR_PATH_LINE = (255, 165, 0)
COLOR_TEXT = (240, 240, 240)
COLOR_PANEL = (28, 33, 42)

def draw_star(surface, color, center, radius):
    pts = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius / 2.0
        angle = i * math.pi / 5 - math.pi / 2
        pts.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)

def run_pygame_gui():
    pygame.init()
    pygame.font.init()

    font_title = pygame.font.SysFont("Tahoma", 20, bold=True)
    font_large = pygame.font.SysFont("Tahoma", 16, bold=True)
    font_small = pygame.font.SysFont("Tahoma", 13)

    WIN_W, WIN_H = 1280, 720
    MARGIN = 20
    WALL_THICKNESS = 4

    MAX_MAP_W = 540
    MAX_MAP_H = 490
    CELL_PIXELS = max(35, min(MAX_MAP_H // GRID_H, MAX_MAP_W // GRID_W, 90))

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RoboMaster EP - SLAM Dashboard (HD Widescreen)")
    clock = pygame.time.Clock()

    map_px_w = GRID_W * CELL_PIXELS
    map_px_h = GRID_H * CELL_PIXELS
    map_origin_x = MARGIN
    map_origin_y = MARGIN + map_px_h

    right_panel_x = map_origin_x + map_px_w + 25
    right_panel_w = WIN_W - right_panel_x - MARGIN
    VIDEO_W = min(640, right_panel_w)
    VIDEO_H = int(VIDEO_W * 9.0 / 16.0)

    while hub.get_state()["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                try:
                    pygame.display.flip()
                    pygame.image.save(screen, "final_map.png")
                    log_event(f"📸 [MAP SAVED] Snapshot saved on exit")
                except: pass
                hub.export_map_data()
                hub.stop_workers()
                return

        with hub.lock:
            if hub.save_map_flag:
                save_fn = hub.pending_map_save_filename
                try:
                    pygame.display.flip()
                    pygame.image.save(screen, save_fn)
                    log_event(f"📸 [MAP SAVED] Snapshot saved cleanly as '{save_fn}'")
                except Exception as e:
                    log_event(f"⚠️ [MAP SAVE ERROR] {e}")
                hub.save_map_flag = False

        state = hub.get_state()
        visited = state["visited"]
        has_target_grid = state["has_target_grid"]
        N_conf, N_pred = state["N_confirmed"], state["N_predicted"]
        E_conf, E_pred = state["E_confirmed"], state["E_predicted"]
        S_out_conf, S_out_pred = state["S_outer_confirmed"], state["S_outer_predicted"]
        W_out_conf, W_out_pred = state["W_outer_confirmed"], state["W_outer_predicted"]
        p = state["pose"]
        path_history = state["robot_path_history"]

        screen.fill(COLOR_BG)

        # --- 1. LEFT PANEL: SLAM MAP ---
        pygame.draw.rect(screen, COLOR_GRID_BG, (map_origin_x, MARGIN, map_px_w, map_px_h))

        for gy in range(GRID_H):
            for gx in range(GRID_W):
                px = map_origin_x + (gx * CELL_PIXELS)
                py = map_origin_y - ((gy + 1) * CELL_PIXELS)
                cell_rect = (px + 2, py + 2, CELL_PIXELS - 4, CELL_PIXELS - 4)

                if has_target_grid[gy][gx]: color = COLOR_HAS_TARGET_GRID
                elif visited[gy][gx] == 1: color = COLOR_VISITED
                else: color = COLOR_UNVISITED

                pygame.draw.rect(screen, color, cell_rect, border_radius=3)
                lbl = font_small.render(f"({gx},{gy})", True, (180, 180, 180))
                screen.blit(lbl, (px + 4, py + 4))

        for gx in range(GRID_W + 1):
            x = map_origin_x + (gx * CELL_PIXELS)
            pygame.draw.line(screen, COLOR_GRID_LINE, (x, MARGIN), (x, MARGIN + map_px_h), 2)
        for gy in range(GRID_H + 1):
            y = map_origin_y - (gy * CELL_PIXELS)
            pygame.draw.line(screen, COLOR_GRID_LINE, (MARGIN, y), (MARGIN + map_px_w, y), 2)

        if len(path_history) >= 2:
            px_points = [
                (int(map_origin_x + (gx + 0.5) * CELL_PIXELS), int(map_origin_y - (gy + 0.5) * CELL_PIXELS))
                for gx, gy in path_history
            ]
            pygame.draw.lines(screen, COLOR_PATH_LINE, False, px_points, 3)

        for gy in range(GRID_H):
            for gx in range(GRID_W):
                x_left = map_origin_x + (gx * CELL_PIXELS)
                x_right = x_left + CELL_PIXELS
                y_bottom = map_origin_y - (gy * CELL_PIXELS)
                y_top = y_bottom - CELL_PIXELS

                if N_conf[gy, gx]: pygame.draw.line(screen, COLOR_WALL_CONFIRMED, (x_left, y_top), (x_right, y_top), WALL_THICKNESS)
                elif N_pred[gy, gx]: pygame.draw.line(screen, COLOR_WALL_PREDICTED, (x_left, y_top), (x_right, y_top), WALL_THICKNESS)

                if E_conf[gy, gx]: pygame.draw.line(screen, COLOR_WALL_CONFIRMED, (x_right, y_top), (x_right, y_bottom), WALL_THICKNESS)
                elif E_pred[gy, gx]: pygame.draw.line(screen, COLOR_WALL_PREDICTED, (x_right, y_top), (x_right, y_bottom), WALL_THICKNESS)

        for gx in range(GRID_W):
            x_left, x_right = map_origin_x + (gx * CELL_PIXELS), map_origin_x + ((gx + 1) * CELL_PIXELS)
            if S_out_conf[gx]: pygame.draw.line(screen, COLOR_WALL_CONFIRMED, (x_left, map_origin_y), (x_right, map_origin_y), WALL_THICKNESS)
            elif S_out_pred[gx]: pygame.draw.line(screen, COLOR_WALL_PREDICTED, (x_left, map_origin_y), (x_right, map_origin_y), WALL_THICKNESS)

        for gy in range(GRID_H):
            y_bottom, y_top = map_origin_y - (gy * CELL_PIXELS), map_origin_y - ((gy + 1) * CELL_PIXELS)
            if W_out_conf[gy]: pygame.draw.line(screen, COLOR_WALL_CONFIRMED, (map_origin_x, y_top), (map_origin_x, y_bottom), WALL_THICKNESS)
            elif W_out_pred[gy]: pygame.draw.line(screen, COLOR_WALL_PREDICTED, (map_origin_x, y_top), (map_origin_x, y_bottom), WALL_THICKNESS)

        sorted_targets = sorted(state["targets_found"], key=lambda t: 1 if t.get("is_main_target") else 0)
        for t in sorted_targets:
            raw_px = map_origin_x + ((t["world_pos_cm"]["x"] / CELL_SIZE_CM) * CELL_PIXELS)
            raw_py = map_origin_y - ((t["world_pos_cm"]["y"] / CELL_SIZE_CM) * CELL_PIXELS)

            star_px = max(map_origin_x, min(raw_px, map_origin_x + map_px_w))
            star_py = max(map_origin_y - map_px_h, min(raw_py, map_origin_y))

            is_main = t.get("is_main_target")
            star_color = COLOR_MAIN_STAR if is_main else COLOR_SUB_STAR
            star_radius = max(8, int(CELL_PIXELS * 0.16)) if is_main else max(5, int(CELL_PIXELS * 0.10))

            draw_star(screen, star_color, (int(star_px), int(star_py)), radius=star_radius)
            t_lbl = font_small.render(f"T{t['target_id']}:{t['color'][0].upper()}", True, (255, 255, 255))
            screen.blit(t_lbl, (int(star_px) + 8, int(star_py) - 8))

        r_gx, r_gy = state["grid_x"] + 0.5, state["grid_y"] + 0.5
        rx_px = map_origin_x + (r_gx * CELL_PIXELS)
        ry_px = map_origin_y - (r_gy * CELL_PIXELS)
        pygame.draw.circle(screen, COLOR_ROBOT, (int(rx_px), int(ry_px)), max(10, int(CELL_PIXELS * 0.18)))

        total_yaw = normalize_angle(p["chassis_yaw"] + state["gimbal_yaw"])
        yaw_rad = math.radians(total_yaw)
        dir_x = rx_px + (CELL_PIXELS * 0.3) * math.sin(yaw_rad)
        dir_y = ry_px - (CELL_PIXELS * 0.3) * math.cos(yaw_rad)
        pygame.draw.line(screen, (255, 255, 255), (rx_px, ry_px), (dir_x, dir_y), 3)

        status_panel_y = map_origin_y + 12
        status_panel_h = WIN_H - status_panel_y - MARGIN
        pygame.draw.rect(screen, COLOR_PANEL, (map_origin_x, status_panel_y, map_px_w, status_panel_h), border_radius=8)
        
        screen.blit(font_large.render("ROBOT TELEMETRY", True, (241, 196, 15)), (map_origin_x + 12, status_panel_y + 8))
        sub_info1 = f"Grid: ({state['grid_x']}, {state['grid_y']})  |  Heading: {p['chassis_yaw']:.1f}°"
        sub_info2 = f"ToF Ground: {calculate_ground_distance(state['filtered_tof']):.1f} cm  |  Targets: {len(state['targets_found'])}"
        sub_info3 = f"Sharp L/R: {state['sharp_left']:.1f} / {state['sharp_right']:.1f} cm"
        screen.blit(font_small.render(sub_info1, True, COLOR_TEXT), (map_origin_x + 12, status_panel_y + 32))
        screen.blit(font_small.render(sub_info2, True, COLOR_TEXT), (map_origin_x + 12, status_panel_y + 54))
        screen.blit(font_small.render(sub_info3, True, COLOR_TEXT), (map_origin_x + 12, status_panel_y + 76))

        # --- 2. RIGHT PANEL: LARGE VIDEO STREAM & RECENT LOGS ---
        video_rect = (right_panel_x, MARGIN, VIDEO_W, VIDEO_H)
        pygame.draw.rect(screen, (10, 10, 10), video_rect, border_radius=8)

        latest_img = hub.get_frame_safe(retries=1, delay=0.005)
        if latest_img is not None:
            try:
                rgb_img = cv2.cvtColor(latest_img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(rgb_img, (VIDEO_W, VIDEO_H))
                surface_img = pygame.surfarray.make_surface(resized_img.swapaxes(0, 1))
                screen.blit(surface_img, (right_panel_x, MARGIN))

                with hub.lock:
                    tof_val = hub.tof_filtered
                ground_dist = calculate_ground_distance(tof_val) if tof_val < 800.0 else 999.0
                
                if ground_dist < 800.0 and ground_dist > 0.0:
                    crop_ratio = float(np.clip(0.12 + (ground_dist - 30.0) * 0.0040, 0.10, 0.48))
                else:
                    crop_ratio = 0.35
                
                crop_h = int(VIDEO_H * crop_ratio)
                overlay = pygame.Surface((VIDEO_W, VIDEO_H), pygame.SRCALPHA)

                if crop_h > 0:
                    trap_pts = [
                        (0, 0),
                        (VIDEO_W, 0),
                        (int(VIDEO_W * 0.88), crop_h),
                        (int(VIDEO_W * 0.12), crop_h)
                    ]
                    pygame.draw.polygon(overlay, (255, 0, 0, 60), trap_pts)
                    pygame.draw.polygon(overlay, (255, 230, 0, 255), trap_pts, width=2)

                # Removed Floor Reflection Mask from GUI


                screen.blit(overlay, (right_panel_x, MARGIN))
                lbl_crop = font_small.render("MULTI-CANDIDATE SCORE + FLOOR REFLECTION MASK", True, (255, 230, 0))
                screen.blit(lbl_crop, (right_panel_x + 10, MARGIN + 5))

            except Exception: pass

        pygame.draw.rect(screen, (241, 196, 15), video_rect, width=2, border_radius=8)

        logs_panel_y = MARGIN + VIDEO_H + 15
        logs_panel_h = WIN_H - logs_panel_y - MARGIN
        pygame.draw.rect(screen, COLOR_PANEL, (right_panel_x, logs_panel_y, VIDEO_W, logs_panel_h), border_radius=8)

        screen.blit(font_title.render("SYSTEM STATUS & LOGS", True, (241, 196, 15)), (right_panel_x + 12, logs_panel_y + 10))
        screen.blit(font_small.render(f"Current Action: {state['status_msg']}", True, (52, 152, 219)), (right_panel_x + 12, logs_panel_y + 36))

        log_y = logs_panel_y + 62
        for log_msg in list(recent_logs)[::-1]:
            screen.blit(font_small.render(log_msg[-70:], True, (200, 200, 200)), (right_panel_x + 12, log_y))
            log_y += 22

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# ==========================================
# 12. GLOBAL CRASH-HANDLING & ENTRY POINT
# ==========================================
def handle_global_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log_event(f"💥 [FATAL CRASH DETECTED] Unhandled exception occurred:\n{err_msg}")
    save_map_image("final_map_crash.png")
    hub.export_map_data()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_global_exception

if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_robot_ref = ep_robot

    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_blaster_ref = getattr(ep_robot, 'blaster', None)

    log_event("[SYSTEM INIT] Setting Gimbal to Front (Fast Moveto)...")
    recenter_gimbal_hardware(ep_gimbal)

    t1 = threading.Thread(target=mapping_worker, args=(ep_sensor, ep_gimbal, ep_chassis), daemon=True)
    t2 = threading.Thread(target=exploration_worker, args=(ep_chassis, ep_gimbal), daemon=True)
    t3 = threading.Thread(target=safe_camera_worker, args=(ep_camera,), daemon=True)
    t4 = threading.Thread(target=sharp_sensor_worker, args=(ep_robot,), daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    log_event("[CALIBRATION] Calibrating Local Frame...")
    if hub.calibrate_origin():
        log_event("[CALIBRATION] Calibration Complete!")
    else:
        log_event("[CALIBRATION] Calibration timeout, proceeding fallback.")

    try:
        run_pygame_gui()
    except KeyboardInterrupt:
        log_event("[USER STOP] Emergency stop triggered by user...")
    except Exception as e:
        log_event(f"💥 [GUI CRASH] Pygame error: {e}")
    finally:
        log_event("[SYSTEM TERMINATION] Saving final map snapshot and telemetry...")
        save_map_image("final_map.png")
        hub.export_map_data()
        stop_chassis_active_brake(ep_chassis, brake_duration=0.2)
        hub.stop_workers()
        time.sleep(0.3)
        try:
            ep_sensor.unsub_distance()
            ep_gimbal.unsub_angle()
            ep_chassis.unsub_position()
            ep_chassis.unsub_attitude()
            ep_robot.close()
        except Exception: pass
        log_event("[SYSTEM] Connection closed cleanly. Final map saved as 'final_map.png'.")