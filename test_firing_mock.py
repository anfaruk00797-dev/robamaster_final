"""
╔══════════════════════════════════════════════════════════════════════╗
║   test_firing_mock.py  – Adaptive Speed Aiming & Dynamic ROI Mask    ║
║   - Adaptive Speed Aiming: Fast when far, smooth slow when close     ║
║   - Dynamic ROI Mask: Blackout shot targets to prevent re-aiming     ║
║   - Pygame GUI Checkbox Setup: Per-Color Shape Rules Configurator    ║
║   - Strict Target Pair Matching: Only fires target matching rules    ║
║   - Clears Queue completely for EACH wall before turning to next wall║
║   - Tracks chassis retreat distance during cut-off resolution        ║
║   - Moves Chassis BACK to origin position after firing per wall      ║
║   - Sequential Gimbal Move & Real-time GUI ROI Overlay               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from collections import Counter, deque
import queue as _queue
import gc

import cv2
import numpy as np
import pygame
from robomaster import robot, blaster

# ==========================================
# 1. LOGGING & SYSTEM CONFIG
# ==========================================
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_FILE = "firing_test_log.txt"
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
stream_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[stream_handler, file_handler],
)

def log(msg):
    logging.info(msg)
    file_handler.flush()

TARGET_IMG_DIR = "firing_test_targets"
os.makedirs(TARGET_IMG_DIR, exist_ok=True)

sdk_lock = threading.Lock()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. TUNABLE PARAMETERS & PER-COLOR MAPPING
# ==========================================
TARGET_RULES = {
    "red":    [],
    "yellow": [],
    "green":  [],
    "blue":   []
}
BURST_FIRE_COUNT = 2

MAX_FIRING_DIST_CM        = 110.0
MAX_TARGET_DETECT_DIST_CM = 130.0

SAFE_CLEARANCE_PITCH = 0.0  # ยก Pitch มาขนานพื้นก่อนหมุน Yaw

AIM_ACCEPT_PX   = 18.0
AIM_TIMEOUT_SEC  = 3.5
AIM_YAW_GAIN    = 0.10
AIM_PITCH_GAIN  = 0.10
AIM_Y_OFFSET_PX  = 32

REQUIRED_MATCH_FRAMES = 7
TOTAL_CHECK_FRAMES    = 10

GIMBAL_FAST_SPEED   = 540
GIMBAL_MEDIUM_SPEED = 320
GIMBAL_SLOW_SPEED   = 140
TOF_OFFSET_HORIZ_CM = 7.5
CAMERA_FOCAL_LENGTH_PX = 650.0

SCAN_DIRECTIONS = [
    ("AHEAD",  0),
    ("RIGHT", 90),
    ("BACK",  180),
    ("LEFT",  -90),
]

WALL_THRESHOLD_CM  = 110.0
INITIAL_SCAN_PITCH = -3.5
WALL_DOWN_PITCH    = -22.0  # ก้มสุดเมื่อเจอกำแพง
RETREAT_SAFE_MIN_CM = 15.0  # ระยะปลอดภัยขั้นต่ำด้านหลังหุ่นยนต์
VALID_SHAPES = ["RECT_V", "RECT_H", "SQUARE", "CIRCLE"]
ALL_COLORS   = ["red", "yellow", "green", "blue"]

# ==========================================
# 3. ASSET LOADERS & CALIBRATION
# ==========================================
def get_asset_path(fn):
    p = os.path.join(SCRIPT_DIR, fn)
    return p if os.path.exists(p) else fn

TOF_A, TOF_B, TOF_C = 0.0, 1.0, 0.0
TOF_SLOPE, TOF_INTERCEPT = 1.0, 0.0
MAX_CALIB_RANGE_CM = 180.0

tof_json = get_asset_path("tof_calib_params.json")
if os.path.exists(tof_json):
    try:
        with open(tof_json) as f:
            cd = json.load(f)
            TOF_A = cd.get("a", 0.0); TOF_B = cd.get("b", 1.0); TOF_C = cd.get("c", 0.0)
            TOF_SLOPE = cd.get("slope", TOF_B); TOF_INTERCEPT = cd.get("intercept", TOF_C)
            MAX_CALIB_RANGE_CM = cd.get("max_calib_range_cm", 180.0)
        log(f"[ToF Calib] Loaded '{tof_json}'")
    except Exception as e:
        log(f"[ToF Calib ERROR] {e}")

def apply_tof_calibration(raw_cm):
    if raw_cm <= 0 or raw_cm >= 400.0: return 999.0
    if raw_cm <= MAX_CALIB_RANGE_CM:
        return round(max(0.0, TOF_A*(raw_cm**2) + TOF_B*raw_cm + TOF_C), 2)
    return round(max(0.0, TOF_SLOPE*raw_cm + TOF_INTERCEPT), 2)

def calculate_ground_distance(raw_cm):
    if raw_cm <= 0 or raw_cm >= 800.0: return 999.0
    return raw_cm + TOF_OFFSET_HORIZ_CM

DEFAULT_HSV = {
    "red":    [{"lower":[0,90,70],    "upper":[12,255,255]},
               {"lower":[165,90,70],  "upper":[180,255,255]}],
    "yellow": [{"lower":[15,100,100], "upper":[35,255,255]}],
    "green":  [{"lower":[35,70,50],   "upper":[85,255,255]}],
    "blue":   [{"lower":[95,80,50],   "upper":[135,255,255]}],
}

def load_hsv_ranges(fn="hsv_config.json"):
    data = DEFAULT_HSV
    p = get_asset_path(fn)
    if os.path.exists(p):
        try:
            with open(p) as f: data = json.load(f)
        except: pass
    return {c: [(np.array(r["lower"]), np.array(r["upper"])) for r in rs]
            for c, rs in data.items()}

COLOR_RANGES = load_hsv_ranges()

roi_mask_path = get_asset_path("roi_mask.png")
ROI_MASK_IMG = None
if os.path.exists(roi_mask_path):
    ROI_MASK_IMG = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    log(f"[ROI Mask] Loaded '{roi_mask_path}'")

def normalize_angle(deg):
    return (deg + 180.0) % 360.0 - 180.0

# ==========================================
# 4. STATE HUB
# ==========================================
class MockHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.running   = True
        self.latest_frame = None
        self.tof_raw      = 0.0
        self.tof_filtered = 0.0
        self.tof_fast_ema = 0.0
        self.tof_buffer   = deque(maxlen=5)
        self.tof_last_update_time = 0.0
        self.gimbal_yaw   = 0.0
        self.gimbal_pitch = 0.0

    def append_tof(self, val):
        with self.lock:
            self.tof_raw = val
            self.tof_last_update_time = time.time()
            if val < 800.0:
                self.tof_fast_ema = (0.85*val + 0.15*self.tof_fast_ema) if self.tof_fast_ema > 0 else val
                self.tof_buffer.append(val)
            else:
                self.tof_fast_ema = 999.0
            valid = [v for v in self.tof_buffer if v < 800.0]
            self.tof_filtered = float(np.median(valid)) if valid else val

    def clear_tof_buffer(self):
        with self.lock: self.tof_buffer.clear()

    def get_frame_safe(self, retries=3, delay=0.02):
        for _ in range(retries):
            with self.lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            time.sleep(delay)
        return None

hub = MockHub()

# ==========================================
# 5. HARDWARE HANDLERS & ADAPTIVE GIMBAL MOVE
# ==========================================
def tof_data_handler(sub_info):
    try:
        raw_mm = sub_info[0]
        if raw_mm is None or float(raw_mm) <= 0: return
        raw_cm = float(raw_mm) / 10.0
        val = apply_tof_calibration(raw_cm)
        hub.append_tof(val)
    except: pass

def gimbal_data_handler(sub_info):
    try:
        with hub.lock:
            hub.gimbal_pitch = float(sub_info[0])
            hub.gimbal_yaw   = float(sub_info[1])
    except: pass

def camera_worker(ep_camera):
    try: ep_camera.start_video_stream(display=False)
    except Exception as e: log(f"[Camera] start failed: {e}")
    while hub.running:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest")
            if frame is not None:
                resized = cv2.resize(frame, (640, 360))
                with hub.lock:
                    hub.latest_frame = resized
        except: pass
        time.sleep(0.04)
    try: ep_camera.stop_video_stream()
    except: pass

def safe_gimbal_move(ep_gimbal, target_pitch, target_yaw, speed=None):
    try:
        with hub.lock:
            current_pitch = hub.gimbal_pitch
            current_yaw   = hub.gimbal_yaw

        yaw_diff = abs(normalize_angle(target_yaw - current_yaw))

        # 🌟 ADAPTIVE SPEED: คำนวณความเร็วตามระยะทางองศาถ้าระบุ speed เป็น None
        if speed is None:
            if yaw_diff > 30.0:
                speed = GIMBAL_FAST_SPEED      # 540 deg/s
            elif yaw_diff > 10.0:
                speed = GIMBAL_MEDIUM_SPEED    # 320 deg/s
            else:
                speed = GIMBAL_SLOW_SPEED      # 140 deg/s

        if yaw_diff > 3.0:
            if abs(current_pitch - SAFE_CLEARANCE_PITCH) > 2.0:
                log(f"[GIMBAL STEP 1] Lifting Pitch ({current_pitch:.1f}° -> {SAFE_CLEARANCE_PITCH}°) before Yaw turn")
                with sdk_lock:
                    ep_gimbal.moveto(pitch=SAFE_CLEARANCE_PITCH, yaw=current_yaw,
                                     pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=0.8)
                time.sleep(0.10)

            log(f"[GIMBAL STEP 2] Rotating Yaw ({current_yaw:.1f}° -> {target_yaw:.1f}°) at speed {speed} deg/s")
            with sdk_lock:
                ep_gimbal.moveto(pitch=SAFE_CLEARANCE_PITCH, yaw=target_yaw,
                                 pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=1.0)
            
            time.sleep(0.20)

            if abs(target_pitch - SAFE_CLEARANCE_PITCH) > 1.0:
                log(f"[GIMBAL STEP 3] Yaw settled -> Pitching ({SAFE_CLEARANCE_PITCH}° -> {target_pitch:.1f}°)")
                with sdk_lock:
                    ep_gimbal.moveto(pitch=target_pitch, yaw=target_yaw,
                                     pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=0.8)
                time.sleep(0.15)
        else:
            with sdk_lock:
                ep_gimbal.moveto(pitch=target_pitch, yaw=target_yaw,
                                 pitch_speed=speed, yaw_speed=speed).wait_for_completed(timeout=0.6)

    except Exception as e:
        log(f"[SAFE GIMBAL ERROR] {e}")

# ==========================================
# 6. SHAPE EVALUATOR & MULTI-TARGET DETECTION
# ==========================================
def get_color_mask(hsv_img, target_color):
    ranges = COLOR_RANGES.get(target_color, [])
    mask = None
    for (lo, hi) in ranges:
        m = cv2.inRange(hsv_img, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask

def check_color_presence_blobs(img, active_colors, min_area=100):
    blobs = []
    if img is None: return blobs
    try:
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for c in active_colors:
            mask = get_color_mask(hsv, c)
            if mask is None: continue

            kernel = np.ones((5,5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        rx, ry, bw, bh = cv2.boundingRect(cnt)
                        blobs.append({'color': c, 'cx': cx, 'cy': cy, 'area': area, 'bbox': (rx, ry, bw, bh)})
    except Exception as e:
        log(f"[COLOR BLOBS ERROR] {e}")
    return blobs

def evaluate_multi_shape_candidates(bw, bh, area, perimeter, num_vertices, circularity, extent, is_far=False):
    if bh == 0 or bw == 0 or perimeter == 0: return None, 0.0
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

def process_frame_multi_targets(img, distance_cm, target_rules_input=None, current_yaw=None, fired_yaws=None):
    if img is None or not isinstance(img, np.ndarray) or img.size == 0: return []
    if target_rules_input is None: target_rules_input = TARGET_RULES

    if current_yaw is None:
        with hub.lock: current_yaw = hub.gimbal_yaw

    ground_dist = calculate_ground_distance(distance_cm) if distance_cm < 800.0 else 999.0
    is_far = ground_dist > 65.0
    min_contour_area = 120 if is_far else 250

    active_colors = [c for c, shapes in target_rules_input.items() if len(shapes) > 0]
    if not active_colors: return []

    detected_list = []
    try:
        h, w, _ = img.shape
        crop_ratio = float(np.clip(0.10 + (ground_dist - 30.0) * 0.0035, 0.08, 0.40)) if 0 < ground_dist < 800.0 else 0.25

        crop_top = int(h * crop_ratio)
        img_masked = img.copy()
        if crop_top > 0:
            trap = np.array([[(0,0),(w,0),(int(w*0.88),crop_top),(int(w*0.12),crop_top)]], dtype=np.int32)
            cv2.fillPoly(img_masked, trap, (0,0,0))

        if ROI_MASK_IMG is not None:
            rm = cv2.resize(ROI_MASK_IMG, (w, h))
            img_masked = cv2.bitwise_and(img_masked, img_masked, mask=rm)

        if fired_yaws:
            for fy in fired_yaws:
                yaw_diff = normalize_angle(fy - current_yaw)
                if abs(yaw_diff) < 32.0:
                    px_offset = math.tan(math.radians(yaw_diff)) * CAMERA_FOCAL_LENGTH_PX
                    mask_cx = int((w / 2.0) + px_offset)
                    mask_w = 90
                    cv2.rectangle(img_masked, (mask_cx - mask_w//2, 0), (mask_cx + mask_w//2, h), (0, 0, 0), -1)

        hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)

        for c in active_colors:
            allowed_shapes = target_rules_input[c]
            mask = get_color_mask(hsv, c)
            if mask is None: continue

            kernel = np.ones((5,5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_contour_area: continue
                rx, ry, bw, bh = cv2.boundingRect(cnt)
                if bw == 0 or bh == 0: continue

                cx = rx + (bw // 2)
                cy = ry + (bh // 2)

                item_yaw = current_yaw + (cx - w / 2.0) / CAMERA_FOCAL_LENGTH_PX * (180.0 / math.pi)
                if fired_yaws and any(abs(normalize_angle(item_yaw - fy)) < 8.0 for fy in fired_yaws):
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0.0
                if solidity < 0.70: continue
                if cy > int(h * 0.98): continue

                bbox_mask = mask_clean[ry : ry + bh, rx : rx + bw]
                fill_ratio = (cv2.countNonZero(bbox_mask) / float(bw * bh)) * 100.0
                if fill_ratio < 25.0: continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue

                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                num_vertices = len(approx)
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                extent = float(area) / (bw * bh)

                subtype, score = evaluate_multi_shape_candidates(bw, bh, area, perimeter, num_vertices, circularity, extent, is_far=is_far)
                if subtype is None or subtype not in VALID_SHAPES: continue

                if "ALL_SHAPES" not in allowed_shapes and subtype not in allowed_shapes:
                    continue

                detected_list.append({
                    "color": c, "shape": subtype,
                    "cx": cx, "cy": cy,
                    "bbox": (rx, ry, bw, bh),
                    "area": area,
                    "fill_ratio": fill_ratio,
                    "confidence": score,
                    "is_far": is_far,
                    "is_main": True,
                })

        detected_list.sort(key=lambda x: -x["confidence"])
    except Exception as e:
        log(f"[DETECTION ERROR] {e}")
    return detected_list

# ==========================================
# 7. SAFE RETREAT WITH DISTANCE TRACKING
# ==========================================
def check_is_target_cut_off_at_bottom(img, bbox, target_color):
    if img is None: return False
    h, w, _ = img.shape
    
    if bbox is not None:
        rx, ry, bw, bh = bbox
        bottom_y = ry + bh
        if bottom_y >= int(h * 0.82):
            return True

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, target_color)
    if mask is not None:
        if ROI_MASK_IMG is not None:
            rm = cv2.resize(ROI_MASK_IMG, (w, h))
            mask = cv2.bitwise_and(mask, mask, mask=rm)
        bottom_zone = mask[int(h * 0.80):h, :]
        if cv2.countNonZero(bottom_zone) > 100:
            return True

    return False

def resolve_bottom_cut_off_by_backing_up(ep_chassis, ep_gimbal, target_color, target_shape, target_yaw, current_pitch, fired_yaws=None):
    total_moved_x = 0.0
    total_moved_y = 0.0

    print(f"\n🔍 [CHECK BOTTOM EDGE] ตรวจสอบเป้าติดขอบล่าง/ปากกระบอกปืนบัง (Gimbal Yaw={target_yaw}°, Pitch={current_pitch}°)...")

    if current_pitch > WALL_DOWN_PITCH:
        print(f"  👇 [PITCH DOWN FIRST] สั่งก้มหน้า Gimbal ลงสุดที่ Pitch = {WALL_DOWN_PITCH}° เพื่อเช็กสีเป้าหมาย...")
        log(f"[PITCH DOWN] Pitching down to {WALL_DOWN_PITCH}° before cut-off check")
        safe_gimbal_move(ep_gimbal, target_pitch=WALL_DOWN_PITCH, target_yaw=target_yaw, speed=200)
        time.sleep(0.20)

    rad = math.radians(target_yaw)
    retreat_dist = 0.06
    move_x = round(-retreat_dist * math.cos(rad), 3)
    move_y = round(-retreat_dist * math.sin(rad), 3)

    single_rule = {target_color: [target_shape]}

    for backup_step in range(1, 6):
        if not hub.running: break
        img = hub.get_frame_safe(retries=3, delay=0.02)
        if img is None: break

        with hub.lock: tof_dist = hub.tof_filtered
        targets = process_frame_multi_targets(img, tof_dist, target_rules_input=single_rule, current_yaw=target_yaw, fired_yaws=fired_yaws)
        
        target_bbox = targets[0]["bbox"] if targets else None
        is_cut_off = check_is_target_cut_off_at_bottom(img, target_bbox, target_color)

        if is_cut_off:
            opposite_yaw = normalize_angle(target_yaw + 180.0)
            rear_tof = get_fresh_tof(ep_gimbal, target_yaw=opposite_yaw, pitch=0.0)
            safe_gimbal_move(ep_gimbal, target_pitch=WALL_DOWN_PITCH, target_yaw=target_yaw)

            if rear_tof <= RETREAT_SAFE_MIN_CM:
                print(f"  🛑 [COLLISION GUARDRAIL] ระยะด้านหลังชิดกำแพง/Sensor เกินไป ({rear_tof:.1f} cm <= {RETREAT_SAFE_MIN_CM} cm)! ยกเลิกการถอยเพื่อป้องกันการชน")
                log(f"[SAFETY STOP] Rear ToF ({rear_tof:.1f}cm) too close -> Canceled retreat move")
                break

            print(f"  ⚠ [CUT-OFF DETECTED AT PITCH -22°] สียังติดขอบล่าง/ปากกระบอก! (ระยะหลังปลอดภัย {rear_tof:.1f}cm) -> ถอยหลบ (x={move_x:+.2f}m, y={move_y:+.2f}m) step {backup_step}...")
            log(f"[RETREAT] Cut-off at pitch -22° -> Moving Chassis (x={move_x}, y={move_y})")
            
            try:
                with sdk_lock:
                    ep_chassis.move(x=move_x, y=move_y, z=0, xy_speed=0.15).wait_for_completed(timeout=1.2)
                total_moved_x += move_x
                total_moved_y += move_y
            except Exception as e:
                log(f"[CHASSIS MOVE ERROR] {e}")

            time.sleep(0.20)
        else:
            print("  ✅ [FULL TARGET VISIBLE] เป้าหมายหลุดจากขอบล่าง/ปากกระบอก และเห็นเต็มแผ่นเรียบร้อย!")
            log("[RETREAT] Target fully clear in frame!")
            break

    return total_moved_x, total_moved_y

# ==========================================
# 8. TOF FETCH & RECENTER
# ==========================================
def recenter_gimbal(ep_gimbal):
    safe_gimbal_move(ep_gimbal, target_pitch=0, target_yaw=0, speed=GIMBAL_FAST_SPEED)

TOF_FILTER_SIZE = 5
TOF_SAMPLE_PERIOD_SEC    = 0.02
GIMBAL_LATENCY_FLUSH_SEC = 0.15

def get_fresh_tof(ep_gimbal, target_yaw, pitch=0):
    safe_gimbal_move(ep_gimbal, target_pitch=pitch, target_yaw=target_yaw)
    time.sleep(GIMBAL_LATENCY_FLUSH_SEC)
    hub.clear_tof_buffer()
    samples = []
    st = time.time(); last_t = st
    while len(samples) < TOF_FILTER_SIZE and (time.time() - st < 0.45):
        with hub.lock:
            if hub.tof_last_update_time > last_t:
                v = hub.tof_raw
                if 2.0 <= v <= 300.0:
                    samples.append(v); last_t = hub.tof_last_update_time
        time.sleep(TOF_SAMPLE_PERIOD_SEC / 2.0)
    return float(np.median(samples)) if samples else (hub.tof_filtered if hub.tof_filtered > 0 else 999.0)

# ==========================================
# 9. FINE-AIMING ABOVE TARGET & REAL BLASTER FIRE (WITH ADAPTIVE SPEED)
# ==========================================
def fine_aim_above_target_and_fire(ep_gimbal, ep_blaster, color, shape, fired_yaws=None):
    print(f"\n🎯 [FINE AIMING] ปรับ Gimbal เล็งเหนือจุดศูนย์กลางเป้า ({color.upper()} {shape}) (ชดเชยกล้องสูงกว่าปืน)...")
    log(f"[AIM] Fine-Aiming start for {color.upper()} {shape}")

    aim_start = time.time()
    target_locked = False
    single_rule = {color: [shape]}

    while time.time() - aim_start < AIM_TIMEOUT_SEC:
        if not hub.running: break
        img = hub.get_frame_safe(retries=2, delay=0.01)
        if img is None: break

        with hub.lock:
            tof_dist = hub.tof_filtered
            c_yaw = hub.gimbal_yaw

        found_targets = process_frame_multi_targets(img, tof_dist, target_rules_input=single_rule, current_yaw=c_yaw, fired_yaws=fired_yaws)

        if found_targets:
            t = found_targets[0]
            cx, cy = t["cx"], t["cy"]
            h_frame, w_frame, _ = img.shape
            center_x, center_y = w_frame / 2.0, h_frame / 2.0

            ex = cx - center_x
            ey = cy - (center_y + AIM_Y_OFFSET_PX)

            accept_y_px = 32.0 if tof_dist <= 45.0 else AIM_ACCEPT_PX

            print(f"   ├─ Aim Error: dX={ex:+.1f}px | dY={ey:+.1f}px (Accepted if dX<={AIM_ACCEPT_PX}px, dY<={accept_y_px}px)")

            if abs(ex) <= AIM_ACCEPT_PX and abs(ey) <= accept_y_px:
                print(f"🎯 [LOCK ENGAGED] ล็อกเป้าหมายสำเร็จ! dX={ex:.1f}px dY={ey:.1f}px (ยกระดับลำกล้องขึ้นเหนือเป้า {AIM_Y_OFFSET_PX}px)")
                log(f"[AIM] Locked target! ex={ex:.1f}px ey={ey:.1f}px")
                target_locked = True
                break

            # 🌟 ADAPTIVE SPEED CONTROL FOR FINE AIMING:
            # คำนวณระยะห่างรวม (dist_err) ถ้าไกลขยับเร็ว พอใกล้เข้าหาเป้าจะชะลอความเร็วให้สมูท
            dist_err = math.hypot(ex, ey)
            if dist_err > 80.0:
                max_speed_yaw, max_speed_pitch = 35.0, 28.0  # ไกล: ปรับความเร็วเล็งสูงสุด
            elif dist_err > 30.0:
                max_speed_yaw, max_speed_pitch = 18.0, 14.0  # ปานกลาง
            else:
                max_speed_yaw, max_speed_pitch = 8.0, 6.0    # ใกล้: ชะลอความเร็วลงเพื่อความแม่นยำสูง

            yaw_corr   = float(np.clip(ex  *  AIM_YAW_GAIN,  -max_speed_yaw, max_speed_yaw))
            pitch_corr = float(np.clip(-ey * AIM_PITCH_GAIN, -max_speed_pitch, max_speed_pitch))

            with sdk_lock:
                ep_gimbal.drive_speed(pitch_speed=pitch_corr, yaw_speed=yaw_corr)

        time.sleep(0.03)

    try:
        with sdk_lock: ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
    except: pass

    if target_locked and ep_blaster is not None and hub.running:
        print(f"\n🔥 [REAL BLASTER FIRING] สั่งยิงเลเซอร์ IR จำนวน {BURST_FIRE_COUNT} นัด!")
        log(f"🔥 FIRE ({color.upper()} {shape}) x{BURST_FIRE_COUNT}")
        try:
            for b_i in range(1, BURST_FIRE_COUNT + 1):
                with sdk_lock:
                    ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=1)
                print(f"   💥 นัดที่ {b_i}: PEW! (IR Laser Fired Successfully)")
                time.sleep(0.15)
            log("[BLASTER] IR Burst completed cleanly")
            return True
        except Exception as e:
            print(f"❌ [FIRING ERROR] เกิดข้อผิดพลาดขณะยิง: {e}")
            log(f"[FIRE ERROR] {e}")
            return False
    elif not target_locked:
        print(f"❌ [AIM TIMEOUT] ไม่สามารถล็อกเป้าหมายได้ในเวลาที่กำหนด (ข้ามการยิง)")
        log("[AIM] NOT locked within timeout -> Skipped Firing")
    return False

def verify_and_fire_pipeline(ep_gimbal, ep_blaster, ep_chassis, target_yaw, target_pitch, color, shape, fired_yaws=None):
    safe_gimbal_move(ep_gimbal, target_pitch=target_pitch, target_yaw=target_yaw)

    backed_x, backed_y = resolve_bottom_cut_off_by_backing_up(ep_chassis, ep_gimbal, target_color=color, target_shape=shape, target_yaw=target_yaw, current_pitch=target_pitch, fired_yaws=fired_yaws)

    print(f"\n==================================================")
    print(f"🔍 [VERIFY TARGET IN QUEUE] สแกนยืนยันเป้าสงสัย {color.upper()} ({shape}) ที่มุม Yaw:{target_yaw:.1f}°")
    print(f"==================================================")

    print("⏳ [WARM-UP] รอกล้องและ ToF เล็งนิ่งกลางวัตถุ 3 เฟรม...")
    consecutive_hits = 0
    warmup_start = time.time()
    single_rule = {color: [shape]}

    while time.time() - warmup_start < 0.45:
        w_img = hub.get_frame_safe(retries=2, delay=0.01)
        with hub.lock:
            w_tof = hub.tof_filtered
            c_yaw = hub.gimbal_yaw
        if w_img is not None:
            w_targets = process_frame_multi_targets(w_img, w_tof, target_rules_input=single_rule, current_yaw=c_yaw, fired_yaws=fired_yaws)
            if w_targets:
                consecutive_hits += 1
                if consecutive_hits >= 3:
                    print("✅ [WARM-UP SUCCESS] ภาพนิ่งพร้อมสแกนตัวอย่าง!")
                    break
            else:
                consecutive_hits = 0
        time.sleep(0.03)

    print("\n📷 [10-FRAME SAMPLING] จ้องนิ่งเพื่อเก็บค่าระยะ ToF และยืนยันเป้าหมาย...")
    valid_samples, tof_samples = [], []
    last_bbox, last_img_sample = None, None

    for frame_idx in range(1, TOTAL_CHECK_FRAMES + 1):
        if not hub.running: break
        time.sleep(0.03)
        s_img = hub.get_frame_safe(retries=2, delay=0.01)
        with hub.lock:
            s_tof = hub.tof_filtered
            c_yaw = hub.gimbal_yaw

        if s_img is not None:
            s_tgts = process_frame_multi_targets(s_img, s_tof, target_rules_input=single_rule, current_yaw=c_yaw, fired_yaws=fired_yaws)
            if s_tgts:
                r = s_tgts[0]
                valid_samples.append((r["color"], r["shape"]))
                if 10.0 <= s_tof <= MAX_TARGET_DETECT_DIST_CM:
                    tof_samples.append(s_tof)
                last_bbox = r["bbox"]
                last_img_sample = s_img.copy()
                print(f"   ├─ Frame {frame_idx:02d}/10: ✅ PASS -> {r['color'].upper()} {r['shape']} (ToF Target: {s_tof:.1f} cm)")
            else:
                print(f"   ├─ Frame {frame_idx:02d}/10: ❌ FAIL (Detect ไม่เจอเป้าหมายในเฟรมนี้)")
        else:
            print(f"   ├─ Frame {frame_idx:02d}/10: ❌ FAIL (ไม่พบภาพจากกล้อง)")

    success_count = len(valid_samples)
    print(f"\n📊 สรุปผลสแกน 10 เฟรม: ผ่าน {success_count}/{TOTAL_CHECK_FRAMES} เฟรม (ต้องการอย่างน้อย {REQUIRED_MATCH_FRAMES} เฟรม)")
    log(f"[VERIFY] Frames={success_count}/{TOTAL_CHECK_FRAMES}")

    fired_success = False
    if success_count >= REQUIRED_MATCH_FRAMES and hub.running:
        conf_color, conf_shape = Counter(valid_samples).most_common(1)[0][0]
        raw_tof_final = float(np.median(tof_samples)) if tof_samples else hub.tof_filtered
        ground_dist = calculate_ground_distance(raw_tof_final)

        if last_img_sample is not None and last_bbox is not None:
            rx, ry, bw, bh = last_bbox
            ann = last_img_sample.copy()
            cv2.rectangle(ann, (rx,ry), (rx+bw,ry+bh), (0,255,0), 2)
            cv2.putText(ann, f"{conf_color.upper()} {conf_shape} | {ground_dist:.1f}cm",
                        (rx, max(20,ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fpath = os.path.join(TARGET_IMG_DIR, f"target_{conf_color}_{conf_shape}_{ts}.png")
            cv2.imwrite(fpath, ann)

        print(f"\n🎉 [TARGET CONFIRMED] ยืนยันเป้าหมายสำเร็จ!")
        print(f"   ├─ ชนิดเป้าหมาย: {conf_color.upper()} ({conf_shape})")
        print(f"   ├─ ระยะทางราบ : {ground_dist:.1f} cm (ToF: {raw_tof_final:.1f} cm)")
        print(f"   └─ พิกัด Gimbal: Yaw={target_yaw:.1f}° | Pitch={WALL_DOWN_PITCH:.1f}°")

        if ground_dist <= (MAX_FIRING_DIST_CM + TOF_OFFSET_HORIZ_CM):
            fired_success = fine_aim_above_target_and_fire(ep_gimbal, ep_blaster, conf_color, conf_shape, fired_yaws=fired_yaws)
        else:
            print(f"⚠️ [OUT OF RANGE] ระยะทาง {ground_dist:.1f}cm ไกลกว่าระยะยิงหวังผลสูงสุด ({MAX_FIRING_DIST_CM}cm) -> ยกเลิกการยิง")
            log(f"[VERIFY] {ground_dist:.1f}cm > MAX {MAX_FIRING_DIST_CM}cm -> Skipped Fire")
    else:
        print("❌ [FALSE ALARM REJECTED] ตรวจจับผ่านน้อยกว่า 7 เฟรม -> ปฏิเสธการยิงเป้าหมายนี้")
        log(f"[VERIFY] FAILED: {success_count}/{REQUIRED_MATCH_FRAMES}")

    return backed_x, backed_y, fired_success

# ==========================================
# 10. RUN FIRING PIPELINE (PER-WALL QUEUE & RETURN HOME)
# ==========================================
def run_firing_test(ep_chassis, ep_gimbal, ep_blaster, ep_sensor, sub_already=False):
    if not sub_already:
        ep_sensor.sub_distance(freq=50, callback=tof_data_handler)
        ep_gimbal.sub_angle(freq=20, callback=gimbal_data_handler)
        time.sleep(0.5)

    rules_summary = [f"{c.upper()}:{','.join(s)}" for c, s in TARGET_RULES.items() if s]
    rule_str = " | ".join(rules_summary) if rules_summary else "NONE"

    log("=" * 60)
    log(f"SCAN START | Target Rules: {rule_str}")
    log("=" * 60)

    print(f"\n{'='*60}")
    print(f"🔄 เริ่มสแกนกวาดและเคลียร์ Queue ทีละทิศทางกำแพง (เงื่อนไขเป้าหมาย: {rule_str})")
    print("="*60)

    for dir_name, rel_yaw in SCAN_DIRECTIONS:
        if not hub.running: break
        print(f"\n👉 [{dir_name}] Gimbal Yaw = {rel_yaw}° | Target Default Pitch = {INITIAL_SCAN_PITCH}°")

        fired_yaws_on_wall = []

        safe_gimbal_move(ep_gimbal, target_pitch=INITIAL_SCAN_PITCH, target_yaw=rel_yaw)
        time.sleep(0.20)

        raw_tof = get_fresh_tof(ep_gimbal, target_yaw=rel_yaw, pitch=INITIAL_SCAN_PITCH)
        ground_dist = calculate_ground_distance(raw_tof)

        current_pitch = INITIAL_SCAN_PITCH
        wall_nearby = ground_dist <= WALL_THRESHOLD_CM

        if not wall_nearby:
            print(f"   ℹ️ [NO WALL DETECTED] ไม่พบกำแพงระยะใกล้ ({ground_dist:.1f}cm > {WALL_THRESHOLD_CM}cm) -> ข้ามการสแกนเป้าหมายฝั่งนี้")
            continue

        print(f"   🧱 [WALL DETECTED] พบกำแพงระยะใกล้ ({ground_dist:.1f}cm) -> สั่ง Gimbal ก้มลงสุดที่ Pitch = {WALL_DOWN_PITCH}°")
        log(f"[WALL DOWN] Wall at {ground_dist:.1f}cm -> Pitch down to {WALL_DOWN_PITCH}°")
        current_pitch = WALL_DOWN_PITCH
        
        safe_gimbal_move(ep_gimbal, target_pitch=current_pitch, target_yaw=rel_yaw)
        time.sleep(0.20)
        raw_tof = get_fresh_tof(ep_gimbal, target_yaw=rel_yaw, pitch=current_pitch)

        img = hub.get_frame_safe(retries=3, delay=0.02)
        if img is None:
            print("   └─ ไม่ได้รับภาพจากกล้อง -> ข้าม")
            continue

        targets = process_frame_multi_targets(img, raw_tof, target_rules_input=TARGET_RULES, current_yaw=rel_yaw, fired_yaws=fired_yaws_on_wall)
        active_colors = [c for c, shapes in TARGET_RULES.items() if len(shapes) > 0]
        color_blobs = check_color_presence_blobs(img, active_colors, min_area=120)

        img_h, img_w = img.shape[:2]
        queued_cxs = []
        wall_queue = _queue.Queue()

        for idx, t in enumerate(targets, 1):
            deg_off  = (t["cx"] - img_w / 2.0) / CAMERA_FOCAL_LENGTH_PX * (180.0 / math.pi)
            item_yaw = rel_yaw + deg_off
            
            wall_queue.put({
                "dir_name": dir_name, "yaw": item_yaw,
                "pitch": current_pitch, "color": t["color"],
                "shape": t["shape"],   "raw_tof": raw_tof,
            })
            queued_cxs.append(t["cx"])
            dist_tag = "FAR" if t["is_far"] else "CLOSE"
            print(f"   ├─ พบเป้าสมบูรณ์ที่ {idx}: {t['color'].upper()} {t['shape']} (Area: {t['area']}px | {dist_tag}) -> เก็บเข้า Queue ทิศ {dir_name} (Yaw: {item_yaw:.1f}°)")

        for blob in color_blobs:
            if any(abs(blob["cx"] - qcx) < 30 for qcx in queued_cxs):
                continue

            deg_off  = (blob["cx"] - img_w / 2.0) / CAMERA_FOCAL_LENGTH_PX * (180.0 / math.pi)
            item_yaw = rel_yaw + deg_off

            allowed_s = TARGET_RULES.get(blob["color"], [])
            if not allowed_s: continue
            default_shape_sub = allowed_s[0]

            wall_queue.put({
                "dir_name": dir_name, "yaw": item_yaw,
                "pitch": current_pitch, "color": blob["color"],
                "shape": default_shape_sub,
                "raw_tof": raw_tof,
            })
            queued_cxs.append(blob["cx"])
            print(f"   🎨 [COLOR TRIGGER QUEUED] เห็นก้อนสี {blob['color'].upper()} ที่ cx={blob['cx']} -> ดันเข้า Queue ทิศ {dir_name} (Yaw: {item_yaw:.1f}°)")

        if not wall_queue.empty():
            print(f"\n🎯 [PROCESS WALL QUEUE] เริ่มยืนยันและยิงเป้าหมายกำแพงทิศ {dir_name} ({wall_queue.qsize()} เป้าหมาย)...")
            accumulated_x = 0.0
            accumulated_y = 0.0

            while not wall_queue.empty() and hub.running:
                item = wall_queue.get()
                backed_x, backed_y, fired_ok = verify_and_fire_pipeline(
                    ep_gimbal, ep_blaster, ep_chassis,
                    target_yaw=item["yaw"], target_pitch=item["pitch"],
                    color=item["color"],    shape=item["shape"],
                    fired_yaws=fired_yaws_on_wall
                )
                accumulated_x += backed_x
                accumulated_y += backed_y

                if fired_ok:
                    fired_yaws_on_wall.append(item["yaw"])
                    print(f"🔒 [DYNAMIC MASK ACTIVATED] บันทึกมุม Yaw={item['yaw']:.1f}° เข้า ROI Mask ป้องกันการยิงซ้ำ")

            if abs(accumulated_x) > 0.001 or abs(accumulated_y) > 0.001:
                return_x = round(-accumulated_x, 3)
                return_y = round(-accumulated_y, 3)
                print(f"\n🔄 [RETURN HOME] สั่งหุ่นยนต์เดินกลับตำแหน่งเดิมก่อนถอย (x={return_x:+.2f}m, y={return_y:+.2f}m)...")
                log(f"[RETURN HOME] Returning chassis back (x={return_x}, y={return_y})")
                try:
                    with sdk_lock:
                        ep_chassis.move(x=return_x, y=return_y, z=0, xy_speed=0.15).wait_for_completed(timeout=2.0)
                except Exception as e:
                    log(f"[CHASSIS RETURN ERROR] {e}")
                time.sleep(0.20)

        else:
            print(f"   └─ ไม่พบเป้าหมายในฝั่งกำแพง {dir_name}")

        gc.collect()

    recenter_gimbal(ep_gimbal)
    gc.collect()
    log("[SCAN] Test pipeline complete.")

# ==========================================
# 11. PYGAME GUI CONFIGURATOR (PER-COLOR SELECTION)
# ==========================================
def setup_menu_pygame():
    global TARGET_RULES, BURST_FIRE_COUNT

    pygame.init()
    WIDTH, HEIGHT = 780, 620
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RoboMaster Per-Color Target Configurator")
    clock = pygame.time.Clock()

    FONT_TITLE = pygame.font.SysFont("Arial", 22, bold=True)
    FONT_SEC   = pygame.font.SysFont("Arial", 16, bold=True)
    FONT_TXT   = pygame.font.SysFont("Arial", 14)

    gui_state = {
        "red":    {"RECT_V": False, "RECT_H": False, "SQUARE": False, "CIRCLE": False, "all": False},
        "yellow": {"RECT_V": False, "RECT_H": False, "SQUARE": False, "CIRCLE": False, "all": False},
        "green":  {"RECT_V": False, "RECT_H": False, "SQUARE": False, "CIRCLE": False, "all": False},
        "blue":   {"RECT_V": False, "RECT_H": False, "SQUARE": False, "CIRCLE": False, "all": False},
    }

    colors_ui_info = [
        ("RED TARGETS", "red", (220, 50, 50), 60),
        ("YELLOW TARGETS", "yellow", (220, 190, 30), 150),
        ("GREEN TARGETS", "green", (40, 190, 80), 240),
        ("BLUE TARGETS", "blue", (50, 120, 240), 330),
    ]

    shape_options = [
        ("RECT_V", "RECT_V"),
        ("RECT_H", "RECT_H"),
        ("SQUARE", "SQUARE"),
        ("CIRCLE", "CIRCLE"),
        ("ALL SHAPES", "all"),
    ]

    running_menu = True
    while running_menu:
        screen.fill((25, 27, 33))

        title_surf = FONT_TITLE.render("Per-Color Target Rules Setup", True, (255, 255, 255))
        screen.blit(title_surf, (20, 15))

        click_targets = []

        for title, col_key, draw_col, y_pos in colors_ui_info:
            pygame.draw.rect(screen, (38, 42, 53), (20, y_pos, 740, 78), border_radius=8)

            pygame.draw.circle(screen, draw_col, (38, y_pos + 22), 8)
            sec_surf = FONT_SEC.render(title, True, (230, 230, 230))
            screen.blit(sec_surf, (54, y_pos + 12))

            x_pos = 50
            for label, sh_key in shape_options:
                box_rect = pygame.Rect(x_pos, y_pos + 42, 20, 20)
                click_targets.append((col_key, sh_key, box_rect, pygame.Rect(x_pos, y_pos + 40, 120, 25)))

                pygame.draw.rect(screen, (70, 75, 90), box_rect, border_radius=4)
                if gui_state[col_key][sh_key]:
                    pygame.draw.rect(screen, (0, 220, 120), box_rect.inflate(-6, -6), border_radius=2)

                lbl_surf = FONT_TXT.render(label, True, (210, 210, 210))
                screen.blit(lbl_surf, (x_pos + 26, y_pos + 43))

                x_pos += 135

        pygame.draw.rect(screen, (38, 42, 53), (20, 425, 740, 65), border_radius=8)
        sec3_surf = FONT_SEC.render("IR Laser Burst Count:", True, (200, 220, 255))
        screen.blit(sec3_surf, (35, 445))

        minus_btn = pygame.Rect(320, 440, 35, 35)
        plus_btn  = pygame.Rect(430, 440, 35, 35)
        pygame.draw.rect(screen, (60, 65, 80), minus_btn, border_radius=6)
        pygame.draw.rect(screen, (60, 65, 80), plus_btn, border_radius=6)

        m_txt = FONT_SEC.render("-", True, (255, 255, 255))
        p_txt = FONT_SEC.render("+", True, (255, 255, 255))
        screen.blit(m_txt, (332, 445))
        screen.blit(p_txt, (441, 445))

        b_val_txt = FONT_TITLE.render(str(BURST_FIRE_COUNT), True, (255, 215, 0))
        screen.blit(b_val_txt, (380, 442))

        start_btn = pygame.Rect(210, 515, 360, 55)
        pygame.draw.rect(screen, (0, 180, 90), start_btn, border_radius=12)
        start_txt = FONT_TITLE.render("START SCAN PIPELINE", True, (255, 255, 255))
        screen.blit(start_txt, (start_btn.x + 40, start_btn.y + 12))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                for col_key, sh_key, box_r, zone_r in click_targets:
                    if box_r.collidepoint(pos) or zone_r.collidepoint(pos):
                        if sh_key == "all":
                            new_val = not gui_state[col_key]["all"]
                            for k in gui_state[col_key]: gui_state[col_key][k] = new_val
                        else:
                            gui_state[col_key][sh_key] = not gui_state[col_key][sh_key]
                            gui_state[col_key]["all"] = all(gui_state[col_key][k] for k in VALID_SHAPES)

                if minus_btn.collidepoint(pos):
                    BURST_FIRE_COUNT = max(1, BURST_FIRE_COUNT - 1)
                elif plus_btn.collidepoint(pos):
                    BURST_FIRE_COUNT = min(5, BURST_FIRE_COUNT + 1)

                if start_btn.collidepoint(pos):
                    running_menu = False

        pygame.display.flip()
        clock.tick(30)

    for c in ALL_COLORS:
        if gui_state[c]["all"]:
            TARGET_RULES[c] = ["ALL_SHAPES"]
        else:
            TARGET_RULES[c] = [s for s in VALID_SHAPES if gui_state[c][s]]

    pygame.quit()

# ==========================================
# 12. MAIN CAMERA GUI LOOP
# ==========================================
def gui_camera_loop(ep_chassis, ep_gimbal, ep_blaster, ep_sensor):
    ep_sensor.sub_distance(freq=50, callback=tof_data_handler)
    ep_gimbal.sub_angle(freq=20, callback=gimbal_data_handler)
    time.sleep(0.5)

    debug_mode = "NORMAL"
    scanning = False
    win_name = "Firing Mock - Camera Monitor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    try:
        while hub.running:
            display_img = hub.get_frame_safe(retries=1, delay=0.005)

            if display_img is not None:
                show = cv2.resize(display_img, (1280, 720), interpolation=cv2.INTER_CUBIC)
                h_s, w_s, _ = show.shape

                with hub.lock:
                    tof_v = hub.tof_filtered
                    g_yaw = hub.gimbal_yaw
                    g_pit = hub.gimbal_pitch

                ground_dist = calculate_ground_distance(tof_v) if tof_v < 800.0 else 999.0
                crop_ratio = float(np.clip(0.10 + (ground_dist - 30.0) * 0.0035, 0.08, 0.40)) if 0 < ground_dist < 800.0 else 0.25

                crop_top_s = int(h_s * crop_ratio)
                if crop_top_s > 0:
                    trap_pts_s = np.array([[(0,0), (w_s,0), (int(w_s*0.88), crop_top_s), (int(w_s*0.12), crop_top_s)]], dtype=np.int32)
                    overlay = show.copy()
                    cv2.fillPoly(overlay, trap_pts_s, (0, 100, 255))
                    cv2.addWeighted(overlay, 0.25, show, 0.75, 0, show)
                    cv2.polylines(show, trap_pts_s, isClosed=True, color=(0, 165, 255), thickness=2)
                    cv2.putText(show, f"ROI MASK TOP CROP ({crop_ratio*100:.1f}%)", (int(w_s*0.15), max(25, crop_top_s - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                if debug_mode == "NORMAL":
                    targets_detected = process_frame_multi_targets(display_img, tof_v, target_rules_input=TARGET_RULES)
                    scale_x = 1280.0 / 640.0
                    scale_y = 720.0 / 360.0

                    for t_obj in targets_detected:
                        rx, ry, bw, bh = t_obj["bbox"]
                        rx_s, ry_s = int(rx * scale_x), int(ry * scale_y)
                        bw_s, bh_s = int(bw * scale_x), int(bh * scale_y)

                        color_box = (0, 255, 0) if t_obj["confidence"] >= 0.40 else (0, 255, 255)
                        cv2.rectangle(show, (rx_s, ry_s), (rx_s + bw_s, ry_s + bh_s), color_box, 3)
                        
                        label = f"{t_obj['color'].upper()} {t_obj['shape']} ({t_obj['confidence']*100:.0f}%)"
                        cv2.putText(show, label, (rx_s, max(25, ry_s - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                elif debug_mode == "MASK":
                    hsv = cv2.cvtColor(show, cv2.COLOR_BGR2HSV)
                    combined_mask = None
                    active_cols = [c for c, shapes in TARGET_RULES.items() if len(shapes) > 0]
                    for c_item in active_cols:
                        m = get_color_mask(hsv, c_item)
                        if m is not None:
                            combined_mask = m if combined_mask is None else cv2.bitwise_or(combined_mask, m)
                    if combined_mask is not None:
                        show = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

                status = "SCANNING & AIMING..." if scanning else "READY (Press 'R' to Start Scan & Fire)"

                center_x_s, center_y_s = w_s // 2, h_s // 2
                cv2.drawMarker(show, (center_x_s, center_y_s), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                target_aim_y = center_y_s + int(AIM_Y_OFFSET_PX * (720 / 360))
                cv2.circle(show, (center_x_s, target_aim_y), 8, (0, 0, 255), 2)
                cv2.putText(show, "AIM POINT (ABOVE TARGET)", (center_x_s + 15, target_aim_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                rules_summary = [f"{c.upper()}:{','.join(s)}" for c, s in TARGET_RULES.items() if s]
                rule_disp_gui = " | ".join(rules_summary) if rules_summary else "NONE"

                cv2.rectangle(show, (10, 10), (880, 150), (0, 0, 0), -1)
                cv2.putText(show, f"Rules:[{rule_disp_gui}]  Burst:{BURST_FIRE_COUNT}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(show, f"ToF:{tof_v:.1f}cm  Gimbal Yaw:{g_yaw:.1f}  Pitch:{g_pit:.1f}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
                cv2.putText(show, f"Status: {status}",
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 100, 255) if scanning else (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow(win_name, show)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('r'), ord('R')):
                if scanning:
                    print("\n⚠️ หุ่นยนต์กำลังสแกน/ยิงอยู่ กรุณารอให้เสร็จสิ้นก่อน")
                else:
                    with hub.lock:
                        hub.tof_buffer.clear()
                        hub.tof_filtered = 0.0
                        hub.tof_fast_ema = 0.0
                    gc.collect()

                    scanning = True

                    def _scan_job():
                        nonlocal scanning
                        try:
                            run_firing_test(ep_chassis, ep_gimbal, ep_blaster, ep_sensor, sub_already=True)
                        except Exception as e:
                            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                            log(f"[SCAN ERROR] {err}")
                        finally:
                            scanning = False
                            gc.collect()
                            print("\n🟢 [READY] สแกนเสร็จสิ้น! สามารถกด 'R' เพื่อทดสอบใหม่ได้ตลอดเวลา")

                    scan_thread = threading.Thread(target=_scan_job, daemon=True)
                    scan_thread.start()

            elif key in (ord('m'), ord('M')):
                debug_mode = "MASK" if debug_mode == "NORMAL" else "NORMAL"
                print(f"[GUI] Debug mode → {debug_mode}")

            elif key in (ord('q'), ord('Q'), 27):
                print("[GUI] กำลังปิดโปรแกรม...")
                hub.running = False
                break

    finally:
        cv2.destroyAllWindows()
        try: ep_sensor.unsub_distance()
        except: pass
        try: ep_gimbal.unsub_angle()
        except: pass

# ==========================================
# 13. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    setup_menu_pygame()

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_sensor      = ep_robot.sensor
    ep_gimbal      = ep_robot.gimbal
    ep_chassis     = ep_robot.chassis
    ep_camera      = ep_robot.camera
    ep_blaster_ref = getattr(ep_robot, "blaster", None)

    try:
        recenter_gimbal(ep_gimbal)
    except: pass

    cam_t = threading.Thread(target=camera_worker, args=(ep_camera,), daemon=True)
    cam_t.start()
    time.sleep(0.6)

    try:
        gui_camera_loop(ep_chassis, ep_gimbal, ep_blaster_ref, ep_sensor)
    except KeyboardInterrupt:
        log("[USER] Ctrl+C Triggered")
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        log(f"[CRASH] {err}")
    finally:
        hub.running = False
        time.sleep(0.4)
        cv2.destroyAllWindows()
        try: ep_chassis.drive_speed(x=0, y=0, z=0)
        except: pass
        try: ep_robot.close()
        except: pass
        log("[SYSTEM] Connection closed cleanly.")