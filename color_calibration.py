import os
import json
import cv2
import numpy as np
import time
from robomaster import robot

CONFIG_FILE = "hsv_config.json"
GIMBAL_FAST_SPEED = 250
# กำหนดขนาดภาพสำหรับประมวลผลเพื่อลดภาระ CPU (ช่วยให้ไม่ค้าง/ดับ)
PROC_WIDTH = 640
PROC_HEIGHT = 360

DEFAULT_CONFIG = {
    "red": [
        {"lower": [0, 100, 80], "upper": [10, 255, 255]},
        {"lower": [165, 100, 80], "upper": [180, 255, 255]}
    ],
    "yellow": [{"lower": [15, 100, 100], "upper": [35, 255, 255]}],
    "green": [{"lower": [38, 80, 50], "upper": [75, 255, 255]}],
    "blue": [{"lower": [95, 90, 50], "upper": [135, 255, 255]}]
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "red" in data and len(data["red"]) < 2:
                    data["red"].append({"lower": [165, 100, 80], "upper": [180, 255, 255]})
                print(f"✅ โหลดค่าคอนฟิกจาก '{CONFIG_FILE}' สำเร็จ!")
                return data
        except Exception as e:
            print(f"⚠️ อ่านไฟล์คอนฟิกไม่สำเร็จ: {e} | ใช้ค่า Default แทน")
    return DEFAULT_CONFIG.copy()

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        print(f"🎉 [SUCCESS] บันทึกค่าสีลงไฟล์ '{CONFIG_FILE}' เรียบร้อยแล้ว!")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

def nothing(x): pass

def setup_trackbars():
    cv2.namedWindow("HSV Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Control", 500, 320)

    cv2.createTrackbar("H_min", "HSV Control", 0, 180, nothing)
    cv2.createTrackbar("H_max", "HSV Control", 180, 180, nothing)
    cv2.createTrackbar("S_min", "HSV Control", 0, 255, nothing)
    cv2.createTrackbar("S_max", "HSV Control", 255, 255, nothing)
    cv2.createTrackbar("V_min", "HSV Control", 0, 255, nothing)
    cv2.createTrackbar("V_max", "HSV Control", 255, 255, nothing)

def update_trackbars(range_data):
    lower = range_data["lower"]
    upper = range_data["upper"]
    cv2.setTrackbarPos("H_min", "HSV Control", lower[0])
    cv2.setTrackbarPos("H_max", "HSV Control", upper[0])
    cv2.setTrackbarPos("S_min", "HSV Control", lower[1])
    cv2.setTrackbarPos("S_max", "HSV Control", upper[1])
    cv2.setTrackbarPos("V_min", "HSV Control", lower[2])
    cv2.setTrackbarPos("V_max", "HSV Control", upper[2])

def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H_min", "HSV Control")
    h_max = cv2.getTrackbarPos("H_max", "HSV Control")
    s_min = cv2.getTrackbarPos("S_min", "HSV Control")
    s_max = cv2.getTrackbarPos("S_max", "HSV Control")
    v_min = cv2.getTrackbarPos("V_min", "HSV Control")
    v_max = cv2.getTrackbarPos("V_max", "HSV Control")
    return [h_min, s_min, v_min], [h_max, s_max, v_max]

# ==========================================
# 📐 GEOMETRIC SHAPE DETECTION (OPTIMIZED)
# ==========================================
def detect_geometric_shapes(img, mask, view_mode):
    display_img = img.copy()
    
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    roi = cv2.bitwise_and(img, img, mask=mask_clean)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # คำนวณ Edge เท่าที่จำเป็นตาม view_mode
    canny_edges = cv2.Canny(blur, 50, 150)
    sobel_edges = None
    
    if view_mode == "SOBEL":
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        combined_edges = cv2.bitwise_or(sobel_edges, canny_edges)
    else:
        combined_edges = canny_edges

    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0: continue

        bbox_mask = mask_clean[y:y+bh, x:x+bw]
        color_pixel_count = cv2.countNonZero(bbox_mask)
        fill_ratio = (color_pixel_count / float(bw * bh)) * 100.0

        if fill_ratio < 40.0: continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue

        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        num_vertices = len(approx)

        extent = float(area) / (bw * bh)
        aspect_ratio = float(bw) / float(bh)

        subtype = "SQUARE"
        if num_vertices >= 5 and extent < 0.83:
            subtype = "CIRCLE"
        elif num_vertices == 4 or extent >= 0.80:
            if 0.85 <= aspect_ratio <= 1.18:
                subtype = "SQUARE"
            elif aspect_ratio > 1.18:
                subtype = "RECT_H"
            elif aspect_ratio < 0.85:
                subtype = "RECT_V"

        color_box = (0, 255, 0) if subtype != "CIRCLE" else (255, 0, 255)
        cv2.rectangle(display_img, (x, y), (x + bw, y + bh), color_box, 2)
        lbl = f"{subtype} ({fill_ratio:.0f}%)"
        cv2.putText(display_img, lbl, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

    return display_img, mask_clean, sobel_edges, canny_edges

def safe_gimbal_move_async(ep_gimbal, yaw_deg):
    """ส่งคำสั่งเคลื่อนที่ Gimbal แบบ Non-blocking (ไม่หยุดค้าง Main Loop)"""
    try:
        ep_gimbal.moveto(pitch=0, yaw=yaw_deg, pitch_speed=GIMBAL_FAST_SPEED, yaw_speed=GIMBAL_FAST_SPEED)
    except Exception as e:
        print(f"⚠️ Gimbal Move Notice: {e}")

def print_instructions():
    print("\n==================================================")
    print("🎨 ROBOMASTER EP - ADVANCED COLOR & SHAPE CALIBRATION")
    print("==================================================")
    print("เลือกปรับสี:")
    print("  [1]: แดง Range 1 (Low)  | [5]: แดง Range 2 (High)")
    print("  [2]: เหลือง             | [3]: เขียว")
    print("  [4]: น้ำเงิน")
    print("--------------------------------------------------")
    print("ควบคุมทิศทาง Gimbal:")
    print("  [W] : หันหน้า (0°)  | [A] : หันซ้าย (-90°)")
    print("  [D] : หันขวา (+90°) | [X] : หันหลัง (180°)")
    print("--------------------------------------------------")
    print("ฟังก์ชันอื่นๆ:")
    print("  [M] : สลับ Filter View")
    print("  [S] : บันทึกค่าสีลงไฟล์ 'hsv_config.json'")
    print("  [Q] : ออกจากโปรแกรม")
    print("==================================================\n")

def main():
    config_data = load_config()
    current_color = "red"
    range_idx = 0
    view_mode = "COLOR_RESULT"
    filter_modes = ["COLOR_RESULT", "MASK", "SOBEL", "CANNY", "RAW"]
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    
    ep_camera.start_video_stream(display=False)
    safe_gimbal_move_async(ep_gimbal, 0)

    setup_trackbars()
    update_trackbars(config_data[current_color][range_idx])
    print_instructions()

    try:
        while True:
            img_raw = ep_camera.read_cv2_image(strategy="newest")
            if img_raw is None:
                time.sleep(0.005)
                continue

            # Resize ภาพเพื่อป้องกันปัญหา CPU 100% / Memory Stack
            img = cv2.resize(img_raw, (PROC_WIDTH, PROC_HEIGHT))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_val, upper_val = get_trackbar_values()
            config_data[current_color][range_idx]["lower"] = lower_val
            config_data[current_color][range_idx]["upper"] = upper_val

            if current_color == "red" and len(config_data["red"]) >= 2:
                r1_low = np.array(config_data["red"][0]["lower"])
                r1_up  = np.array(config_data["red"][0]["upper"])
                r2_low = np.array(config_data["red"][1]["lower"])
                r2_up  = np.array(config_data["red"][1]["upper"])

                mask1 = cv2.inRange(hsv, r1_low, r1_up)
                mask2 = cv2.inRange(hsv, r2_low, r2_up)
                mask_combined = cv2.bitwise_or(mask1, mask2)
            else:
                mask_combined = cv2.inRange(hsv, np.array(lower_val), np.array(upper_val))

            evaluated_img, clean_mask, sobel_img, canny_img = detect_geometric_shapes(img, mask_combined, view_mode)

            osd_orig = evaluated_img.copy()
            status_text = f"COLOR: {current_color.upper()} (R{range_idx+1})"
            cv2.putText(osd_orig, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if view_mode == "COLOR_RESULT":
                filter_img = cv2.bitwise_and(img, img, mask=clean_mask)
            elif view_mode == "MASK":
                filter_img = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            elif view_mode == "SOBEL" and sobel_img is not None:
                filter_img = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR)
            elif view_mode == "CANNY":
                filter_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
            else:
                filter_img = img.copy()

            cv2.putText(filter_img, f"FILTER VIEW: {view_mode} ('M' to switch)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Original (Evaluated Target)", osd_orig)
            cv2.imshow("Filtered Process View", filter_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                current_color, range_idx = "red", 0
                update_trackbars(config_data[current_color][range_idx])
            elif key == ord('5'):
                current_color, range_idx = "red", 1
                update_trackbars(config_data[current_color][range_idx])
            elif key == ord('2'):
                current_color, range_idx = "yellow", 0
                update_trackbars(config_data[current_color][range_idx])
            elif key == ord('3'):
                current_color, range_idx = "green", 0
                update_trackbars(config_data[current_color][range_idx])
            elif key == ord('4'):
                current_color, range_idx = "blue", 0
                update_trackbars(config_data[current_color][range_idx])

            # --- Gimbal Control (Non-blocking) ---
            elif key in (ord('w'), ord('W')):
                safe_gimbal_move_async(ep_gimbal, 0)
            elif key in (ord('a'), ord('A')):
                safe_gimbal_move_async(ep_gimbal, -90)
            elif key in (ord('d'), ord('D')):
                safe_gimbal_move_async(ep_gimbal, 90)
            elif key in (ord('x'), ord('X')):
                safe_gimbal_move_async(ep_gimbal, 180)

            elif key in (ord('m'), ord('M')):
                view_mode = filter_modes[(filter_modes.index(view_mode) + 1) % len(filter_modes)]
            elif key in (ord('s'), ord('S')):
                save_config(config_data)
            elif key in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()