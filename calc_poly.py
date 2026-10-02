import json
import numpy as np
#แปลงค่าเป็นสมการกำลังสองจากไฟล์ calibration_sharp.json เป็น calibration_sharp_poly.json เพื่อให้สามารถคำนวณระยะทางจากค่า ADC ได้ง่ายขึ้น
def generate_poly_json(input_filename="calibration_sharp.json", output_filename="calibration_sharp_poly.json"):
    # 1. อ่านข้อมูลจากไฟล์ JSON ต้นทาง
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. คำนวณสมการกำลังสองฝั่งซ้าย
    left_x = [pt[0] for pt in data["LEFT_LUT"]]
    left_y = [pt[1] for pt in data["LEFT_LUT"]]
    a_l, b_l, c_l = np.polyfit(left_x, left_y, 2)

    # 3. คำนวณสมการกำลังสองฝั่งขวา
    right_x = [pt[0] for pt in data["RIGHT_LUT"]]
    right_y = [pt[1] for pt in data["RIGHT_LUT"]]
    a_r, b_r, c_r = np.polyfit(right_x, right_y, 2)

    # 4. ดึงค่า Noise Threshold
    noise_cfg = data.get("NOISE_CONFIG", {})
    left_noise_thresh = noise_cfg.get("LEFT_NOISE", {}).get("threshold", int(min(left_x)))
    right_noise_thresh = noise_cfg.get("RIGHT_NOISE", {}).get("threshold", int(min(right_x)))

    # 5. จัดโครงสร้างข้อมูลใหม่
    poly_data = {
        "LEFT_SENSOR": {
            "a": round(float(a_l), 8),
            "b": round(float(b_l), 8),
            "c": round(float(c_l), 8),
            "min_adc": int(min(left_x)),
            "max_adc": int(max(left_x)),
            "noise_threshold": int(left_noise_thresh)
        },
        "RIGHT_SENSOR": {
            "a": round(float(a_r), 8),
            "b": round(float(b_r), 8),
            "c": round(float(c_r), 8),
            "min_adc": int(min(right_x)),
            "max_adc": int(max(right_x)),
            "noise_threshold": int(right_noise_thresh)
        }
    }

    # 6. บันทึกลงไฟล์ปลายทาง
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(poly_data, f, indent=4, ensure_ascii=False)

    print(f"✅ อ่านจาก '{input_filename}' -> บันทึกเป็น '{output_filename}' เรียบร้อย!")


if __name__ == "__main__":
    # ใส่ชื่อไฟล์ที่ต้องการดึงมาอ่าน และชื่อไฟล์ที่ต้องการเซฟออกไป
    generate_poly_json(
        input_filename="calibration_sharp.json", 
        output_filename="calibration_sharp_poly.json"
    )