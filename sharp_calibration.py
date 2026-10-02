import json
import os
import time
from robomaster import robot
## Calibration Sharp IR Sensor -> บันทึกลงไฟล์ JSON เพื่อใช้ในการคำนวณระยะทางจากค่า ADC
JSON_FILE = "calibration_sharp.json"

def get_average_adc(sensor_adaptor, s_id, s_port, samples=30):
    """อ่านค่า ADC หลายๆ ครั้งเพื่อหาค่าเฉลี่ยกรอง Noise"""
    total = 0
    valid_samples = 0
    for _ in range(samples):
        try:
            val = sensor_adaptor.get_adc(id=s_id, port=s_port)
            if val is not None and val > 0:
                total += val
                valid_samples += 1
        except Exception:
            pass
        time.sleep(0.005)
    return int(total / valid_samples) if valid_samples > 0 else 0

def collect_noise_data(sensor_adaptor, s_id, s_port, samples=150):
    """เก็บค่า Noise ในพื้นที่โล่งจำนวนหลายๆ ตัวอย่าง เพื่อหาค่า Max, Min, Avg"""
    raw_values = []
    print(f"  กำลังเก็บตัวอย่าง Noise จำนวน {samples} ค่า...")
    for i in range(samples):
        try:
            val = sensor_adaptor.get_adc(id=s_id, port=s_port)
            if val is not None and val > 0:
                raw_values.append(val)
        except Exception:
            pass
        time.sleep(0.005)

    if not raw_values:
        return {"avg": 0, "max": 0, "min": 0, "threshold": 0}

    avg_val = int(sum(raw_values) / len(raw_values))
    max_val = max(raw_values)
    min_val = min(raw_values)
    
    # กำหนดค่า Threshold สำหรับบล็อก (ใช้ค่า Max + Safety Margin 10-15 ADC)
    safety_threshold = max_val + 15

    return {
        "avg": avg_val,
        "max": max_val,
        "min": min_val,
        "threshold": safety_threshold
    }

def load_existing_json():
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "LEFT_LUT": [], 
        "RIGHT_LUT": [],
        "NOISE_CONFIG": {
            "LEFT_NOISE": {"avg": 0, "max": 0, "threshold": 0},
            "RIGHT_NOISE": {"avg": 0, "max": 0, "threshold": 0}
        }
    }

def save_to_json(data):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ บันทึกค่า Calibration เรียบร้อยลงไฟล์: {os.path.abspath(JSON_FILE)}")

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    sensor = ep_robot.sensor_adaptor

    RIGHT_ID, RIGHT_PORT = 1, 1
    LEFT_ID, LEFT_PORT = 2, 1

    test_distances = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    print("============================================")
    print("  Calibration Sharp IR -> บันทึกลง JSON")
    print("============================================")
    print("  1 - วัดเฉพาะข้างซ้าย (Left)")
    print("  2 - วัดเฉพาะข้างขวา (Right)")
    print("  3 - วัดพร้อมกันทั้งสองข้าง (Both)")
    print("  4 - วัดค่า Noise พื้นที่โล่ง/ไม่มีกำแพง (Open Space Noise)")
    
    mode = input("\nเลือกโหมด (1/2/3/4): ").strip()

    json_data = load_existing_json()
    temp_left = []
    temp_right = []

    try:
        if mode == '4':
            print("\n-------------------------------------------------")
            print("⚠️ กรุณาหันหุ่นยนต์/เซนเซอร์ไปทางพื้นที่โล่ง (ไม่มีกำแพงในระยะ 1 เมตร)")
            input("กด Enter เมื่อพร้อมเริ่มเก็บค่า Noise...")

            print("\n[1/2] เก็บค่า Noise เซนเซอร์ซ้าย (ID:2)...")
            left_noise = collect_noise_data(sensor, LEFT_ID, LEFT_PORT, samples=150)
            print(f"  -> ซ้าย: Avg={left_noise['avg']}, Max={left_noise['max']} => Block Threshold={left_noise['threshold']}")

            print("\n[2/2] เก็บค่า Noise เซนเซอร์ขวา (ID:1)...")
            right_noise = collect_noise_data(sensor, RIGHT_ID, RIGHT_PORT, samples=150)
            print(f"  -> ขวา: Avg={right_noise['avg']}, Max={right_noise['max']} => Block Threshold={right_noise['threshold']}")

            if "NOISE_CONFIG" not in json_data:
                json_data["NOISE_CONFIG"] = {}

            json_data["NOISE_CONFIG"]["LEFT_NOISE"] = left_noise
            json_data["NOISE_CONFIG"]["RIGHT_NOISE"] = right_noise
            
            save_to_json(json_data)

        elif mode in ['1', '2', '3']:
            for dist_cm in test_distances:
                prompt = input(f"\n👉 ระยะ [ {dist_cm} cm ] - วางวัตถุแล้วกด Enter (พิมพ์ 'q' เพื่อหยุด): ")
                if prompt.lower() == 'q':
                    break

                print("  กำลังอ่านค่า ADC...")

                if mode == '1':
                    adc_l = get_average_adc(sensor, LEFT_ID, LEFT_PORT)
                    print(f"  [ซ้าย ID:2] => ADC: {adc_l}")
                    temp_left.append([adc_l, dist_cm])

                elif mode == '2':
                    adc_r = get_average_adc(sensor, RIGHT_ID, RIGHT_PORT)
                    print(f"  [ขวา ID:1] => ADC: {adc_r}")
                    temp_right.append([adc_r, dist_cm])

                elif mode == '3':
                    adc_l = get_average_adc(sensor, LEFT_ID, LEFT_PORT)
                    adc_r = get_average_adc(sensor, RIGHT_ID, RIGHT_PORT)
                    print(f"  [ซ้าย ID:2] => ADC: {adc_l} | [ขวา ID:1] => ADC: {adc_r}")
                    temp_left.append([adc_l, dist_cm])
                    temp_right.append([adc_r, dist_cm])

            if temp_left:
                json_data["LEFT_LUT"] = temp_left
            if temp_right:
                json_data["RIGHT_LUT"] = temp_right

            save_to_json(json_data)

        else:
            print("เลือกโหมดไม่ถูกต้อง")

    except KeyboardInterrupt:
        print("\nยกเลิกการสแกน")

    ep_robot.close()