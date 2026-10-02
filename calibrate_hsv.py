"""Live HSV lower/upper calibration for target detection."""

import cv2
from robomaster import robot

from target_config import COLOR_NAMES, color_mask, load_ranges, save_ranges, reset_gimbal
from target_shooter import DISPLAY_COLORS, SHAPE_NAMES, classify

WINDOW = "HSV controls"
SLIDERS = ("H low", "S low", "V low", "H high", "S high", "V high")
MAX_VALUES = (179, 255, 255, 179, 255, 255)


def main():
    ranges = load_ranges()
    bot = robot.Robot()
    bot.initialize(conn_type="ap")
    camera = bot.camera
    reset_gimbal(bot.gimbal)
    camera.start_video_stream(display=False)
    cv2.namedWindow(WINDOW)
    for name, maximum in zip(SLIDERS, MAX_VALUES):
        cv2.createTrackbar(name, WINDOW, 0, maximum, lambda _value: None)

    color, slot = "r", 0

    def show_values():
        for name, value in zip(SLIDERS, ranges[color][slot]):
            cv2.setTrackbarPos(name, WINDOW, value)

    show_values()
    print("Colors: r/y/g/b | n: next range (if present) | s: save | q: quit")
    try:
        while True:
            image = camera.read_cv2_image(strategy="newest")
            if image is None:
                continue
            values = [cv2.getTrackbarPos(name, WINDOW) for name in SLIDERS]
            ranges[color][slot] = values
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = color_mask(hsv, color, ranges)
            preview = cv2.bitwise_and(image, image, mask=mask)
            height, width = image.shape[:2]
            aim_x, aim_y = width // 2, height // 2
            for code_color in COLOR_NAMES:
                all_mask = color_mask(hsv, code_color, ranges)
                contours, _ = cv2.findContours(all_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    result = classify(contour)
                    if result is None or result[2] < width * height * 0.00015:
                        continue
                    shape, (x, y, bw, bh), _area = result
                    paint = DISPLAY_COLORS[code_color]
                    cv2.drawContours(image, [contour], -1, paint, 2)
                    cv2.rectangle(image, (x, y), (x + bw, y + bh), paint, 1)
                    cv2.circle(image, (x + bw // 2, y + bh // 2), 5, paint, -1)
                    cv2.putText(image, f"{shape}{code_color} {SHAPE_NAMES[shape]} {COLOR_NAMES[code_color]}",
                                (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, paint, 1)
            cv2.drawMarker(image, (aim_x, aim_y), (255, 255, 255),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.rectangle(image, (aim_x - 18, aim_y - 18),
                          (aim_x + 18, aim_y + 18), (0, 255, 0), 1)
            cv2.putText(image, f"{COLOR_NAMES[color]} range {slot + 1}/{len(ranges[color])}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera", image)
            cv2.imshow("HSV mask", mask)
            cv2.imshow("Selected pixels", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in map(ord, COLOR_NAMES):
                color, slot = chr(key), 0
                show_values()
            elif key == ord("n"):
                slot = (slot + 1) % len(ranges[color])
                show_values()
            elif key == ord("s"):
                invalid = [name for name, intervals in ranges.items()
                           for item in intervals
                           if any(item[i] > item[i + 3] for i in range(3))]
                if invalid:
                    print(f"Cannot save: lower exceeds upper for {', '.join(sorted(set(invalid)))}")
                else:
                    save_ranges(ranges)
                    print("Saved HSV values to hsv_ranges.json")
    finally:
        camera.stop_video_stream()
        bot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
