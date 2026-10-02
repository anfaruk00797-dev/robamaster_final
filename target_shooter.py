"""RoboMaster EP camera preview with color and shape classification."""

from collections import Counter
import math

import cv2
from robomaster import robot

from target_config import COLOR_NAMES, SHAPE_NAMES, color_mask, load_ranges, reset_gimbal

MIN_AREA_FRACTION = 0.00015
PITCH_AIM_OFFSET = 0
ACQUIRE_HALF_WIDTH_FRACTION = 0.08
ACQUIRE_HALF_HEIGHT_FRACTION = 0.18

DISPLAY_COLORS = {"r": (0, 0, 255), "y": (0, 255, 255),
                  "g": (0, 255, 0), "b": (255, 0, 0)}


def parse_targets(raw):
    items = [part.strip().lower() for part in raw.split(",")]
    if not items or any(len(s) != 2 or s[0] not in SHAPE_NAMES or
                            s[1] not in COLOR_NAMES for s in items):
        raise ValueError("Enter shape+color pairs such as ug,ir,ob,or")
    return Counter(items)


def classify(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if min(w, h) <= 0:
        return None
    ratio = w / h
    vertices = len(cv2.approxPolyDP(contour, 0.025 * perimeter, True))
    circle = 4 * math.pi * area / (perimeter * perimeter)
    extent = area / (w * h)
    if vertices >= 6 and circle >= 0.84 and 0.82 <= ratio <= 1.18:
        shape = "o"
    elif vertices == 4 and extent >= 0.65 and 1.25 <= ratio <= 1.85:
        shape = "i"
    elif vertices == 4 and extent >= 0.65 and 0.54 <= ratio <= 0.80:
        shape = "p"
    elif vertices == 4 and extent >= 0.65 and 0.82 <= ratio <= 1.18:
        shape = "u"
    else:
        return None
    return shape, (x, y, w, h), area


def find_matches(image, needed, ranges):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    matches = []
    for color in {code[1] for code in needed}:
        mask = color_mask(hsv, color, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            result = classify(contour)
            if result is None:
                continue
            shape, box, area = result
            code = shape + color
            if code in needed and area >= width * height * MIN_AREA_FRACTION:
                x, y, w, h = box
                matches.append((code, box, x + w // 2, y + h // 2 + PITCH_AIM_OFFSET))
    return matches


def preview_main():
    """Camera preview; center gimbal once, without chassis or blaster commands."""
    while True:
        try:
            requested = parse_targets(input("Targets to highlight (e.g. ug,ir,ob,or): "))
            break
        except ValueError as exc:
            print(exc)

    ranges = load_ranges()
    all_codes = {shape + color for shape in SHAPE_NAMES for color in COLOR_NAMES}
    bot = robot.Robot()
    bot.initialize(conn_type="ap")
    camera = bot.camera
    reset_gimbal(bot.gimbal)
    camera.start_video_stream(display=False)
    print("Preview only: no movement and no firing. h=reload HSV, q=quit.")
    try:
        while True:
            image = camera.read_cv2_image(timeout=0.5, strategy="newest")
            if image is None:
                continue
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            tol_x = max(8, round(18 * width / 1280))
            tol_y = max(8, round(18 * height / 720))
            acquire_x = max(20, round(width * ACQUIRE_HALF_WIDTH_FRACTION))
            acquire_y = max(20, round(height * ACQUIRE_HALF_HEIGHT_FRACTION))
            found = find_matches(image, all_codes, ranges)
            candidates = [item for item in found if item[0] in requested]
            chosen = min(candidates, key=lambda item: (
                (item[2] - center_x) ** 2 + (item[3] - center_y) ** 2)) if candidates else None

            for item in found:
                code, (x, y, bw, bh), tx, ty = item
                color = DISPLAY_COLORS[code[1]]
                is_chosen = item is chosen
                cv2.rectangle(image, (x, y), (x + bw, y + bh),
                              (255, 0, 255) if is_chosen else color, 3 if is_chosen else 2)
                cv2.circle(image, (tx, ty), 6, color, -1)
                label = f"{code}: {SHAPE_NAMES[code[0]]} {COLOR_NAMES[code[1]]}"
                if is_chosen:
                    label = "SELECTED " + label
                cv2.putText(image, label, (x, max(18, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.drawMarker(image, (center_x, center_y), (255, 255, 255),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.rectangle(image, (center_x - acquire_x, center_y - acquire_y),
                          (center_x + acquire_x, center_y + acquire_y), (0, 165, 255), 1)
            cv2.rectangle(image, (center_x - tol_x, center_y - tol_y),
                          (center_x + tol_x, center_y + tol_y), (0, 255, 0), 1)
            status = chosen[0] if chosen else "NONE"
            cv2.putText(image, f"PREVIEW ONLY | chosen: {status} | detected: {len(found)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, "Targets: " + ",".join(requested.elements()),
                        (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("RoboMaster detection preview", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                ranges = load_ranges()
                print("HSV settings reloaded")
    finally:
        camera.stop_video_stream()
        bot.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    preview_main()

