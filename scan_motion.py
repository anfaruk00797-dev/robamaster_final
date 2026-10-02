"""Alignment and temporary 10 cm chassis offsets for target scanning."""
import logging
import time

import alig


def align_before_scan(bot):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if bot.set_robot_mode(mode="free") is False:
        raise RuntimeError("Cannot select FREE mode before alignment")
    reader = alig.DistanceReader()
    if not bot.sensor.sub_distance(freq=alig.TOF_FREQUENCY_HZ, callback=reader.callback):
        raise RuntimeError("Cannot subscribe to ToF for alignment")
    try:
        alig.wait_action(bot.gimbal.recenter(
            pitch_speed=alig.GIMBAL_SPEED_DEG_S, yaw_speed=alig.GIMBAL_SPEED_DEG_S),
            "Initial alignment recenter")
        walls = alig.scan_walls(bot, reader)
        results = alig.align_detected_walls(bot, reader, walls)
        alig.point_gimbal(bot, 0)
        print("Alignment results:", results)
        if not any(results.values()):
            raise RuntimeError("Alignment not confirmed; target scan stopped")
        return results
    finally:
        try:
            bot.chassis.drive_speed(x=0, y=0, z=0)
        finally:
            try:
                bot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            finally:
                bot.sensor.unsub_distance()


def away_vector(bearing_deg):
    """SDK axes: x forward, y right. Select nearest body side."""
    sector = int(((bearing_deg % 360) + 45) // 90) % 4
    return ((-0.10, 0.0), (0.0, -0.10),
            (0.10, 0.0), (0.0, 0.10))[sector]


class OffsetMove:
    def __init__(self, bot):
        self.bot = bot
        self.offset = (0.0, 0.0)
        self.away = False
        self.action = None
        self.started = None
        self.completed_at = None
        self.returning = False

    def begin(self, bearing_deg=None, returning=False):
        self.returning = returning
        if returning:
            x, y = (-self.offset[0], -self.offset[1])
        else:
            self.offset = away_vector(bearing_deg)
            x, y = self.offset
        self.action = self.bot.chassis.move(x=x, y=y, z=0, xy_speed=0.15)
        self.started = time.monotonic()
        self.completed_at = None
        print(f"{'RETURN' if returning else 'STEP AWAY'}: x={x:+.2f} y={y:+.2f} m")

    def poll(self):
        now = time.monotonic()
        if not self.action.is_completed:
            if now - self.started > 5:
                self.bot.chassis.drive_speed(x=0, y=0, z=0)
                raise RuntimeError("Chassis offset timed out; position uncertain, scan stopped")
            return False
        if not self.action.has_succeeded:
            raise RuntimeError("Chassis offset failed; position uncertain, scan stopped")
        if self.completed_at is None:
            self.completed_at = now
        if now - self.completed_at < 0.2:
            return False
        self.away = not self.returning
        return True
