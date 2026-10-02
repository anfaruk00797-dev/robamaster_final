import pygame
import heapq
import itertools
import json
import sys
import math

# ==========================================
# CONFIGURATION (BOTTOM-LEFT ORIGIN & HEADING)
# ==========================================
print("==========================================")
print(" 🗺️  MOCK MAP PLANNER (BOTTOM-LEFT (0,0)) ")
print("==========================================")
GRID_W = int(input("กรอกความกว้างสนาม (จำนวนช่อง X): "))
GRID_H = int(input("กรอกความยาวสนาม (จำนวนช่อง Y): "))

CELL_SIZE = 70 if max(GRID_W, GRID_H) > 6 else 80
MARGIN = 20
PANEL_WIDTH = 360

WIN_WIDTH = (GRID_W * CELL_SIZE) + (MARGIN * 2) + PANEL_WIDTH
WIN_HEIGHT = max((GRID_H * CELL_SIZE) + (MARGIN * 2), 700)

# Colors
COLOR_BG = (25, 28, 36)
COLOR_GRID_BG = (40, 44, 52)
COLOR_GRID_LINE = (60, 64, 72)
COLOR_WALL = (231, 76, 60)
COLOR_TARGET = (241, 196, 15)
COLOR_ROBOT = (46, 204, 113)
COLOR_PATH = (52, 152, 219)
COLOR_TEXT = (236, 240, 241)
COLOR_PANEL = (33, 37, 43)

class MockMapPlanner:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption(f"RoboMaster EP - Mock Map (Bottom-Left 0,0) [{GRID_W}x{GRID_H}]")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("Tahoma", 15, bold=True)
        self.font_small = pygame.font.SysFont("Tahoma", 13)
        
        # 🌟 จุดเริ่มต้นของหุ่นยนต์ (สามารถเปลี่ยนได้โดยการคลิกขวา)
        self.start_pos = (0, 0)
        self.robot_pos = (0, 0)
        self.robot_heading = 0.0  
        
        self.targets = set()
        self.walls = set()
        
        self.path = []
        self.route_order = []
        self.direction_steps = []
        
        self.simulating = False
        self.robot_path_index = 0
        self.save_message_timer = 0

    def has_wall(self, c1, c2):
        return (c1, c2) in self.walls or (c2, c1) in self.walls

    def toggle_wall(self, c1, c2):
        if (c1, c2) in self.walls:
            self.walls.remove((c1, c2))
        elif (c2, c1) in self.walls:
            self.walls.remove((c2, c1))
        else:
            self.walls.add((c1, c2))

    def export_map_to_json(self, filename="map_config.json"):
        formatted_walls = [[[int(c1[0]), int(c1[1])], [int(c2[0]), int(c2[1])]] for c1, c2 in self.walls]
        formatted_targets = [{"x": int(t[0]), "y": int(t[1])} for t in self.targets]
            
        # 🌟 บันทึก start position ลงในไฟล์ JSON อัตโนมัติ
        data = {
            "grid_size": {"w": int(GRID_W), "h": int(GRID_H)},
            "start": {"x": int(self.start_pos[0]), "y": int(self.start_pos[1])},
            "targets": formatted_targets,
            "walls": formatted_walls
        }
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 [SUCCESS] บันทึกไฟล์แผนที่ลง '{filename}' เรียบร้อยแล้ว! (จุดเริ่มหุ่น: {self.start_pos})")
            self.save_message_timer = pygame.time.get_ticks()
        except Exception as e:
            print(f"❌ [ERROR] บันทึกไฟล์ไม่สำเร็จ: {e}")

    def a_star(self, start, target):
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
                neighbor = (nx, ny)
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                    if not self.has_wall(current, neighbor):
                        tentative_g = g_score[current] + 1
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + heuristic(neighbor, target)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def convert_path_to_directions(self, full_path):
        directions = []
        for i in range(len(full_path) - 1):
            x1, y1 = full_path[i]
            x2, y2 = full_path[i+1]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 1: directions.append("เดินขวา (Right)")
            elif dx == -1: directions.append("เดินซ้าย (Left)")
            elif dy == 1: directions.append("เดินขึ้น (North)")
            elif dy == -1: directions.append("เดินลง (South)")
        return directions

    def calculate_tsp_path(self):
        if not self.targets:
            self.path = []
            self.route_order = []
            self.direction_steps = []
            return
            
        start = self.start_pos
        targets = list(self.targets)
        best_full_path = None
        best_route_order = None
        min_total_len = float('inf')
        
        for perm in itertools.permutations(targets):
            current = start
            current_path = [start]
            valid_perm = True
            total_len = 0
            for t in perm:
                subpath = self.a_star(current, t)
                if subpath is None:
                    valid_perm = False
                    break
                total_len += len(subpath)
                current_path.extend(subpath)
                current = t
                
            if valid_perm and total_len < min_total_len:
                min_total_len = total_len
                best_full_path = current_path
                best_route_order = list(perm)
                
        if best_full_path:
            self.path = best_full_path
            self.route_order = best_route_order
            self.direction_steps = self.convert_path_to_directions(best_full_path)
            print(f"🎯 ลำดับเก็บเป้าหมายที่ดีที่สุด: {self.route_order}")
        else:
            print("❌ ไม่สามารถหาเส้นทางไปเก็บเป้าหมายทั้งหมดได้")
            self.path = []
            self.route_order = []
            self.direction_steps = []

    def run(self):
        running = True
        while running:
            self.screen.fill(COLOR_BG)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.simulating:
                    mx, my = event.pos
                    map_origin_x = MARGIN
                    map_origin_y = MARGIN
                    grid_pixel_w = GRID_W * CELL_SIZE
                    grid_pixel_h = GRID_H * CELL_SIZE
                    
                    if map_origin_x <= mx <= map_origin_x + grid_pixel_w and map_origin_y <= my <= map_origin_y + grid_pixel_h:
                        gx = (mx - map_origin_x) // CELL_SIZE
                        rel_y_pixel = my - map_origin_y
                        gy = GRID_H - 1 - (rel_y_pixel // CELL_SIZE)
                        
                        # 🌟 [คลิกขวา]: กำหนดตำแหน่งเริ่มต้นของหุ่นยนต์
                        if event.button == 3:
                            self.start_pos = (gx, gy)
                            self.robot_pos = (gx, gy)
                            self.robot_heading = 0.0
                            if (gx, gy) in self.targets:
                                self.targets.remove((gx, gy))
                            self.path.clear()
                            self.route_order.clear()
                            self.direction_steps.clear()
                            print(f"🤖 ย้ายตำแหน่งเริ่มต้นหุ่นยนต์ไปที่: ({gx}, {gy})")

                        # 🌟 [คลิกซ้าย]: จัดการกำแพง หรือ วางเป้าหมาย
                        elif event.button == 1:
                            rel_x = (mx - map_origin_x) % CELL_SIZE
                            rel_y = rel_y_pixel % CELL_SIZE
                            edge_threshold = 12
                            
                            if rel_x < edge_threshold and gx > 0:
                                self.toggle_wall((gx, gy), (gx - 1, gy))
                            elif rel_x > CELL_SIZE - edge_threshold and gx < GRID_W - 1:
                                self.toggle_wall((gx, gy), (gx + 1, gy))
                            elif rel_y < edge_threshold and gy < GRID_H - 1:
                                self.toggle_wall((gx, gy), (gx, gy + 1))
                            elif rel_y > CELL_SIZE - edge_threshold and gy > 0:
                                self.toggle_wall((gx, gy), (gx, gy - 1))
                            else:
                                if (gx, gy) != self.start_pos:
                                    if (gx, gy) in self.targets:
                                        self.targets.remove((gx, gy))
                                    else:
                                        self.targets.add((gx, gy))
                                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.targets.clear()
                        self.walls.clear()
                        self.path.clear()
                        self.route_order.clear()
                        self.direction_steps.clear()
                        self.robot_pos = self.start_pos
                        self.robot_heading = 0.0
                    elif event.key == pygame.K_SPACE:
                        self.robot_pos = self.start_pos
                        self.calculate_tsp_path()
                    elif event.key == pygame.K_s:
                        self.export_map_to_json("map_config.json")
                    elif event.key == pygame.K_v:
                        if self.path:
                            self.simulating = True
                            self.robot_path_index = 0
                            self.robot_pos = self.start_pos

            # --- Simulation & Heading Update ---
            if self.simulating:
                pygame.time.delay(350)
                if self.robot_path_index < len(self.path) - 1:
                    curr_cell = self.path[self.robot_path_index]
                    next_cell = self.path[self.robot_path_index + 1]
                    
                    dx = next_cell[0] - curr_cell[0]
                    dy = next_cell[1] - curr_cell[1]
                    
                    if dx == 1: self.robot_heading = 90.0
                    elif dx == -1: self.robot_heading = -90.0
                    elif dy == 1: self.robot_heading = 0.0
                    elif dy == -1: self.robot_heading = 180.0
                    
                    self.robot_path_index += 1
                    self.robot_pos = self.path[self.robot_path_index]
                else:
                    self.simulating = False
                    print("🎉 จำลองการวิ่งเก็บเป้าหมายครบทุกจุดเรียบร้อย!")

            # --- Drawing Grid (Bottom-Left Origin 0,0) ---
            map_origin_x = MARGIN
            map_origin_y = MARGIN
            
            for gy in range(GRID_H):
                for gx in range(GRID_W):
                    px = map_origin_x + (gx * CELL_SIZE)
                    py = map_origin_y + ((GRID_H - 1 - gy) * CELL_SIZE)
                    
                    rect = (px, py, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, COLOR_GRID_BG, rect)
                    pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)
                    
                    lbl = self.font_small.render(f"{gx},{gy}", True, (120, 130, 140))
                    self.screen.blit(lbl, (px + 4, py + 4))
                    
                    if (gx, gy) in self.targets:
                        pygame.draw.circle(self.screen, COLOR_TARGET, (px + CELL_SIZE//2, py + CELL_SIZE//2), min(20, CELL_SIZE//3))

            # วาดเส้นทาง Path
            if len(self.path) > 1:
                points = []
                for pt in self.path:
                    pt_px = map_origin_x + pt[0]*CELL_SIZE + CELL_SIZE//2
                    pt_py = map_origin_y + ((GRID_H - 1 - pt[1])*CELL_SIZE) + CELL_SIZE//2
                    points.append((pt_px, pt_py))
                pygame.draw.lines(self.screen, COLOR_PATH, False, points, 4)

            # วาดกำแพง (Walls)
            for (c1, c2) in self.walls:
                x1, y1 = c1
                x2, y2 = c2
                if x1 == x2:
                    max_y = max(y1, y2)
                    py = map_origin_y + ((GRID_H - max_y) * CELL_SIZE)
                    px1 = map_origin_x + (x1 * CELL_SIZE)
                    px2 = px1 + CELL_SIZE
                    pygame.draw.line(self.screen, COLOR_WALL, (px1, py), (px2, py), 6)
                else:
                    max_x = max(x1, x2)
                    px = map_origin_x + (max_x * CELL_SIZE)
                    py1 = map_origin_y + ((GRID_H - 1 - y1) * CELL_SIZE)
                    py2 = py1 + CELL_SIZE
                    pygame.draw.line(self.screen, COLOR_WALL, (px, py1), (px, py2), 6)

            # วาดหุ่นยนต์
            rx, ry = self.robot_pos
            r_px = map_origin_x + (rx * CELL_SIZE) + CELL_SIZE//2
            r_py = map_origin_y + ((GRID_H - 1 - ry) * CELL_SIZE) + CELL_SIZE//2
            pygame.draw.circle(self.screen, COLOR_ROBOT, (r_px, r_py), min(18, CELL_SIZE//3))
            
            heading_rad = math.radians(self.robot_heading)
            dir_x = r_px + 24 * math.sin(heading_rad)
            dir_y = r_py - 24 * math.cos(heading_rad)
            pygame.draw.line(self.screen, (255, 255, 255), (r_px, r_py), (dir_x, dir_y), 3)

            # --- Drawing Control Panel ---
            panel_x = MARGIN + (GRID_W * CELL_SIZE) + MARGIN
            panel_rect = (panel_x, MARGIN, PANEL_WIDTH, WIN_HEIGHT - (MARGIN * 2))
            pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect, border_radius=8)
            
            p_y = MARGIN + 15
            
            if self.robot_heading == 0: heading_name = "North (N)"
            elif self.robot_heading == 90: heading_name = "East (E)"
            elif abs(self.robot_heading) == 180: heading_name = "South (S)"
            else: heading_name = "West (W)"
            
            lines = [
                f"MAP SIZE: {GRID_W} x {GRID_H} (Origin: Bottom-Left)",
                f"Start Pos: ({self.start_pos[0]}, {self.start_pos[1]})",
                f"Robot Heading: {heading_name} ({self.robot_heading}°)",
                "----------------------------------------",
                "• คลิกขวา: กำหนดตำแหน่งเริ่มของหุ่น",
                "• คลิกซ้าย (กลางช่อง): วาง/ลบ เป้าหมาย",
                "• คลิกซ้าย (ริมขอบ): สร้าง/ลบ กำแพง",
                "• [SPACEBAR]: คำนวณเส้นทางสั้นที่สุด",
                "• [S]: บันทึกแผนลง 'map_config.json'",
                "• [V]: จำลองการวิ่ง (แสดงทิศทางหันหน้า)",
                "• [C]: ล้างข้อมูลทั้งหมด",
                "----------------------------------------",
                f"เป้าหมายคงเหลือ: {len(self.targets)} จุด",
                f"สถานะ: {'กำลังเดิน...' if self.simulating else 'พร้อม'}",
                "----------------------------------------",
                "ลำดับทิศทางการเดิน:"
            ]
            
            for line in lines:
                color = (241, 196, 15) if "MAP SIZE" in line or "Start Pos" in line or "Robot Heading" in line or "ลำดับทิศทาง" in line else COLOR_TEXT
                txt_surf = self.font_small.render(line, True, color)
                self.screen.blit(txt_surf, (panel_x + 10, p_y))
                p_y += 22

            if pygame.time.get_ticks() - self.save_message_timer < 2000:
                save_notify = self.font.render("💾 บันทึก map_config.json แล้ว!", True, (46, 204, 113))
                self.screen.blit(save_notify, (panel_x + 10, p_y + 5))
                p_y += 25

            max_show = 6
            start_idx = max(0, len(self.direction_steps) - max_show)
            for step_idx in range(start_idx, len(self.direction_steps)):
                step_text = f"  {step_idx+1}. {self.direction_steps[step_idx]}"
                txt_surf = self.font_small.render(step_text, True, (100, 220, 130))
                self.screen.blit(txt_surf, (panel_x + 10, p_y))
                p_y += 20

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = MockMapPlanner()
    app.run()