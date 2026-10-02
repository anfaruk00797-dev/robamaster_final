"""map_scan sweep/detection adapter for final_round1's shared robot and map."""
import math
import time
import map_scan as scan

SHAPES = {'u': 'SQUARE', 'o': 'CIRCLE', 'i': 'RECT_H', 'p': 'RECT_V'}
COLORS = {'r': 'red', 'y': 'yellow', 'g': 'green', 'b': 'blue'}


def requested_codes(ctx):
    return {s+c for s,shape in SHAPES.items() for c,color in COLORS.items()
            if (ctx.TARGET_COLOR == 'ALL_COLORS' or color == ctx.TARGET_COLOR)
            and (ctx.TARGET_SHAPE == 'ALL_SHAPES' or shape == ctx.TARGET_SHAPE)}


def saved_marks(ctx, width, height, yaw, pitch):
    with ctx.hub.lock:
        records = list(getattr(ctx.hub, 'scan_shot_marks', []))
        rx, ry = ctx.hub.get_robot_world_pos()
        chassis_yaw = ctx.hub.pose['chassis_yaw']
    marks = []
    for item in records:
        dx,dy = item['world_x_cm']-rx,item['world_y_cm']-ry
        distance = math.hypot(dx,dy)
        direction = ctx.normalize_angle(math.degrees(math.atan2(dx,dy))-chassis_yaw)
        elevation = math.degrees(math.atan2(item['height_delta_cm'],max(distance,1)))
        projected = dict(bearing_deg=direction,elevation_deg=elevation)
        marks.append((item, scan.project_saved_mark(projected,width,height,yaw,pitch)))
    return marks


def already_shot(detection, marks):
    return any(item['code'] == detection[0] and scan.mark_inside_box(point,detection[1],scan.MARK_MATCH_MARGIN_PX)
               for item,point in marks)


def publish(ctx, image, found, state='SCAN', chosen=None):
    height,width = image.shape[:2]
    with ctx.hub.lock:
        yaw,pitch = ctx.hub.gimbal_yaw,ctx.hub.gimbal_pitch
    marks = saved_marks(ctx,width,height,yaw,pitch)
    requested = requested_codes(ctx)
    view = image.copy()
    center = (width//2,height//2)
    inner_x = max(8,round(width*scan.INNER_HALF_WIDTH_FRACTION))
    inner_y = max(8,round(height*scan.INNER_HALF_HEIGHT_FRACTION))
    accept_x = max(24,round(width*scan.ACCEPT_HALF_WIDTH_FRACTION))
    accept_y = max(20,round(height*scan.ACCEPT_HALF_HEIGHT_FRACTION))
    cv2 = scan.cv2
    for detection in found:
        code,(x,y,w,h),cx,cy = detection
        fired = already_shot(detection,marks)
        selected = chosen is not None and detection == chosen
        paint = (0,0,0) if fired else ((255,0,255) if selected else scan.DISPLAY_COLORS[code[1]])
        reason = 'IGNORED: IR SENT' if fired else ('NOT REQUESTED' if code not in requested else ('LOCKED' if selected else 'READY'))
        cv2.rectangle(view,(x,y),(x+w,y+h),paint,-1 if fired else 2)
        cv2.putText(view,code+' '+reason,(x,max(18,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255) if fired else paint,2)
        if selected:
            cv2.line(view,center,(cx,cy),(255,0,255),2)
    cv2.drawMarker(view,center,(255,255,255),cv2.MARKER_CROSS,30,2)
    cv2.rectangle(view,(center[0]-accept_x,center[1]-accept_y),(center[0]+accept_x,center[1]+accept_y),(0,165,255),2)
    cv2.rectangle(view,(center[0]-inner_x,center[1]-inner_y),(center[0]+inner_x,center[1]+inner_y),(0,255,0),1)
    cv2.putText(view,state+' | requested '+','.join(sorted(requested)),(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    view.flags.writeable = False
    with ctx.hub.lock:
        ctx.hub.scan_preview = view
    ctx.hub.set_active_detections([])



class ScanStopped(RuntimeError):
    pass


def check(ctx):
    if not ctx.hub.running or ctx.hub.emergency_stop:
        raise ScanStopped('Grid scan stopped')
    if time.monotonic() >= getattr(ctx.hub, 'scan_round_deadline', float('inf')):
        raise ScanStopped('Round 1 time limit reached during grid scan')


def fresh_frame(ctx, sequence, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check(ctx)
        with ctx.hub.lock:
            if ctx.hub.camera_sequence > sequence and ctx.hub.latest_frame is not None:
                return ctx.hub.camera_sequence, ctx.hub.latest_frame
        time.sleep(.015)
    raise RuntimeError('No fresh camera frame during grid scan')


def point(ctx, gimbal, yaw, pitch):
    check(ctx)
    with ctx.sdk_lock:
        gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        action = gimbal.moveto(yaw=yaw, pitch=pitch, yaw_speed=120, pitch_speed=90)
    if not action.wait_for_completed(timeout=5) or getattr(action, 'has_succeeded', True) is False:
        raise RuntimeError('Grid scan gimbal action failed')
    completed_at = time.monotonic()
    deadline = time.monotonic()+1.5
    while time.monotonic() < deadline:
        check(ctx)
        with ctx.hub.lock:
            good = (ctx.hub.gimbal_sample_time > completed_at
                    and abs(ctx.normalize_angle(ctx.hub.gimbal_yaw-yaw)) <= 2.5
                    and abs(ctx.hub.gimbal_pitch-pitch) <= 2.5)
        if good:
            return
        time.sleep(.02)
    with ctx.hub.lock:
        measured_yaw = ctx.hub.gimbal_yaw
        measured_pitch = ctx.hub.gimbal_pitch
        body_pitch = getattr(ctx.hub,'gimbal_pitch_body',measured_pitch)
        age = time.monotonic()-ctx.hub.gimbal_sample_time
    raise RuntimeError('Grid scan gimbal did not settle: requested yaw=%.1f pitch=%.1f; '
                       'actual yaw=%.1f pitch_ground=%.1f pitch_body=%.1f; age=%.3fs' %
                       (yaw,pitch,measured_yaw,measured_pitch,body_pitch,age))


def fresh_range(ctx, timeout=1.0):
    with ctx.hub.lock:
        mark = ctx.hub.tof_last_update_time
    values = []
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        check(ctx)
        with ctx.hub.lock:
            stamp, value = ctx.hub.tof_last_update_time, ctx.hub.tof_raw
        if stamp > mark:
            mark = stamp
            if math.isfinite(value) and 2 <= value <= 300:
                values.append(value)
                if len(values) == 3:
                    return sorted(values)[1]
        time.sleep(.01)
    return None


def aim_confirm(ctx, gimbal, candidate, ranges, all_codes):
    code, box, cx, cy, width = candidate
    if code not in requested_codes(ctx):
        return
    with ctx.hub.lock:
        bearing = scan.target_bearing(cx,width,ctx.hub.gimbal_yaw)
        sequence = ctx.hub.camera_sequence
    deadline = time.monotonic()+scan.AIM_TIMEOUT_S
    last_seen = time.monotonic()
    try:
        while time.monotonic() < deadline:
            sequence,image = fresh_frame(ctx,sequence)
            with ctx.hub.lock:
                yaw,pitch = ctx.hub.gimbal_yaw,ctx.hub.gimbal_pitch
            found = scan.find_matches(image,all_codes,ranges)
            chosen = scan.match_locked_target(found,code,bearing,image.shape[1],yaw)
            marks = saved_marks(ctx,image.shape[1],image.shape[0],yaw,pitch)
            if chosen is not None and already_shot(chosen,marks):
                publish(ctx,image,found,'ALREADY FIRED')
                return
            publish(ctx,image,found,'AIM',chosen)
            if chosen is None:
                gimbal.drive_speed(pitch_speed=0,yaw_speed=0)
                if time.monotonic()-last_seen > scan.TARGET_LOST_S:
                    return
                continue
            last_seen = time.monotonic()
            ex,ey = chosen[2]-image.shape[1]/2,chosen[3]-image.shape[0]/2
            # Same orange acceptance box and green precision box as map_scan.
            accept_x = max(24,round(image.shape[1]*scan.ACCEPT_HALF_WIDTH_FRACTION))
            accept_y = max(20,round(image.shape[0]*scan.ACCEPT_HALF_HEIGHT_FRACTION))
            if abs(ex) <= accept_x and abs(ey) <= accept_y:
                gimbal.drive_speed(pitch_speed=0,yaw_speed=0)
                raw = fresh_range(ctx)
                if raw is None or not ctx.SAFE_MIN_TOF_CM <= raw <= ctx.MAX_FIRING_DIST_CM:
                    publish(ctx,image,found,'LOCKED: OUT OF RANGE',chosen)
                    ctx.log_event('[MAP SCAN] %s locked; no valid firing range' % code)
                    return
                # Range acquisition must not permit a lost/moved target to fire.
                sequence,image = fresh_frame(ctx,sequence)
                with ctx.hub.lock:
                    yaw,pitch = ctx.hub.gimbal_yaw,ctx.hub.gimbal_pitch
                    chassis_yaw = ctx.hub.pose['chassis_yaw']
                    rx,ry = ctx.hub.get_robot_world_pos()
                found = scan.find_matches(image,all_codes,ranges)
                chosen = scan.match_locked_target(found,code,bearing,image.shape[1],yaw)
                if chosen is None or abs(chosen[2]-image.shape[1]/2)>accept_x or abs(chosen[3]-image.shape[0]/2)>accept_y:
                    continue
                color,shape = COLORS[code[1]],SHAPES[code[0]]
                distance = ctx.calculate_ground_distance(raw)
                world_yaw = chassis_yaw + scan.target_bearing(chosen[2],image.shape[1],yaw)
                focal = image.shape[1]/(2*math.tan(math.radians(scan.CAMERA_HFOV_DEG/2)))
                elevation = pitch + math.degrees(math.atan((image.shape[0]/2-chosen[3])/focal))
                rad = math.radians(world_yaw)
                wx,wy = rx+distance*math.sin(rad),ry+distance*math.cos(rad)
                ctx.hub.register_target_spatial(ctx.normalize_angle(world_yaw),raw,color,shape,is_main_target=True)
                with ctx.hub.lock:
                    record = next((t for t in reversed(ctx.hub.targets_found)
                                   if t['color']==color and t['shape']==shape
                                   and math.hypot(t['world_pos_cm']['x']-wx,t['world_pos_cm']['y']-wy)<=25),None)
                    if record is not None and (record.get('is_knocked_down') or record.get('ir_status') in ('pending','uncertain','sent')):
                        return
                if ctx.ep_blaster_ref is None:
                    raise RuntimeError('IR blaster unavailable')
                mark = dict(code=code,world_x_cm=wx,world_y_cm=wy,
                            height_delta_cm=distance*math.tan(math.radians(elevation)),status='pending')
                with ctx.hub.lock:
                    ctx.hub.scan_shot_marks.append(mark)
                    if record is not None:
                        record['ir_status']='pending'
                try:
                    for _ in range(ctx.BURST_FIRE_COUNT):
                        check(ctx)
                        if ctx.ep_blaster_ref.fire(fire_type=ctx.blaster.INFRARED_FIRE,times=1) is False:
                            raise RuntimeError('IR command rejected; no retry')
                        time.sleep(.15)
                except BaseException:
                    with ctx.hub.lock:
                        mark['status']='uncertain'
                        if record is not None:record['ir_status']='uncertain'
                    raise
                with ctx.hub.lock:
                    mark['status']='sent'
                    if record is not None:
                        record['ir_status']='sent'
                        record['is_knocked_down']=True
                publish(ctx,image,found,'IR SENT: '+code)
                ctx.log_event('[MAP SCAN] IR sent '+code+'; black mask saved')
                ctx.save_target_inspection_image(image,dict(target_id=record['target_id'] if record else -1,color=color,shape=shape),
                                                 chosen[1],distance,status='IR_SENT')
                return
            py = max(-scan.MAX_PITCH_AIM_DPS,min(scan.MAX_PITCH_AIM_DPS,-ey*scan.PITCH_GAIN))
            if py<0:py=max(py,-max(0,pitch-scan.SCAN_PITCH_LOW_DEG)*2)
            ys = max(-scan.MAX_YAW_AIM_DPS,min(scan.MAX_YAW_AIM_DPS,ex*scan.YAW_GAIN))
            if abs(yaw)>245 and ys*yaw>0:ys=0
            gimbal.drive_speed(pitch_speed=py,yaw_speed=ys)
    finally:
        gimbal.drive_speed(pitch_speed=0,yaw_speed=0)



def scan_grid(ctx, gimbal, heading, gx, gy):
    ranges = scan.load_ranges()
    all_codes = {s+c for s in SHAPES for c in COLORS}
    results = {}
    skipped = []
    with ctx.hub.lock:
        if not hasattr(ctx.hub,'scan_shot_marks'):ctx.hub.scan_shot_marks=[]
    ctx.hub.set_scanning(True)
    ctx.hub.set_status_msg('map_scan: walls + target sweep')
    ctx.log_event('[MAP SCAN] Grid(%d,%d) start' % (gx, gy))
    try:
        # map_scan's three legs, with exact cardinal pauses for wall ToF.
        point(ctx, gimbal, 0, 0)
        for name, angle in (('AHEAD',0), ('RIGHT',90), ('BACK',180), ('AHEAD_RETURN',0), ('LEFT',-90), ('BACK_RETURN',-180)):
            with ctx.hub.lock:
                current = ctx.hub.gimbal_yaw
                sequence = ctx.hub.camera_sequence
            end = time.monotonic()+15
            pitch_direction = -1
            sweep_sign = 1 if angle-current >= 0 else -1
            while abs(current-angle) > scan.STOP_TOLERANCE_DEG:
                check(ctx)
                if time.monotonic() > end:
                    raise RuntimeError('Grid sweep timed out')
                sequence, image = fresh_frame(ctx, sequence)
                with ctx.hub.lock:
                    current, pitch = ctx.hub.gimbal_yaw, ctx.hub.gimbal_pitch
                # Camera acquisition may take us past the endpoint. Never send
                # another sweep command using the previous frame's yaw.
                remaining = angle-current
                if sweep_sign*remaining <= scan.STOP_TOLERANCE_DEG:
                    gimbal.drive_speed(pitch_speed=0,yaw_speed=0)
                    break
                found = scan.find_matches(image, all_codes, ranges)
                publish(ctx, image, found)
                marks = saved_marks(ctx,image.shape[1],image.shape[0],current,pitch)
                candidates = [d for d in found if d[0] in requested_codes(ctx) and not already_shot(d,marks) and not any(d[0] == code and abs(ctx.normalize_angle(scan.target_bearing(d[2], image.shape[1], current)-bearing)) < scan.TRACK_BEARING_DEG for code,bearing in skipped)]
                if candidates:
                    chosen = scan.pick_in_sweep_order(candidates, angle-current > 0)
                    bearing = scan.target_bearing(chosen[2], image.shape[1], current)
                    gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    skipped.append((chosen[0], bearing))
                    aim_confirm(ctx, gimbal, (*chosen, image.shape[1]), ranges, all_codes)
                    point(ctx, gimbal, current, pitch)
                    end = time.monotonic()+15
                remaining = angle-current
                ps, pitch_direction = scan.scan_pitch_velocity(pitch, pitch_direction)
                speed = min(scan.SCAN_SPEED_DPS,max(4.0,abs(remaining)*2.0))
                if abs(current) >= 240 and current*remaining > 0:
                    gimbal.drive_speed(pitch_speed=0,yaw_speed=0)
                    raise RuntimeError('Gimbal mechanical limit approached during sweep: %.1f' % current)
                gimbal.drive_speed(pitch_speed=ps, yaw_speed=math.copysign(speed, remaining))
            point(ctx, gimbal, angle, 0)
            if name.endswith('RETURN'):
                continue
            raw = fresh_range(ctx)
            results[name] = ctx.calculate_ground_distance(raw) if raw is not None else 999.0
            if raw is None:
                ctx.log_event('[MAP SCAN] %s ToF unknown; wall unchanged' % name)
                continue
            update_wall(ctx, heading, gx, gy, angle, results[name])
            # Inspect cardinal view too (including the initial front view).
            point(ctx, gimbal, angle, scan.HOME_PITCH_DEG)
            with ctx.hub.lock:
                sequence = ctx.hub.camera_sequence
            sequence, image = fresh_frame(ctx, sequence)
            found = scan.find_matches(image,all_codes,ranges)
            publish(ctx,image,found)
            for chosen in found:
                if chosen[0] not in requested_codes(ctx) or already_shot(chosen,saved_marks(ctx,image.shape[1],image.shape[0],angle,scan.HOME_PITCH_DEG)):
                    continue
                bearing = scan.target_bearing(chosen[2], image.shape[1], angle)
                if any(chosen[0] == c and abs(ctx.normalize_angle(bearing-b)) < scan.TRACK_BEARING_DEG for c,b in skipped):
                    continue
                skipped.append((chosen[0],bearing))
                point(ctx,gimbal,angle,scan.HOME_PITCH_DEG)
                aim_confirm(ctx,gimbal,(*chosen,image.shape[1]),ranges,all_codes)
        if any(value == 999.0 for value in results.values()):
            raise ScanStopped('Unknown wall range; hold current grid')
        return results
    finally:
        try:
            gimbal.drive_speed(pitch_speed=0,yaw_speed=0)
            if (ctx.hub.running and not ctx.hub.emergency_stop
                    and time.monotonic() < getattr(ctx.hub, 'scan_round_deadline', float('inf'))):
                point(ctx,gimbal,0,0)
        finally:
            skipped.clear()
            ctx.hub.set_active_detections([])
            ctx.hub.set_scanning(False)
            with ctx.hub.lock:
                ctx.hub.scan_preview = None
            ctx.log_event('[MAP SCAN] Grid(%d,%d) released temporary detections' % (gx,gy))


def update_wall(ctx, heading, gx, gy, angle, distance):
    direction = ctx.deg_to_cardinal_exact(ctx.normalize_angle(heading+angle))
    dx,dy = {'N':(0,1),'E':(1,0),'S':(0,-1),'W':(-1,0)}[direction]
    steps = max(1,min(int(round(distance/ctx.CELL_SIZE_CM)),3))
    for step in range(steps):
        x,y = gx+dx*step, gy+dy*step
        if not (0 <= x < ctx.hub.w and 0 <= y < ctx.hub.h):
            break
        wall = (distance <= ctx.WALL_THRESHOLD_CM if steps == 1 else distance < steps*ctx.CELL_SIZE_CM-10) if step == steps-1 else False
        ctx.hub.update_wall_prob(x,y,direction,is_wall=wall,is_direct_adjacent=step==0)
    ctx.log_event('[MAP SCAN] Wall %s distance %.1f cm' % (direction,distance))
