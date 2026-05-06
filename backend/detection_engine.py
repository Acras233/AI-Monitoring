import cv2
import torch
import threading
import time
import os
import json
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from backend.config import MODELS, WARNING_DIR, DETECTION_CONFIG
from backend.database import Database


def segment_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 == 0 and on_segment(p1, p2, q1)) or (o2 == 0 and on_segment(p1, p2, q2)) or \
       (o3 == 0 and on_segment(q1, q2, p1)) or (o4 == 0 and on_segment(q1, q2, p2)):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def polyline_intersect(a, b, polyline_points):
    if len(polyline_points) < 2:
        return False
    for i in range(len(polyline_points) - 1):
        if segment_intersect(a, b, polyline_points[i], polyline_points[i + 1]):
            return True
    return False


def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return True
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class BaseDetector:
    def __init__(self, model_path, detection_type):
        self.model = YOLO(model_path)
        self.detection_type = detection_type
        self.class_names = self.model.names
        self.is_running = False
        self.video_source = 0
        self.conf_threshold = DETECTION_CONFIG.get(detection_type, {}).get('conf_threshold', 0.5)
        self.db = Database()
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._current_frame = None
        self._annotated_frame = None
        self.camera_id = None
        self.roi_points = None
        self.enabled_detectors = []
    
    def get_current_frame(self):
        with self._frame_lock:
            if self._annotated_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self._annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
        return None
    
    def start(self, video_source=0, camera_id=None, roi_points=None, enabled_detectors=None):
        if self.is_running:
            return False, "检测器已在运行中"
        
        self.video_source = video_source
        self.camera_id = camera_id
        self.roi_points = roi_points
        self.enabled_detectors = enabled_detectors or [self.detection_type]
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_detection, daemon=True)
        self._thread.start()
        self.is_running = True
        self.db.update_detection_status(self.detection_type, running=True)
        return True, "检测器已启动"
    
    def stop(self):
        if not self.is_running:
            return False, "检测器未在运行"
        
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_running = False
        with self._frame_lock:
            self._current_frame = None
            self._annotated_frame = None
        self.db.update_detection_status(self.detection_type, running=False)
        return True, "检测器已停止"
    
    def _run_detection(self):
        raise NotImplementedError
    
    def _prepare_video_source(self):
        if isinstance(self.video_source, str):
            if self.video_source.isdigit():
                self.video_source = int(self.video_source)
            elif self.video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
                pass
            elif not os.path.exists(self.video_source):
                self.db.add_log('ERROR', self.detection_type, f"视频文件不存在：{self.video_source}")
                return None
        
        self.db.add_log('INFO', self.detection_type, f"正在打开视频源：{self.video_source}")
        
        cap = cv2.VideoCapture(self.video_source)
        
        if self.video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            self.db.add_log('ERROR', self.detection_type, f"无法打开视频源：{self.video_source}")
            return None
        
        return cap
    
    def _is_in_roi(self, x1, y1, x2, y2):
        if not self.roi_points or len(self.roi_points) < 3:
            return True
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return point_in_polygon((center_x, center_y), self.roi_points)
    
    def _save_warning(self, message, frame, confidence=None, bbox=None, extra_data=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.detection_type}_{timestamp}.jpg"
        screenshot_path = os.path.join(WARNING_DIR, filename)
        
        cv2.imwrite(screenshot_path, frame)
        
        warning_id = self.db.add_warning(
            warning_type=self.detection_type,
            message=message,
            screenshot_path=screenshot_path,
            confidence=confidence,
            bbox=bbox,
            extra_data=extra_data,
            camera_id=self.camera_id
        )
        
        self.db.update_daily_stats()
        
        return warning_id, screenshot_path


class FireDetector(BaseDetector):
    def __init__(self):
        super().__init__(MODELS['fire'], 'fire')
        self.warning_cooldown = DETECTION_CONFIG['fire'].get('warning_cooldown', 5)
        self.last_warning_time = 0
    
    def _run_detection(self):
        cap = self._prepare_video_source()
        if cap is None:
            self.is_running = False
            return
        
        frame_count = 0
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = datetime.now()
            
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            if frame_count % 2 == 0:
                try:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    if self.roi_points and len(self.roi_points) >= 3:
                        pts = np.array(self.roi_points, np.int32)
                        cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)
                    
                    boxes = result.boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            class_name = self.class_names[cls]
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            
                            if class_name.lower() in ['fire', 'smoke']:
                                if self._is_in_roi(x1, y1, x2, y2):
                                    elapsed_since_last = (current_time - datetime.fromtimestamp(self.last_warning_time)).total_seconds() if self.last_warning_time else float('inf')
                                    
                                    if elapsed_since_last >= self.warning_cooldown:
                                        message = f"检测到{class_name}！位置：({x1},{y1})-({x2},{y2})"
                                        self._save_warning(
                                            message=message,
                                            frame=annotated_frame,
                                            confidence=conf,
                                            bbox=[x1, y1, x2, y2],
                                            extra_data={'class': class_name}
                                        )
                                        self.last_warning_time = current_time.timestamp()
                                        self.db.add_log('WARNING', self.detection_type, message)
                    
                    with self._frame_lock:
                        self._annotated_frame = annotated_frame.copy()
                        
                except Exception as e:
                    self.db.add_log('ERROR', self.detection_type, f"检测错误：{str(e)}")
            else:
                with self._frame_lock:
                    self._annotated_frame = frame.copy()
        
        cap.release()
        self.is_running = False
        self.db.update_detection_status(self.detection_type, running=False)


class PersonLoiteringDetector(BaseDetector):
    def __init__(self):
        super().__init__(MODELS['general'], 'person_loitering')
        self.loitering_threshold = DETECTION_CONFIG['person_loitering'].get('loitering_threshold', 10)
        self.person_tracks = {}
        self.next_person_id = 0
        self.warned_persons = set()
    
    def _run_detection(self):
        cap = self._prepare_video_source()
        if cap is None:
            self.is_running = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        start_time = datetime.now()
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            if frame_count % 2 == 0:
                try:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    if self.roi_points and len(self.roi_points) >= 3:
                        pts = np.array(self.roi_points, np.int32)
                        cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)
                    
                    boxes = result.boxes
                    current_person_ids = set()
                    
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls == 0:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                if not self._is_in_roi(x1, y1, x2, y2):
                                    continue
                                
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                person_id = None
                                min_distance = float('inf')
                                
                                for pid, track in self.person_tracks.items():
                                    if pid not in self.warned_persons or pid in current_person_ids:
                                        last_center = track.get('last_center')
                                        if last_center:
                                            distance = ((center_x - last_center[0])**2 + 
                                                       (center_y - last_center[1])**2) ** 0.5
                                            if distance < min_distance and distance < 100:
                                                min_distance = distance
                                                person_id = pid
                                
                                if person_id is None:
                                    person_id = self.next_person_id
                                    self.next_person_id += 1
                                    self.person_tracks[person_id] = {
                                        'start_time': elapsed_time,
                                        'last_center': (center_x, center_y),
                                        'bbox': (x1, y1, x2, y2)
                                    }
                                else:
                                    self.person_tracks[person_id]['last_center'] = (center_x, center_y)
                                    self.person_tracks[person_id]['bbox'] = (x1, y1, x2, y2)
                                
                                current_person_ids.add(person_id)
                                
                                start_time_person = self.person_tracks[person_id]['start_time']
                                stay_duration = elapsed_time - start_time_person
                                
                                if stay_duration >= self.loitering_threshold:
                                    if person_id not in self.warned_persons:
                                        self.warned_persons.add(person_id)
                                        message = f"人员滞留超过{self.loitering_threshold}秒！滞留时长：{stay_duration:.1f}秒"
                                        self._save_warning(
                                            message=message,
                                            frame=annotated_frame,
                                            bbox=[x1, y1, x2, y2],
                                            extra_data={'stay_duration': stay_duration, 'person_id': person_id}
                                        )
                                        self.db.add_log('WARNING', self.detection_type, message)
                    
                    with self._frame_lock:
                        self._annotated_frame = annotated_frame.copy()
                    
                    persons_to_remove = [pid for pid in self.person_tracks if pid not in current_person_ids]
                    for pid in persons_to_remove:
                        del self.person_tracks[pid]
                        if pid in self.warned_persons:
                            self.warned_persons.remove(pid)
                            
                except Exception as e:
                    self.db.add_log('ERROR', self.detection_type, f"检测错误：{str(e)}")
            else:
                with self._frame_lock:
                    self._annotated_frame = frame.copy()
        
        cap.release()
        self.is_running = False
        self.db.update_detection_status(self.detection_type, running=False)


class VehicleLoiteringDetector(BaseDetector):
    def __init__(self):
        super().__init__(MODELS['general'], 'vehicle_loitering')
        self.loitering_threshold = DETECTION_CONFIG['vehicle_loitering'].get('loitering_threshold', 10)
        self.vehicle_class_ids = {2, 3, 5, 7}
        self.vehicle_tracks = {}
        self.next_vehicle_id = 0
        self.warned_vehicles = set()
    
    def _run_detection(self):
        cap = self._prepare_video_source()
        if cap is None:
            self.is_running = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        start_time = datetime.now()
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            if frame_count % 2 == 0:
                try:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    if self.roi_points and len(self.roi_points) >= 3:
                        pts = np.array(self.roi_points, np.int32)
                        cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)
                    
                    boxes = result.boxes
                    current_vehicle_ids = set()
                    
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls in self.vehicle_class_ids:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                if not self._is_in_roi(x1, y1, x2, y2):
                                    continue
                                
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                vehicle_id = None
                                min_distance = float('inf')
                                
                                for vid, track in self.vehicle_tracks.items():
                                    if vid not in self.warned_vehicles or vid in current_vehicle_ids:
                                        last_center = track.get('last_center')
                                        if last_center:
                                            distance = ((center_x - last_center[0])**2 + 
                                                       (center_y - last_center[1])**2) ** 0.5
                                            if distance < min_distance and distance < 100:
                                                min_distance = distance
                                                vehicle_id = vid
                                
                                if vehicle_id is None:
                                    vehicle_id = self.next_vehicle_id
                                    self.next_vehicle_id += 1
                                    self.vehicle_tracks[vehicle_id] = {
                                        'start_time': elapsed_time,
                                        'last_center': (center_x, center_y),
                                        'bbox': (x1, y1, x2, y2),
                                        'cls': cls
                                    }
                                else:
                                    self.vehicle_tracks[vehicle_id]['last_center'] = (center_x, center_y)
                                    self.vehicle_tracks[vehicle_id]['bbox'] = (x1, y1, x2, y2)
                                    self.vehicle_tracks[vehicle_id]['cls'] = cls
                                
                                current_vehicle_ids.add(vehicle_id)
                                
                                start_time_vehicle = self.vehicle_tracks[vehicle_id]['start_time']
                                stay_duration = elapsed_time - start_time_vehicle
                                
                                if stay_duration >= self.loitering_threshold:
                                    if vehicle_id not in self.warned_vehicles:
                                        self.warned_vehicles.add(vehicle_id)
                                        label = self.class_names.get(cls, 'Vehicle')
                                        message = f"车辆({label})滞留超过{self.loitering_threshold}秒！滞留时长：{stay_duration:.1f}秒"
                                        self._save_warning(
                                            message=message,
                                            frame=annotated_frame,
                                            bbox=[x1, y1, x2, y2],
                                            extra_data={'stay_duration': stay_duration, 'vehicle_id': vehicle_id, 'vehicle_type': label}
                                        )
                                        self.db.add_log('WARNING', self.detection_type, message)
                    
                    with self._frame_lock:
                        self._annotated_frame = annotated_frame.copy()
                    
                    vehicles_to_remove = [vid for vid in self.vehicle_tracks if vid not in current_vehicle_ids]
                    for vid in vehicles_to_remove:
                        del self.vehicle_tracks[vid]
                        if vid in self.warned_vehicles:
                            self.warned_vehicles.remove(vid)
                            
                except Exception as e:
                    self.db.add_log('ERROR', self.detection_type, f"检测错误：{str(e)}")
            else:
                with self._frame_lock:
                    self._annotated_frame = frame.copy()
        
        cap.release()
        self.is_running = False
        self.db.update_detection_status(self.detection_type, running=False)


class ShutterDetector(BaseDetector):
    def __init__(self):
        super().__init__(MODELS['shutter'], 'shutter')
        self.state_names = {0: 'closed', 1: 'opened', 2: 'partially_opened'}
        self.state_map = {
            'closed': 'CLOSED',
            'opened': 'OPENED',
            'partially_opened': 'PARTIALLY OPENED'
        }
        self.shutter_tracks = {}
        self.next_shutter_id = 0
        self.last_shutter_states = {}
    
    def _run_detection(self):
        cap = self._prepare_video_source()
        if cap is None:
            self.is_running = False
            return
        
        frame_count = 0
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            if frame_count % 2 == 0:
                try:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    if self.roi_points and len(self.roi_points) >= 3:
                        pts = np.array(self.roi_points, np.int32)
                        cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)
                    
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            if not self._is_in_roi(x1, y1, x2, y2):
                                continue
                            
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            state = self.state_names.get(cls, 'unknown')
                            state_name = self.state_map.get(state, state)
                            
                            min_distance = float('inf')
                            shutter_id = None
                            
                            for sid, track in self.shutter_tracks.items():
                                last_center = track.get('last_center')
                                if last_center:
                                    distance = ((center_x - last_center[0])**2 + 
                                               (center_y - last_center[1])**2) ** 0.5
                                    if distance < min_distance and distance < 100:
                                        min_distance = distance
                                        shutter_id = sid
                            
                            if shutter_id is None:
                                shutter_id = self.next_shutter_id
                                self.next_shutter_id += 1
                            
                            self.shutter_tracks[shutter_id] = {
                                'last_center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'state': state
                            }
                            
                            last_state = self.last_shutter_states.get(shutter_id)
                            
                            if last_state != state and last_state is not None:
                                from_state = self.state_map.get(last_state, last_state)
                                message = f"卷帘门#{shutter_id}状态变化：{from_state} -> {state_name}"
                                self._save_warning(
                                    message=message,
                                    frame=annotated_frame,
                                    confidence=conf,
                                    bbox=[x1, y1, x2, y2],
                                    extra_data={'shutter_id': shutter_id, 'from_state': last_state, 'to_state': state}
                                )
                                self.db.add_log('INFO', self.detection_type, message)
                            
                            self.last_shutter_states[shutter_id] = state
                    
                    with self._frame_lock:
                        self._annotated_frame = annotated_frame.copy()
                            
                except Exception as e:
                    self.db.add_log('ERROR', self.detection_type, f"检测错误：{str(e)}")
            else:
                with self._frame_lock:
                    self._annotated_frame = frame.copy()
        
        cap.release()
        self.is_running = False
        self.db.update_detection_status(self.detection_type, running=False)


class FenceCrossingDetector(BaseDetector):
    def __init__(self):
        super().__init__(MODELS['general'], 'fence_crossing')
        self.track_distance_threshold = DETECTION_CONFIG['fence_crossing'].get('track_distance_threshold', 200)
        self.max_track_lost_frames = DETECTION_CONFIG['fence_crossing'].get('max_track_lost_frames', 20)
        self.tracks = {}
        self.next_id = 0
        self.warned_ids = set()
        self.trail_len = 30
    
    def _run_detection(self):
        cap = self._prepare_video_source()
        if cap is None:
            self.is_running = False
            return
        
        frame_count = 0
        fence_points = self.roi_points
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated = frame.copy()
            
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            if fence_points and len(fence_points) >= 2:
                for i in range(len(fence_points) - 1):
                    cv2.line(annotated, fence_points[i], fence_points[i + 1], (255, 0, 255), 3)
                for p in fence_points:
                    cv2.circle(annotated, p, 4, (255, 0, 255), -1)
            
            if frame_count % 2 == 0:
                try:
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    result = results[0]
                    boxes = result.boxes
                    
                    current_ids = set()
                    pending_warnings = []
                    
                    if boxes is not None:
                        detections = []
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = (x1 + x2) // 2
                            cy = y2
                            detections.append(((cx, cy), (x1, y1, x2, y2)))
                        
                        assigned_track_ids = set()
                        for (center, bbox) in detections:
                            best_id = None
                            best_dist = float("inf")
                            for tid, t in self.tracks.items():
                                if tid in assigned_track_ids:
                                    continue
                                if t["lost"] > self.max_track_lost_frames:
                                    continue
                                last = t["last_center"]
                                dist = ((center[0] - last[0]) ** 2 + (center[1] - last[1]) ** 2) ** 0.5
                                if dist < best_dist and dist < self.track_distance_threshold:
                                    best_dist = dist
                                    best_id = tid
                            
                            if best_id is None:
                                tid = self.next_id
                                self.next_id += 1
                                self.tracks[tid] = {
                                    "last_center": center,
                                    "bbox": bbox,
                                    "lost": 0,
                                    "trail": deque([center], maxlen=self.trail_len),
                                }
                                best_id = tid
                            else:
                                if fence_points and len(fence_points) >= 2:
                                    prev_center = self.tracks[best_id]["last_center"]
                                    if polyline_intersect(prev_center, center, fence_points):
                                        if best_id not in self.warned_ids:
                                            self.warned_ids.add(best_id)
                                            pending_warnings.append((best_id, bbox))
                                
                                self.tracks[best_id]["last_center"] = center
                                self.tracks[best_id]["bbox"] = bbox
                                self.tracks[best_id]["lost"] = 0
                                self.tracks[best_id]["trail"].append(center)
                            
                            assigned_track_ids.add(best_id)
                            current_ids.add(best_id)
                        
                        to_delete = []
                        for tid, t in self.tracks.items():
                            if tid not in current_ids:
                                t["lost"] += 1
                            if t["lost"] > self.max_track_lost_frames:
                                to_delete.append(tid)
                                if tid in self.warned_ids:
                                    self.warned_ids.remove(tid)
                        for tid in to_delete:
                            del self.tracks[tid]
                    
                    for tid, t in self.tracks.items():
                        x1, y1, x2, y2 = t["bbox"]
                        color = (0, 0, 255) if tid in self.warned_ids else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"ID {tid}", (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        trail_pts = list(t["trail"])
                        for i in range(1, len(trail_pts)):
                            cv2.line(annotated, trail_pts[i - 1], trail_pts[i], color, 2)
                    
                    for tid, bbox in pending_warnings:
                        x1, y1, x2, y2 = bbox
                        message = f"围栏翻越检测：ID={tid} 位置=({x1},{y1})-({x2},{y2})"
                        self._save_warning(
                            message=message,
                            frame=annotated,
                            bbox=[x1, y1, x2, y2],
                            extra_data={'track_id': tid}
                        )
                        self.db.add_log('WARNING', self.detection_type, message)
                    
                    with self._frame_lock:
                        self._annotated_frame = annotated.copy()
                        
                except Exception as e:
                    self.db.add_log('ERROR', self.detection_type, f"检测错误：{str(e)}")
            else:
                with self._frame_lock:
                    self._annotated_frame = annotated.copy()
        
        cap.release()
        self.is_running = False
        self.db.update_detection_status(self.detection_type, running=False)


class CameraDetector:
    def __init__(self, camera_id, detection_types, roi_points, configs, db):
        self.camera_id = camera_id
        self.detection_types = detection_types
        self.roi_points = roi_points
        self.configs = configs
        self.db = db
        
        self.models = {
            'fire': YOLO(MODELS['fire']),
            'general': YOLO(MODELS['general']),
            'shutter': YOLO(MODELS['shutter'])
        }
        
        self.person_tracks = {}
        self.next_person_id = 0
        self.warned_persons = set()
        
        self.vehicle_tracks = {}
        self.next_vehicle_id = 0
        self.warned_vehicles = set()
        self.vehicle_class_ids = {2, 3, 5, 7}
        
        self.shutter_tracks = {}
        self.next_shutter_id = 0
        self.last_shutter_states = {}
        self.state_names = {0: 'closed', 1: 'opened', 2: 'partially_opened'}
        self.state_map = {'closed': 'CLOSED', 'opened': 'OPENED', 'partially_opened': 'PARTIALLY OPENED'}
        
        self.fence_tracks = {}
        self.next_fence_id = 0
        self.warned_fence_ids = set()
        
        self.fire_last_warning_time = 0
        self.start_time = None
        
        self.loitering_threshold = self._get_config('person_loitering', 'loitering_threshold', 10)
        self.vehicle_loitering_threshold = self._get_config('vehicle_loitering', 'loitering_threshold', 10)
        self.fire_cooldown = self._get_config('fire', 'warning_cooldown', 5)
        self.fence_track_distance = self._get_config('fence_crossing', 'track_distance_threshold', 200)
    
    def _get_config(self, det_type, key, default):
        config = self.configs.get(det_type, {})
        return config.get(key, default)
    
    def _is_in_roi(self, x1, y1, x2, y2):
        if not self.roi_points or len(self.roi_points) < 3:
            return True
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return point_in_polygon((center_x, center_y), self.roi_points)
    
    def _save_warning(self, det_type, message, frame, confidence=None, bbox=None, extra_data=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{det_type}_{self.camera_id}_{timestamp}.jpg"
        screenshot_path = os.path.join(WARNING_DIR, filename)
        cv2.imwrite(screenshot_path, frame)
        
        self.db.add_warning(
            warning_type=det_type,
            message=message,
            screenshot_path=screenshot_path,
            confidence=confidence,
            bbox=bbox,
            extra_data=extra_data,
            camera_id=self.camera_id
        )
        self.db.update_daily_stats()
        self.db.add_log('WARNING', det_type, message)
    
    def process_frame(self, frame, elapsed_time):
        annotated = frame.copy()
        
        if self.roi_points and len(self.roi_points) >= 3:
            pts = np.array(self.roi_points, np.int32)
            cv2.polylines(annotated, [pts], True, (255, 255, 0), 2)
        
        if 'fire' in self.detection_types:
            annotated = self._detect_fire(frame, annotated)
        
        if 'person_loitering' in self.detection_types:
            annotated = self._detect_person_loitering(frame, annotated, elapsed_time)
        
        if 'vehicle_loitering' in self.detection_types:
            annotated = self._detect_vehicle_loitering(frame, annotated, elapsed_time)
        
        if 'shutter' in self.detection_types:
            annotated = self._detect_shutter(frame, annotated)
        
        if 'fence_crossing' in self.detection_types:
            annotated = self._detect_fence_crossing(frame, annotated)
        
        return annotated
    
    def _detect_fire(self, frame, annotated):
        try:
            conf_threshold = self._get_config('fire', 'conf_threshold', 0.5)
            results = self.models['fire'](frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = self.models['fire'].names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    if class_name.lower() in ['fire', 'smoke']:
                        if self._is_in_roi(x1, y1, x2, y2):
                            current_time = datetime.now()
                            elapsed_since_last = (current_time - datetime.fromtimestamp(self.fire_last_warning_time)).total_seconds() if self.fire_last_warning_time else float('inf')
                            
                            if elapsed_since_last >= self.fire_cooldown:
                                message = f"检测到{class_name}！位置：({x1},{y1})-({x2},{y2})"
                                self._save_warning('fire', message, annotated, conf, [x1, y1, x2, y2], {'class': class_name})
                                self.fire_last_warning_time = current_time.timestamp()
            
            annotated = result.plot()
        except Exception as e:
            self.db.add_log('ERROR', 'fire', f"检测错误：{str(e)}")
        
        return annotated
    
    def _detect_person_loitering(self, frame, annotated, elapsed_time):
        try:
            conf_threshold = self._get_config('person_loitering', 'conf_threshold', 0.5)
            results = self.models['general'](frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            boxes = result.boxes
            current_person_ids = set()
            
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        if not self._is_in_roi(x1, y1, x2, y2):
                            continue
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        person_id = None
                        min_distance = float('inf')
                        
                        for pid, track in self.person_tracks.items():
                            if pid not in self.warned_persons or pid in current_person_ids:
                                last_center = track.get('last_center')
                                if last_center:
                                    distance = ((center_x - last_center[0])**2 + (center_y - last_center[1])**2) ** 0.5
                                    if distance < min_distance and distance < 100:
                                        min_distance = distance
                                        person_id = pid
                        
                        if person_id is None:
                            person_id = self.next_person_id
                            self.next_person_id += 1
                            self.person_tracks[person_id] = {
                                'start_time': elapsed_time,
                                'last_center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2)
                            }
                        else:
                            self.person_tracks[person_id]['last_center'] = (center_x, center_y)
                            self.person_tracks[person_id]['bbox'] = (x1, y1, x2, y2)
                        
                        current_person_ids.add(person_id)
                        
                        start_time_person = self.person_tracks[person_id]['start_time']
                        stay_duration = elapsed_time - start_time_person
                        
                        if stay_duration >= self.loitering_threshold:
                            if person_id not in self.warned_persons:
                                self.warned_persons.add(person_id)
                                message = f"人员滞留超过{self.loitering_threshold}秒！滞留时长：{stay_duration:.1f}秒"
                                self._save_warning('person_loitering', message, annotated, bbox=[x1, y1, x2, y2], 
                                                  extra_data={'stay_duration': stay_duration, 'person_id': person_id})
            
            persons_to_remove = [pid for pid in self.person_tracks if pid not in current_person_ids]
            for pid in persons_to_remove:
                del self.person_tracks[pid]
                if pid in self.warned_persons:
                    self.warned_persons.remove(pid)
            
            annotated = result.plot()
            
            for pid, track in self.person_tracks.items():
                x1, y1, x2, y2 = track['bbox']
                color = (0, 0, 255) if pid in self.warned_persons else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"Person {pid}", (x1, max(20, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        except Exception as e:
            self.db.add_log('ERROR', 'person_loitering', f"检测错误：{str(e)}")
        
        return annotated
    
    def _detect_vehicle_loitering(self, frame, annotated, elapsed_time):
        try:
            conf_threshold = self._get_config('vehicle_loitering', 'conf_threshold', 0.5)
            results = self.models['general'](frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            boxes = result.boxes
            current_vehicle_ids = set()
            
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_class_ids:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        if not self._is_in_roi(x1, y1, x2, y2):
                            continue
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        vehicle_id = None
                        min_distance = float('inf')
                        
                        for vid, track in self.vehicle_tracks.items():
                            if vid not in self.warned_vehicles or vid in current_vehicle_ids:
                                last_center = track.get('last_center')
                                if last_center:
                                    distance = ((center_x - last_center[0])**2 + (center_y - last_center[1])**2) ** 0.5
                                    if distance < min_distance and distance < 100:
                                        min_distance = distance
                                        vehicle_id = vid
                        
                        if vehicle_id is None:
                            vehicle_id = self.next_vehicle_id
                            self.next_vehicle_id += 1
                            self.vehicle_tracks[vehicle_id] = {
                                'start_time': elapsed_time,
                                'last_center': (center_x, center_y),
                                'bbox': (x1, y1, x2, y2),
                                'cls': cls
                            }
                        else:
                            self.vehicle_tracks[vehicle_id]['last_center'] = (center_x, center_y)
                            self.vehicle_tracks[vehicle_id]['bbox'] = (x1, y1, x2, y2)
                            self.vehicle_tracks[vehicle_id]['cls'] = cls
                        
                        current_vehicle_ids.add(vehicle_id)
                        
                        start_time_vehicle = self.vehicle_tracks[vehicle_id]['start_time']
                        stay_duration = elapsed_time - start_time_vehicle
                        
                        if stay_duration >= self.vehicle_loitering_threshold:
                            if vehicle_id not in self.warned_vehicles:
                                self.warned_vehicles.add(vehicle_id)
                                label = self.models['general'].names.get(cls, 'Vehicle')
                                message = f"车辆({label})滞留超过{self.vehicle_loitering_threshold}秒！滞留时长：{stay_duration:.1f}秒"
                                self._save_warning('vehicle_loitering', message, annotated, bbox=[x1, y1, x2, y2],
                                                  extra_data={'stay_duration': stay_duration, 'vehicle_id': vehicle_id, 'vehicle_type': label})
            
            vehicles_to_remove = [vid for vid in self.vehicle_tracks if vid not in current_vehicle_ids]
            for vid in vehicles_to_remove:
                del self.vehicle_tracks[vid]
                if vid in self.warned_vehicles:
                    self.warned_vehicles.remove(vid)
            
            annotated = result.plot()
            
            for vid, track in self.vehicle_tracks.items():
                x1, y1, x2, y2 = track['bbox']
                color = (0, 0, 255) if vid in self.warned_vehicles else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"Vehicle {vid}", (x1, max(20, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        except Exception as e:
            self.db.add_log('ERROR', 'vehicle_loitering', f"检测错误：{str(e)}")
        
        return annotated
    
    def _detect_shutter(self, frame, annotated):
        try:
            conf_threshold = self._get_config('shutter', 'conf_threshold', 0.5)
            results = self.models['shutter'](frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    if not self._is_in_roi(x1, y1, x2, y2):
                        continue
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    state = self.state_names.get(cls, 'unknown')
                    state_name = self.state_map.get(state, state)
                    
                    min_distance = float('inf')
                    shutter_id = None
                    
                    for sid, track in self.shutter_tracks.items():
                        last_center = track.get('last_center')
                        if last_center:
                            distance = ((center_x - last_center[0])**2 + (center_y - last_center[1])**2) ** 0.5
                            if distance < min_distance and distance < 100:
                                min_distance = distance
                                shutter_id = sid
                    
                    if shutter_id is None:
                        shutter_id = self.next_shutter_id
                        self.next_shutter_id += 1
                    
                    self.shutter_tracks[shutter_id] = {
                        'last_center': (center_x, center_y),
                        'bbox': (x1, y1, x2, y2),
                        'state': state
                    }
                    
                    last_state = self.last_shutter_states.get(shutter_id)
                    
                    if last_state != state and last_state is not None:
                        from_state = self.state_map.get(last_state, last_state)
                        message = f"卷帘门#{shutter_id}状态变化：{from_state} -> {state_name}"
                        self._save_warning('shutter', message, annotated, conf, [x1, y1, x2, y2],
                                          extra_data={'shutter_id': shutter_id, 'from_state': last_state, 'to_state': state})
                    
                    self.last_shutter_states[shutter_id] = state
            
            annotated = result.plot()
            
        except Exception as e:
            self.db.add_log('ERROR', 'shutter', f"检测错误：{str(e)}")
        
        return annotated
    
    def _detect_fence_crossing(self, frame, annotated):
        try:
            if not self.roi_points or len(self.roi_points) < 2:
                return annotated
            
            conf_threshold = self._get_config('fence_crossing', 'conf_threshold', 0.5)
            results = self.models['general'](frame, conf=conf_threshold, verbose=False)
            result = results[0]
            
            boxes = result.boxes
            current_ids = set()
            
            if boxes is not None:
                detections = []
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = y2
                    detections.append(((cx, cy), (x1, y1, x2, y2)))
                
                assigned_ids = set()
                for (center, bbox) in detections:
                    best_id = None
                    best_dist = float('inf')
                    
                    for tid, t in self.fence_tracks.items():
                        if tid in assigned_ids:
                            continue
                        last = t['last_center']
                        dist = ((center[0] - last[0])**2 + (center[1] - last[1])**2) ** 0.5
                        if dist < best_dist and dist < self.fence_track_distance:
                            best_dist = dist
                            best_id = tid
                    
                    if best_id is None:
                        tid = self.next_fence_id
                        self.next_fence_id += 1
                        self.fence_tracks[tid] = {'last_center': center, 'bbox': bbox}
                        best_id = tid
                    else:
                        prev_center = self.fence_tracks[best_id]['last_center']
                        if polyline_intersect(prev_center, center, self.roi_points):
                            if best_id not in self.warned_fence_ids:
                                self.warned_fence_ids.add(best_id)
                                x1, y1, x2, y2 = bbox
                                message = f"围栏翻越检测：ID={best_id} 位置=({x1},{y1})-({x2},{y2})"
                                self._save_warning('fence_crossing', message, annotated, bbox=[x1, y1, x2, y2],
                                                  extra_data={'track_id': best_id})
                        
                        self.fence_tracks[best_id]['last_center'] = center
                        self.fence_tracks[best_id]['bbox'] = bbox
                    
                    assigned_ids.add(best_id)
                    current_ids.add(best_id)
                
                to_delete = [tid for tid in self.fence_tracks if tid not in current_ids]
                for tid in to_delete:
                    del self.fence_tracks[tid]
                    if tid in self.warned_fence_ids:
                        self.warned_fence_ids.remove(tid)
            
            annotated = result.plot()
            
            for tid, t in self.fence_tracks.items():
                x1, y1, x2, y2 = t['bbox']
                color = (0, 0, 255) if tid in self.warned_fence_ids else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"ID {tid}", (x1, max(20, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.roi_points and len(self.roi_points) >= 2:
                for i in range(len(self.roi_points) - 1):
                    cv2.line(annotated, self.roi_points[i], self.roi_points[i + 1], (255, 0, 255), 3)
                for p in self.roi_points:
                    cv2.circle(annotated, p, 4, (255, 0, 255), -1)
                
        except Exception as e:
            self.db.add_log('ERROR', 'fence_crossing', f"检测错误：{str(e)}")
        
        return annotated


class MultiDetector:
    def __init__(self):
        self.db = Database()
        self._camera_threads = {}
        self._camera_frames = {}
        self._camera_detectors = {}
        self._frame_lock = threading.Lock()
    
    def start_camera(self, camera_id, video_source, detection_types, roi_points=None):
        if camera_id in self._camera_threads:
            return False, "摄像头已在运行"
        
        if isinstance(video_source, str):
            if video_source.isdigit():
                video_source = int(video_source)
        
        self.db.add_log('INFO', 'MultiDetector', f"正在打开摄像头 {camera_id}，视频源：{video_source}")
        
        cap = cv2.VideoCapture(video_source)
        
        if isinstance(video_source, str) and video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            self.db.add_log('ERROR', 'MultiDetector', f"无法打开视频源：{video_source}")
            return False, "无法打开视频源"
        
        configs = {}
        for det_type in detection_types:
            status = self.db.get_detection_status(det_type)
            if status and status.get('config'):
                try:
                    configs[det_type] = json.loads(status['config'])
                except:
                    configs[det_type] = DETECTION_CONFIG.get(det_type, {})
            else:
                configs[det_type] = DETECTION_CONFIG.get(det_type, {})
        
        camera_detector = CameraDetector(camera_id, detection_types, roi_points, configs, self.db)
        
        stop_event = threading.Event()
        is_video_file = isinstance(video_source, str) and not video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')) and not video_source.isdigit()
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        frame_delay = 1.0 / video_fps
        
        def run_camera():
            nonlocal cap
            frame_count = 0
            start_time = datetime.now()
            last_frame_time = datetime.now()
            
            while not stop_event.is_set():
                current_frame_time = datetime.now()
                elapsed_since_last = (current_frame_time - last_frame_time).total_seconds()
                
                if elapsed_since_last < frame_delay:
                    import time
                    time.sleep(frame_delay - elapsed_since_last)
                    continue
                
                last_frame_time = datetime.now()
                
                ret, frame = cap.read()
                
                if not ret:
                    if is_video_file:
                        cap.release()
                        cap = cv2.VideoCapture(video_source)
                        if cap.isOpened():
                            frame_count = 0
                            start_time = datetime.now()
                            continue
                    break
                
                frame_count += 1
                current_time = datetime.now()
                elapsed_time = (current_time - start_time).total_seconds()
                
                if frame_count % 2 == 0:
                    try:
                        annotated = camera_detector.process_frame(frame, elapsed_time)
                    except Exception as e:
                        self.db.add_log('ERROR', 'MultiDetector', f"处理帧错误：{str(e)}")
                        annotated = frame.copy()
                else:
                    annotated = frame.copy()
                
                with self._frame_lock:
                    ret_encode, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret_encode:
                        self._camera_frames[camera_id] = buffer.tobytes()
            
            cap.release()
            self.db.add_log('INFO', 'MultiDetector', f"摄像头 {camera_id} 已停止")
        
        thread = threading.Thread(target=run_camera, daemon=True)
        thread.start()
        
        self._camera_threads[camera_id] = {
            'thread': thread,
            'stop_event': stop_event,
            'video_source': video_source,
            'detection_types': detection_types,
            'roi_points': roi_points,
            'name': f'摄像头 {len(self._camera_threads) + 1}'
        }
        self._camera_detectors[camera_id] = camera_detector
        
        return True, "摄像头已启动"
    
    def stop_camera(self, camera_id):
        if camera_id not in self._camera_threads:
            return False, "摄像头未运行"
        
        self._camera_threads[camera_id]['stop_event'].set()
        del self._camera_threads[camera_id]
        
        if camera_id in self._camera_detectors:
            del self._camera_detectors[camera_id]
        
        with self._frame_lock:
            if camera_id in self._camera_frames:
                del self._camera_frames[camera_id]
        
        return True, "摄像头已停止"
    
    def get_camera_details(self):
        cameras = {}
        for camera_id, info in self._camera_threads.items():
            cameras[camera_id] = {
                'name': info.get('name', camera_id),
                'source': info.get('video_source'),
                'detection_types': info.get('detection_types', [])
            }
        return cameras
    
    def get_camera_frame(self, camera_id):
        with self._frame_lock:
            return self._camera_frames.get(camera_id)
    
    def get_running_cameras(self):
        return list(self._camera_threads.keys())


class DetectionEngine:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.detectors = {
            'fire': FireDetector(),
            'person_loitering': PersonLoiteringDetector(),
            'vehicle_loitering': VehicleLoiteringDetector(),
            'shutter': ShutterDetector(),
            'fence_crossing': FenceCrossingDetector()
        }
        
        self.multi_detector = MultiDetector()
        self.db = Database()
    
    def start_detection(self, detection_type, video_source=0, camera_id=None, roi_points=None):
        if detection_type not in self.detectors:
            return False, f"未知的检测类型：{detection_type}"
        
        status = self.db.get_detection_status(detection_type)
        if not status or not status.get('enabled', True):
            return False, f"检测功能 {detection_type} 已被禁用"
        
        return self.detectors[detection_type].start(video_source, camera_id, roi_points)
    
    def stop_detection(self, detection_type):
        if detection_type not in self.detectors:
            return False, f"未知的检测类型：{detection_type}"
        
        return self.detectors[detection_type].stop()
    
    def stop_all(self):
        for detection_type in self.detectors:
            self.detectors[detection_type].stop()
    
    def get_status(self):
        status = {}
        for detection_type, detector in self.detectors.items():
            db_status = self.db.get_detection_status(detection_type)
            status[detection_type] = {
                'enabled': db_status.get('enabled', True) if db_status else True,
                'running': detector.is_running,
                'last_detection_time': db_status.get('last_detection_time') if db_status else None
            }
        return status
    
    def get_frame(self, detection_type):
        if detection_type not in self.detectors:
            return None
        return self.detectors[detection_type].get_current_frame()
    
    def start_multi_detection(self, camera_id, video_source, detection_types, roi_points=None):
        return self.multi_detector.start_camera(camera_id, video_source, detection_types, roi_points)
    
    def stop_multi_detection(self, camera_id):
        return self.multi_detector.stop_camera(camera_id)
    
    def get_multi_frame(self, camera_id):
        return self.multi_detector.get_camera_frame(camera_id)
    
    def get_running_cameras(self):
        return self.multi_detector.get_running_cameras()
    
    def get_camera_details(self):
        return self.multi_detector.get_camera_details()
    
    def enable_detection(self, detection_type, enabled=True):
        if detection_type not in self.detectors:
            return False, f"未知的检测类型：{detection_type}"
        
        if not enabled:
            self.detectors[detection_type].stop()
        
        self.db.update_detection_status(detection_type, enabled=enabled)
        return True, f"检测功能 {detection_type} 已{'启用' if enabled else '禁用'}"
    
    def update_config(self, detection_type, config):
        if detection_type not in self.detectors:
            return False, f"未知的检测类型：{detection_type}"
        
        detector = self.detectors[detection_type]
        
        if 'conf_threshold' in config:
            detector.conf_threshold = config['conf_threshold']
        
        if detection_type in ['person_loitering', 'vehicle_loitering'] and 'loitering_threshold' in config:
            detector.loitering_threshold = config['loitering_threshold']
        
        if detection_type == 'fire' and 'warning_cooldown' in config:
            detector.warning_cooldown = config['warning_cooldown']
        
        if detection_type == 'fence_crossing':
            if 'track_distance_threshold' in config:
                detector.track_distance_threshold = config['track_distance_threshold']
            if 'max_track_lost_frames' in config:
                detector.max_track_lost_frames = config['max_track_lost_frames']
        
        self.db.update_detection_status(detection_type, config=config)
        return True, f"检测配置已更新"
