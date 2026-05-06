import cv2
import os
import threading
import uuid
from datetime import datetime
from ultralytics import YOLO
from backend.config import MODELS, UPLOAD_DIR, WARNING_DIR, ALLOWED_IMAGE_FORMATS, ALLOWED_VIDEO_FORMATS
from backend.database import Database


class OfflineProcessor:
    def __init__(self):
        self.models = {
            'fire': YOLO(MODELS['fire']),
            'general': YOLO(MODELS['general']),
            'shutter': YOLO(MODELS['shutter'])
        }
        self.db = Database()
        self._tasks = {}
    
    def _get_model(self, detection_type):
        if detection_type == 'fire':
            return self.models['fire']
        elif detection_type in ['person_loitering', 'vehicle_loitering', 'fence_crossing']:
            return self.models['general']
        elif detection_type == 'shutter':
            return self.models['shutter']
        return None
    
    def process_image(self, task_id, file_path, detection_types, conf_threshold=0.5):
        self.db.update_offline_task(task_id, status='processing', progress=0)
        
        try:
            frame = cv2.imread(file_path)
            if frame is None:
                self.db.update_offline_task(task_id, status='failed', result={'error': '无法读取图片'})
                return
            
            results_list = []
            annotated = frame.copy()
            total_detections = len(detection_types)
            
            for idx, det_type in enumerate(detection_types):
                model = self._get_model(det_type)
                if model is None:
                    continue
                
                results = model(frame, conf=conf_threshold, verbose=False)
                result = results[0]
                annotated = result.plot()
                
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names.get(cls, str(cls))
                        
                        results_list.append({
                            'type': det_type,
                            'class': class_name,
                            'confidence': round(conf, 2),
                            'bbox': [x1, y1, x2, y2]
                        })
                
                progress = int((idx + 1) / total_detections * 100)
                self.db.update_offline_task(task_id, progress=progress)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"offline_{timestamp}_{task_id}.jpg"
            result_path = os.path.join(WARNING_DIR, result_filename)
            cv2.imwrite(result_path, annotated)
            
            self.db.update_offline_task(
                task_id, 
                status='completed', 
                progress=100,
                result={
                    'detections': results_list,
                    'result_image': result_filename,
                    'total_detections': len(results_list)
                }
            )
            
        except Exception as e:
            self.db.update_offline_task(task_id, status='failed', result={'error': str(e)})
    
    def process_video(self, task_id, file_path, detection_types, conf_threshold=0.5):
        self.db.update_offline_task(task_id, status='processing', progress=0)
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.db.update_offline_task(task_id, status='failed', result={'error': '无法打开视频'})
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"offline_{timestamp}_{task_id}.mp4"
            result_path = os.path.join(WARNING_DIR, result_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            results_list = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                annotated = frame.copy()
                
                if frame_count % 3 == 0:
                    for det_type in detection_types:
                        model = self._get_model(det_type)
                        if model is None:
                            continue
                        
                        results = model(frame, conf=conf_threshold, verbose=False)
                        result = results[0]
                        annotated = result.plot()
                        
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                class_name = model.names.get(cls, str(cls))
                                
                                results_list.append({
                                    'type': det_type,
                                    'class': class_name,
                                    'confidence': round(conf, 2),
                                    'bbox': [x1, y1, x2, y2],
                                    'frame': frame_count
                                })
                
                out.write(annotated)
                
                if total_frames > 0:
                    progress = int(frame_count / total_frames * 100)
                    self.db.update_offline_task(task_id, progress=progress)
            
            cap.release()
            out.release()
            
            self.db.update_offline_task(
                task_id,
                status='completed',
                progress=100,
                result={
                    'detections': results_list[:100],
                    'result_video': result_filename,
                    'total_detections': len(results_list),
                    'total_frames': frame_count
                }
            )
            
        except Exception as e:
            self.db.update_offline_task(task_id, status='failed', result={'error': str(e)})
    
    def start_task(self, file_path, detection_types, conf_threshold=0.5):
        task_id = self.db.create_offline_task(
            task_type='image' if self._is_image(file_path) else 'video',
            detection_types=detection_types,
            file_path=file_path
        )
        
        if self._is_image(file_path):
            thread = threading.Thread(
                target=self.process_image,
                args=(task_id, file_path, detection_types, conf_threshold),
                daemon=True
            )
        else:
            thread = threading.Thread(
                target=self.process_video,
                args=(task_id, file_path, detection_types, conf_threshold),
                daemon=True
            )
        
        thread.start()
        self._tasks[task_id] = thread
        
        return task_id
    
    def get_task_status(self, task_id):
        return self.db.get_offline_task(task_id)
    
    def get_all_tasks(self):
        return self.db.get_offline_tasks()
    
    def _is_image(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ALLOWED_IMAGE_FORMATS
    
    def _is_video(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ALLOWED_VIDEO_FORMATS
    
    def validate_file(self, file_path):
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ALLOWED_IMAGE_FORMATS and ext not in ALLOWED_VIDEO_FORMATS:
            return False, f"不支持的文件格式: {ext}"
        
        return True, "文件有效"
