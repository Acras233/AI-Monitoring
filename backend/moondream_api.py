import os
import sys
import threading
import time
import json
import base64
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
from backend.database import Database

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model")


class MoondreamAPI:
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
        self.model = None
        self.model_loaded = False
        self.model_loading = False
        self.model_error = None
        self.db = Database()
        self._analysis_lock = threading.Lock()
        self._active_streams = {}
        self._stream_threads = {}
        self._stream_stop_events = {}
        self._stream_results = {}
        self._stream_frames = {}
        self._stream_frame_lock = {}
        self._load_lock = threading.Lock()
    
    def get_status(self):
        return {
            'loaded': self.model_loaded,
            'loading': self.model_loading,
            'error': self.model_error,
            'gpu_available': self._check_gpu()
        }
    
    def _check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self):
        if self.model_loaded:
            return True, "模型已加载"
        
        if self.model_loading:
            return False, "模型正在加载中，请稍候"
        
        with self._load_lock:
            if self.model_loaded:
                return True, "模型已加载"
            
            self.model_loading = True
            self.model_error = None
            
            try:
                import torch
                import torch.nn as nn
                
                if not torch.cuda.is_available():
                    self.model_loading = False
                    self.model_error = "未检测到CUDA设备，此功能需要NVIDIA GPU"
                    return False, self.model_error
                
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                from Model.config import MoondreamConfig
                from Model.moondream import MoondreamModel
                import safetensors
                
                self.db.add_log('INFO', 'MoondreamAPI', "正在加载 Moondream2 模型...")
                
                config = MoondreamConfig()
                model = MoondreamModel(config, dtype=torch.bfloat16)
                
                weights_file = os.path.join(MODEL_DIR, "model.safetensors")
                if not os.path.exists(weights_file):
                    self.model_loading = False
                    self.model_error = f"模型权重文件不存在: {weights_file}"
                    return False, self.model_error
                
                state_dict = {}
                with safetensors.safe_open(weights_file, framework="pt") as f:
                    for key in f.keys():
                        new_key = key
                        if key.startswith("model."):
                            new_key = key[len("model."):]
                        state_dict[new_key] = f.get_tensor(key)
                
                model.load_state_dict(state_dict, strict=False)
                model = model.to("cuda")
                model.eval()
                
                try:
                    from torchao.quantization import Int4WeightOnlyQuantizer
                    quantizer = Int4WeightOnlyQuantizer(groupsize=128)
                    for name, module in list(model.named_modules()):
                        if isinstance(module, nn.Linear):
                            parts = name.split(".")
                            parent = model
                            for part in parts[:-1]:
                                if part.isdigit():
                                    parent = parent[int(part)]
                                else:
                                    parent = getattr(parent, part)
                            if parts[-1].isdigit():
                                parent[int(parts[-1])] = quantizer.quantize(module)
                            else:
                                setattr(parent, parts[-1], quantizer.quantize(module))
                    
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except ImportError:
                    self.db.add_log('WARNING', 'MoondreamAPI', "torchao未安装，跳过量化")
                
                self.model = model
                self.model_loaded = True
                self.model_loading = False
                
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                self.db.add_log('INFO', 'MoondreamAPI', f"模型加载完成，显存占用: {gpu_mem:.2f} GB")
                return True, f"模型加载完成，显存占用: {gpu_mem:.2f} GB"
                
            except Exception as e:
                self.model_loading = False
                self.model_error = str(e)
                self.db.add_log('ERROR', 'MoondreamAPI', f"模型加载失败: {str(e)}")
                return False, f"模型加载失败: {str(e)}"
    
    def unload_model(self):
        if not self.model_loaded:
            return True, "模型未加载"
        
        if self.model_loading:
            return False, "模型正在加载中，无法卸载"
        
        try:
            for stream_id in list(self._active_streams.keys()):
                self.stop_stream(stream_id)
            
            import torch
            import gc
            
            del self.model
            self.model = None
            self.model_loaded = False
            self.model_error = None
            
            gc.collect()
            torch.cuda.empty_cache()
            
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            self.db.add_log('INFO', 'MoondreamAPI', f"模型已卸载，显存占用: {gpu_mem:.2f} GB")
            return True, f"模型已卸载，显存已释放"
            
        except Exception as e:
            self.db.add_log('ERROR', 'MoondreamAPI', f"模型卸载失败: {str(e)}")
            return False, f"模型卸载失败: {str(e)}"
    
    def analyze_frame(self, frame, question):
        if not self.model_loaded:
            return None, "模型未加载"
        
        try:
            import torch
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            start_time = time.time()
            with torch.inference_mode():
                encoded_image = self.model.encode_image(pil_image)
                answer = self.model.query(encoded_image, question)["answer"]
            elapsed = time.time() - start_time
            
            torch.cuda.empty_cache()
            
            return {
                'answer': answer,
                'inference_time': round(elapsed, 2),
                'timestamp': datetime.now().isoformat()
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def analyze_image_base64(self, image_base64, question):
        if not self.model_loaded:
            return None, "模型未加载，请先加载模型"
        
        try:
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None, "无法解码图片"
            
            return self.analyze_frame(frame, question)
        except Exception as e:
            return None, str(e)
    
    def start_stream(self, stream_id, video_source, question, interval=3):
        if stream_id in self._stream_threads:
            return False, "该流已在运行"
        
        if not self.model_loaded:
            return False, "模型未加载"
        
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return False, "无法打开视频源"
        
        is_video_file = isinstance(video_source, str) and not video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')) and not video_source.isdigit()
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30
        frame_delay = 1.0 / video_fps
        
        stop_event = threading.Event()
        self._stream_stop_events[stream_id] = stop_event
        self._stream_results[stream_id] = {
            'status': 'running',
            'question': question,
            'interval': interval,
            'answers': [],
            'current_frame': None,
            'analysis_frame': None
        }
        
        self._stream_frames[stream_id] = None
        self._stream_frame_lock[stream_id] = threading.Lock()
        
        def run_display_stream():
            nonlocal cap
            frame_count = 0
            last_frame_time = time.time()
            
            while not stop_event.is_set():
                current_time = time.time()
                elapsed_since_last = current_time - last_frame_time
                
                if elapsed_since_last < frame_delay:
                    time.sleep(min(frame_delay - elapsed_since_last, 0.005))
                    continue
                
                last_frame_time = time.time()
                
                ret, frame = cap.read()
                
                if not ret:
                    if is_video_file:
                        cap.release()
                        cap = cv2.VideoCapture(video_source)
                        if cap.isOpened():
                            frame_count = 0
                            continue
                    break
                
                frame_count += 1
                
                ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret_encode:
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    self._stream_results[stream_id]['current_frame'] = frame_b64
                
                with self._stream_frame_lock[stream_id]:
                    self._stream_frames[stream_id] = frame.copy()
            
            cap.release()
            self._stream_results[stream_id]['status'] = 'stopped'
        
        def run_analysis_stream():
            last_analysis_time = 0
            
            while not stop_event.is_set():
                current_time = time.time()
                
                if current_time - last_analysis_time < interval:
                    time.sleep(0.1)
                    continue
                
                with self._stream_frame_lock.get(stream_id, threading.Lock()):
                    frame = self._stream_frames.get(stream_id)
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                last_analysis_time = current_time
                
                with self._analysis_lock:
                    try:
                        result, error = self.analyze_frame(frame, question)
                        if result:
                            result_entry = {
                                'answer': result['answer'],
                                'inference_time': result['inference_time'],
                                'timestamp': result['timestamp'],
                                'question': question
                            }
                            self._stream_results[stream_id]['answers'].append(result_entry)
                            if len(self._stream_results[stream_id]['answers']) > 100:
                                self._stream_results[stream_id]['answers'] = self._stream_results[stream_id]['answers'][-50:]
                    except Exception as e:
                        self.db.add_log('ERROR', 'MoondreamAPI', f"分析帧错误: {str(e)}")
        
        display_thread = threading.Thread(target=run_display_stream, daemon=True)
        display_thread.start()
        
        analysis_thread = threading.Thread(target=run_analysis_stream, daemon=True)
        analysis_thread.start()
        
        self._stream_threads[stream_id] = display_thread
        self._active_streams[stream_id] = {
            'video_source': video_source,
            'question': question,
            'interval': interval
        }
        
        return True, "流已启动"
    
    def stop_stream(self, stream_id):
        if stream_id not in self._stream_stop_events:
            return False, "流不存在"
        
        self._stream_stop_events[stream_id].set()
        
        if stream_id in self._active_streams:
            del self._active_streams[stream_id]
        if stream_id in self._stream_threads:
            del self._stream_threads[stream_id]
        if stream_id in self._stream_stop_events:
            del self._stream_stop_events[stream_id]
        if stream_id in self._stream_frames:
            del self._stream_frames[stream_id]
        if stream_id in self._stream_frame_lock:
            del self._stream_frame_lock[stream_id]
        
        if stream_id in self._stream_results:
            self._stream_results[stream_id]['status'] = 'stopped'
        
        return True, "流已停止"
    
    def get_stream_result(self, stream_id):
        if stream_id not in self._stream_results:
            return None
        return self._stream_results[stream_id]
    
    def get_stream_frame(self, stream_id):
        if stream_id not in self._stream_results:
            return None
        return self._stream_results[stream_id].get('current_frame')
    
    def get_active_streams(self):
        return list(self._active_streams.keys())
    
    def update_question(self, stream_id, question):
        if stream_id in self._active_streams:
            self._active_streams[stream_id]['question'] = question
            if stream_id in self._stream_results:
                self._stream_results[stream_id]['question'] = question
            return True
        return False
    
    def update_interval(self, stream_id, interval):
        if stream_id in self._active_streams:
            self._active_streams[stream_id]['interval'] = interval
            if stream_id in self._stream_results:
                self._stream_results[stream_id]['interval'] = interval
            return True
        return False
