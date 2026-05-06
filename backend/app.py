from flask import Flask, jsonify, request, send_from_directory, send_file, Response
from flask_cors import CORS
import os
import uuid
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename
from backend.config import WARNING_DIR, BASE_DIR, UPLOAD_DIR, ALLOWED_IMAGE_FORMATS, ALLOWED_VIDEO_FORMATS, DETECTION_TYPES
from backend.database import Database
from backend.detection_engine import DetectionEngine
from backend.siliconflow_api import SiliconFlowAPI
from backend.offline_processor import OfflineProcessor
from backend.assistant_api import AssistantAPI
from backend.moondream_api import MoondreamAPI

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')
CORS(app)

db = Database()
engine = DetectionEngine()
ai_api = SiliconFlowAPI()
offline_processor = OfflineProcessor()
assistant = AssistantAPI()
moondream = MoondreamAPI()


@app.route('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')


@app.route('/api/detection_types')
def get_detection_types():
    return jsonify({
        'success': True,
        'data': DETECTION_TYPES
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    status = engine.get_status()
    return jsonify({
        'success': True,
        'data': status
    })


@app.route('/api/detection/<detection_type>/start', methods=['POST'])
def start_detection(detection_type):
    data = request.get_json() or {}
    video_source = data.get('video_source', 0)
    camera_id = data.get('camera_id')
    roi_points = data.get('roi_points')
    
    success, message = engine.start_detection(detection_type, video_source, camera_id, roi_points)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/detection/<detection_type>/stop', methods=['POST'])
def stop_detection(detection_type):
    success, message = engine.stop_detection(detection_type)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/detection/<detection_type>/enable', methods=['POST'])
def enable_detection(detection_type):
    data = request.get_json() or {}
    enabled = data.get('enabled', True)
    
    success, message = engine.enable_detection(detection_type, enabled)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/detection/<detection_type>/config', methods=['POST'])
def update_detection_config(detection_type):
    config = request.get_json() or {}
    
    success, message = engine.update_config(detection_type, config)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/video_feed/<detection_type>')
def video_feed(detection_type):
    def generate():
        while True:
            frame = engine.get_frame(detection_type)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                import time
                time.sleep(0.1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/frame/<detection_type>')
def get_frame(detection_type):
    frame = engine.get_frame(detection_type)
    if frame:
        return Response(frame, mimetype='image/jpeg')
    return jsonify({
        'success': False,
        'message': '无可用帧'
    }), 404


@app.route('/api/camera/preview', methods=['POST'])
def get_camera_preview():
    data = request.get_json() or {}
    video_source = data.get('video_source', 0)
    
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    cap = cv2.VideoCapture(video_source)
    
    if isinstance(video_source, str) and video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        cap.release()
        return jsonify({
            'success': False,
            'message': f'无法打开视频源：{video_source}'
        }), 400
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({
            'success': False,
            'message': '无法读取视频帧'
        }), 400
    
    ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret_encode:
        return jsonify({
            'success': False,
            'message': '编码失败'
        }), 500
    
    import base64
    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    return jsonify({
        'success': True,
        'data': {
            'frame': frame_base64,
            'width': frame.shape[1],
            'height': frame.shape[0]
        }
    })


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    data = request.get_json() or {}
    camera_id = data.get('camera_id', str(uuid.uuid4()))
    video_source = data.get('video_source', 0)
    detection_types = data.get('detection_types', [])
    roi_points = data.get('roi_points')
    
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    if not detection_types:
        return jsonify({
            'success': False,
            'message': '请选择至少一个检测类型'
        }), 400
    
    success, message = engine.start_multi_detection(camera_id, video_source, detection_types, roi_points)
    return jsonify({
        'success': success,
        'message': message,
        'data': {'camera_id': camera_id} if success else None
    })


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    cameras = engine.get_camera_details()
    return jsonify({
        'success': True,
        'data': cameras
    })


@app.route('/api/camera/<camera_id>/stop', methods=['POST'])
def stop_camera(camera_id):
    success, message = engine.stop_multi_detection(camera_id)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/camera/<camera_id>/feed')
def camera_feed(camera_id):
    def generate():
        while True:
            frame = engine.get_multi_frame(camera_id)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                import time
                time.sleep(0.1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/<camera_id>/frame')
def get_camera_frame(camera_id):
    frame = engine.get_multi_frame(camera_id)
    if frame:
        return Response(frame, mimetype='image/jpeg')
    return jsonify({
        'success': False,
        'message': '无可用帧'
    }), 404


@app.route('/api/cameras/running')
def get_running_cameras():
    cameras = engine.get_running_cameras()
    return jsonify({
        'success': True,
        'data': cameras
    })


@app.route('/api/warnings', methods=['GET'])
def get_warnings():
    warning_type = request.args.get('type')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    unread_only = request.args.get('unread_only', False, type=bool)
    camera_id = request.args.get('camera_id', type=int)
    
    warnings = db.get_warnings(warning_type, limit, offset, unread_only, camera_id)
    total = db.get_warning_count(warning_type)
    unread = db.get_unread_count(warning_type)
    
    return jsonify({
        'success': True,
        'data': {
            'warnings': warnings,
            'total': total,
            'unread': unread
        }
    })


@app.route('/api/warnings/<int:warning_id>', methods=['GET'])
def get_warning(warning_id):
    warning = db.get_warning_by_id(warning_id)
    if warning:
        return jsonify({
            'success': True,
            'data': warning
        })
    return jsonify({
        'success': False,
        'message': '警告记录不存在'
    }), 404


@app.route('/api/warnings/<int:warning_id>/read', methods=['POST'])
def mark_warning_read(warning_id):
    db.mark_warning_read(warning_id)
    return jsonify({
        'success': True,
        'message': '已标记为已读'
    })


@app.route('/api/warnings/read-all', methods=['POST'])
def mark_all_warnings_read():
    warning_type = request.args.get('type')
    db.mark_all_warnings_read(warning_type)
    return jsonify({
        'success': True,
        'message': '已全部标记为已读'
    })


@app.route('/api/warnings/<int:warning_id>', methods=['DELETE'])
def delete_warning(warning_id):
    warning = db.get_warning_by_id(warning_id)
    if warning and warning.get('screenshot_path'):
        try:
            if os.path.exists(warning['screenshot_path']):
                os.remove(warning['screenshot_path'])
        except:
            pass
    
    db.delete_warning(warning_id)
    return jsonify({
        'success': True,
        'message': '已删除'
    })


@app.route('/api/warnings/<int:warning_id>/analyze', methods=['POST'])
def analyze_warning(warning_id):
    description, error = ai_api.analyze_warning(warning_id)
    if error:
        return jsonify({
            'success': False,
            'message': error
        }), 500
    return jsonify({
        'success': True,
        'data': {
            'description': description
        }
    })


@app.route('/api/screenshots/<path:filename>')
def get_screenshot(filename):
    filepath = os.path.join(WARNING_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({
        'success': False,
        'message': '文件不存在'
    }), 404


@app.route('/api/stats/daily', methods=['GET'])
def get_daily_stats():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    stats = db.get_daily_stats(date)
    return jsonify({
        'success': True,
        'data': stats
    })


@app.route('/api/stats/range', methods=['GET'])
def get_stats_range():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({
            'success': False,
            'message': '请提供开始和结束日期'
        }), 400
    
    stats = db.get_stats_range(start_date, end_date)
    return jsonify({
        'success': True,
        'data': stats
    })


@app.route('/api/stats/summary', methods=['GET'])
def get_stats_summary():
    today = datetime.now().strftime('%Y-%m-%d')
    today_stats = db.get_daily_stats(today)
    
    total_warnings = db.get_warning_count()
    unread_warnings = db.get_unread_count()
    
    status = engine.get_status()
    running_count = sum(1 for s in status.values() if s['running'])
    
    return jsonify({
        'success': True,
        'data': {
            'today': today_stats or {
                'fire_count': 0,
                'person_loitering_count': 0,
                'vehicle_loitering_count': 0,
                'shutter_change_count': 0,
                'fence_crossing_count': 0
            },
            'total_warnings': total_warnings,
            'unread_warnings': unread_warnings,
            'running_detectors': running_count,
            'total_detectors': len(status)
        }
    })


@app.route('/api/ai/daily-report', methods=['GET'])
def generate_daily_report():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    result, error = ai_api.generate_daily_report(date)
    if error:
        return jsonify({
            'success': False,
            'message': error
        }), 500
    return jsonify({
        'success': True,
        'data': result
    })


@app.route('/api/reports/list', methods=['GET'])
def get_report_list():
    reports = ai_api.get_report_list()
    return jsonify({
        'success': True,
        'data': reports
    })


@app.route('/api/reports/<path:filename>', methods=['GET'])
def get_report(filename):
    result, error = ai_api.get_report_content(filename)
    if error:
        return jsonify({
            'success': False,
            'message': error
        }), 404
    return jsonify({
        'success': True,
        'data': result
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    limit = request.args.get('limit', 100, type=int)
    level = request.args.get('level')
    logs = db.get_logs(limit, level)
    return jsonify({
        'success': True,
        'data': logs
    })


@app.route('/api/cameras', methods=['POST'])
def add_camera():
    data = request.get_json()
    name = data.get('name')
    source = data.get('source')
    
    if not name or not source:
        return jsonify({
            'success': False,
            'message': '请提供摄像头名称和源'
        }), 400
    
    camera_id = db.add_camera(name, source)
    return jsonify({
        'success': True,
        'data': {
            'id': camera_id
        }
    })


@app.route('/api/camera/<int:camera_id>/detectors', methods=['GET'])
def get_camera_detectors(camera_id):
    detectors = db.get_camera_detectors(camera_id)
    return jsonify({
        'success': True,
        'data': detectors
    })


@app.route('/api/camera/<int:camera_id>/detectors', methods=['POST'])
def update_camera_detectors(camera_id):
    data = request.get_json() or {}
    detection_type = data.get('detection_type')
    enabled = data.get('enabled')
    roi_points = data.get('roi_points')
    config = data.get('config')
    
    db.update_camera_detectors(camera_id, detection_type, enabled, roi_points, config)
    return jsonify({
        'success': True,
        'message': '配置已更新'
    })


@app.route('/api/roi', methods=['POST'])
def add_roi_region():
    data = request.get_json()
    camera_id = data.get('camera_id')
    name = data.get('name')
    detection_type = data.get('detection_type')
    points = data.get('points')
    
    if not all([camera_id, name, detection_type, points]):
        return jsonify({
            'success': False,
            'message': '请提供完整信息'
        }), 400
    
    region_id = db.add_roi_region(camera_id, name, detection_type, points)
    return jsonify({
        'success': True,
        'data': {
            'id': region_id
        }
    })


@app.route('/api/roi', methods=['GET'])
def get_roi_regions():
    camera_id = request.args.get('camera_id', type=int)
    detection_type = request.args.get('detection_type')
    
    regions = db.get_roi_regions(camera_id, detection_type)
    return jsonify({
        'success': True,
        'data': regions
    })


@app.route('/api/roi/<int:region_id>', methods=['DELETE'])
def delete_roi_region(region_id):
    db.delete_roi_region(region_id)
    return jsonify({
        'success': True,
        'message': '已删除'
    })


@app.route('/api/offline/upload', methods=['POST'])
def upload_offline_file():
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': '未找到上传文件'
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': '未选择文件'
        }), 400
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_FORMATS and ext not in ALLOWED_VIDEO_FORMATS:
        return jsonify({
            'success': False,
            'message': f'不支持的文件格式: {ext}'
        }), 400
    
    filename = secure_filename(f"{uuid.uuid4()}{ext}")
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'data': {
            'filename': filename,
            'filepath': filepath,
            'type': 'image' if ext in ALLOWED_IMAGE_FORMATS else 'video'
        }
    })


@app.route('/api/offline/start', methods=['POST'])
def start_offline_task():
    data = request.get_json() or {}
    file_path = data.get('file_path')
    detection_types = data.get('detection_types', [])
    conf_threshold = data.get('conf_threshold', 0.5)
    
    if not file_path or not detection_types:
        return jsonify({
            'success': False,
            'message': '请提供文件路径和检测类型'
        }), 400
    
    valid, msg = offline_processor.validate_file(file_path)
    if not valid:
        return jsonify({
            'success': False,
            'message': msg
        }), 400
    
    task_id = offline_processor.start_task(file_path, detection_types, conf_threshold)
    
    return jsonify({
        'success': True,
        'data': {
            'task_id': task_id
        }
    })


@app.route('/api/offline/<int:task_id>', methods=['GET'])
def get_offline_task(task_id):
    task = offline_processor.get_task_status(task_id)
    if task:
        return jsonify({
            'success': True,
            'data': task
        })
    return jsonify({
        'success': False,
        'message': '任务不存在'
    }), 404


@app.route('/api/offline/tasks', methods=['GET'])
def get_offline_tasks():
    tasks = offline_processor.get_all_tasks()
    return jsonify({
        'success': True,
        'data': tasks
    })


@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    data = request.get_json() or {}
    message = data.get('message', '')
    
    if not message:
        return jsonify({
            'success': False,
            'message': '请输入消息'
        }), 400
    
    parsed, error = assistant.process_message(message)
    
    if error:
        return jsonify({
            'success': False,
            'message': error
        }), 500
    
    print(f"[Assistant] Parsed response: {parsed}")
    
    if parsed.get('type') == 'command':
        action = parsed.get('action')
        params = parsed.get('params', {})
        print(f"[Assistant] Executing command: {action} with params: {params}")
        result = assistant.execute_command(action, params, engine)
        print(f"[Assistant] Command result: {result}")
        
        return jsonify({
            'success': True,
            'data': {
                'type': 'command_result',
                'action': action,
                'result': result,
                'message': parsed.get('message', '')
            }
        })
    else:
        return jsonify({
            'success': True,
            'data': {
                'type': 'response',
                'message': parsed.get('message', '')
            }
        })


@app.route('/api/assistant/clear', methods=['POST'])
def assistant_clear():
    result = assistant.clear_history()
    return jsonify(result)


@app.route('/api/assistant/execute', methods=['POST'])
def assistant_execute():
    data = request.get_json() or {}
    action = data.get('action')
    params = data.get('params', {})
    
    if not action:
        return jsonify({
            'success': False,
            'message': '请指定操作类型'
        }), 400
    
    result = assistant.execute_command(action, params, engine)
    return jsonify(result)


@app.route('/api/moondream/status', methods=['GET'])
def moondream_status():
    status = moondream.get_status()
    return jsonify({
        'success': True,
        'data': status
    })


@app.route('/api/moondream/load', methods=['POST'])
def moondream_load():
    success, message = moondream.load_model()
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/moondream/unload', methods=['POST'])
def moondream_unload():
    success, message = moondream.unload_model()
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/moondream/analyze', methods=['POST'])
def moondream_analyze():
    data = request.get_json() or {}
    image_base64 = data.get('image')
    question = data.get('question', '请用中文详细描述你在这张图片中看到的内容。')
    
    if not image_base64:
        return jsonify({
            'success': False,
            'message': '请提供图片数据'
        }), 400
    
    result, error = moondream.analyze_image_base64(image_base64, question)
    if error:
        return jsonify({
            'success': False,
            'message': error
        }), 500
    
    return jsonify({
        'success': True,
        'data': result
    })


@app.route('/api/moondream/stream/start', methods=['POST'])
def moondream_stream_start():
    data = request.get_json() or {}
    video_source = data.get('video_source', '0')
    question = data.get('question', '请用中文详细描述你在这张图片中看到的内容。')
    interval = data.get('interval', 3)
    
    stream_id = f"md_{uuid.uuid4().hex[:8]}"
    success, message = moondream.start_stream(stream_id, video_source, question, interval)
    
    return jsonify({
        'success': success,
        'message': message,
        'data': {'stream_id': stream_id} if success else None
    })


@app.route('/api/moondream/stream/<stream_id>/stop', methods=['POST'])
def moondream_stream_stop(stream_id):
    success, message = moondream.stop_stream(stream_id)
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/moondream/stream/<stream_id>/result', methods=['GET'])
def moondream_stream_result(stream_id):
    result = moondream.get_stream_result(stream_id)
    if result is None:
        return jsonify({
            'success': False,
            'message': '流不存在'
        }), 404
    
    return jsonify({
        'success': True,
        'data': result
    })


@app.route('/api/moondream/stream/<stream_id>/frame', methods=['GET'])
def moondream_stream_frame(stream_id):
    frame = moondream.get_stream_frame(stream_id)
    if frame is None:
        return jsonify({
            'success': False,
            'message': '无可用帧'
        }), 404
    
    return jsonify({
        'success': True,
        'data': {'frame': frame}
    })


@app.route('/api/moondream/stream/<stream_id>/update', methods=['POST'])
def moondream_stream_update(stream_id):
    data = request.get_json() or {}
    question = data.get('question')
    interval = data.get('interval')
    
    if question:
        moondream.update_question(stream_id, question)
    if interval:
        moondream.update_interval(stream_id, interval)
    
    return jsonify({
        'success': True,
        'message': '已更新'
    })


@app.route('/api/moondream/translate', methods=['POST'])
def moondream_translate():
    data = request.get_json() or {}
    text = data.get('text', '')
    target_lang = data.get('target_lang', 'zh')
    
    if not text:
        return jsonify({
            'success': False,
            'message': '请提供待翻译文本'
        }), 400
    
    try:
        import requests as req
        headers = {
            "Authorization": f"Bearer {ai_api.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"请将以下文本翻译为{'中文' if target_lang == 'zh' else '英文'}，只返回翻译结果，不要添加任何解释或额外内容：\n\n{text}"
        
        payload = {
            "model": "Qwen/Qwen3-32B",
            "messages": [
                {"role": "system", "content": "你是一个专业的翻译助手，只返回翻译结果。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.3
        }
        
        response = req.post(ai_api.api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        translated = result["choices"][0]["message"]["content"].strip()
        
        return jsonify({
            'success': True,
            'data': {
                'original': text,
                'translated': translated,
                'target_lang': target_lang
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'翻译失败: {str(e)}'
        }), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
