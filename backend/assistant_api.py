import json
import threading
from datetime import datetime, timedelta
from backend.config import SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_MODEL
from backend.database import Database
import requests
import time
import re


class AssistantAPI:
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
        self.api_key = SILICONFLOW_API_KEY
        self.api_url = SILICONFLOW_API_URL
        self.model = SILICONFLOW_MODEL
        self.db = Database()
        self.max_retries = 3
        self.retry_delay = 2
        self.conversation_history = []
        self.max_history = 10
    
    def _make_request(self, messages, max_tokens=1024):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise Exception(f"API请求失败：{str(e)}")
        
        raise Exception("API请求重试次数已用尽")
    
    def _get_system_prompt(self):
        return """你是一个智能监控系统的AI助手。你可以帮助用户：
1. 控制摄像头（开启、停止、查看状态）
2. 管理检测功能（火焰检测、人员滞留检测、车辆滞留检测、卷帘门状态检测、围栏翻越检测）
3. 查询警告记录和统计数据
4. 生成分析报告

当用户发送指令时，你需要解析用户的意图并以JSON格式返回操作指令。

**重要：你的回复必须严格遵循以下JSON格式，不要添加任何其他文字：**

对于需要执行操作的指令：
```json
{
    "type": "command",
    "action": "操作类型",
    "params": {},
    "message": "给用户的确认消息"
}
```

对于纯查询或对话：
```json
{
    "type": "response",
    "message": "回复内容"
}
```

**支持的操作类型(action)：**
- `get_status` - 获取系统状态，params: {}
- `get_cameras` - 获取摄像头列表，params: {}
- `get_warnings` - 获取警告记录，params: {"type": "类型(可选)", "limit": 数量}
- `get_stats` - 获取统计数据，params: {"date": "日期(可选)"}
- `start_camera` - 启动摄像头，params: {"source": "视频源", "detection_types": ["类型1","类型2"]}
- `stop_camera` - 停止摄像头，params: {"camera_id": "摄像头ID"}
- `stop_all_cameras` - 停止所有摄像头，params: {}
- `generate_report` - 生成报告，params: {}
- `get_detection_status` - 获取检测功能状态，params: {}
- `update_config` - 更新检测配置，params: {"detection_type": "类型", "config": {}}

**检测类型包括：**
- fire - 火焰检测
- person_loitering - 人员滞留检测
- vehicle_loitering - 车辆滞留检测
- shutter - 卷帘门状态检测
- fence_crossing - 围栏翻越检测

**示例对话：**

用户: "查看系统状态"
回复: {"type": "command", "action": "get_status", "params": {}, "message": "正在查询系统状态..."}

用户: "今日有多少警告？"
回复: {"type": "command", "action": "get_stats", "params": {}, "message": "正在查询今日统计数据..."}

用户: "开启火焰检测"
回复: {"type": "command", "action": "start_camera", "params": {"source": "0", "detection_types": ["fire"]}, "message": "正在启动火焰检测..."}

用户: "停止所有摄像头"
回复: {"type": "command", "action": "stop_all_cameras", "params": {}, "message": "正在停止所有摄像头..."}

用户: "你好"
回复: {"type": "response", "message": "您好！我是智能监控助手，有什么可以帮您的吗？"}

请始终返回有效的JSON格式，不要包含任何其他内容。"""
    
    def _get_context_info(self):
        context = {}
        
        try:
            status = {}
            detection_status = self.db.get_detection_status()
            for det_type, det_info in detection_status.items():
                status[det_type] = {
                    'enabled': det_info.get('enabled', True),
                    'running': det_info.get('running', False)
                }
            context['detection_status'] = status
            
            today = datetime.now().strftime('%Y-%m-%d')
            stats = self.db.get_daily_stats(today)
            if stats:
                context['today_stats'] = {
                    'fire_count': stats.get('fire_count', 0),
                    'person_loitering_count': stats.get('person_loitering_count', 0),
                    'vehicle_loitering_count': stats.get('vehicle_loitering_count', 0),
                    'shutter_change_count': stats.get('shutter_change_count', 0),
                    'fence_crossing_count': stats.get('fence_crossing_count', 0)
                }
            
            warnings = self.db.get_warnings(limit=5)
            context['recent_warnings'] = len(warnings)
            
        except Exception as e:
            context['error'] = str(e)
        
        return context
    
    def _extract_json(self, response):
        response = response.strip()
        
        if '```' in response:
            start_idx = response.find('```')
            end_idx = response.rfind('```')
            if start_idx != end_idx:
                json_part = response[start_idx+3:end_idx].strip()
                if json_part.startswith('json'):
                    json_part = json_part[4:].strip()
                return json_part
        
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            return response[start:end+1]
        
        return response
    
    def process_message(self, user_message):
        try:
            context = self._get_context_info()
            
            context_message = f"""当前系统上下文信息：
{json.dumps(context, ensure_ascii=False, indent=2)}

用户消息：{user_message}

请根据用户消息和系统上下文，返回相应的JSON响应。只返回JSON，不要包含其他文字。"""
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": context_message}
            ]
            
            if self.conversation_history:
                recent_history = self.conversation_history[-self.max_history:]
                messages = messages[:1] + recent_history + messages[1:]
            
            response = self._make_request(messages)
            
            print(f"[Assistant] Raw AI response: {response[:500]}...")
            
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            json_str = self._extract_json(response)
            print(f"[Assistant] Extracted JSON: {json_str[:300]}...")
            
            try:
                parsed = json.loads(json_str)
                print(f"[Assistant] Parsed type: {parsed.get('type')}")
                
                if parsed.get('type') == 'command':
                    parsed['action'] = parsed.get('action', '')
                    parsed['params'] = parsed.get('params', {})
                
                return parsed, None
            except json.JSONDecodeError as e:
                print(f"[Assistant] JSON decode error: {e}")
                return {"type": "response", "message": f"解析响应失败，原始响应：{response[:200]}..."}, None
                
        except Exception as e:
            return None, str(e)
    
    def execute_command(self, action, params, engine):
        try:
            if action == 'get_status':
                return self._get_system_status(engine)
            elif action == 'get_cameras':
                return self._get_cameras(engine)
            elif action == 'get_warnings':
                return self._get_warnings(params)
            elif action == 'get_stats':
                return self._get_stats(params)
            elif action == 'start_camera':
                return self._start_camera(params, engine)
            elif action == 'stop_camera':
                return self._stop_camera(params, engine)
            elif action == 'stop_all_cameras':
                return self._stop_all_cameras(engine)
            elif action == 'generate_report':
                return self._generate_report()
            elif action == 'get_detection_status':
                return self._get_detection_status()
            elif action == 'update_config':
                return self._update_config(params)
            else:
                return {"success": False, "message": f"未知操作：{action}"}
        except Exception as e:
            return {"success": False, "message": f"执行操作失败：{str(e)}"}
    
    def _get_system_status(self, engine):
        status = engine.get_status()
        running_count = sum(1 for s in status.values() if s['running'])
        cameras = engine.get_running_cameras()
        
        return {
            "success": True,
            "data": {
                "running_detectors": running_count,
                "total_detectors": len(status),
                "running_cameras": len(cameras),
                "status_details": status
            },
            "message": f"系统状态：{running_count}个检测器运行中，{len(cameras)}个摄像头在线"
        }
    
    def _get_cameras(self, engine):
        cameras = engine.get_running_cameras()
        return {
            "success": True,
            "data": {"cameras": cameras},
            "message": f"当前有{len(cameras)}个摄像头运行中"
        }
    
    def _get_warnings(self, params):
        warning_type = params.get('type')
        limit = params.get('limit', 10)
        warnings = self.db.get_warnings(warning_type=warning_type, limit=limit)
        
        return {
            "success": True,
            "data": {"warnings": warnings, "count": len(warnings)},
            "message": f"查询到{len(warnings)}条警告记录"
        }
    
    def _get_stats(self, params):
        date = params.get('date', datetime.now().strftime('%Y-%m-%d'))
        stats = self.db.get_daily_stats(date)
        
        if stats:
            total = (stats.get('fire_count', 0) + stats.get('person_loitering_count', 0) + 
                    stats.get('vehicle_loitering_count', 0) + stats.get('shutter_change_count', 0) +
                    stats.get('fence_crossing_count', 0))
            return {
                "success": True,
                "data": {"stats": stats, "date": date},
                "message": f"{date} 统计：共{total}条警告（火焰:{stats.get('fire_count',0)}，人员滞留:{stats.get('person_loitering_count',0)}，车辆滞留:{stats.get('vehicle_loitering_count',0)}）"
            }
        return {
            "success": True,
            "data": {"stats": {}, "date": date},
            "message": f"{date} 暂无统计数据"
        }
    
    def _start_camera(self, params, engine):
        import uuid
        camera_id = f"cam_{uuid.uuid4().hex[:8]}"
        source = params.get('source', '0')
        detection_types = params.get('detection_types', ['fire'])
        
        success, message = engine.start_multi_detection(camera_id, source, detection_types)
        return {
            "success": success,
            "data": {"camera_id": camera_id} if success else None,
            "message": message
        }
    
    def _stop_camera(self, params, engine):
        camera_id = params.get('camera_id')
        if not camera_id:
            cameras = engine.get_running_cameras()
            if cameras:
                camera_id = cameras[0]
            else:
                return {"success": False, "message": "没有运行中的摄像头"}
        
        success, message = engine.stop_multi_detection(camera_id)
        return {"success": success, "message": message}
    
    def _stop_all_cameras(self, engine):
        cameras = engine.get_running_cameras()
        stopped = 0
        for camera_id in cameras:
            success, _ = engine.stop_multi_detection(camera_id)
            if success:
                stopped += 1
        
        return {
            "success": True,
            "message": f"已停止{stopped}个摄像头"
        }
    
    def _generate_report(self):
        from backend.siliconflow_api import SiliconFlowAPI
        api = SiliconFlowAPI()
        result, error = api.generate_daily_report()
        
        if error:
            return {"success": False, "message": error}
        return {
            "success": True,
            "data": result,
            "message": "报告生成成功"
        }
    
    def _get_detection_status(self):
        status = self.db.get_detection_status()
        return {
            "success": True,
            "data": {"status": status},
            "message": "检测状态查询成功"
        }
    
    def _update_config(self, params):
        detection_type = params.get('detection_type')
        config = params.get('config', {})
        
        if not detection_type:
            return {"success": False, "message": "请指定检测类型"}
        
        self.db.update_detection_status(detection_type, config=config)
        return {
            "success": True,
            "message": f"{detection_type} 配置已更新"
        }
    
    def clear_history(self):
        self.conversation_history = []
        return {"success": True, "message": "对话历史已清空"}
