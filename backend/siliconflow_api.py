import base64
import requests
import time
import threading
import os
from datetime import datetime
from backend.config import SILICONFLOW_API_KEY, SILICONFLOW_API_URL, SILICONFLOW_MODEL, REPORT_DIR
from backend.database import Database


class SiliconFlowAPI:
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
    
    def _encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _make_request(self, image_base64, prompt, max_tokens=512):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise Exception("API请求超时")
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise Exception(f"API请求失败：{str(e)}")
        
        raise Exception("API请求重试次数已用尽")
    
    def describe_image(self, image_path, prompt="请用简练的语言描述画面内容，50字以内，只说关键信息"):
        try:
            image_base64 = self._encode_image_to_base64(image_path)
            return self._make_request(image_base64, prompt)
        except Exception as e:
            self.db.add_log('ERROR', 'siliconflow', f"图像描述失败：{str(e)}")
            raise
    
    def analyze_warning(self, warning_id):
        warning = self.db.get_warning_by_id(warning_id)
        if not warning:
            return None, "警告记录不存在"
        
        screenshot_path = warning.get('screenshot_path')
        if not screenshot_path or not os.path.exists(screenshot_path):
            return None, "截图文件不存在"
        
        warning_type = warning.get('type')
        warning_message = warning.get('message')
        
        prompt = f"""这是一个监控系统的警告截图。
警告类型：{warning_type}
警告信息：{warning_message}

请分析这个画面：
1. 描述画面中的主要内容和场景
2. 指出可能存在的安全隐患或异常情况
3. 给出建议的处理措施

请用简洁的中文回答，不超过100字。"""
        
        try:
            description = self.describe_image(screenshot_path, prompt)
            self.db.update_warning_ai_description(warning_id, description)
            return description, None
        except Exception as e:
            return None, str(e)
    
    def generate_daily_report(self, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        stats = self.db.get_daily_stats(date)
        if not stats:
            return None, "当日无统计数据"
        
        warnings = self.db.get_warnings(limit=50)
        today_warnings = [w for w in warnings if w['created_at'].startswith(date)]
        
        if not today_warnings:
            report = "今日无警告事件，系统运行正常。"
            self._save_report_to_file(date, report)
            return {'report': report, 'date': date}, None
        
        warning_summary = {}
        for w in today_warnings:
            w_type = w['type']
            if w_type not in warning_summary:
                warning_summary[w_type] = []
            warning_summary[w_type].append(w['message'])
        
        prompt = f"""请分析以下监控系统的当日数据，生成一份简要的安全报告：

日期：{date}
统计数据：
- 火焰警告：{stats.get('fire_count', 0)} 次
- 人员滞留警告：{stats.get('person_loitering_count', 0)} 次
- 车辆滞留警告：{stats.get('vehicle_loitering_count', 0)} 次
- 卷帘门状态变化：{stats.get('shutter_change_count', 0)} 次

警告详情：
{self._format_warning_summary(warning_summary)}

请生成一份包含以下内容的报告：
1. 当日安全状况概述
2. 主要风险点分析
3. 改进建议

请使用Markdown格式，包含标题、列表等格式，不超过500字。"""
        
        try:
            report = self._make_request_with_text(prompt, max_tokens=1024)
            self.db.update_daily_ai_analysis(date, report)
            self._save_report_to_file(date, report)
            return {'report': report, 'date': date}, None
        except Exception as e:
            return None, str(e)
    
    def _save_report_to_file(self, date, report):
        os.makedirs(REPORT_DIR, exist_ok=True)
        filename = f"report_{date}.md"
        filepath = os.path.join(REPORT_DIR, filename)
        
        header = f"# AI分析报告\n\n**生成日期：** {date}\n\n---\n\n"
        content = header + report
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def get_report_list(self):
        reports = []
        if os.path.exists(REPORT_DIR):
            for filename in sorted(os.listdir(REPORT_DIR), reverse=True):
                if filename.endswith('.md') and filename.startswith('report_'):
                    filepath = os.path.join(REPORT_DIR, filename)
                    stat = os.stat(filepath)
                    date_str = filename.replace('report_', '').replace('.md', '')
                    reports.append({
                        'filename': filename,
                        'title': f'安全报告 ({date_str})',
                        'date': date_str,
                        'size': stat.st_size
                    })
        return reports
    
    def get_report_content(self, filename):
        filepath = os.path.join(REPORT_DIR, filename)
        if not os.path.exists(filepath):
            return None, "报告文件不存在"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        date_str = filename.replace('report_', '').replace('.md', '')
        return {'content': content, 'date': date_str}, None
    
    def _make_request_with_text(self, prompt, max_tokens=512):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens
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
    
    def _format_warning_summary(self, warning_summary):
        lines = []
        for w_type, messages in warning_summary.items():
            lines.append(f"- {w_type}：{len(messages)} 次")
            for msg in messages[:3]:
                lines.append(f"  - {msg}")
        return "\n".join(lines)
