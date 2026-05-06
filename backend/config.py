import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'monitoring.db')
WARNING_DIR = os.path.join(BASE_DIR, 'data', 'warnings')
SCREENSHOT_DIR = os.path.join(BASE_DIR, 'data', 'screenshots')
UPLOAD_DIR = os.path.join(BASE_DIR, 'data', 'uploads')
ROI_DIR = os.path.join(BASE_DIR, 'data', 'roi')
REPORT_DIR = os.path.join(BASE_DIR, 'data', 'reports')

MODELS = {
    'fire': os.path.join(BASE_DIR, 'yolov8n-fire-smoke.pt'),
    'general': os.path.join(BASE_DIR, 'yolov8n.pt'),
    'shutter': os.path.join(BASE_DIR, 'roller_shutter.pt')
}

SILICONFLOW_API_KEY = "Your API KEY"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_MODEL = "Pro/moonshotai/Kimi-K2.5"

DETECTION_CONFIG = {
    'fire': {
        'enabled': True,
        'conf_threshold': 0.5,
        'warning_cooldown': 5
    },
    'person_loitering': {
        'enabled': True,
        'conf_threshold': 0.5,
        'loitering_threshold': 10
    },
    'vehicle_loitering': {
        'enabled': True,
        'conf_threshold': 0.5,
        'loitering_threshold': 10
    },
    'shutter': {
        'enabled': True,
        'conf_threshold': 0.5
    },
    'fence_crossing': {
        'enabled': True,
        'conf_threshold': 0.5,
        'track_distance_threshold': 200,
        'max_track_lost_frames': 20
    }
}

DETECTION_TYPES = {
    'fire': {'name': '火焰检测', 'icon': 'fire', 'color': '#FF4D4F'},
    'person_loitering': {'name': '人员滞留', 'icon': 'person', 'color': '#FAAD14'},
    'vehicle_loitering': {'name': '车辆滞留', 'icon': 'car-front', 'color': '#4A90E2'},
    'shutter': {'name': '卷帘门状态', 'icon': 'door-closed', 'color': '#52C41A'},
    'fence_crossing': {'name': '围栏翻越', 'icon': 'bounding-box', 'color': '#722ED1'}
}

ALLOWED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

for dir_path in [os.path.dirname(DATABASE_PATH), WARNING_DIR, SCREENSHOT_DIR, UPLOAD_DIR, ROI_DIR, REPORT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
