import sqlite3
import os
import json
from datetime import datetime
from threading import Lock
from backend.config import DATABASE_PATH


class Database:
    _instance = None
    _lock = Lock()
    
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
        self._connection = None
        self._init_db()
    
    def _get_connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                screenshot_path TEXT,
                confidence REAL,
                bbox TEXT,
                extra_data TEXT,
                ai_description TEXT,
                camera_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_read INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_type TEXT NOT NULL UNIQUE,
                enabled INTEGER DEFAULT 1,
                running INTEGER DEFAULT 0,
                last_detection_time TIMESTAMP,
                config TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                module TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                fire_count INTEGER DEFAULT 0,
                person_loitering_count INTEGER DEFAULT 0,
                vehicle_loitering_count INTEGER DEFAULT 0,
                shutter_change_count INTEGER DEFAULT 0,
                fence_crossing_count INTEGER DEFAULT 0,
                ai_analysis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_detectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                detection_type TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                roi_points TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(camera_id, detection_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS roi_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                detection_type TEXT NOT NULL,
                points TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS offline_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                detection_types TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        detection_types = ['fire', 'person_loitering', 'vehicle_loitering', 'shutter', 'fence_crossing']
        for det_type in detection_types:
            cursor.execute('''
                INSERT OR IGNORE INTO detection_status (detection_type, enabled, running, config)
                VALUES (?, 1, 0, '{}')
            ''', (det_type,))
        
        cursor.execute("PRAGMA table_info(warnings)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'camera_id' not in columns:
            cursor.execute("ALTER TABLE warnings ADD COLUMN camera_id INTEGER")
        
        cursor.execute("PRAGMA table_info(daily_stats)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'fence_crossing_count' not in columns:
            cursor.execute("ALTER TABLE daily_stats ADD COLUMN fence_crossing_count INTEGER DEFAULT 0")
        
        cursor.execute("PRAGMA table_info(offline_tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'detection_types' not in columns:
            cursor.execute("ALTER TABLE offline_tasks ADD COLUMN detection_types TEXT DEFAULT '[]'")
        
        conn.commit()
    
    def add_warning(self, warning_type, message, screenshot_path=None, confidence=None, bbox=None, extra_data=None, camera_id=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        bbox_json = json.dumps(bbox) if bbox else None
        extra_json = json.dumps(extra_data) if extra_data else None
        
        cursor.execute('''
            INSERT INTO warnings (type, message, screenshot_path, confidence, bbox, extra_data, camera_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (warning_type, message, screenshot_path, confidence, bbox_json, extra_json, camera_id))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_warning_ai_description(self, warning_id, description):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE warnings SET ai_description = ? WHERE id = ?
        ''', (description, warning_id))
        conn.commit()
    
    def get_warnings(self, warning_type=None, limit=100, offset=0, unread_only=False, camera_id=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM warnings WHERE 1=1'
        params = []
        
        if warning_type:
            query += ' AND type = ?'
            params.append(warning_type)
        
        if unread_only:
            query += ' AND is_read = 0'
        
        if camera_id:
            query += ' AND camera_id = ?'
            params.append(camera_id)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_warning_by_id(self, warning_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM warnings WHERE id = ?', (warning_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def mark_warning_read(self, warning_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE warnings SET is_read = 1 WHERE id = ?', (warning_id,))
        conn.commit()
    
    def mark_all_warnings_read(self, warning_type=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        if warning_type:
            cursor.execute('UPDATE warnings SET is_read = 1 WHERE type = ?', (warning_type,))
        else:
            cursor.execute('UPDATE warnings SET is_read = 1')
        conn.commit()
    
    def delete_warning(self, warning_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM warnings WHERE id = ?', (warning_id,))
        conn.commit()
    
    def get_warning_count(self, warning_type=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if warning_type:
            cursor.execute('SELECT COUNT(*) FROM warnings WHERE type = ?', (warning_type,))
        else:
            cursor.execute('SELECT COUNT(*) FROM warnings')
        
        return cursor.fetchone()[0]
    
    def get_unread_count(self, warning_type=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if warning_type:
            cursor.execute('SELECT COUNT(*) FROM warnings WHERE type = ? AND is_read = 0', (warning_type,))
        else:
            cursor.execute('SELECT COUNT(*) FROM warnings WHERE is_read = 0')
        
        return cursor.fetchone()[0]
    
    def update_detection_status(self, detection_type, enabled=None, running=None, config=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if enabled is not None:
            cursor.execute('UPDATE detection_status SET enabled = ? WHERE detection_type = ?', 
                          (1 if enabled else 0, detection_type))
        
        if running is not None:
            cursor.execute('UPDATE detection_status SET running = ?, last_detection_time = ? WHERE detection_type = ?',
                          (1 if running else 0, datetime.now().isoformat() if running else None, detection_type))
        
        if config is not None:
            cursor.execute('UPDATE detection_status SET config = ? WHERE detection_type = ?',
                          (json.dumps(config), detection_type))
        
        conn.commit()
    
    def get_detection_status(self, detection_type=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if detection_type:
            cursor.execute('SELECT * FROM detection_status WHERE detection_type = ?', (detection_type,))
            row = cursor.fetchone()
            return dict(row) if row else None
        else:
            cursor.execute('SELECT * FROM detection_status')
            rows = cursor.fetchall()
            return {row['detection_type']: dict(row) for row in rows}
    
    def add_log(self, level, module, message):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (level, module, message)
            VALUES (?, ?, ?)
        ''', (level, module, message))
        conn.commit()
    
    def get_logs(self, limit=100, level=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if level:
            cursor.execute('SELECT * FROM system_logs WHERE level = ? ORDER BY created_at DESC LIMIT ?', 
                          (level, limit))
        else:
            cursor.execute('SELECT * FROM system_logs ORDER BY created_at DESC LIMIT ?', (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def update_daily_stats(self, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT type, COUNT(*) as count FROM warnings 
            WHERE date(created_at) = ? 
            GROUP BY type
        ''', (date,))
        
        stats = {row['type']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats 
            (date, fire_count, person_loitering_count, vehicle_loitering_count, shutter_change_count, fence_crossing_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            date,
            stats.get('fire', 0),
            stats.get('person_loitering', 0),
            stats.get('vehicle_loitering', 0),
            stats.get('shutter', 0),
            stats.get('fence_crossing', 0)
        ))
        
        conn.commit()
    
    def get_daily_stats(self, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (date,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_stats_range(self, start_date, end_date):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM daily_stats 
            WHERE date BETWEEN ? AND ? 
            ORDER BY date
        ''', (start_date, end_date))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def update_daily_ai_analysis(self, date, analysis):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE daily_stats SET ai_analysis = ? WHERE date = ?
        ''', (analysis, date))
        conn.commit()
    
    def add_camera(self, name, source):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO camera_config (name, source)
            VALUES (?, ?)
        ''', (name, source))
        conn.commit()
        return cursor.lastrowid
    
    def get_cameras(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM camera_config WHERE is_active = 1')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_camera_by_id(self, camera_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM camera_config WHERE id = ?', (camera_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def delete_camera(self, camera_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE camera_config SET is_active = 0 WHERE id = ?', (camera_id,))
        conn.commit()
    
    def update_camera_detectors(self, camera_id, detection_type, enabled=None, roi_points=None, config=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO camera_detectors (camera_id, detection_type, enabled, roi_points, config)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(camera_id, detection_type) DO UPDATE SET
                enabled = COALESCE(?, enabled),
                roi_points = COALESCE(?, roi_points),
                config = COALESCE(?, config)
        ''', (camera_id, detection_type, 
              1 if enabled else 0 if enabled is not None else 1,
              json.dumps(roi_points) if roi_points else None,
              json.dumps(config) if config else None,
              1 if enabled else 0 if enabled is not None else None,
              json.dumps(roi_points) if roi_points else None,
              json.dumps(config) if config else None))
        
        conn.commit()
    
    def get_camera_detectors(self, camera_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM camera_detectors WHERE camera_id = ?', (camera_id,))
        rows = cursor.fetchall()
        result = {}
        for row in rows:
            data = dict(row)
            if data['roi_points']:
                data['roi_points'] = json.loads(data['roi_points'])
            if data['config']:
                data['config'] = json.loads(data['config'])
            result[data['detection_type']] = data
        return result
    
    def add_roi_region(self, camera_id, name, detection_type, points):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO roi_regions (camera_id, name, detection_type, points)
            VALUES (?, ?, ?, ?)
        ''', (camera_id, name, detection_type, json.dumps(points)))
        conn.commit()
        return cursor.lastrowid
    
    def get_roi_regions(self, camera_id=None, detection_type=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM roi_regions WHERE is_active = 1'
        params = []
        
        if camera_id:
            query += ' AND camera_id = ?'
            params.append(camera_id)
        
        if detection_type:
            query += ' AND detection_type = ?'
            params.append(detection_type)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            data = dict(row)
            data['points'] = json.loads(data['points'])
            result.append(data)
        return result
    
    def delete_roi_region(self, region_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE roi_regions SET is_active = 0 WHERE id = ?', (region_id,))
        conn.commit()
    
    def create_offline_task(self, task_type, detection_types, file_path):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO offline_tasks (task_type, detection_types, file_path)
            VALUES (?, ?, ?)
        ''', (task_type, json.dumps(detection_types), file_path))
        conn.commit()
        return cursor.lastrowid
    
    def update_offline_task(self, task_id, status=None, progress=None, result=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute('UPDATE offline_tasks SET status = ? WHERE id = ?', (status, task_id))
        
        if progress is not None:
            cursor.execute('UPDATE offline_tasks SET progress = ? WHERE id = ?', (progress, task_id))
        
        if result:
            cursor.execute('UPDATE offline_tasks SET result = ? WHERE id = ?', (json.dumps(result), task_id))
        
        if status == 'completed':
            cursor.execute('UPDATE offline_tasks SET completed_at = ? WHERE id = ?', 
                          (datetime.now().isoformat(), task_id))
        
        conn.commit()
    
    def get_offline_task(self, task_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM offline_tasks WHERE id = ?', (task_id,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            if 'detection_types' in data and data['detection_types']:
                try:
                    data['detection_types'] = json.loads(data['detection_types'])
                except:
                    data['detection_types'] = []
            else:
                data['detection_types'] = []
            if data.get('result'):
                try:
                    data['result'] = json.loads(data['result'])
                except:
                    pass
            return data
        return None
    
    def get_offline_tasks(self, limit=50):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM offline_tasks ORDER BY created_at DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            data = dict(row)
            if 'detection_types' in data and data['detection_types']:
                try:
                    data['detection_types'] = json.loads(data['detection_types'])
                except:
                    data['detection_types'] = []
            else:
                data['detection_types'] = []
            if data.get('result'):
                try:
                    data['result'] = json.loads(data['result'])
                except:
                    pass
            result.append(data)
        return result
