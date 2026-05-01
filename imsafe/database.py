"""SQLite 表结构初始化（桌面端与 Web 共用同一库文件）。"""
import sqlite3

from imsafe.paths import get_db_path


def connect_db(row_factory: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def init_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            rfid_card TEXT UNIQUE,
            permission_level INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS face_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            sample_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            area TEXT NOT NULL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            access_granted BOOLEAN,
            reason TEXT,
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            area TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            violation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT '未处理',
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
        """
    )
    cursor.execute("PRAGMA table_info(violations)")
    columns = [row[1] for row in cursor.fetchall()]
    if "screenshot_path" not in columns:
        try:
            cursor.execute("ALTER TABLE violations ADD COLUMN screenshot_path TEXT")
        except Exception as e:
            print(f"添加 screenshot_path 字段失败: {e}")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS salary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            month TEXT NOT NULL,
            working_hours REAL DEFAULT 0,
            daily_rate REAL DEFAULT 0,
            hourly_rate REAL DEFAULT 0,
            performance_factor REAL DEFAULT 1.0,
            total_salary REAL DEFAULT 0,
            status TEXT DEFAULT '未结算',
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS face_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            features TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
        """
    )


def row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}
