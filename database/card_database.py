"""
database/card_database.py — Инициализация SQLite базы данных.

Таблицы:
    1. projects — метаданные проектов (PK id)
    2. individuals — карточки особей (FK project_id)
    3. photos — фотографии особей (FK individual_id)
    4. uploads — временные загрузки (FK project_id)

Архитектурные принципы:
    - ✅ ТОЛЬКО создание таблиц и индексов
    - ✅ НЕТ CRUD функций (это в card_service.py)
    - ✅ НЕТ утилит (get_project_by_id и т.д. — это в card_service.py)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("database/cards.db")

def init_database():
    """
    Инициализировать базу данных (создать таблицы и индексы).
    Идемпотентно: можно вызывать многократно без побочных эффектов.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # -------------------------------------------------------------------------
    # ТАБЛИЦА 1: projects (метаданные проектов)
    # -------------------------------------------------------------------------
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            species_filter TEXT,
            territory_filter TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # -------------------------------------------------------------------------
    # ТАБЛИЦА 2: individuals (карточки особей) — FK project_id
    # -------------------------------------------------------------------------
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            individual_id TEXT PRIMARY KEY,
            template_type TEXT,
            species TEXT,
            project_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            notes TEXT,
            length_body REAL,
            length_tail REAL,
            length_total REAL,
            weight REAL,
            sex TEXT,
            birth_year_exact TEXT,
            birth_year_approx TEXT,
            origin_region TEXT,
            length_device TEXT,
            weight_device TEXT,
            parent_male_id TEXT,
            parent_female_id TEXT,
            release_date TEXT,
            water_body_name TEXT,
            meeting_time TEXT,
            status TEXT,
            water_body_number TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_male_id) REFERENCES individuals(individual_id),
            FOREIGN KEY(parent_female_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    # -------------------------------------------------------------------------
    # ТАБЛИЦА 3: photos (фотографии)
    # -------------------------------------------------------------------------
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id TEXT,
            photo_type TEXT,
            photo_number TEXT,
            photo_path TEXT,
            date_taken TEXT,
            time_taken TEXT,
            embedding_index INTEGER,
            is_main BOOLEAN DEFAULT 0,
            is_processed BOOLEAN DEFAULT 0,
            is_legacy BOOLEAN DEFAULT 0,
            notes TEXT,
            FOREIGN KEY(individual_id) REFERENCES individuals(individual_id) ON DELETE CASCADE
        )
    ''')
    
    # -------------------------------------------------------------------------
    # ТАБЛИЦА 4: uploads (временные загрузки) — FK project_id
    # -------------------------------------------------------------------------
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            embedding TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            card_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        )
    ''')
    
    # -------------------------------------------------------------------------
    # ИНДЕКСЫ
    # -------------------------------------------------------------------------
    
    # --- projects ---
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(is_active)')
    
    # --- individuals ---
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_individuals_project ON individuals(project_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_individuals_template ON individuals(template_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_individuals_species ON individuals(species)')
    
    # --- photos ---
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_embedding ON photos(embedding_index)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_individual ON photos(individual_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_type ON photos(photo_type)')
    
    # --- uploads ---
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_uploads_status_expires ON uploads(status, expires_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_uploads_project ON uploads(project_id)')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("🦎 Инициализация базы данных...")
    init_database()
    print(f"✅ База создана: {DB_PATH}")