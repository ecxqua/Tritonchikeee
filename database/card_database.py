"""
database/cards_database.py — Инициализация SQLite базы данных.

Таблицы:
    1. individuals — карточки особей (ИК-1, ИК-2, КВ-1, КВ-2)
    2. photos — фотографии особей (full, cropped)
    3. uploads — временные загрузки (Two-Phase Commit) ⭐ НОВОЕ

Индексы:
    - Оптимизированы для поиска по FAISS embedding_index
    - Оптимизированы для очистки просроченных загрузок
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
    
    # =============================================================================
    # ТАБЛИЦА 1: individuals (карточки особей)
    # =============================================================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            -- === СЛУЖЕБНЫЕ ПОЛЯ (для всех шаблонов) ===
            individual_id TEXT PRIMARY KEY,      -- 1. ID-номер особи
            template_type TEXT,                  -- ИК-1, ИК-2, КВ-1, КВ-2
            species TEXT,                        -- Карелина / Гребенчатый
            project_name TEXT,                   -- Название проекта
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- === ОБЩИЕ ПОЛЯ (есть в большинстве шаблонов) ===
            date TEXT,                           -- 2/2/2/2. Дата заполнения/встречи
            notes TEXT,                          -- 12/9/13/7. Примечания
            
            -- === БИОМЕТРИЯ (ИК-1, КВ-1) ===
            length_body REAL,                    -- 3/4. Длина тела (L), мм
            length_tail REAL,                    -- 3/5. Длина хвоста (Lcd), мм
            length_total REAL,                   -- 6/4. Общая длина (L + Lcd), см
            weight REAL,                         -- 4/7/6. Масса, г
            sex TEXT,                            -- 5/7. Пол (М/Ж)
            
            -- === РОЖДЕНИЕ И ПРОИСХОЖДЕНИЕ (ИК-1) ===
            birth_year_exact TEXT,               -- 6. Точный год рождения
            birth_year_approx TEXT,              -- 7. Условный год рождения
            origin_region TEXT,                  -- 9. Регион происхождения
            length_device TEXT,                  -- 10/11. Марка устройства
            weight_device TEXT,                  -- 11/12. Марка весов
            
            -- === РОДИТЕЛИ (ИК-2) ===
            parent_male_id TEXT,                 -- 4. ID самца (родитель)
            parent_female_id TEXT,               -- 5. ID самки (родитель)
            release_date TEXT,                   -- 3. Дата выпуска в водоем
            water_body_name TEXT,                -- 8/6. Название водоема
            
            -- === ВСТРЕЧА (КВ-1, КВ-2) ===
            meeting_time TEXT,                   -- 3/3. Время встречи
            status TEXT,                         -- 9/5. Статус (жив/мертв)
            water_body_number TEXT,              -- 10. Номер водоема
            
            -- === ОГРАНИЧЕНИЯ ===
            FOREIGN KEY(parent_male_id) REFERENCES individuals(individual_id),
            FOREIGN KEY(parent_female_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    # =============================================================================
    # ТАБЛИЦА 2: photos (все фотографии)
    # =============================================================================
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            -- PRIMARY KEY
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Связь с карточкой особи
            individual_id TEXT,                  -- Ссылка на individuals.individual_id
            
            -- Типы фото
            photo_type TEXT,                     -- 'full' (полное) / 'cropped' (брюшко)
            
            -- Порядковый номер фото в серии (из ТЗ)
            photo_number TEXT,                   -- 01, 02, 03...
            
            -- Пути к файлам
            photo_path TEXT,                     -- Полный путь к файлу
            
            -- Метаданные съёмки
            date_taken TEXT,                     -- Дата съёмки (дд.мм.гггг)
            time_taken TEXT,                     -- Время съёмки (чч:мм)
            
            -- FAISS эмбеддинг (только для cropped)
            embedding_index INTEGER,             -- Позиция в векторной базе
            
            -- Статус обработки
            is_main BOOLEAN DEFAULT 0,           -- Основное фото карточки (1/0)
            is_processed BOOLEAN DEFAULT 0,      -- Обработано YOLO + ViT
            
            is_legacy BOOLEAN DEFAULT 0,         -- 1 = фото из старого датасета (нет оригинала)      

            -- Примечания к фото
            notes TEXT,
            
            -- Ссылка на individuals
            FOREIGN KEY(individual_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    # =============================================================================
    # ТАБЛИЦА 3: uploads (временные загрузки) ⭐ НОВОЕ
    # =============================================================================
    # Для Two-Phase Commit: анализ → подтверждение
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,         -- Для изоляции проектов
            file_path TEXT NOT NULL,             -- Путь к кропу брюшка
            embedding TEXT NOT NULL,             -- JSON строка: "[0.1, 0.2, ...]"
            status TEXT DEFAULT 'pending',       -- pending, completed, cancelled
            card_id TEXT,                        -- Ссылка на созданную карточку (после confirm)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL        -- created_at + 24h (для автоочистки)
        )
    ''')
    
    # =============================================================================
    # ИНДЕКСЫ
    # =============================================================================
    
    # --- individuals ---
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_individuals_template 
        ON individuals(template_type)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_individuals_project 
        ON individuals(project_name)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_individuals_species 
        ON individuals(species)
    ''')
    
    # --- photos ---
    # 🔥 КРИТИЧЕСКИЙ: Поиск фото по вектору из FAISS
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_photos_embedding 
        ON photos(embedding_index)
    ''')

    # 🔥 КРИТИЧЕСКИЙ: Связь фото → особь
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_photos_individual 
        ON photos(individual_id)
    ''')

    # 🟡 ПОЛЕЗНЫЙ: Фильтрация полных/кроп фото
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_photos_type 
        ON photos(photo_type)
    ''')
    
    # --- uploads ⭐ НОВЫЕ ИНДЕКСЫ ---
    # 🔥 КРИТИЧЕСКИЙ: Быстрая очистка просроченных загрузок
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_uploads_status_expires 
        ON uploads(status, expires_at)
    ''')
    
    # 🔥 КРИТИЧЕСКИЙ: Поиск загрузок по проекту
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_uploads_project 
        ON uploads(project_id)
    ''')
    
    # 🟡 ПОЛЕЗНЫЙ: Поиск по статусу (pending/completed/cancelled)
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_uploads_status 
        ON uploads(status)
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ База карточек создана/обновлена: {DB_PATH}")

# Первый запуск
if __name__ == "__main__":
    init_database()