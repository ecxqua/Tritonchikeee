import sqlite3
from pathlib import Path

DB_PATH = Path("database/cards.db")

def init_database():
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # === ТАБЛИЦА 1: individuals (ваша текущая структура) ===
    # Одна запись = одна карточка (ИК-1, ИК-2, КВ-1, КВ-2)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            -- === СЛУЖЕБНЫЕ ПОЛЯ (для всех шаблонов) ===
            individual_id TEXT PRIMARY KEY,      -- 1. ID-номер особи
            template_type TEXT,                  -- ИК-1, ИК-2, КВ-1, КВ-2
            species TEXT,                        -- Карелина / Гребенчатый
            project_name TEXT,                   -- Название проекта
            photo_path TEXT,                     -- Путь к основному фото
            photo_number TEXT,                   -- 8/8. Номер фото индивидуального рисунка
            embedding_index INTEGER,             -- Индекс в векторной базе (FAISS)
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
    
    # === ТАБЛИЦА 2: photos (все фотографии) ===
    # Множество записей на одну карточку (multiple photos per card)
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
    
    # === ИНДЕКСЫ для ускорения поиска ===
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_photos_individual 
        ON photos(individual_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_photos_type 
        ON photos(photo_type)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_individuals_template 
        ON individuals(template_type)
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ База карточек создана: {DB_PATH}")

# Первый запуск
if __name__ == "__main__":
    init_database()