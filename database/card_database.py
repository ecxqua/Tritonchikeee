import sqlite3
from pathlib import Path

DB_PATH = Path("database/cards.db")

def init_database():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # === ТАБЛИЦА 1: individuals (Паспорт особи) ===
    # Одна запись на животное. НЕТ векторов и путей к фото.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            -- PRIMARY KEY
            individual_id TEXT PRIMARY KEY,      -- NT-K-47
            
            -- Вид и проект
            species TEXT,                        -- Карелина / Гребенчатый
            project_name TEXT,                   -- ООПТ_Исеть
            
            -- Происхождение (заполняется один раз)
            birth_year_exact TEXT,
            birth_year_approx TEXT,
            origin_region TEXT,
            
            -- Родители (ссылки на других особей)
            parent_male_id TEXT,
            parent_female_id TEXT,
            
            -- Метаданные
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            
            FOREIGN KEY(parent_male_id) REFERENCES individuals(individual_id),
            FOREIGN KEY(parent_female_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    # === ТАБЛИЦА 2: photos (Все фотографии + Векторы) ===
    # Множество записей на особь. ЗДЕСЬ хранится embedding_index.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            -- PRIMARY KEY
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Связь с особью
            individual_id TEXT,
            
            -- Типы фото
            photo_type TEXT,                     -- 'full' / 'cropped'
            
            -- Порядковый номер фото в серии (из ТЗ)
            photo_number TEXT,                   -- 01, 02, 03...
            
            -- Пути к файлам
            photo_path TEXT,
            
            -- Метаданные съёмки
            date_taken TEXT,
            time_taken TEXT,
            
            -- === ВЕКТОРНАЯ СВЯЗЬ (ТОЛЬКО ЗДЕСЬ) ===
            embedding_index INTEGER,             -- Позиция в FAISS
            
            -- Статус обработки
            is_main BOOLEAN DEFAULT 0,
            is_processed BOOLEAN DEFAULT 0,
            is_legacy BOOLEAN DEFAULT 0,
            
            -- Примечания к фото
            notes TEXT,
            
            FOREIGN KEY(individual_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    # Индексы для ускорения поиска
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_individual ON photos(individual_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_embedding ON photos(embedding_index)')
    
    conn.commit()
    conn.close()
    print(f"✅ База данных инициализирована: {DB_PATH}")

# Первый запуск
if __name__ == "__main__":
    init_database()