import sqlite3
from pathlib import Path

DB_PATH = Path("database/cards.db")

def init_database():
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            individual_id TEXT PRIMARY KEY,
            species TEXT,                    -- Карелина или Гребенчатый
            project_name TEXT,
            photo_path TEXT,
            embedding_index INTEGER,         -- индекс в векторной базе
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Поля из всех карточек
            date TEXT,
            length_body REAL,
            length_tail REAL,
            weight REAL,
            sex TEXT,
            birth_year TEXT,
            water_body TEXT,
            notes TEXT,
            template_type TEXT               -- ИК-1, ИК-2, КВ-1, КВ-2
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ База карточек создана: {DB_PATH}")

# Первый запуск
if __name__ == "__main__":
    init_database()