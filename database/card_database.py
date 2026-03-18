import sqlite3
from pathlib import Path

DB_PATH = Path("database/cards.db")

def init_database():
    DB_PATH.parent.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS individuals (
            -- === СЛУЖЕБНЫЕ ПОЛЯ (для всех шаблонов) ===
            individual_id TEXT PRIMARY KEY,      -- 1. ID-номер особи
            template_type TEXT,                  -- ИК-1, ИК-2, КВ-1, КВ-2
            species TEXT,                        -- Карелина / Гребенчатый
            project_name TEXT,                   -- Название проекта
            photo_path TEXT,                     -- Путь к фото
            photo_number TEXT,                   -- 8/8. Номер фото индивидуального рисунка
            embedding_index INTEGER,             -- Индекс в векторной базе (FAISS)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- === ОБЩИЕ ПОЛЯ (есть в большинстве шаблонов) ===
            date TEXT,                           -- 2/2/2/2. Дата заполнения/встречи
            notes TEXT,                          -- 12/9/13/7. Примечания
            
            -- === БИОМЕТРИЯ (ИК-1, КВ-1) ===
            length_body REAL,                    -- 3/4. Длина тела (L), мм
            length_tail REAL,                    -- 3/5. Длина хвоста (Lcd), мм
            length_total REAL,                   -- 6/4. Общая длина (L + Lcd), см (для ИК-2, КВ-2)
            weight REAL,                         -- 4/7/6. Масса, г
            sex TEXT,                            -- 5/7. Пол (М/Ж)
            
            -- === РОЖДЕНИЕ И ПРОИСХОЖДЕНИЕ (ИК-1) ===
            birth_year_exact TEXT,               -- 6. Точный год рождения (дд.мм.гггг)
            birth_year_approx TEXT,              -- 7. Условный год рождения (дд.мм.гггг)
            origin_region TEXT,                  -- 9. Регион происхождения
            length_device TEXT,                  -- 10/11. Марка устройства для измерения длины
            weight_device TEXT,                  -- 11/12. Марка весов
            
            -- === РОДИТЕЛИ (ИК-2) ===
            parent_male_id TEXT,                 -- 4. ID самца (родитель)
            parent_female_id TEXT,               -- 5. ID самки (родитель)
            release_date TEXT,                   -- 3. Дата выпуска в водоем
            water_body_name TEXT,                -- 8/6. Название водоема (для ИК-2, КВ-2)
            
            -- === ВСТРЕЧА (КВ-1, КВ-2) ===
            meeting_time TEXT,                   -- 3/3. Время встречи (ч.мин.)
            status TEXT,                         -- 9/5. Статус (жив/мертв)
            water_body_number TEXT,              -- 10. Номер водоема (для КВ-1)
            
            -- === ОГРАНИЧЕНИЯ (ссылки на самих себя) ===
            FOREIGN KEY(parent_male_id) REFERENCES individuals(individual_id),
            FOREIGN KEY(parent_female_id) REFERENCES individuals(individual_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ База карточек создана: {DB_PATH}")

# Первый запуск
if __name__ == "__main__":
    init_database()