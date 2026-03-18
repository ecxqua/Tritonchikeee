import sys
from pathlib import Path
from datetime import datetime
import sqlite3

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from database.card_database import DB_PATH, init_database

def save_new_individual(
    embedding,                    # numpy-вектор от ViT (обязательно)
    photo_path: str,              # путь к оригинальному фото
    species: str = "Карелина",    # "Карелина" или "Гребенчатый"
    project_name: str = "Основной",
    template_type: str = "ИК-1",   # ИК-1, ИК-2, КВ-1, КВ-2
    **card_data                   # сюда передаём все поля карточки
):
    """
    Сохраняет новую особь в базу (карточка + эмбеддинг)
    """
    init_database() 

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    individual_id = f"NT-{datetime.now().strftime('%y%m%d%H%M')}"

    cursor.execute('''
        INSERT INTO individuals (
            individual_id, species, project_name, photo_path, embedding_index,
            created_at, template_type, date, length_body, length_tail, 
            weight, sex, birth_year, water_body, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        individual_id,
        species,
        project_name,
        photo_path,
        -1,                                 # индекс в FAISS добавим позже
        datetime.now().isoformat(),
        template_type,
        card_data.get('date'),
        card_data.get('length_body'),
        card_data.get('length_tail'),
        card_data.get('weight'),
        card_data.get('sex'),
        card_data.get('birth_year'),
        card_data.get('water_body'),
        card_data.get('notes')
    ))

    conn.commit()
    conn.close()

    print(f"✅ Новая особь успешно сохранена!")
    print(f"   ID: {individual_id}")
    print(f"   Вид: {species}")
    print(f"   Шаблон: {template_type}")
    return individual_id


# === Тест функции ===
if __name__ == "__main__":
    print("Тестируем сохранение новой особи...\n")
    
    # Пример вызова
    save_new_individual(
        embedding=None,                    # пока заглушка
        photo_path="data/input/test.jpg",
        species="Карелина",
        template_type="ИК-1",
        date="2026-03-17",
        length_body=14.2,
        weight=52.0,
        sex="М",
        notes="Первый тритон в новом проекте"
    )