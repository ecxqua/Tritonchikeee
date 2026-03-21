import sys
from pathlib import Path
from datetime import datetime
import sqlite3

sys.path.append(str(Path(__file__).parent.parent))
from database.card_database import DB_PATH, init_database

# === КОНФИГУРАЦИЯ ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ ===
REQUIRED_FIELDS = {
    "ИК-1": ["length_body", "weight", "sex"],
    "ИК-2": ["parent_male_id", "parent_female_id", "water_body_name", "release_date"],
    "КВ-1": ["status", "water_body_number", "length_body", "length_tail"],
    "КВ-2": ["status", "water_body_name"]
}


def _get_next_photo_number(cursor, individual_id: str) -> str:
    """
    Автоматически генерирует порядковый номер фото (01, 02, 03...)
    """
    base_id = individual_id.replace("NT-", "")
    cursor.execute(
        "SELECT COUNT(*) FROM photos WHERE individual_id LIKE ?",
        (f"NT-{base_id}%",)
    )
    count = cursor.fetchone()[0]
    return f"{count + 1:02d}"


def save_new_individual(
    embedding,
    photo_path_full: str = None,
    photo_path_cropped: str = None,
    species: str = "Карелина",
    project_name: str = "Основной",
    template_type: str = "ИК-1",
    individual_id: str = None,
    photo_number: str = None,
    is_legacy: bool = False,
    **card_data
):
    """
    Сохраняет новую особь в базу (карточка + все фотографии)
    """
    init_database() 
    _validate_template_fields(template_type, card_data)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if individual_id is None:
        individual_id = f"NT-{datetime.now().strftime('%y%m%d%H%M%S')}"
    
    if photo_number is None:
        photo_number = _get_next_photo_number(cursor, individual_id)
    
    try:
        # === 1. ТАБЛИЦА individuals (Карточка) ===
        # ← ИЗМЕНЕНО: Убрали photo_path, photo_number, embedding_index
        cursor.execute('''
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                created_at,
                date, notes,
                length_body, length_tail, length_total, weight, sex,
                birth_year_exact, birth_year_approx, origin_region,
                length_device, weight_device,
                parent_male_id, parent_female_id, release_date, water_body_name,
                meeting_time, status, water_body_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            individual_id, template_type, species, project_name,
            datetime.now().isoformat(),
            card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
            card_data.get('notes'),
            card_data.get('length_body'), card_data.get('length_tail'),
            card_data.get('length_total'), card_data.get('weight'),
            card_data.get('sex'), card_data.get('birth_year_exact'),
            card_data.get('birth_year_approx'), card_data.get('origin_region'),
            card_data.get('length_device'), card_data.get('weight_device'),
            card_data.get('parent_male_id'), card_data.get('parent_female_id'),
            card_data.get('release_date'), card_data.get('water_body_name'),
            card_data.get('meeting_time'), card_data.get('status'),
            card_data.get('water_body_number')
        ))
        
        # === 2. ТАБЛИЦА photos (Полное фото) ===
        if photo_path_full and not is_legacy:
            cursor.execute('''
                INSERT INTO photos (
                    individual_id, photo_type, photo_number, photo_path,
                    date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, 'full', photo_number, photo_path_full,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('meeting_time'), 1, 0, None, 0
            ))
        
        # === 3. ТАБЛИЦА photos (Кроп брюшка) ===
        if photo_path_cropped:
            cursor.execute('''
                INSERT INTO photos (
                    individual_id, photo_type, photo_number, photo_path,
                    date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, 'cropped', photo_number, photo_path_cropped,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('meeting_time'), 
                1 if not photo_path_full else 0,
                1, -1, 1 if is_legacy else 0
            ))
        
        conn.commit()
        
        # === 4. Обновление embedding_index (FAISS) ===
        if embedding is not None and photo_path_cropped:
            # Здесь будет вызов FAISS: index.add(embedding)
            # embedding_idx = index.add(embedding)
            # cursor.execute("UPDATE photos SET embedding_index = ? WHERE photo_path = ?", 
            #                (embedding_idx, photo_path_cropped))
            # conn.commit()
            pass
        
        conn.close()
        
        print(f"✅ Особь сохранена: {individual_id} ({template_type})")
        print(f"   Фото: {photo_number} | Legacy: {is_legacy}")
        return individual_id
        
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def _validate_template_fields(template_type: str, card_data: dict):  
    """
    Проверяет наличие обязательных полей для выбранного шаблона
    """
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [field for field in required if card_data.get(field) is None]
    
    if missing:
        raise ValueError(
            f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
            f"Переданные данные: {list(card_data.keys())}"
        )


def update_individual(individual_id: str, **kwargs):
    """Обновляет данные карточки особи"""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    fields = [f"{key} = ?" for key in kwargs.keys()]
    values = list(kwargs.values()) + [individual_id]

    if not fields:
        print("⚠️ Нет полей для обновления")
        return

    query = f"UPDATE individuals SET {', '.join(fields)} WHERE individual_id = ?"
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    print(f"✅ Особь {individual_id} обновлена.")


def add_encounter(
    individual_id: str,
    template_type: str,
    photo_path_full: str = None,
    photo_path_cropped: str = None,
    embedding=None,
    **card_data
):
    """
    Добавляет НОВУЮ ВСТРЕЧУ (КВ-1/КВ-2) для существующей особи.
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if template_type not in ["КВ-1", "КВ-2"]:
        raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")
    
    _validate_template_fields(template_type, card_data)
    photo_number = _get_next_photo_number(cursor, individual_id)
    
    try:
        # 1. Создаем запись о встрече в individuals
        # ← ИЗМЕНЕНО: Убрали photo_path, photo_number, embedding_index
        cursor.execute('''
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                date, meeting_time, status, water_body_number, water_body_name,
                length_body, length_tail, length_total, weight, sex, notes,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            individual_id, template_type, "Карелина", "Основной",
            card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
            card_data.get('time'), card_data.get('status'),
            card_data.get('water_body_number'), card_data.get('water_body_name'),
            card_data.get('length_body'), card_data.get('length_tail'),
            card_data.get('length_total'), card_data.get('weight'),
            card_data.get('sex'), card_data.get('notes'),
            datetime.now().isoformat()
        ))
        
        # 2. Сохраняем фото (полное)
        if photo_path_full:
            cursor.execute('''
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (individual_id, 'full', photo_number, photo_path_full, card_data.get('date'), 0, 0, None, 0))
        
        # 3. Сохраняем фото (кроп)
        if photo_path_cropped:
            cursor.execute('''
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (individual_id, 'cropped', photo_number, photo_path_cropped, card_data.get('date'), 0, 1, -1, 0))
        
        conn.commit()
        conn.close()
        print(f"✅ Встреча {template_type} добавлена к особи {individual_id}")
        return photo_number
        
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def get_individual_photos(individual_id: str):
    """Получает все фотографии особи"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT photo_id, photo_type, photo_number, photo_path, 
               date_taken, is_main, is_legacy
        FROM photos
        WHERE individual_id = ?
        ORDER BY photo_number ASC, photo_type DESC
    ''', (individual_id,))
    
    photos = cursor.fetchall()
    conn.close()
    return photos


# === ТЕСТЫ ===
if __name__ == "__main__":
    print("🧪 Тестирование сохранения (исправленная версия)...\n")
    
    # Тест 1: Новая особь (ИК-1)
    print("1️⃣ Тест ИК-1 (Новая особь)")
    try:
        save_new_individual(
            embedding=None,
            photo_path_full="data/photos/full/newt_001.jpg",
            photo_path_cropped="data/crop/newt_001.jpg",
            template_type="ИК-1",
            length_body=42.5, weight=3.2, sex="М",
            notes="Тестовая особь"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Тест 2: Legacy особь (только кроп)
    print("2️⃣ Тест Legacy (Только кроп из датасета)")
    try:
        save_new_individual(
            embedding=None,
            photo_path_full=None,
            photo_path_cropped="data/dataset_crop/karelin/47/IMG_001.jpg",
            template_type="ИК-1",
            individual_id="NT-47",
            is_legacy=True,
            length_body=40.0, weight=3.0, sex="Ж"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Тест 3: Повторная встреча (КВ-1)
    print("3️⃣ Тест КВ-1 (Повторная встреча)")
    try:
        add_encounter(
            individual_id="NT-47",
            template_type="КВ-1",
            photo_path_full="data/photos/full/newt_001_v2.jpg",
            photo_path_cropped="data/crop/newt_001_v2.jpg",
            status="жив",
            water_body_number="Пруд №3",
            length_body=45.0,
            date="20.06.2024"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")