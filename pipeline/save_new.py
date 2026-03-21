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
    "КВ-2": ["status", "water_body_name"],
}


def get_next_animal_number(cursor, species: str) -> int:
    """
    Возвращает следующий порядковый номер для животного данного вида.
    Считает уникальные номера особей (без пропусков).
    """
    species_prefix = {
        "Карелина": "K",
        "Гребенчатый": "R",
        "Ребристый": "R"
    }.get(species, "X")
    
    # Считаем УНИКАЛЬНЫЕ номера особей этого вида
    # Например: NT-K-1, NT-K-2, NT-K-3 → вернёт 4
    cursor.execute('''
        SELECT COUNT(DISTINCT CAST(SUBSTR(individual_id, 5, INSTR(SUBSTR(individual_id, 5), '-') - 1) AS INTEGER))
        FROM individuals
        WHERE individual_id LIKE ?
    ''', (f"NT-{species_prefix}-%",))
    
    count = cursor.fetchone()[0]
    return (count or 0) + 1


def generate_card_id(cursor, species: str, template_type: str, animal_id: str = None) -> str:
    """
    Генерирует ID карточки.
    
    Если animal_id передан → использует его (повторная встреча).
    Если animal_id НЕ передан → создаёт новый номер особи (новая особь).
    
    Формат: NT-{Вид}-{Номер}-{Тип}
    Пример: NT-K-47-ИК1
    """
    species_prefix = {
        "Карелина": "K",
        "Гребенчатый": "R",
        "Ребристый": "R"
    }.get(species, "X")
    
    template_short = template_type.replace("-", "")
    
    # 1. Если ID особи не передан → генерируем новый номер (автоинкремент)
    if animal_id is None:
        new_num = get_next_animal_number(cursor, species)
        animal_id = f"NT-{species_prefix}-{new_num}"
    
    # 2. Формируем ID карточки
    card_id = f"{animal_id}-{template_short}"
    
    # 3. Проверяем, нет ли уже такой карточки
    cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (card_id,))
    if cursor.fetchone():
        print(f"⚠️ Карточка {card_id} уже существует. Возвращаем существующий ID.")
        return card_id
    
    return card_id

def _get_next_photo_number(cursor, individual_id: str) -> str:
    base_id = individual_id.replace("NT-", "")
    cursor.execute(
        "SELECT COUNT(*) FROM photos WHERE individual_id LIKE ?", (f"NT-{base_id}%",)
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
    **card_data,
):
    init_database()
    _validate_template_fields(template_type, card_data)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if individual_id is None:
        individual_id = f"NT-{datetime.now().strftime('%y%m%d%H%M%S')}"

    if photo_number is None:
        photo_number = _get_next_photo_number(cursor, individual_id)

    try:
        # === 1. ТАБЛИЦА individuals (24 колонки = 24 значения) ===
        cursor.execute(
            """
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                created_at,
                date, notes,
                length_body, length_tail, length_total, weight, sex,
                birth_year_exact, birth_year_approx, origin_region,
                length_device, weight_device,
                parent_male_id, parent_female_id, release_date, water_body_name,
                meeting_time, status, water_body_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                individual_id,  # 1
                template_type,  # 2
                species,  # 3
                project_name,  # 4
                datetime.now().isoformat(),  # 5
                card_data.get("date", datetime.now().strftime("%d.%m.%Y")),  # 6
                card_data.get("notes"),  # 7
                card_data.get("length_body"),  # 8
                card_data.get("length_tail"),  # 9
                card_data.get("length_total"),  # 10
                card_data.get("weight"),  # 11
                card_data.get("sex"),  # 12
                card_data.get("birth_year_exact"),  # 13
                card_data.get("birth_year_approx"),  # 14
                card_data.get("origin_region"),  # 15
                card_data.get("length_device"),  # 16
                card_data.get("weight_device"),  # 17
                card_data.get("parent_male_id"),  # 18
                card_data.get("parent_female_id"),  # 19
                card_data.get("release_date"),  # 20
                card_data.get("water_body_name"),  # 21
                card_data.get("meeting_time"),  # 22
                card_data.get("status"),  # 23
                card_data.get("water_body_number"),  # 24
            ),
        )

        # === 2. ТАБЛИЦА photos (Полное фото) ===
        if photo_path_full and not is_legacy:
            cursor.execute(
                """
                INSERT INTO photos (
                    individual_id, photo_type, photo_number, photo_path,
                    date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    individual_id,
                    "full",
                    photo_number,
                    photo_path_full,
                    card_data.get("date", datetime.now().strftime("%d.%m.%Y")),
                    card_data.get("meeting_time"),
                    1,
                    0,
                    None,
                    0,
                ),
            )

        # === 3. ТАБЛИЦА photos (Кроп брюшка) ===
        if photo_path_cropped:
            cursor.execute(
                """
                INSERT INTO photos (
                    individual_id, photo_type, photo_number, photo_path,
                    date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    individual_id,
                    "cropped",
                    photo_number,
                    photo_path_cropped,
                    card_data.get("date", datetime.now().strftime("%d.%m.%Y")),
                    card_data.get("meeting_time"),
                    1 if not photo_path_full else 0,
                    1,
                    -1,
                    1 if is_legacy else 0,
                ),
            )

        conn.commit()
        conn.close()

        print(f"✅ Особь сохранена: {individual_id} ({template_type})")
        print(f"   Фото: {photo_number} | Legacy: {is_legacy}")
        return individual_id

    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def _validate_template_fields(template_type: str, card_data: dict):
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [field for field in required if card_data.get(field) is None]

    if missing:
        raise ValueError(
            f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
            f"Переданные данные: {list(card_data.keys())}"
        )


def update_individual(individual_id: str, **kwargs):
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
    **card_data,
):
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if template_type not in ["КВ-1", "КВ-2"]:
        raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")

    _validate_template_fields(template_type, card_data)
    photo_number = _get_next_photo_number(cursor, individual_id)

    try:
        # 1. Создаем запись о встрече в individuals (16 колонок = 16 значений)
        cursor.execute(
            """
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                date, meeting_time, status, water_body_number, water_body_name,
                length_body, length_tail, length_total, weight, sex, notes,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                individual_id,  # 1
                template_type,  # 2
                "Карелина",  # 3
                "Основной",  # 4
                card_data.get("date", datetime.now().strftime("%d.%m.%Y")),  # 5
                card_data.get("time"),  # 6
                card_data.get("status"),  # 7
                card_data.get("water_body_number"),  # 8
                card_data.get("water_body_name"),  # 9
                card_data.get("length_body"),  # 10
                card_data.get("length_tail"),  # 11
                card_data.get("length_total"),  # 12
                card_data.get("weight"),  # 13
                card_data.get("sex"),  # 14
                card_data.get("notes"),  # 15
                datetime.now().isoformat(),  # 16
            ),
        )

        # 2. Сохраняем фото (полное)
        if photo_path_full:
            cursor.execute(
                """
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    individual_id,
                    "full",
                    photo_number,
                    photo_path_full,
                    card_data.get("date"),
                    0,
                    0,
                    None,
                    0,
                ),
            )

        # 3. Сохраняем фото (кроп)
        if photo_path_cropped:
            cursor.execute(
                """
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    individual_id,
                    "cropped",
                    photo_number,
                    photo_path_cropped,
                    card_data.get("date"),
                    0,
                    1,
                    -1,
                    0,
                ),
            )

        conn.commit()
        conn.close()
        print(f"✅ Встреча {template_type} добавлена к особи {individual_id}")
        return photo_number

    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def get_individual_photos(individual_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT photo_id, photo_type, photo_number, photo_path, 
               date_taken, is_main, is_legacy
        FROM photos
        WHERE individual_id = ?
        ORDER BY photo_number ASC, photo_type DESC
    """,
        (individual_id,),
    )

    photos = cursor.fetchall()
    conn.close()
    return photos


# === ТЕСТЫ ===
if __name__ == "__main__":
    print("🧪 Тестирование сохранения с авто-нумерацией...\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # === 1. Первая особь Карелина ===
    print("1️⃣ Первая особь Карелина (ИК-1)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
        # → NT-K-1-ИК1
        
        save_new_individual(
            embedding=None,
            photo_path_full="data/photos/full/test_001.jpg",
            photo_path_cropped="data/crop/test_001.jpg",
            species="Карелина",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=42.5,
            length_tail=38.0,
            weight=3.2,
            sex="М",
            notes="Первичная регистрация"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === 2. Вторая особь Карелина ===
    print("2️⃣ Вторая особь Карелина (ИК-1)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
        # → NT-K-2-ИК1
        
        save_new_individual(
            embedding=None,
            photo_path_full="data/photos/full/test_002.jpg",
            photo_path_cropped="data/crop/test_002.jpg",
            species="Карелина",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=45.0,
            length_tail=40.0,
            weight=3.5,
            sex="Ж",
            notes="Вторая особь"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === 3. Первая особь Гребенчатый ===
    print("3️⃣ Первая особь Гребенчатый (ИК-1)")
    try:
        card_id = generate_card_id(cursor, species="Гребенчатый", template_type="ИК-1")
        # → NT-R-1-ИК1
        
        save_new_individual(
            embedding=None,
            photo_path_full="data/photos/full/test_003.jpg",
            photo_path_cropped="data/crop/test_003.jpg",
            species="Гребенчатый",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=50.0,
            length_tail=45.0,
            weight=4.5,
            sex="М",
            notes="Гребенчатый тритон"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === 4. Повторная встреча для NT-K-1 ===
    print("4️⃣ Повторная встреча для NT-K-1 (КВ-1)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="КВ-1", animal_id="NT-K-1")
        # → NT-K-1-КВ1
        
        add_encounter(
            individual_id=card_id,
            template_type="КВ-1",
            photo_path_full="data/photos/full/test_001_v2.jpg",
            photo_path_cropped="data/crop/test_001_v2.jpg",
            status="жив",
            water_body_number="Пруд №3",
            length_body=44.0,
            length_tail=39.0,
            weight=3.4,
            sex="М",
            date="20.06.2024",
            notes="Повторная поимка"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === 5. Проверка: попытка создать дубликат ИК-1 для NT-K-1 ===
    print("5️⃣ Проверка дубля: ИК-1 для NT-K-1 (должен вернуть тот же ID)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1", animal_id="NT-K-1")
        # → NT-K-1-ИК1 (уже существует)
        print(f"   ✅ Вернут существующий ID: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*50 + "\n")
    print("📊 Тесты завершены! Проверьте базу данных.")
