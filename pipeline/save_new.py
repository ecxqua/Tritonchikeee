import sys
from pathlib import Path
from datetime import datetime
import sqlite3

# Добавляем корень проекта в путь для импортов
sys.path.append(str(Path(__file__).parent.parent))
from database.card_database import DB_PATH, init_database

# ============================================================================
# КОНФИГУРАЦИЯ ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ
# Для каждого шаблона карточки указаны обязательные поля для валидации
# ============================================================================
REQUIRED_FIELDS = {
    "ИК-1": ["length_body", "weight", "sex"],
    "ИК-2": ["parent_male_id", "parent_female_id", "water_body_name", "release_date"],
    "КВ-1": ["status", "water_body_number", "length_body", "length_tail"],
    "КВ-2": ["status", "water_body_name"]
}


# ============================================================================
# ГЕНЕРАЦИЯ ID ЖИВОТНОГО (автоинкремент по виду)
# ============================================================================
def get_next_animal_number(cursor, species: str) -> int:
    """
    Возвращает следующий порядковый номер для животного данного вида.
    Считает уникальные номера особей (без пропусков).
    
    Args:
        cursor: Курсор SQLite
        species: Вид тритона ("Карелина" / "Гребенчатый")
    
    Returns:
        int: Следующий номер (например, 47 → 48)
    """
    # Маппинг видов на префиксы
    species_prefix = {
        "Карелина": "K",
        "Гребенчатый": "R",
        "Ребристый": "R"
    }.get(species, "X")
    
    # Считаем УНИКАЛЬНЫЕ номера особей этого вида
    # Например: NT-K-1, NT-K-2, NT-K-3 → вернёт 4
    cursor.execute('''
        SELECT COUNT(DISTINCT CAST(
            SUBSTR(individual_id, 5, INSTR(SUBSTR(individual_id, 5), '-') - 1) 
            AS INTEGER)
        )
        FROM individuals
        WHERE individual_id LIKE ?
    ''', (f"NT-{species_prefix}-%",))
    
    count = cursor.fetchone()[0]
    return (count or 0) + 1


# ============================================================================
# ГЕНЕРАЦИЯ ID КАРТОЧКИ
# ============================================================================
def generate_card_id(cursor, species: str, template_type: str, animal_id: str = None) -> str:
    """
    Генерирует ID карточки.
    
    Args:
        cursor: Курсор SQLite
        species: Вид тритона
        template_type: Тип карточки (ИК-1, ИК-2, КВ-1, КВ-2)
        animal_id: ID животного (если None → создаётся новый)
    
    Returns:
        str: ID карточки (например, "NT-K-47-ИК1")
    """
    species_prefix = {
        "Карелина": "K",
        "Гребенчатый": "R",
        "Ребристый": "R"
    }.get(species, "X")
    
    # Убираем дефис из типа шаблона (ИК-1 → ИК1)
    template_short = template_type.replace("-", "")
    
    # 1. Если ID особи не передан → генерируем новый номер (автоинкремент)
    if animal_id is None:
        new_num = get_next_animal_number(cursor, species)
        animal_id = f"NT-{species_prefix}-{new_num}"
    
    # 2. Формируем ID карточки
    card_id = f"{animal_id}-{template_short}"
    
    # 3. Проверяем, нет ли уже такой карточки (защита от дублей)
    cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (card_id,))
    if cursor.fetchone():
        print(f"⚠️ Карточка {card_id} уже существует. Возвращаем существующий ID.")
        return card_id
    
    return card_id


# ============================================================================
# ГЕНЕРАЦИЯ НОМЕРА ФОТО
# ============================================================================
def _get_next_photo_number(cursor, individual_id: str) -> str:
    """
    Автоматически генерирует порядковый номер фото (01, 02, 03...).
    
    ✅ ИСПРАВЛЕНО: Считает фото ТОЛЬКО для этой карточки (точное совпадение).
    Каждая карточка начинает нумерацию с 01.
    
    Args:
        cursor: Курсор SQLite
        individual_id: ID карточки (например, "NT-K-47-ИК1")
    
    Returns:
        str: Номер фото с ведущим нулём ("01", "02", ...)
    """
    # ← ИСПРАВЛЕНО: Точное совпадение (=), а не поиск по подстроке (LIKE)
    cursor.execute(
        "SELECT COUNT(*) FROM photos WHERE individual_id = ?",
        (individual_id,)
    )
    count = cursor.fetchone()[0]
    return f"{count + 1:02d}"


# ============================================================================
# СОХРАНЕНИЕ НОВОЙ ОСОБИ (основная функция)
# ============================================================================
def save_new_individual(
    embedding,                    # Вектор от ViT (512 чисел)
    photo_path_full: str = None,  # Путь к полному фото
    photo_path_cropped: str = None,  # Путь к кропу брюшка
    species: str = "Карелина",
    project_name: str = "Основной",
    template_type: str = "ИК-1",
    individual_id: str = None,
    photo_number: str = None,
    is_legacy: bool = False,      # Флаг для старых данных из датасета
    **card_data                   # Остальные поля карточки (length_body, weight...)
):
    """
    Сохраняет новую особь в базу данных (карточка + фотографии).
    
    Args:
        embedding: Вектор от ViT модели
        photo_path_full: Путь к полному фото (оригинал)
        photo_path_cropped: Путь к обрезанному брюшку (для ViT)
        species: Вид тритона
        project_name: Название проекта
        template_type: Тип карточки (ИК-1, ИК-2, КВ-1, КВ-2)
        individual_id: ID карточки (генерируется автоматически если None)
        photo_number: Номер фото (генерируется автоматически если None)
        is_legacy: Флаг "из старого датасета" (нет полного фото)
        **card_data: Дополнительные поля (length_body, weight, sex...)
    
    Returns:
        str: individual_id сохранённой карточки
    """
    # 1. Инициализация БД (создаёт таблицы если нет)
    init_database()
    
    # 2. Валидация обязательных полей для шаблона
    _validate_template_fields(template_type, card_data)
    
    # 3. Подключение к БД
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 4. Генерация ID карточки если не передан
    if individual_id is None:
        individual_id = f"NT-{datetime.now().strftime('%y%m%d%H%M%S')}"
    
    # 5. Генерация номера фото если не передан
    if photo_number is None:
        photo_number = _get_next_photo_number(cursor, individual_id)
    
    try:
        # === ТАБЛИЦА 1: individuals (карточка особи) ===
        # 24 колонки = 24 значения
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            individual_id,                    # 1
            template_type,                    # 2
            species,                          # 3
            project_name,                     # 4
            datetime.now().isoformat(),       # 5 created_at
            card_data.get('date', datetime.now().strftime("%d.%m.%Y")),  # 6
            card_data.get('notes'),           # 7
            card_data.get('length_body'),     # 8
            card_data.get('length_tail'),     # 9
            card_data.get('length_total'),    # 10
            card_data.get('weight'),          # 11
            card_data.get('sex'),             # 12
            card_data.get('birth_year_exact'), # 13
            card_data.get('birth_year_approx'), # 14
            card_data.get('origin_region'),   # 15
            card_data.get('length_device'),   # 16
            card_data.get('weight_device'),   # 17
            card_data.get('parent_male_id'),  # 18
            card_data.get('parent_female_id'), # 19
            card_data.get('release_date'),    # 20
            card_data.get('water_body_name'), # 21
            card_data.get('meeting_time'),    # 22
            card_data.get('status'),          # 23
            card_data.get('water_body_number') # 24
        ))
        
        # === ТАБЛИЦА 2: photos (полное фото) ===
        # Сохраняем ТОЛЬКО если есть полное фото (не legacy)
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
        
        # === ТАБЛИЦА 2: photos (кроп брюшка) ===
        # Сохраняем всегда, если кроп есть (нужен для ViT)
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
                1 if not photo_path_full else 0,  # is_main=1 если нет полного
                1, -1, 1 if is_legacy else 0      # embedding_index=-1 (заглушка)
            ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Особь сохранена: {individual_id} ({template_type})")
        print(f"   Фото: {photo_number} | Legacy: {is_legacy}")
        return individual_id
        
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


# ============================================================================
# ВАЛИДАЦИЯ ПОЛЕЙ ШАБЛОНА
# ============================================================================
def _validate_template_fields(template_type: str, card_data: dict):
    """
    Проверяет наличие обязательных полей для выбранного шаблона.
    
    Args:
        template_type: Тип карточки (ИК-1, ИК-2, КВ-1, КВ-2)
        card_data: Словарь с данными карточки
    
    Raises:
        ValueError: Если отсутствуют обязательные поля
    """
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [field for field in required if card_data.get(field) is None]
    
    if missing:
        raise ValueError(
            f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
            f"Переданные данные: {list(card_data.keys())}"
        )

def delete_individual(individual_id: str, delete_photos: bool = True, confirm: bool = False):
    """
    Полностью удаляет особь и все её фото (hard delete).
    
    ⚠️ ВНИМАНИЕ: Это действие необратимо!
    ⚠️ Используйте ТОЛЬКО для тестов или исправления ошибок ввода.
    
    Args:
        individual_id: ID карточки для удаления
        delete_photos: Если True → удаляем фото из БД и с диска
        confirm: Если True → требует подтверждения
    """
    if not confirm:
        raise ValueError(
            f"⚠️ ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ!\n"
            f"Вы уверены, что хотите удалить {individual_id}?\n"
            f"Передайте confirm=True для подтверждения."
        )
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Проверяем, существует ли особь
        cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (individual_id,))
        if not cursor.fetchone():
            raise ValueError(f"Особь {individual_id} не найдена в базе.")
        
        # 2. Получаем пути к фото перед удалением
        photo_paths = []
        if delete_photos:
            cursor.execute('SELECT photo_path FROM photos WHERE individual_id = ?', (individual_id,))
            photo_paths = [row[0] for row in cursor.fetchall()]
        
        # 3. Удаляем фото из БД
        if delete_photos:
            cursor.execute('DELETE FROM photos WHERE individual_id = ?', (individual_id,))
        
        # 4. Удаляем карточку
        cursor.execute('DELETE FROM individuals WHERE individual_id = ?', (individual_id,))
        
        conn.commit()
        conn.close()
        
        # 5. Удаляем файлы с диска
        if delete_photos:
            for photo_path in photo_paths:
                try:
                    Path(photo_path).unlink()
                    print(f"   🗑 Удалён файл: {photo_path}")
                except FileNotFoundError:
                    pass
        
        print(f"✅ Особь {individual_id} удалена (hard delete).")
        
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e

# ============================================================================
# ОБНОВЛЕНИЕ ДАННЫХ ОСОБИ
# ============================================================================
def update_individual(individual_id: str, **kwargs):
    """
    Обновляет данные существующей особи (редактирование карточки).
    
    Args:
        individual_id: ID карточки для обновления
        **kwargs: Поля для обновления (например, weight=3.5, status="жив")
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Формируем динамический запрос только для переданных полей
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


# ============================================================================
# ДОБАВЛЕНИЕ ПОВТОРНОЙ ВСТРЕЧИ
# ============================================================================
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
    
    Args:
        individual_id: ID животного (например, "NT-K-47")
        template_type: Тип встречи (КВ-1 или КВ-2)
        photo_path_full: Путь к полному фото
        photo_path_cropped: Путь к кропу брюшка
        embedding: Вектор от ViT
        **card_data: Данные встречи (status, length_body...)
    
    Returns:
        str: Номер фото
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Проверка шаблона (только встречи)
    if template_type not in ["КВ-1", "КВ-2"]:
        raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")
    
    # Валидация полей встречи
    _validate_template_fields(template_type, card_data)
    
    # Генерация номера фото
    photo_number = _get_next_photo_number(cursor, individual_id)
    
    try:
        # 1. Создаем запись о встрече в individuals (16 колонок = 16 значений)
        cursor.execute('''
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                date, meeting_time, status, water_body_number, water_body_name,
                length_body, length_tail, length_total, weight, sex, notes,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            individual_id,                    # 1
            template_type,                    # 2
            "Карелина",                       # 3
            "Основной",                       # 4
            card_data.get('date', datetime.now().strftime("%d.%m.%Y")),  # 5
            card_data.get('time'),            # 6
            card_data.get('status'),          # 7
            card_data.get('water_body_number'), # 8
            card_data.get('water_body_name'), # 9
            card_data.get('length_body'),     # 10
            card_data.get('length_tail'),     # 11
            card_data.get('length_total'),    # 12
            card_data.get('weight'),          # 13
            card_data.get('sex'),             # 14
            card_data.get('notes'),           # 15
            datetime.now().isoformat()        # 16
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


# ============================================================================
# ПОЛУЧЕНИЕ ВСЕХ ФОТО ОСОБИ
# ============================================================================
def get_individual_photos(individual_id: str):
    """
    Получает все фотографии особи из базы данных.
    
    Args:
        individual_id: ID карточки
    
    Returns:
        list: Список кортежей с данными фото
    """
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


# ============================================================================
# ТЕСТЫ
# ============================================================================
if __name__ == "__main__":
    print("🧪 Тестирование сохранения с авто-нумерацией...\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # === 1. Первая особь Карелина ===
    print("1️⃣ Первая особь Карелина (ИК-1)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
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
    
    # === 5. Проверка дубля ===
    print("5️⃣ Проверка дубля: ИК-1 для NT-K-1 (должен вернуть тот же ID)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1", animal_id="NT-K-1")
        print(f"   ✅ Вернут существующий ID: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*50 + "\n")
    print("📊 Тесты завершены! Проверьте базу данных.")