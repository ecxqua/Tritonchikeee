"""
⚠️ Проблемы
- Если embedding = None → особь сохраняется в БД, но не добавляется в FAISS (embedding_index = -1)
- Если FAISS индекс не существует → создаётся новый автоматически
- Ошибки FAISS не прерывают сохранение → БД сохранится даже если FAISS упадёт
- Транзакции → FAISS сохраняется после commit БД (возмоен рассинхрон)
"""


import sys
from pathlib import Path
from datetime import datetime
import sqlite3
import faiss  # 🔥 ДОБАВЛЕНО
import numpy as np  # 🔥 ДОБАВЛЕНО
from typing import Optional  # 🔥 ДОБАВЛЕНО

# Добавляем корень проекта в путь для импортов
sys.path.append(str(Path(__file__).parent.parent))
from database.card_database import DB_PATH, init_database

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
FAISS_INDEX_PATH = Path("data/embeddings/database_embeddings.pkl")
EMBEDDING_DIM = 512  # Размер вектора ViT

REQUIRED_FIELDS = {
    "ИК-1": ["length_body", "weight", "sex"],
    "ИК-2": ["parent_male_id", "parent_female_id", "water_body_name", "release_date"],
    "КВ-1": ["status", "water_body_number", "length_body", "length_tail"],
    "КВ-2": ["status", "water_body_name"]
}


# ============================================================================
# 🔥 НОВАЯ ФУНКЦИЯ: Добавление эмбеддинга в FAISS
# ============================================================================
def add_embedding_to_faiss(embedding: np.ndarray) -> int:
    """
    Добавить эмбеддинг в FAISS индекс.
    
    Args:
        embedding: Вектор размерности (512,), dtype: float32, L2 нормализован
    
    Returns:
        int: Позиция вектора в индексе (embedding_index)
    
    Raises:
        ValueError: Если embedding имеет неверный формат
    """
    # 1. Проверка и подготовка embedding
    if embedding is None:
        raise ValueError("Embedding не может быть None")
    
    # Конвертация если torch tensor
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()
    
    # Убедиться что это numpy array
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    # Проверка размерности
    if embedding.shape == (EMBEDDING_DIM,):
        embedding = embedding.reshape(1, -1)
    elif embedding.shape != (1, EMBEDDING_DIM):
        raise ValueError(f"Неверный размер embedding: {embedding.shape}, ожидалось ({EMBEDDING_DIM},)")
    
    # Конвертация в float32 (требование FAISS)
    embedding = embedding.astype('float32')
    
    # 2. Загрузить существующий индекс или создать новый
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if FAISS_INDEX_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product для косинусного сходства
    
    # 3. Получить текущий индекс перед добавлением
    embedding_index = index.ntotal
    
    # 4. Добавить вектор
    index.add(embedding)
    
    # 5. Сохранить индекс
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    return embedding_index


# ============================================================================
# 🔥 НОВАЯ ФУНКЦИЯ: Обновление embedding_index в БД
# ============================================================================
def update_photo_embedding_index(
    cursor: sqlite3.Cursor, 
    photo_path: str, 
    embedding_index: int
):
    """
    Обновить embedding_index для фотографии в БД.
    
    Args:
        cursor: Курсор SQLite
        photo_path: Путь к фото
        embedding_index: Позиция в FAISS индексе
    """
    cursor.execute('''
        UPDATE photos 
        SET embedding_index = ?, 
            is_processed = 1
        WHERE photo_path = ?
    ''', (embedding_index, photo_path))


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
    species_prefix = {
        "Карелина": "K",
        "Гребенчатый": "R",
        "Ребристый": "R"
    }.get(species, "X")
    
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
    
    template_short = template_type.replace("-", "")
    
    if animal_id is None:
        new_num = get_next_animal_number(cursor, species)
        animal_id = f"NT-{species_prefix}-{new_num}"
    
    card_id = f"{animal_id}-{template_short}"
    
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
    
    Args:
        cursor: Курсор SQLite
        individual_id: ID карточки
    
    Returns:
        str: Номер фото с ведущим нулём ("01", "02", ...)
    """
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
    embedding: Optional[np.ndarray],    # 🔥 Тип указан
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
    Сохраняет новую особь в базу данных (карточка + фотографии).
    🔥 Автоматически добавляет embedding в FAISS если он предоставлен
    
    Args:
        embedding: Вектор от ViT модели (512,) или None
        photo_path_full: Путь к полному фото
        photo_path_cropped: Путь к кропу брюшка
        species: Вид тритона
        project_name: Название проекта
        template_type: Тип карточки
        individual_id: ID карточки
        photo_number: Номер фото
        is_legacy: Флаг "из старого датасета"
        **card_data: Дополнительные поля
    
    Returns:
        str: individual_id сохранённой карточки
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
        # === ТАБЛИЦА 1: individuals ===
        cursor.execute('''
            INSERT INTO individuals (
                individual_id, template_type, species, project_name,
                created_at, date, notes,
                length_body, length_tail, length_total, weight, sex,
                birth_year_exact, birth_year_approx, origin_region,
                length_device, weight_device,
                parent_male_id, parent_female_id, release_date, water_body_name,
                meeting_time, status, water_body_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        
        # === ТАБЛИЦА 2: photos (полное фото) ===
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
        # 🔥 embedding_index = -1 (временная заглушка, обновится ниже)
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
                1,  # is_processed = 1 (будет обновлено если embedding есть)
                -1, # 🔥 embedding_index заглушка
                1 if is_legacy else 0
            ))
            
            # 🔥 ИНТЕГРАЦИЯ С FAISS: После сохранения в БД
            if embedding is not None:
                try:
                    embedding_index = add_embedding_to_faiss(embedding)
                    update_photo_embedding_index(cursor, photo_path_cropped, embedding_index)
                    print(f"   📦 Добавлено в FAISS: индекс {embedding_index}")
                except Exception as faiss_error:
                    print(f"⚠️ Ошибка добавления в FAISS: {faiss_error}")
                    print(f"   Особь сохранена в БД, но не добавлена в поиск")
                    # Не прерываем выполнение, БД уже сохранена
        
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
    """
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [field for field in required if card_data.get(field) is None]
    
    if missing:
        raise ValueError(
            f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
            f"Переданные данные: {list(card_data.keys())}"
        )


# ============================================================================
# УДАЛЕНИЕ ОСОБИ
# ============================================================================
def delete_individual(individual_id: str, delete_photos: bool = True, confirm: bool = False):
    """
    Полностью удаляет особь и все её фото (hard delete).
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
        cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (individual_id,))
        if not cursor.fetchone():
            raise ValueError(f"Особь {individual_id} не найдена в базе.")
        
        photo_paths = []
        if delete_photos:
            cursor.execute('SELECT photo_path FROM photos WHERE individual_id = ?', (individual_id,))
            photo_paths = [row[0] for row in cursor.fetchall()]
            cursor.execute('DELETE FROM photos WHERE individual_id = ?', (individual_id,))
        
        cursor.execute('DELETE FROM individuals WHERE individual_id = ?', (individual_id,))
        
        conn.commit()
        conn.close()
        
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
    Обновляет данные существующей особи.
    """
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


# ============================================================================
# ДОБАВЛЕНИЕ ПОВТОРНОЙ ВСТРЕЧИ
# ============================================================================
def add_encounter(
    individual_id: str,
    template_type: str,
    photo_path_full: str = None,
    photo_path_cropped: str = None,
    embedding: Optional[np.ndarray] = None,  # 🔥 Тип указан
    **card_data
):
    """
    Добавляет НОВУЮ ВСТРЕЧУ (КВ-1/КВ-2) для существующей особи.
    🔥 Автоматически добавляет embedding в FAISS если он предоставлен
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if template_type not in ["КВ-1", "КВ-2"]:
        raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")
    
    _validate_template_fields(template_type, card_data)
    photo_number = _get_next_photo_number(cursor, individual_id)
    
    try:
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
        
        if photo_path_full:
            cursor.execute('''
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (individual_id, 'full', photo_number, photo_path_full, card_data.get('date'), 0, 0, None, 0))
        
        if photo_path_cropped:
            cursor.execute('''
                INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (individual_id, 'cropped', photo_number, photo_path_cropped, card_data.get('date'), 0, 1, -1, 0))
            
            # 🔥 ИНТЕГРАЦИЯ С FAISS: Тоже добавляем в поиск
            if embedding is not None:
                try:
                    embedding_index = add_embedding_to_faiss(embedding)
                    update_photo_embedding_index(cursor, photo_path_cropped, embedding_index)
                    print(f"   📦 Добавлено в FAISS: индекс {embedding_index}")
                except Exception as faiss_error:
                    print(f"⚠️ Ошибка добавления в FAISS: {faiss_error}")
        
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
    print("🧪 Тестирование интеграции с FAISS...\n")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # === ТЕСТ 1: Сохранение с embedding ===
    print("1️⃣ Сохранение особи с embedding (должно добавить в FAISS)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
        test_embedding = np.random.rand(512).astype('float32')
        test_embedding = test_embedding / np.linalg.norm(test_embedding)  # L2 norm
        
        save_new_individual(
            embedding=test_embedding,
            photo_path_full="data/photos/full/test_faiss_001.jpg",
            photo_path_cropped="data/crop/test_faiss_001.jpg",
            species="Карелина",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=42.5,
            weight=3.2,
            sex="М"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === ТЕСТ 2: Сохранение без embedding ===
    print("2️⃣ Сохранение особи БЕЗ embedding (embedding_index = -1)")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
        
        save_new_individual(
            embedding=None,
            photo_path_full="data/photos/full/test_faiss_002.jpg",
            photo_path_cropped="data/crop/test_faiss_002.jpg",
            species="Карелина",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=45.0,
            weight=3.5,
            sex="Ж"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # === ТЕСТ 3: add_encounter с embedding ===
    print("3️⃣ Добавление встречи с embedding")
    try:
        card_id = generate_card_id(cursor, species="Карелина", template_type="КВ-1", animal_id="NT-K-1")
        test_embedding = np.random.rand(512).astype('float32')
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        add_encounter(
            individual_id=card_id,
            template_type="КВ-1",
            photo_path_cropped="data/crop/test_faiss_001_v2.jpg",
            embedding=test_embedding,
            status="жив",
            water_body_number="Пруд №3",
            length_body=44.0,
            weight=3.4,
            sex="М"
        )
        print(f"   ✅ Сохранено: {card_id}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    conn.commit()
    conn.close()
    
    # === ПРОВЕРКА FAISS ===
    print("\n" + "="*50 + "\n")
    print("📊 Проверка FAISS индекса...")
    try:
        if FAISS_INDEX_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            print(f"   ✅ Векторов в индексе: {index.ntotal}")
        else:
            print(f"   ⚠️ FAISS индекс не создан: {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"   ❌ Ошибка чтения FAISS: {e}")
    
    # === ПРОВЕРКА БД ===
    print("\n📊 Проверка БД...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index != -1")
    print(f"   ✅ Фото с embedding_index: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index = -1")
    print(f"   ⚠️ Фото без embedding_index: {cursor.fetchone()[0]}")
    
    conn.close()
    
    print("\n" + "="*50 + "\n")
    print("📊 Тесты завершены!")