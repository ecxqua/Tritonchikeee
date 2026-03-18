import sys
from pathlib import Path
from datetime import datetime
import sqlite3

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from database.card_database import DB_PATH, init_database

# === КОНФИГУРАЦИЯ ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ ===
# Определяем, какие поля обязательны для каждого шаблона согласно ТЗ
REQUIRED_FIELDS = {
    "ИК-1": ["length_body", "weight", "sex"],  # + date, notes (общие)
    "ИК-2": ["parent_male_id", "parent_female_id", "water_body_name", "release_date"],
    "КВ-1": ["status", "water_body_number", "length_body", "length_tail"],
    "КВ-2": ["status", "water_body_name"]
}

def save_new_individual(
    embedding,                    # numpy-вектор от ViT (обязательно)
    photo_path: str,              # путь к оригинальному фото
    species: str = "Карелина",    # "Карелина" или "Гребенчатый"
    project_name: str = "Основной",
    template_type: str = "ИК-1",  # ИК-1, ИК-2, КВ-1, КВ-2
    individual_id: str = None,    # Если None, генерируется автоматически
    **card_data                   # Все остальные поля карточки
):
    """
    Сохраняет новую особь в базу (карточка + эмбеддинг)
    
    Поддерживает 4 шаблона: ИК-1, ИК-2, КВ-1, КВ-2
    """
    # 1. Инициализация БД
    init_database() 
    
    # 2. Валидация обязательных полей для выбранного шаблона
    _validate_template_fields(template_type, card_data)
    
    # 3. Генерация ID, если не передан
    if individual_id is None:
        individual_id = f"NT-{datetime.now().strftime('%y%m%d%H%M%S')}"
    
    # 4. Подключение и запись
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO individuals (
            -- Служебные поля
            individual_id, template_type, species, project_name, photo_path,
            photo_number, embedding_index, created_at,
            
            -- Общие поля
            date, notes,
            
            -- Биометрия
            length_body, length_tail, length_total, weight, sex,
            
            -- Рождение и происхождение (ИК-1)
            birth_year_exact, birth_year_approx, origin_region,
            length_device, weight_device,
            
            -- Родители (ИК-2)
            parent_male_id, parent_female_id, release_date, water_body_name,
            
            -- Встреча (КВ-1, КВ-2)
            meeting_time, status, water_body_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        # Служебные
        individual_id,
        template_type,
        species,
        project_name,
        photo_path,
        card_data.get('photo_number'),
        -1,  # embedding_index (FAISS добавим позже)
        datetime.now().isoformat(),
        
        # Общие
        card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
        card_data.get('notes'),
        
        # Биометрия
        card_data.get('length_body'),
        card_data.get('length_tail'),
        card_data.get('length_total'),
        card_data.get('weight'),
        card_data.get('sex'),
        
        # Рождение и происхождение
        card_data.get('birth_year_exact'),
        card_data.get('birth_year_approx'),
        card_data.get('origin_region'),
        card_data.get('length_device'),
        card_data.get('weight_device'),
        
        # Родители
        card_data.get('parent_male_id'),
        card_data.get('parent_female_id'),
        card_data.get('release_date'),
        card_data.get('water_body_name'),
        
        # Встреча
        card_data.get('meeting_time'),
        card_data.get('status'),
        card_data.get('water_body_number')
    ))
    
    conn.commit()
    conn.close()

    print(f"✅ Новая особь успешно сохранена!")
    print(f"   ID: {individual_id}")
    print(f"   Шаблон: {template_type}")
    print(f"   Вид: {species}")
    return individual_id


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
    """
    Обновляет данные существующей особи (требование ТЗ: повторное редактирование)
    """
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Формируем динамический запрос только для переданных полей
    fields = [f"{key} = ?" for key in kwargs.keys()]
    values = list(kwargs.values())
    values.append(individual_id)

    if not fields:
        print("⚠️ Нет полей для обновления")
        return

    query = f"UPDATE individuals SET {', '.join(fields)} WHERE individual_id = ?"
    
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    print(f"✅ Особь {individual_id} обновлена.")


# === ТЕСТЫ ФУНКЦИИ ===
if __name__ == "__main__":
    print("🧪 Тестируем сохранение новых особей...\n")
    
    # --- Тест 1: Шаблон ИК-1 (Первичная регистрация) ---
    print("1️⃣ Тест шаблона ИК-1 (Первичная регистрация)")
    try:
        save_new_individual(
            embedding=None,
            photo_path="data/input/test_ik1.jpg",
            template_type="ИК-1",
            length_body=42.5,
            length_tail=38.0,
            weight=3.2,
            sex="М",
            birth_year_exact="15.03.2022",
            origin_region="Свердловская обл.",
            length_device="Mitutoyo",
            weight_device="Ohaus",
            notes="Первичный отлов, хвост регенерирует"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # --- Тест 2: Шаблон ИК-2 (Выпуск с родителями) ---
    print("2️⃣ Тест шаблона ИК-2 (Выпуск с родителями)")
    try:
        save_new_individual(
            embedding=None,
            photo_path="data/input/test_ik2.jpg",
            template_type="ИК-2",
            parent_male_id="NT-24031701",
            parent_female_id="NT-24031702",
            release_date="20.06.2024",
            water_body_name="Пруд №3",
            length_total=8.5,
            weight=4.1,
            notes="Выпуск после зимовки в неволе"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # --- Тест 3: Шаблон КВ-1 (Подробная повторная встреча) ---
    print("3️⃣ Тест шаблона КВ-1 (Повторная встреча)")
    try:
        save_new_individual(
            embedding=None,
            photo_path="data/input/test_kv1.jpg",
            individual_id="NT-24031701",  # Существующий ID
            template_type="КВ-1",
            meeting_time="14:30",
            status="жив",
            water_body_number="Пруд №3",
            length_body=45.0,
            length_tail=40.0,
            weight=3.5,
            sex="М",
            notes="Повторная поимка, вырос на 2.5 мм"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # --- Тест 4: Шаблон КВ-2 (Быстрая встреча) ---
    print("4️⃣ Тест шаблона КВ-2 (Быстрая встреча)")
    try:
        save_new_individual(
            embedding=None,
            photo_path="data/input/test_kv2.jpg",
            individual_id="NT-24031702",
            template_type="КВ-2",
            meeting_time="16:45",
            status="жив",
            water_body_name="Река Исеть",
            notes="Замечен на берегу"
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # --- Тест 5: Проверка валидации (должна выдать ошибку) ---
    print("5️⃣ Тест валидации (попытка сохранить ИК-2 без родителей)")
    try:
        save_new_individual(
            embedding=None,
            photo_path="data/input/test_fail.jpg",
            template_type="ИК-2",
            # parent_male_id и parent_female_id отсутствуют → ошибка
            water_body_name="Пруд №5",
            release_date="01.07.2024"
        )
    except ValueError as e:
        print(f"✅ Валидация сработала: {e}")
    except Exception as e:
        print(f"❌ Другая ошибка: {e}")