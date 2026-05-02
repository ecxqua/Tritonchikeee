"""
🦎 Миграция датасета тритонов в базу данных.

Работает с новой схемой:
    - ✅ projects таблица (id, name, description)
    - ✅ cards.project_id (FK → projects.id)
    - ✅ uploads.project_id (FK → projects.id)

Изменения:
    - ✅ Добавлена проверка фото на дубликаты (photo_exists)
    - ✅ Скрипт идемпотентен: безопасно запускать повторно
    - ✅ Не пропускает всю особь, если карточка уже есть, а проверяет каждое фото отдельно
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("database/cards.db")
DATASET_PATH = Path("data/dataset_crop/dataset_crop_24_new")

SPECIES_CONFIG = {
    "karelin": {
        "species_name": "Карелина",
        "prefix": "K",
        "folder": DATASET_PATH / "karelin"
    },
    "ribbed": {
        "species_name": "Гребенчатый",
        "prefix": "R",
        "folder": DATASET_PATH / "ribbed"
    }
}

PROJECT_NAME = "Миграция_Датасет_2024"
DEFAULT_TEMPLATE = "ИК-1"

def get_connection():
    """Получить соединение с БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_or_create_project(cursor, project_name: str, description: str = None) -> int:
    """
    Получить ID проекта или создать новый.
    ⚠️ Это вспомогательная функция для миграции, НЕ часть CRUD API.
    """
    cursor.execute('SELECT id FROM projects WHERE name = ?', (project_name,))
    row = cursor.fetchone()
    
    if row:
        return row['id']
    
    cursor.execute('''
        INSERT INTO projects (name, description, created_at)
        VALUES (?, ?, ?)
    ''', (project_name, description, datetime.now().isoformat()))
    
    return cursor.lastrowid

def individual_exists(cursor, card_id: str) -> bool:
    """Проверить, существует ли карточка в БД."""
    cursor.execute(
        "SELECT 1 FROM cards WHERE card_id = ?",
        (card_id,)
    )
    return cursor.fetchone() is not None

def photo_exists(cursor, card_id: str, photo_path: str) -> bool:
    """Проверить, существует ли фотография в БД для данной особи."""
    cursor.execute(
        "SELECT 1 FROM photos WHERE card_id = ? AND photo_path = ?",
        (card_id, photo_path)
    )
    return cursor.fetchone() is not None

def create_individual(cursor, card_id: str, species: str, project_id: int, template_type: str = DEFAULT_TEMPLATE):
    """Создать запись о карточке особи. (Идемпотентно благодаря INSERT OR IGNORE)"""
    cursor.execute('''
        INSERT OR IGNORE INTO cards
        (card_id, template_type, species, project_id, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (card_id, template_type, species, project_id, datetime.now()))

def create_photo(cursor, card_id: str, photo_path: Path, photo_number: str, is_main: bool = False, is_legacy: bool = True) -> int:
    """Создать запись о фотографии."""
    cursor.execute('''
        INSERT INTO photos
        (card_id, photo_type, photo_number, photo_path, is_main, is_legacy, embedding_index, is_processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        card_id,
        "cropped",
        photo_number,
        str(photo_path),
        1 if is_main else 0,
        1 if is_legacy else 0,
        -1,
        0
    ))
    return cursor.lastrowid

def migrate_dataset() -> dict:
    """Основная функция миграции."""
    print("🦎 Начинаем миграцию датасета тритонов...")
    print("=" * 60)
    
    if not DATASET_PATH.exists():
        print(f"❌ Папка датасета не найдена: {DATASET_PATH}")
        return {}
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 🔥 Создать или получить проект
    project_id = get_or_create_project(
        cursor,
        PROJECT_NAME,
        "Автоматически импортированные данные из dataset_crop_24_new"
    )
    print(f"✅ Проект: '{PROJECT_NAME}' (ID={project_id})")
    
    total_cards = 0
    total_photos = 0
    skipped_cards = 0
    skipped_photos = 0
    
    for species_key, config in SPECIES_CONFIG.items():
        species_folder = config["folder"]
        species_name = config["species_name"]
        species_prefix = config["prefix"]
        
        print(f"\n📁 Обработка вида: {species_name} ({species_folder})")
        
        if not species_folder.exists():
            print(f"⚠️ Папка не найдена: {species_folder}")
            continue
        
        individual_folders = sorted([
            f for f in species_folder.iterdir()
            if f.is_dir() and f.name.isdigit()
        ])
        
        print(f"   Найдено особей: {len(individual_folders)}")
        
        for individual_folder in individual_folders:
            # Формирование ID: NT-K-1-ИК1
            animal_num = individual_folder.name
            card_id = f"NT-{species_prefix}-{animal_num}-{DEFAULT_TEMPLATE.replace('-', '')}"
            
            # ✅ Проверяем наличие карточки, но НЕ пропускаем фото, если она уже есть
            card_exists = individual_exists(cursor, card_id)
            if not card_exists:
                create_individual(
                    cursor=cursor,
                    card_id=card_id,
                    species=species_name,
                    project_id=project_id
                )
                total_cards += 1
                print(f"   ✅ Добавлена карточка: {card_id}")
            else:
                print(f"   ℹ️ Карточка {card_id} уже существует (проверяем фото)")
                skipped_cards += 1
            
            # Поиск фотографий
            photos = sorted(individual_folder.glob("*.jpg"))
            if not photos: photos = sorted(individual_folder.glob("*.jpeg"))
            if not photos: photos = sorted(individual_folder.glob("*.png"))
            
            print(f"      Найдено фото: {len(photos)}")
            
            for idx, photo_path in enumerate(photos, 1):
                photo_path_str = str(photo_path)
                
                # 🔥 Проверка на дубликат фото перед вставкой
                if photo_exists(cursor, card_id, photo_path_str):
                    print(f"      ⏭️ Пропущено фото: {photo_path.name}")
                    skipped_photos += 1
                    continue
                
                photo_number = f"{idx:02d}"
                is_main = (idx == 1)
                
                create_photo(
                    cursor=cursor,
                    card_id=card_id,
                    photo_path=photo_path,
                    photo_number=photo_number,
                    is_main=is_main,
                    is_legacy=True
                )
                total_photos += 1
                print(f"      ✅ Добавлено фото: {photo_path.name}")
            
            conn.commit()
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("📊 ИТОГИ МИГРАЦИИ:")
    print(f"   ✅ Карточек добавлено: {total_cards}")
    print(f"   ✅ Фото добавлено: {total_photos}")
    print(f"   ⏭️ Карточек пропущено: {skipped_cards}")
    print(f"   ⏭️ Фото пропущено (уже в БД): {skipped_photos}")
    print("=" * 60)
    
    return {
        "cards_added": total_cards,
        "photos_added": total_photos,
        "cards_skipped": skipped_cards,
        "photos_skipped": skipped_photos
    }

def verify_migration():
    """Проверить результаты миграции."""
    print("\n🔍 Проверка миграции...")
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.species, COUNT(*) as count, p.name as project_name
        FROM cards i
        JOIN projects p ON i.project_id = p.id
        WHERE p.name = ?
        GROUP BY i.species
    ''', (PROJECT_NAME,))
    
    print("\n📋 Карточки по видам:")
    for row in cursor.fetchall():
        print(f"   {row['species']} ({row['project_name']}): {row['count']}")
    
    cursor.execute('''
        SELECT photo_type, is_legacy, COUNT(*) as count
        FROM photos
        WHERE card_id LIKE 'NT-%'
        GROUP BY photo_type, is_legacy
    ''')
    
    print("\n📸 Фотографии:")
    for row in cursor.fetchall():
        print(f"   {row['photo_type']} (legacy={row['is_legacy']}): {row['count']}")
    
    cursor.execute('''
        SELECT COUNT(*) as count
        FROM photos
        WHERE embedding_index = -1 AND is_legacy = 1
    ''')
    result = cursor.fetchone()
    print(f"\n⚠️ Фото без эмбеддинга: {result['count']}")
    
    cursor.execute('''
        SELECT i.card_id, p.name as project_name
        FROM cards i
        JOIN projects p ON i.project_id = p.id
        WHERE p.name = ?
        LIMIT 5
    ''', (PROJECT_NAME,))
    
    print("\n📋 Примеры ID карточек:")
    for row in cursor.fetchall():
        print(f"   {row['card_id']} (проект: {row['project_name']})")
    
    conn.close()

if __name__ == "__main__":
    from database.card_database import init_database
    
    # 1. Инициализация БД (создаст таблицу projects)
    init_database()
    
    # 2. Миграция
    stats = migrate_dataset()
    verify_migration()
    
    print("\n✅ Миграция завершена!")
    print("👉 Следующий шаг: запустить build_faiss_index.py")