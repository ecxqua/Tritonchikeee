"""
🦎 Миграция датасета тритонов в базу данных
Сканирует папки karelin/ и ribbed/, создаёт записи в individuals и photos
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("database/cards.db")
DATASET_PATH = Path("data/dataset_crop/dataset_crop_24")

# Конфигурация видов
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
    """Получить соединение с БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def individual_exists(cursor, individual_id):
    """Проверить, существует ли особь в БД"""
    cursor.execute(
        "SELECT individual_id FROM individuals WHERE individual_id = ?",
        (individual_id,)
    )
    return cursor.fetchone() is not None


def create_individual(cursor, individual_id, species, project_name, template_type=DEFAULT_TEMPLATE):
    """Создать запись об особи"""
    cursor.execute('''
        INSERT OR IGNORE INTO individuals 
        (individual_id, template_type, species, project_name, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (individual_id, template_type, species, project_name, datetime.now()))


def create_photo(cursor, individual_id, photo_path, photo_number, is_main=False, is_legacy=True):
    """Создать запись о фотографии"""
    cursor.execute('''
        INSERT INTO photos 
        (individual_id, photo_type, photo_number, photo_path, 
         is_main, is_legacy, embedding_index, is_processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        individual_id,
        "cropped",
        photo_number,
        str(photo_path),
        1 if is_main else 0,
        1 if is_legacy else 0,
        -1,  # embedding_index будет обновлён позже
        0    # is_processed = 0, пока не обработано ViT
    ))
    
    return cursor.lastrowid


def migrate_dataset():
    """Основная функция миграции"""
    print("🦎 Начинаем миграцию датасета тритонов...")
    print("=" * 60)
    
    # Проверка существования папок
    if not DATASET_PATH.exists():
        print(f"❌ Папка датасета не найдена: {DATASET_PATH}")
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    total_individuals = 0
    total_photos = 0
    skipped_individuals = 0
    
    for species_key, config in SPECIES_CONFIG.items():
        species_folder = config["folder"]
        species_name = config["species_name"]
        species_prefix = config["prefix"]
        
        print(f"\n📁 Обработка вида: {species_name} ({species_folder})")
        
        if not species_folder.exists():
            print(f"⚠️ Папка не найдена: {species_folder}")
            continue
        
        # Получаем все папки особей (цифровые имена)
        individual_folders = sorted([
            f for f in species_folder.iterdir() 
            if f.is_dir() and f.name.isdigit()
        ])
        
        print(f"   Найдено особей: {len(individual_folders)}")
        
        for individual_folder in individual_folders:
            individual_id = f"NT-{species_prefix}-{individual_folder.name}"
            
            # Пропускаем если уже в БД
            if individual_exists(cursor, individual_id):
                print(f"   ⏭️ Пропущено: {individual_id} (уже в БД)")
                skipped_individuals += 1
                continue
            
            # Создаём запись об особи
            create_individual(cursor, individual_id, species_name, PROJECT_NAME)
            total_individuals += 1
            print(f"   ✅ Добавлено: {individual_id}")
            
            # Получаем все фото в папке
            photos = sorted(individual_folder.glob("*.jpg"))
            if not photos:
                photos = sorted(individual_folder.glob("*.jpeg"))
            if not photos:
                photos = sorted(individual_folder.glob("*.png"))
            
            print(f"      Фото найдено: {len(photos)}")
            
            # Обрабатываем каждое фото
            for idx, photo_path in enumerate(photos, 1):
                photo_number = f"{idx:02d}"
                is_main = (idx == 1)
                
                create_photo(
                    cursor=cursor,
                    individual_id=individual_id,
                    photo_path=photo_path,
                    photo_number=photo_number,
                    is_main=is_main,
                    is_legacy=True
                )
                total_photos += 1
            
            # Фиксируем изменения после каждой особи
            conn.commit()
    
    conn.close()
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("📊 ИТОГИ МИГРАЦИИ:")
    print(f"   ✅ Особей добавлено: {total_individuals}")
    print(f"   ✅ Фото добавлено: {total_photos}")
    print(f"   ⏭️ Особей пропущено: {skipped_individuals}")
    print("=" * 60)
    
    return {
        "individuals_added": total_individuals,
        "photos_added": total_photos,
        "individuals_skipped": skipped_individuals
    }


def verify_migration():
    """Проверить результаты миграции"""
    print("\n🔍 Проверка миграции...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Сколько особей мигрировано
    cursor.execute('''
        SELECT species, COUNT(*) as count 
        FROM individuals 
        WHERE project_name = ?
        GROUP BY species
    ''', (PROJECT_NAME,))
    
    print("\n📋 Особи по видам:")
    for row in cursor.fetchall():
        print(f"   {row['species']}: {row['count']}")
    
    # 2. Сколько фото мигрировано
    cursor.execute('''
        SELECT photo_type, is_legacy, COUNT(*) as count 
        FROM photos 
        WHERE individual_id LIKE 'NT-%'
        GROUP BY photo_type, is_legacy
    ''')
    
    print("\n📸 Фотографии:")
    for row in cursor.fetchall():
        print(f"   {row['photo_type']} (legacy={row['is_legacy']}): {row['count']}")
    
    # 3. Сколько фото без эмбеддинга
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM photos 
        WHERE embedding_index = -1 AND is_legacy = 1
    ''')
    
    result = cursor.fetchone()
    print(f"\n⚠️ Фото без эмбеддинга: {result['count']}")
    
    conn.close()


if __name__ == "__main__":
    # Запуск миграции
    migrate_dataset()
    
    # Проверка результатов
    verify_migration()
    
    print("\n✅ Миграция завершена!")
    print("👉 Следующий шаг: запустить build_faiss_index.py")