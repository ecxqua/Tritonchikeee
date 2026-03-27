"""
🦎 Тестирование полного цикла работы системы
1. Миграция старого датасета
2. Построение FAISS индекса
3. Анализ нового фото
4. Сохранение новой особи / встречи
"""

import sys
from pathlib import Path
import sqlite3
import numpy as np

# Добавляем корень проекта
sys.path.append(str(Path(__file__).parent.parent))

from database.card_database import DB_PATH, init_database
from pipeline.save_new import (
    save_new_individual,
    add_encounter,
    generate_card_id,
    get_individual_photos,
    add_embedding_to_faiss,
    update_photo_embedding_index
)
from pipeline.deployment_vit import (
    load_model,
    get_embedding,
    find_similar_in_database  # ⚠️ Должна быть добавлена!
)
from pipeline.deployment_yolo_new import process_single_image
from database.migrate_dataset import migrate_dataset, verify_migration
from database.build_faiss_index import build_faiss_index, verify_index

# ============================================================================
# КОНФИГУРАЦИЯ ТЕСТА
# ============================================================================
TEST_CONFIG = {
    "model_path": "models/best_model.pth",
    "faiss_index_path": "data/embeddings/database_embeddings.pkl",
    "db_path": "database/cards.db",
    "test_photo": "data/input/test_triton.jpg",
    "output_dir": "data/output/test"
}


# ============================================================================
# ЭТАП 1: ПРОВЕРКА БАЗЫ
# ============================================================================
def test_database_status():
    """Проверка состояния базы данных"""
    print("\n" + "="*60)
    print("📊 ЭТАП 1: Проверка базы данных")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Сколько особей в базе
    cursor.execute("SELECT COUNT(*) FROM individuals")
    individuals_count = cursor.fetchone()[0]
    
    # Сколько фото с эмбеддингом
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index != -1")
    photos_with_embedding = cursor.fetchone()[0]
    
    # Сколько фото без эмбеддинга
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index = -1")
    photos_without_embedding = cursor.fetchone()[0]
    
    print(f"   ✅ Особей в базе: {individuals_count}")
    print(f"   ✅ Фото с эмбеддингом: {photos_with_embedding}")
    print(f"   ⚠️ Фото без эмбеддинга: {photos_without_embedding}")
    
    conn.close()
    
    return {
        "individuals": individuals_count,
        "photos_with_embedding": photos_with_embedding,
        "photos_without_embedding": photos_without_embedding
    }


# ============================================================================
# ЭТАП 2: МИГРАЦИЯ (если нужно)
# ============================================================================
def test_migration():
    """Миграция старого датасета"""
    print("\n" + "="*60)
    print("📊 ЭТАП 2: Миграция старого датасета")
    print("="*60)
    
    try:
        result = migrate_dataset()
        verify_migration()
        print(f"   ✅ Миграция завершена: {result['individuals_added']} особей")
        return True
    except Exception as e:
        print(f"   ⚠️ Миграция не требуется или ошибка: {e}")
        return False


# ============================================================================
# ЭТАП 3: ПОСТРОЕНИЕ FAISS ИНДЕКСА
# ============================================================================
def test_build_faiss_index():
    """Построение FAISS индекса для легаси-фото"""
    print("\n" + "="*60)
    print("📊 ЭТАП 3: Построение FAISS индекса")
    print("="*60)
    
    try:
        result = build_faiss_index()
        verify_index()
        print(f"   ✅ Индекс построен: {result['total_vectors']} векторов")
        return True
    except Exception as e:
        print(f"   ⚠️ Ошибка построения индекса: {e}")
        return False


# ============================================================================
# ЭТАП 4: АНАЛИЗ НОВОГО ФОТО
# ============================================================================
async def test_photo_analysis():
    """Анализ нового фото через YOLO + ViT + FAISS"""
    print("\n" + "="*60)
    print("📊 ЭТАП 4: Анализ нового фото")
    print("="*60)
    
    from pipeline.analyse import photo_processing, load_config
    
    try:
        config = load_config()
        result = await photo_processing(config, TEST_CONFIG["test_photo"])
        
        print(f"   Статус: {result['status']}")
        print(f"   Сообщение: {result['message']}")
        
        if result.get('matches'):
            print(f"   🔍 Найдено совпадений: {len(result['matches'])}")
            for i, match in enumerate(result['matches'][:3], 1):
                db_status = "✅ в базе" if match.get('in_database', True) else "❌ не в базе"
                print(f"      {i}. {match['individual_id']}: {match['score']:.2%} {db_status}")
        
        return result
    except Exception as e:
        print(f"   ❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ============================================================================
# ЭТАП 5: СОХРАНЕНИЕ НОВОЙ ОСОБИ
# ============================================================================
def test_save_new_individual(embedding, photo_path_cropped):
    """Сохранение новой особи с FAISS интеграцией"""
    print("\n" + "="*60)
    print("📊 ЭТАП 5: Сохранение новой особи")
    print("="*60)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        card_id = generate_card_id(cursor, species="Карелина", template_type="ИК-1")
        conn.close()
        
        print(f"   Сгенерирован ID: {card_id}")
        
        individual_id = save_new_individual(
            embedding=embedding,
            photo_path_full=TEST_CONFIG["test_photo"],
            photo_path_cropped=photo_path_cropped,
            species="Карелина",
            template_type="ИК-1",
            individual_id=card_id,
            length_body=42.5,
            weight=3.2,
            sex="М",
            notes="Тестовая особь"
        )
        
        print(f"   ✅ Сохранено: {individual_id}")
        return individual_id
    except Exception as e:
        print(f"   ❌ Ошибка сохранения: {e}")
        return None


# ============================================================================
# ЭТАП 6: ДОБАВЛЕНИЕ ВСТРЕЧИ
# ============================================================================
def test_add_encounter(individual_id, embedding, photo_path_cropped):
    """Добавление повторной встречи"""
    print("\n" + "="*60)
    print("📊 ЭТАП 6: Добавление повторной встречи")
    print("="*60)
    
    try:
        photo_number = add_encounter(
            individual_id=individual_id,
            template_type="КВ-1",
            photo_path_full=TEST_CONFIG["test_photo"],
            photo_path_cropped=photo_path_cropped,
            embedding=embedding,
            status="жив",
            water_body_number="Пруд №3",
            length_body=44.0,
            length_tail=39.0,
            weight=3.4,
            sex="М",
            date="20.06.2024"
        )
        
        print(f"   ✅ Встреча добавлена: фото #{photo_number}")
        return photo_number
    except Exception as e:
        print(f"   ❌ Ошибка добавления встречи: {e}")
        return None


# ============================================================================
# ЭТАП 7: ПРОВЕРКА РЕЗУЛЬТАТОВ
# ============================================================================
def test_final_verification():
    """Финальная проверка состояния системы"""
    print("\n" + "="*60)
    print("📊 ЭТАП 7: Финальная проверка")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Проверка FAISS индекса
    import faiss
    from pathlib import Path
    if Path(TEST_CONFIG["faiss_index_path"]).exists():
        index = faiss.read_index(str(TEST_CONFIG["faiss_index_path"]))
        print(f"   ✅ Векторов в FAISS: {index.ntotal}")
    else:
        print(f"   ⚠️ FAISS индекс не найден")
    
    # Проверка БД
    cursor.execute("SELECT COUNT(*) FROM individuals")
    print(f"   ✅ Особей в базе: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index != -1")
    print(f"   ✅ Фото с эмбеддингом: {cursor.fetchone()[0]}")
    
    # Последние 5 особей
    cursor.execute("""
        SELECT individual_id, template_type, created_at 
        FROM individuals 
        ORDER BY created_at DESC 
        LIMIT 5
    """)
    print("\n   📋 Последние особи:")
    for row in cursor.fetchall():
        print(f"      {row[0]} ({row[1]})")
    
    conn.close()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
async def run_full_cycle_test():
    """Запуск полного цикла тестирования"""
    print("\n" + "🦎"*30)
    print("🦎 СИМУЛЯЦИЯ ПОЛНОГО ЦИКЛА РАБОТЫ СИСТЕМЫ 🦎")
    print("🦎"*30)
    
    # Этап 1: Проверка базы
    db_status = test_database_status()
    
    # Этап 2: Миграция (если фото без эмбеддинга)
    if db_status['photos_without_embedding'] > 0:
        test_migration()
    
    # Этап 3: Построение FAISS индекса
    test_build_faiss_index()
    
    # Этап 4: Анализ нового фото
    analysis_result = await test_photo_analysis()
    
    # Этап 5/6: Сохранение (в зависимости от результата анализа)
    if analysis_result.get('status') == 'review_required':
        print("\n⚠️ Требуется ручное решение биолога!")
        print("   Для теста сохраняем как новую особь...")
        
        embedding = analysis_result.get('embedding')
        cropped_path = analysis_result.get('cropped_path')
        
        if embedding is not None and cropped_path is not None:
            # Этап 5: Новая особь
            new_id = test_save_new_individual(embedding, cropped_path)
            
            # Этап 6: Повторная встреча (для теста)
            if new_id:
                test_add_encounter(new_id, embedding, cropped_path)
    
    # Этап 7: Финальная проверка
    test_final_verification()
    
    print("\n" + "🦎"*30)
    print("✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА!")
    print("🦎"*30 + "\n")


# ============================================================================
# ЗАПУСК
# ============================================================================
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_full_cycle_test())