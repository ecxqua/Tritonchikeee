"""
Тестирование функции save_new_person()
"""

import sqlite3
from pathlib import Path
from pipeline.analyse import save_new_person
from database.card_database import DB_PATH

# ============================================================================
# КОНФИГУРАЦИЯ ТЕСТА
# ============================================================================

TEST_INDIVIDUAL_ID = "NT-K-TEST-001"
TEST_PHOTO_FULL = "data/photos/full/test_save_new.jpg"
TEST_PHOTO_CROPPED = "data/output/test_save_new_cropped.jpg"

# ============================================================================
# ПОДГОТОВКА
# ============================================================================

def cleanup_test_data():
    """Удалить тестовые данные из БД"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM photos WHERE individual_id LIKE 'NT-K-TEST-%'")
    cursor.execute("DELETE FROM individuals WHERE individual_id LIKE 'NT-K-TEST-%'")
    
    conn.commit()
    conn.close()
    print("✅ Тестовые данные очищены")


def verify_saved_data(individual_id):
    """Проверить что данные сохранились в БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Проверка individuals
    cursor.execute("""
        SELECT individual_id, species, template_type, project_name, created_at
        FROM individuals 
        WHERE individual_id = ?
    """, (individual_id,))
    
    individual = cursor.fetchone()
    
    if individual:
        print("\n✅ individuals:")
        print(f"   individual_id: {individual['individual_id']}")
        print(f"   species: {individual['species']}")
        print(f"   template_type: {individual['template_type']}")
        print(f"   project_name: {individual['project_name']}")
    else:
        print("\n❌ individuals: запись не найдена")
    
    # 2. Проверка photos
    cursor.execute("""
        SELECT photo_id, photo_type, photo_path, is_legacy, embedding_index
        FROM photos 
        WHERE individual_id = ?
    """, (individual_id,))
    
    photos = cursor.fetchall()
    
    if photos:
        print(f"\n✅ photos: найдено {len(photos)} записей")
        for photo in photos:
            print(f"   - {photo['photo_type']}: embedding_index={photo['embedding_index']}, is_legacy={photo['is_legacy']}")
    else:
        print("\n❌ photos: записи не найдены")
    
    conn.close()
    
    return individual is not None and len(photos) > 0


# ============================================================================
# ТЕСТ 1: Сохранение с mock embedding
# ============================================================================

def test_save_with_mock_embedding():
    """Тест: сохранение с фиктивным embedding"""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Сохранение с mock embedding")
    print("=" * 60)
    
    import numpy as np
    
    try:
        # Создать фиктивный embedding (512 чисел)
        mock_embedding = np.random.rand(512).astype('float32')
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)  # L2 norm
        
        individual_id = save_new_person(
            mock_embedding,
            photo_path_full=TEST_PHOTO_FULL,
            photo_path_cropped=TEST_PHOTO_CROPPED,
            species="Гребенчатый",
            project_name="Тест_Сохранение",
            template_type="ИК-1",
            individual_id="NT-R-TEST-001",
            length_body=50.0,
            weight=4.5,
            sex="Ж"
        )
        
        print(f"\n✅ Функция вернула: {individual_id}")
        
        # Проверка БД
        success = verify_saved_data("NT-R-TEST-001")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return False


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

if __name__ == "__main__":
    print("🧪 Тестирование save_new_person()\n")
    
    # 1. Очистка
    cleanup_test_data()
    
    # 2. Запуск тестов
    results = []
    
    results.append(("С mock embedding", test_save_with_mock_embedding()))
    
    # 3. Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {test_name}: {'PASS' if result else 'FAIL'}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Все тесты пройдены!")
    else:
        print("❌ Некоторые тесты не пройдены")
    print("=" * 60)
    
    # 4. Финальная очистка
    cleanup_test_data()