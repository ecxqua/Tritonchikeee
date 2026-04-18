"""
🦎 Построение FAISS индекса для идентификации тритонов
Использует модель EnhancedTripletNet из pipeline/deployment_vit.py

🔥 MIGRATION: Использует IndexIDMap для поддержки удаления по photo_id
🔥 DRY: get_embedding_from_array — единственный источник истины для preprocessing
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import faiss
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from database.card_database import DB_PATH

# Добавляем pipeline в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.deployment_vit_faiss import EnhancedTripletNet, load_model, get_embedding_from_array

MODEL_PATH = Path("models/best_model.pth")
FAISS_INDEX_PATH = Path("data/embeddings/database_embeddings.pkl")

# Параметры
BATCH_SIZE = 32
EMBEDDING_DIM = 512  # Размер вектора из EnhancedTripletNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_connection():
    """Получить соединение с БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_transforms():
    """Создать трансформы для предобработки изображений (как в deployment_vit.py)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_unprocessed_photos(cursor):
    """Получить все кропы без эмбеддинга (is_legacy=1, embedding_index=-1)"""
    cursor.execute('''
        SELECT photo_id, card_id, photo_path, photo_number
        FROM photos
        WHERE is_legacy = 1 
          AND embedding_index = -1
          AND photo_type = 'cropped'
        ORDER BY card_id, photo_number
    ''')
    
    return cursor.fetchall()


def get_embeddings_batch(model, transform, photos, device, batch_size=BATCH_SIZE):
    """Получить эмбеддинги для батча фотографий.
    
    🔥 DRY: Напрямую использует get_embedding_from_array — никакой дубликации логики.
    Гарантирует идентичность пайплайна: BGR→RGB, трансформы, инференс — всё как в продакшене.
    
    Args:
        model: Загруженная модель ViT
        transform: Трансформы для предобработки
        photos: Список dict с ключом 'photo_path'
        device: Устройство для вычислений
        batch_size: (Оставлен для совместимости API)
    
    Returns:
        Tuple[np.ndarray, List[Dict]]: 
            - массив эмбеддингов формы (N, 512) для FAISS, или (0, 512) если пусто
            - список валидных фото
    """
    model.eval()
    embeddings = []
    valid_photos = []
    
    for photo in photos:
        # cv2.imread загружает в BGR — именно такой формат ожидает get_embedding_from_array
        img_array = cv2.imread(photo['photo_path'])
        if img_array is None:
            continue
            
        # 🔥 DRY: прямой вызов — гарантия идентичности пайплайна
        emb = get_embedding_from_array(img_array, model, transform, device)
        
        if emb is not None:
            embeddings.append(emb)  # emb имеет shape (512,)
            valid_photos.append(photo)
            
    if not embeddings:
        # 🔥 Безопасный возврат пустого 2D-массива для FAISS
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
        
    # Преобразуем список (512,) → массив (N, 512) для add_with_ids
    return np.vstack(embeddings), valid_photos


def load_or_create_faiss_index():
    """Загрузить существующий индекс или создать новый.
    
    🔥 Возвращает IndexIDMap, оборачивающий базовый индекс.
    """
    if FAISS_INDEX_PATH.exists():
        print(f"📦 Загрузка существующего индекса: {FAISS_INDEX_PATH}")
        try:
            raw_index = faiss.read_index(str(FAISS_INDEX_PATH))
            
            # 🔥 Если это не IndexIDMap — оборачиваем (для обратной совместимости)
            if not isinstance(raw_index, faiss.IndexIDMap):
                print("   ⚠️ Обнаружен старый формат индекса. Оборачиваем в IndexIDMap.")
                print("   ⚠️ Векторы из старого индекса имеют позиционные ID (0,1,2...).")
                base_index = raw_index
                index = faiss.IndexIDMap(base_index)
            else:
                index = raw_index
                
            print(f"   ✅ Индекс загружен, векторов: {index.ntotal}")
            return index
        except Exception as e:
            print(f"   ⚠️ Ошибка загрузки: {e}. Создаём новый индекс.")
    
    # 🔥 Создаём новый индекс: базовый + обёртка IDMap
    print("📦 Создание нового FAISS индекса (IndexIDMap)...")
    base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIDMap(base_index)
    print(f"   ✅ Индекс создан (размер вектора: {EMBEDDING_DIM})")
    
    return index


def update_photo_embedding(cursor, photo_id, embedding_index):
    """Обновить embedding_index для фотографии.
    
    🔥 После миграции: embedding_index == photo_id (стабильный идентификатор)
    """
    cursor.execute('''
        UPDATE photos 
        SET embedding_index = ?, 
            is_processed = 1
        WHERE photo_id = ?
    ''', (embedding_index, photo_id))


def build_faiss_index():
    """Основная функция построения индекса"""
    print("🦎 Построение FAISS индекса для тритонов...")
    print("=" * 60)
    print(f"🖥️ Устройство: {DEVICE}")
    print(f"📦 Размер батча: {BATCH_SIZE}")
    print(f"🧠 Модель: {MODEL_PATH}")
    print(f"🔥 Индекс: IndexIDMap (поддержка удаления по photo_id)")
    print("=" * 60)
    
    # 1. Загрузка модели
    if not MODEL_PATH.exists():
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        return
    
    print(f"\n📦 Загрузка модели EnhancedTripletNet...")
    model = load_model(str(MODEL_PATH), DEVICE)
    print(f"   ✅ Модель загружена на {DEVICE}")
    
    transform = get_transforms()
    
    # 2. Получение необработанных фото
    conn = get_connection()
    cursor = conn.cursor()
    
    photos = get_unprocessed_photos(cursor)
    total_photos = len(photos)
    
    if total_photos == 0:
        print("\n✅ Все фото уже обработаны!")
        conn.close()
        return
    
    print(f"\n📸 Фото для обработки: {total_photos}")
    
    # 3. Загрузка/создание FAISS индекса
    faiss_index = load_or_create_faiss_index()
    
    # 4. Обработка фото
    processed_count = 0
    error_count = 0
    
    print("\n⏳ Обработка фото...")
    
    # 🔥 Общий прогресс-бар на все фото
    with tqdm(total=total_photos, desc="Всего фото") as pbar:
        for i in range(0, total_photos, BATCH_SIZE):
            batch = photos[i:i + BATCH_SIZE]
            
            # Получаем эмбеддинги (используем get_embedding_from_array внутри)
            embeddings, valid_photos = get_embeddings_batch(model, transform, batch, DEVICE)
            
            if len(embeddings) == 0:
                error_count += len(batch)
                pbar.update(len(batch))
                continue
            
            # 🔥 Фильтрация дубликатов: не добавляем photo_id, который уже есть в индексе
            photo_ids = np.array([p['photo_id'] for p in valid_photos], dtype=np.int64)
            filtered_embeddings = []
            filtered_photo_ids = []
            
            for emb, pid in zip(embeddings, photo_ids):
                try:
                    # Проверяем, есть ли уже этот ID в индексе
                    faiss_index.reconstruct(np.int64(pid))
                    # Если нет ошибки — вектор уже есть, пропускаем
                    print(f"   ⚠️ Пропущен дубликат photo_id={pid}")
                except:
                    # ID нет в индексе — добавляем
                    filtered_embeddings.append(emb)
                    filtered_photo_ids.append(pid)
            
            if not filtered_embeddings:
                pbar.update(len(batch))
                continue
                
            embeddings_filtered = np.vstack(filtered_embeddings)
            photo_ids_filtered = np.array(filtered_photo_ids, dtype=np.int64)
            
            # 🔥 Атомарная транзакция: FAISS + БД
            try:
                # 1. Добавляем в FAISS (может упасть при дубликатах или ошибке памяти)
                faiss_index.add_with_ids(embeddings_filtered, photo_ids_filtered)
                
                # 2. Только если успех — обновляем БД
                for photo, pid in zip(valid_photos, photo_ids):
                    if pid in photo_ids_filtered:
                        update_photo_embedding(cursor, photo['photo_id'], photo['photo_id'])
                
                # 3. Фиксируем всё вместе
                conn.commit()
                
            except Exception as e:
                # 🔥 Откат БД при ошибке FAISS — предотвращаем рассинхрон
                conn.rollback()
                print(f"⚠️ Ошибка батча, откат БД: {e}")
                error_count += len(valid_photos)
            
            # Обновляем прогресс
            processed_count += len([p for p in valid_photos if p['photo_id'] in photo_ids_filtered])
            pbar.update(len(batch))
            
            # Промежуточное сохранение индекса каждые 5 батчей (на случай краха)
            if (i // BATCH_SIZE) % 5 == 0 and i > 0:
                faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
                print(f"   💾 Промежуточное сохранение индекса: {faiss_index.ntotal} векторов")
    
    # 5. Финальное сохранение FAISS индекса
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    
    conn.close()
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ПОСТРОЕНИЯ ИНДЕКСА:")
    print(f"   ✅ Обработано фото: {processed_count}")
    print(f"   ⚠️ Ошибок/пропущено: {error_count}")
    print(f"   📦 Всего векторов в индексе: {faiss_index.ntotal}")
    print(f"   💾 Индекс сохранён: {FAISS_INDEX_PATH}")
    print(f"   🔥 Тип индекса: {type(faiss_index).__name__}")
    print("=" * 60)
    
    return {
        "processed": processed_count,
        "errors": error_count,
        "total_vectors": faiss_index.ntotal
    }


def verify_index():
    """Проверить результаты построения индекса"""
    print("\n🔍 Проверка индекса...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Сколько фото теперь имеют эмбеддинг
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM photos 
        WHERE embedding_index != -1 AND is_legacy = 1
    ''')
    
    result = cursor.fetchone()
    print(f"   ✅ Фото с эмбеддингом: {result['count']}")
    
    # Сколько ещё без эмбеддинга
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM photos 
        WHERE embedding_index = -1 AND is_legacy = 1
    ''')
    
    result = cursor.fetchone()
    print(f"   ⚠️ Фото без эмбеддинга: {result['count']}")
    
    # 🔥 Проверка: совпадают ли embedding_index с photo_id
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM photos 
        WHERE embedding_index != photo_id AND embedding_index != -1 AND is_legacy = 1
    ''')
    mismatch = cursor.fetchone()['count']
    if mismatch > 0:
        print(f"   ⚠️ Несоответствий (embedding_index != photo_id): {mismatch}")
    else:
        print(f"   ✅ Все embedding_index совпадают с photo_id")
    
    # Пример связи
    cursor.execute('''
        SELECT p.photo_id, i.card_id, i.species, p.photo_path, p.embedding_index
        FROM cards i
        JOIN photos p ON i.card_id = p.card_id
        WHERE p.embedding_index != -1
        LIMIT 5
    ''')
    
    print("\n   📋 Примеры записей:")
    for row in cursor.fetchall():
        match = "✓" if row['photo_id'] == row['embedding_index'] else "✗"
        print(f"      {match} photo_id={row['photo_id']} → embedding_index={row['embedding_index']} | {row['card_id']} ({row['species']})")
    
    conn.close()


if __name__ == "__main__":
    # Построение индекса
    build_faiss_index()
    
    # Проверка результатов
    verify_index()
    
    print("\n✅ FAISS индекс готов!")
    print("🔥 Теперь можно удалять векторы по photo_id через embedding_service.delete()")
    print("👉 Следующий шаг: тестирование поиска и удаления особей")