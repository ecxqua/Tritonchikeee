"""
🦎 Построение FAISS индекса для идентификации тритонов
Использует модель EnhancedTripletNet из pipeline/deployment_vit.py
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import faiss
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from database.card_database import DB_PATH

# Добавляем pipeline в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.deployment_vit_faiss import EnhancedTripletNet, load_model

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
        SELECT photo_id, individual_id, photo_path, photo_number
        FROM photos
        WHERE is_legacy = 1 
          AND embedding_index = -1
          AND photo_type = 'cropped'
        ORDER BY individual_id, photo_number
    ''')
    
    return cursor.fetchall()


def load_image(photo_path):
    """Загрузить изображение"""
    try:
        img = Image.open(photo_path).convert('RGB')
        return img
    except Exception as e:
        print(f"⚠️ Ошибка загрузки {photo_path}: {e}")
        return None


def get_embeddings_batch(model, transform, photos, device, batch_size=BATCH_SIZE):
    """Получить эмбеддинги для батча фотографий"""
    model.eval()
    embeddings = []
    valid_photos = []
    
    images = []
    for photo in photos:
        img = load_image(photo['photo_path'])
        if img is not None:
            images.append(transform(img))
            valid_photos.append(photo)
    
    if not images:
        return [], []
    
    # Создаём батч
    batch = torch.stack(images).to(device)
    
    # Получаем эмбеддинги
    with torch.no_grad():
        batch_embeddings = model(batch)
        # Модель уже возвращает нормализованные векторы (p=2, dim=1)
        embeddings = batch_embeddings.cpu().numpy()
    
    return embeddings, valid_photos


def load_or_create_faiss_index():
    """Загрузить существующий индекс или создать новый"""
    if FAISS_INDEX_PATH.exists():
        print(f"📦 Загрузка существующего индекса: {FAISS_INDEX_PATH}")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"   ✅ Индекс загружен, векторов: {index.ntotal}")
    else:
        print("📦 Создание нового FAISS индекса...")
        # IndexFlatIP - inner product (косинусное сходство после нормализации)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        print(f"   ✅ Индекс создан (размер вектора: {EMBEDDING_DIM})")
    
    return index


def update_photo_embedding(cursor, photo_id, embedding_index):
    """Обновить embedding_index для фотографии"""
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
    
    # 4. Обработка фото батчами
    processed_count = 0
    error_count = 0
    
    print("\n⏳ Обработка фото...")
    
    for i in tqdm(range(0, total_photos, BATCH_SIZE), desc="Батчи"):
        batch = photos[i:i + BATCH_SIZE]
        
        # Получаем эмбеддинги
        embeddings, valid_photos = get_embeddings_batch(model, transform, batch, DEVICE)
        
        if len(embeddings) == 0:
            error_count += len(batch)
            continue
        
        # Добавляем в FAISS
        start_index = faiss_index.ntotal
        faiss_index.add(embeddings)
        
        # Обновляем БД
        for j, photo in enumerate(valid_photos):
            embedding_index = start_index + j
            update_photo_embedding(cursor, photo['photo_id'], embedding_index)
        
        processed_count += len(valid_photos)
        
        # Фиксируем изменения каждые 5 батчей
        if (i // BATCH_SIZE) % 5 == 0:
            conn.commit()
    
    # Финальный коммит
    conn.commit()
    
    # 5. Сохранение FAISS индекса
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    
    conn.close()
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ПОСТРОЕНИЯ ИНДЕКСА:")
    print(f"   ✅ Обработано фото: {processed_count}")
    print(f"   ⚠️ Ошибок: {error_count}")
    print(f"   📦 Всего векторов в индексе: {faiss_index.ntotal}")
    print(f"   💾 Индекс сохранён: {FAISS_INDEX_PATH}")
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
    
    # Пример связи
    cursor.execute('''
        SELECT i.individual_id, i.species, p.photo_path, p.embedding_index
        FROM individuals i
        JOIN photos p ON i.individual_id = p.individual_id
        WHERE p.embedding_index != -1
        LIMIT 5
    ''')
    
    print("\n   📋 Примеры записей:")
    for row in cursor.fetchall():
        print(f"      {row['individual_id']} ({row['species']}) → индекс {row['embedding_index']}")
    
    conn.close()


if __name__ == "__main__":
    # Построение индекса
    build_faiss_index()
    
    # Проверка результатов
    verify_index()
    
    print("\n✅ FAISS индекс готов!")
    print("👉 Следующий шаг: тестирование поиска особей")