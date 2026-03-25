"""
Проверка корректности поиска FAISS + SQLite
"""

import torch
import numpy as np
from pathlib import Path
import sqlite3
import faiss
from torchvision import transforms
from PIL import Image

from pipeline.deployment_vit import load_model, get_embedding, find_similar_images

DB_PATH = "database/cards.db"
FAISS_INDEX_PATH = "data/embeddings/database_embeddings.pkl"
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def test_embedding_consistency():
    """
    Проверить что эмбеддинги одинаковые при многократном проходе.
    
    Returns:
        bool: True если тест пройден
    """
    print("1. Проверка консистентности эмбеддингов...")
    
    model = load_model(MODEL_PATH, DEVICE)
    
    test_img = Path("data/dataset_crop/dataset_crop_24/karelin/1/01-01-1224.JPG")
    if not test_img.exists():
        print(f"   Файл не найден: {test_img}")
        return False
    
    emb1 = get_embedding(str(test_img), model, TRANSFORMS, DEVICE)
    emb2 = get_embedding(str(test_img), model, TRANSFORMS, DEVICE)
    
    if emb1 is None or emb2 is None:
        print("   Ошибка получения эмбеддинга")
        return False
    
    diff = np.abs(emb1 - emb2).max()
    print(f"   Макс. разница: {diff:.2e}")
    print(f"   Статус: {'OK' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def test_faiss_index_integrity():
    """
    Проверить целостность FAISS индекса и связь с БД.
    
    Returns:
        bool: True если тест пройден
    """
    print("\n2. Проверка целостности FAISS индекса...")
    
    if not Path(FAISS_INDEX_PATH).exists():
        print(f"   FAISS индекс не найден: {FAISS_INDEX_PATH}")
        return False
    
    index = faiss.read_index(FAISS_INDEX_PATH)
    faiss_count = index.ntotal
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index != -1")
    db_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index >= ?", (faiss_count,))
    invalid_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"   Векторов в FAISS: {faiss_count}")
    print(f"   Записей в БД: {db_count}")
    print(f"   Неверных индексов: {invalid_count}")
    print(f"   Статус: {'OK' if faiss_count == db_count and invalid_count == 0 else 'FAIL'}")
    
    return faiss_count == db_count and invalid_count == 0


def test_search_results():
    """
    Проверить результаты поиска на тестовом изображении.
    
    Returns:
        bool: True если тест пройден
    """
    print("\n3. Проверка поиска...")
    
    query_image = "data/dataset_crop/dataset_crop_24/karelin/1/01-01-1224.JPG"
    
    if not Path(query_image).exists():
        print(f"   Файл не найден: {query_image}")
        return False
    
    results = find_similar_images(
        model_path=MODEL_PATH,
        database_dir="data/dataset_crop/dataset_crop_24",
        query_image_path=query_image,
        output_dir="data/output/test_verify",
        transform=TRANSFORMS,
        device=DEVICE,
        size_answer=5
    )
    
    if not results:
        print("   Поиск не вернул результатов")
        return False
    
    print(f"   Найдено результатов: {len(results)}")
    
    similarities = [r['similarity'] for r in results]
    is_sorted = all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
    
    print(f"   Результаты отсортированы: {'OK' if is_sorted else 'FAIL'}")
    print(f"   Диапазон similarity: {min(similarities):.1f}% - {max(similarities):.1f}%")
    
    print("\n   Топ-3 результата:")
    for r in results[:3]:
        print(f"      {r['rank']}. {r['individual_id']} ({r['species']}) — {r['similarity']:.1f}%")
    
    return len(results) == 5 and is_sorted


def test_same_image_search():
    """
    Поиск изображения самого себя (должно быть ~100% совпадение).
    
    Returns:
        bool: True если тест пройден
    """
    print("\n4. Поиск изображения самого себя...")
    
    query_image = "data/dataset_crop/dataset_crop_24/karelin/1/01-01-1224.JPG"
    
    if not Path(query_image).exists():
        print(f"   Файл не найден: {query_image}")
        return False
    
    results = find_similar_images(
        model_path=MODEL_PATH,
        database_dir="data/dataset_crop/dataset_crop_24",
        query_image_path=query_image,
        output_dir="data/output/test_self",
        transform=TRANSFORMS,
        device=DEVICE,
        size_answer=1
    )
    
    if not results:
        print("   Результаты пустые")
        return False
    
    similarity = results[0]['similarity']
    print(f"   Первое совпадение: {similarity:.1f}%")
    print(f"   Статус: {'OK' if similarity > 95 else 'FAIL'}")
    
    return similarity > 95


if __name__ == "__main__":
    print("Верификация системы поиска тритонов\n")
    print("=" * 60)
    
    test1 = test_embedding_consistency()
    test2 = test_faiss_index_integrity()
    test3 = test_search_results()
    test4 = test_same_image_search()
    
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print(f"   Консистентность эмбеддингов: {'OK' if test1 else 'FAIL'}")
    print(f"   Целостность FAISS индекса: {'OK' if test2 else 'FAIL'}")
    print(f"   Корректность поиска: {'OK' if test3 else 'FAIL'}")
    print(f"   Поиск самого себя: {'OK' if test4 else 'FAIL'}")
    print("=" * 60)
    
    if all([test1, test2, test3, test4]):
        print("\nВсе тесты пройдены. Система работает корректно.")
    else:
        print("\nОбнаружены проблемы. Требуется отладка.")