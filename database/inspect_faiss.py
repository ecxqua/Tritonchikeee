"""
Просмотр содержимого FAISS индекса с векторами
"""

import sqlite3
import faiss
import numpy as np
from pathlib import Path

DB_PATH = Path("database/cards.db")
FAISS_PATH = Path("data/embeddings/database_embeddings.pkl")

# Загрузить индекс
index = faiss.read_index(str(FAISS_PATH))
print(f"Векторов в FAISS: {index.ntotal}")
print(f"Размерность: {index.d}")

# Подключиться к БД
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Показать первые 10 записей с векторами
print("\n" + "=" * 100)
print(f"{'FAISS idx':<10} {'Особь':<15} {'Вид':<12} {'Вектор (первые 8 значений)':<60}")
print("=" * 100)

cursor.execute("""
    SELECT p.embedding_index, p.individual_id, p.photo_path, i.species
    FROM photos p
    JOIN individuals i ON p.individual_id = i.individual_id
    WHERE p.embedding_index != -1
    ORDER BY p.embedding_index
    LIMIT 10
""")

for row in cursor.fetchall():
    idx = row['embedding_index']
    
    # Извлечь вектор из FAISS
    if hasattr(index, 'xb'):
        vec = index.xb[idx].reshape(index.d)
        vec_str = f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, {vec[3]:.3f}, {vec[4]:.3f}, {vec[5]:.3f}, {vec[6]:.3f}, {vec[7]:.3f}...]"
    else:
        # Для некоторых типов индексов нужен reconstruct
        vec = index.reconstruct(idx)
        vec_str = f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, {vec[3]:.3f}, {vec[4]:.3f}, {vec[5]:.3f}, {vec[6]:.3f}, {vec[7]:.3f}...]"
    
    print(f"{idx:<10} {row['individual_id']:<15} {row['species']:<12} {vec_str:<60}")

# Проверка синхронизации
cursor.execute("SELECT COUNT(*) FROM photos WHERE embedding_index != -1")
db_count = cursor.fetchone()[0]

conn.close()

print("=" * 100)
print(f"Записей в БД: {db_count}")
print(f"Векторов в FAISS: {index.ntotal}")
print(f"Статус: {'✅ Синхронизировано' if db_count == index.ntotal else '❌ Рассинхрон'}")

# Статистика векторов
print("\n" + "=" * 100)
print("СТАТИСТИКА ВЕКТОРОВ")
print("=" * 100)

if hasattr(index, 'xb'):
    vectors = index.xb[:index.ntotal].reshape(index.ntotal, index.d)
    print(f"L2 норма (средняя): {np.mean([np.linalg.norm(v) for v in vectors]):.4f}")
    print(f"L2 норма (мин): {np.min([np.linalg.norm(v) for v in vectors]):.4f}")
    print(f"L2 норма (макс): {np.max([np.linalg.norm(v) for v in vectors]):.4f}")
    print(f"Значения (среднее): {np.mean(vectors):.4f}")
    print(f"Значения (std): {np.std(vectors):.4f}")
    print(f"Значения (min): {np.min(vectors):.4f}")
    print(f"Значения (max): {np.max(vectors):.4f}")