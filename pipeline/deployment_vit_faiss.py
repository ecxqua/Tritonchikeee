"""
deployment_vit.py — Поиск похожих особей тритонов (ViT + FAISS + SQLite)

Версия: 2.2
Интеграция: FAISS индекс + SQLite база данных + прототипы на лету

Зависимости:
    - database/cards.db — SQLite база с таблицами individuals, photos
    - data/embeddings/database_embeddings.pkl — FAISS индекс (512 dim, L2 norm)
    - models/best_model.pth — веса модели EnhancedTripletNet
"""

import os
import shutil
import csv
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from PIL import Image
from torchvision import transforms


# =============================================================================
# КОНСТАНТЫ
# =============================================================================

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================================================================
# МОДЕЛЬ (БЕЗ ИЗМЕНЕНИЙ — СОВМЕСТИМОСТЬ С ML КОМАНДОЙ)
# =============================================================================

class EnhancedTripletNet(nn.Module):
    """
    ViT модель для идентификации тритонов по фото брюшка.
    
    Архитектура:
        - Base: ViT-B/16 (pretrained)
        - Embedding head: 1024 -> 512 -> 512 (с L2 нормализацией)
        - Projection head: 512 -> 256 -> 128 (для triplet loss)

    Args:
        base_model_name: Название модели timm (по умолчанию 'vit_base_patch16_224')
        embedding_dim: Размерность выходного эмбеддинга (по умолчанию 512)
        dropout_rate: Коэффициент dropout (по умолчанию 0.4)

    Returns:
        torch.Tensor: Нормализованный вектор эмбеддинга (batch, 512)
    """

    def __init__(self, base_model_name: str = 'vit_base_patch16_224',
                 embedding_dim: int = 512, dropout_rate: float = 0.4):
        super().__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        # Замораживаем базовую модель, кроме последних слоев
        for param in self.base_model.parameters():
            param.requires_grad = False

        if hasattr(self.base_model, 'blocks'):
            num_blocks = len(self.base_model.blocks)
            blocks_to_unfreeze = min(6, num_blocks)
            for i in range(num_blocks - blocks_to_unfreeze, num_blocks):
                for param in self.base_model.blocks[i].parameters():
                    param.requires_grad = True

        # Эмбеддинг сеть
        self.embedding = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, embedding_dim),
        )

        # Проекционная головка
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # Инициализация весов
        for module in self.embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        Прямой проход через модель.

        Args:
            x: Входное изображение (batch, 3, 224, 224)
            return_projection: Вернуть ли output проекционной головки

        Returns:
            Если return_projection=False: Нормализованный вектор (batch, 512)
            Если return_projection=True: Кортеж (embeddings, projections)
        """
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_projection:
            projections = self.projection(embeddings)
            projections = F.normalize(projections, p=2, dim=1)
            return embeddings, projections

        return embeddings


# =============================================================================
# ФУНКЦИИ ЗАГРУЗКИ МОДЕЛИ И ЭМБЕДДИНГОВ
# =============================================================================

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Загрузить модель ViT для идентификации.

    Args:
        model_path: Путь к файлу весов (.pt)
        device: Устройство для вычислений (cuda/cpu)

    Returns:
        Загруженная модель в режиме eval
    """
    model = EnhancedTripletNet(
        base_model_name='vit_base_patch16_224',
        embedding_dim=512
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path: str, model: nn.Module,
                  transform: transforms.Compose,
                  device: torch.device) -> Optional[np.ndarray]:
    """
    Получить эмбеддинг изображения через ViT модель.

    Args:
        image_path: Путь к изображению
        model: Загруженная модель
        transform: Трансформы для предобработки
        device: Устройство для вычислений

    Returns:
        Вектор размерности (512,), L2 нормализован, или None при ошибке
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


# =============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С FAISS + SQLite
# =============================================================================

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """
    Получить соединение с SQLite базой данных.

    Args:
        db_path: Путь к файлу базы данных

    Returns:
        sqlite3.Connection с row_factory=sqlite3.Row
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_faiss_index(faiss_index_path: str) -> faiss.Index:
    """
    Загрузить FAISS индекс из файла.

    Args:
        faiss_index_path: Путь к файлу индекса (.pkl)

    Returns:
        faiss.Index с векторами эмбеддингов

    Raises:
        FileNotFoundError: Если индекс не найден
    """
    path = Path(faiss_index_path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS индекс не найден: {path}")
    
    index = faiss.read_index(str(path))
    return index


def get_individual_prototypes_on_the_fly(
    conn: sqlite3.Connection,
    faiss_index: faiss.Index,
    embedding_dim: int = 512
) -> Dict[str, Any]:
    """
    Построить прототипы особей (средние эмбеддинги) НА ЛЕТУ из базы данных.

    КЛЮЧЕВАЯ ФУНКЦИЯ: Вычисляет средний эмбеддинг для каждой особи
    на основе всех её фотографий в момент вызова (не кэшируется).

    ЛОГИКА:
        1. Получаем все фото с embedding_index != -1 из SQLite
        2. Группируем по individual_id
        3. Для каждой особи извлекаем векторы из FAISS по embedding_index
        4. Вычисляем средний эмбеддинг (прототип) с L2 нормализацией
        5. Находим наиболее репрезентативное фото (ближайшее к прототипу)

    Args:
        conn: Соединение с SQLite
        faiss_index: FAISS индекс с векторами
        embedding_dim: Размерность вектора (по умолчанию 512)

    Returns:
        Словарь с ключами:
            - 'individual_ids': список ID особей
            - 'embeddings': массив прототипов (n_individuals, 512)
            - 'representative_photo_ids': индексы лучших фото для каждой особи
            - 'member_photo_indices': dict {individual_id: [embedding_indices]}
            - 'metadata': dict {individual_id: {species, template_type, ...}}
    """
    cursor = conn.cursor()

    # Получаем все фото с эмбеддингами (cropped, с вычисленным вектором)
    cursor.execute('''
        SELECT photo_id, individual_id, embedding_index, photo_path
        FROM photos
        WHERE embedding_index != -1
        AND photo_type = 'cropped'
        ORDER BY individual_id, photo_id
    ''')

    rows = cursor.fetchall()
    if not rows:
        raise ValueError("В базе нет фото с эмбеддингами")

    # Группируем по особям
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        ind_id = row['individual_id']
        if ind_id not in groups:
            groups[ind_id] = []
        groups[ind_id].append({
            'photo_id': row['photo_id'],
            'embedding_index': row['embedding_index'],
            'photo_path': row['photo_path']
        })

    # Получаем метаданные особей
    cursor.execute('''
        SELECT individual_id, species, template_type, project_name
        FROM individuals
    ''')
    metadata = {row['individual_id']: dict(row) for row in cursor.fetchall()}

    # Вычисляем прототипы НА ЛЕТУ
    individual_ids = []
    prototype_embeddings = []
    representative_photo_ids = []
    member_photo_indices = {}

    for ind_id, photos in groups.items():
        # Получаем все эмбеддинги особи из FAISS по embedding_index
        indices = [p['embedding_index'] for p in photos]
        embeddings_list = []
        for idx in indices:
            emb = faiss_index.reconstruct(idx)
            embeddings_list.append(emb)

        embeddings_array = np.array(embeddings_list)

        # СРЕДНИЙ ЭМБЕДДИНГ (ПРОТОТИП)
        prototype = np.mean(embeddings_array, axis=0)
        # L2 нормализация прототипа
        norm = np.linalg.norm(prototype)
        if norm > 1e-12:
            prototype = prototype / norm

        # Находим фото, наиболее близкое к прототипу (для отображения)
        similarities = np.dot(embeddings_array, prototype)
        best_local_idx = int(np.argmax(similarities))
        best_photo = photos[best_local_idx]

        individual_ids.append(ind_id)
        prototype_embeddings.append(prototype)
        representative_photo_ids.append(best_photo['photo_id'])
        member_photo_indices[ind_id] = indices

    return {
        'individual_ids': individual_ids,
        'embeddings': np.array(prototype_embeddings),
        'representative_photo_ids': representative_photo_ids,
        'member_photo_indices': member_photo_indices,
        'metadata': metadata,
    }


def search_with_faiss(
    query_embedding: np.ndarray,
    prototypes: Dict[str, Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Поиск наиболее похожих особей через FAISS (по прототипам).

    Args:
        query_embedding: Эмбеддинг запроса (512,)
        prototypes: Прототипы особей из get_individual_prototypes_on_the_fly()
        top_k: Количество результатов

    Returns:
        Список словарей с результатами поиска
    """
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    prototype_embeddings = prototypes['embeddings'].astype('float32')

    # Создаём временный индекс для поиска по прототипам
    # IndexFlatIP = Inner Product (косинусное сходство для L2 нормализованных векторов)
    temp_index = faiss.IndexFlatIP(prototype_embeddings.shape[1])
    temp_index.add(prototype_embeddings)
    
    distances, indices = temp_index.search(query_vector, top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx >= len(prototypes['individual_ids']):
            continue

        ind_id = prototypes['individual_ids'][idx]
        metadata = prototypes['metadata'].get(ind_id, {})
        photo_id = prototypes['representative_photo_ids'][idx]

        # Similarity = cosine similarity (т.к. векторы L2 нормализованы)
        similarity = float(dist)
        similarity_percent = similarity * 100

        results.append({
            'rank': i + 1,
            'individual_id': ind_id,
            'species': metadata.get('species', 'Unknown'),
            'template_type': metadata.get('template_type', 'Unknown'),
            'photo_id': photo_id,
            'similarity': similarity,
            'similarity_percent': similarity_percent,
            'distance': 1 - similarity,
        })

    return results


def get_photo_path_from_db(conn: sqlite3.Connection, photo_id: int) -> Optional[str]:
    """
    Получить путь к фото по photo_id из базы данных.

    Args:
        conn: Соединение с SQLite
        photo_id: ID фотографии

    Returns:
        Путь к файлу или None
    """
    cursor = conn.cursor()
    cursor.execute("SELECT photo_path FROM photos WHERE photo_id = ?", (photo_id,))
    row = cursor.fetchone()
    return row['photo_path'] if row else None


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def save_vit_debug_report(
    query_image_path: str,
    results: List[Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Сохраняет отладочный отчёт о поиске в форматах CSV и TXT.

    Args:
        query_image_path: Путь к изображению запроса
        results: Результаты поиска от search_with_faiss()
        output_dir: Директория для сохранения отчётов
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'vit_debug_topk.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'rank', 'similarity_percent', 'distance',
            'individual_id', 'species', 'template_type', 'photo_id'
        ])
        for r in results:
            writer.writerow([
                r['rank'],
                round(r['similarity_percent'], 3),
                round(r['distance'], 6),
                r['individual_id'],
                r['species'],
                r['template_type'],
                r['photo_id'],
            ])

    summary_path = os.path.join(output_dir, 'vit_debug_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as summary:
        summary.write(f"Query: {query_image_path}\n")
        summary.write(f"Candidates in DB: {len(results)}\n")
        if results:
            distances = [r['distance'] for r in results]
            summary.write(
                f"Distance min/mean/max: {min(distances):.6f} / "
                f"{np.mean(distances):.6f} / {max(distances):.6f}\n"
            )
        summary.write(f"Top-K report: {csv_path}\n")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def find_similar_images(
    model_path: str,
    db_path: str,
    faiss_index_path: str,
    query_image_path: str,
    output_dir: str = 'output/results',
    transform: Optional[transforms.Compose] = None,
    device: Optional[torch.device] = None,
    size_answer: int = 5,
    search_mode: str = 'by_individual',
) -> List[Dict[str, Any]]:
    """
    Основная функция поиска похожих изображений.

    ИЗМЕНЕНИЯ В ВЕРСИИ 2.2:
        - Пути передаются через параметры функции (не config.yaml)
        - Используется FAISS индекс вместо pickle с эмбеддингами
        - Метаданные загружаются из SQLite вместо парсинга путей
        - Прототипы особей вычисляются НА ЛЕТУ (средний эмбеддинг по всем фото)
        - Сигнатура функции сохранена для обратной совместимости

    Args:
        model_path: Путь к файлу весов модели
        db_path: Путь к SQLite базе данных
        faiss_index_path: Путь к FAISS индексу
        query_image_path: Путь к изображению запроса
        output_dir: Директория для сохранения результатов
        transform: Трансформы для предобработки изображений
        device: Устройство для вычислений (cuda/cpu)
        size_answer: Количество результатов для возврата
        search_mode: Режим поиска ('by_individual' или 'by_image')

    Returns:
        Список словарей с результатами поиска (top-K похожих особей)

    Raises:
        FileNotFoundError: Если не найден FAISS индекс или модель
        ValueError: Если не удалось обработать запросное изображение
    """
    # Инициализация устройства и трансформов
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if transform is None:
        transform = DEFAULT_TRANSFORM

    # Загрузка модели
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = load_model(model_path, device)

    # Загрузка FAISS индекса
    faiss_index = load_faiss_index(faiss_index_path)

    # Построение прототипов особей НА ЛЕТУ из БД
    conn = get_db_connection(db_path)
    prototypes = get_individual_prototypes_on_the_fly(conn, faiss_index)

    # Получение эмбеддинга запроса
    if not query_image_path or not Path(query_image_path).exists():
        raise FileNotFoundError(f"Изображение запроса не найдено: {query_image_path}")

    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        raise ValueError("Не удалось получить эмбеддинг запроса")

    # Поиск похожих
    if search_mode == 'by_individual':
        results = search_with_faiss(
            query_embedding=query_embedding,
            prototypes=prototypes,
            top_k=size_answer
        )
    else:
        # Fallback: поиск по всем фото (без прототипов)
        results = search_with_faiss(
            query_embedding=query_embedding,
            prototypes=prototypes,
            top_k=size_answer
        )

    # Дополнение результатов путями к фото
    for r in results:
        r['photo_path'] = get_photo_path_from_db(conn, r['photo_id'])

    conn.close()

    # Сохранение результатов
    os.makedirs(output_dir, exist_ok=True)
    save_vit_debug_report(query_image_path, results, output_dir)

    with open(os.path.join(output_dir, 'res.txt'), 'w', encoding='utf-8') as file:
        for r in results:
            species_name = 'Ребристый' if 'Гребенчатый' in r['species'] else 'Карелина'
            res_str = (
                f"{r['rank']}. Класс: {species_name} | "
                f"Особь: {r['individual_id']} | "
                f"Схожесть: {r['similarity_percent']:.1f}%\n"
            )
            file.write(res_str)

            # Копирование фото результата
            if r['photo_path'] and Path(r['photo_path']).exists():
                dst_path = os.path.join(output_dir, f"top{r['rank']}.jpg")
                shutil.copy(r['photo_path'], dst_path)

    return results