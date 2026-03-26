import hashlib
import os
import pickle
import shutil
import csv
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torchvision import transforms
from typing import Optional, List, Dict
import sqlite3
import faiss


# ============================================================================
# МОДЕЛЬ
# ============================================================================

class EnhancedTripletNet(nn.Module):
    """
    ViT модель для идентификации тритонов по фото брюшка.
    
    Архитектура:
        - Base: ViT-B/16 (pretrained)
        - Embedding head: 1024 -> 512 -> 512 (с нормализацией)
        - Projection head: 512 -> 256 -> 128 (для triplet loss)
    
    Args:
        base_model_name: Название модели timm
        embedding_dim: Размерность выходного эмбеддинга
        dropout_rate: Коэффициент dropout
    """
    
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=512, dropout_rate=0.4):
        """
        Инициализация модели EnhancedTripletNet.
        
        Загружает базовую модель ViT, настраивает голову эмбеддинга и проекции,
        а также управляет заморозкой/разморозкой слоев базовой модели.
        
        Args:
            base_model_name (str): Название архитектуры модели из библиотеки timm.
            embedding_dim (int): Размерность выходного вектора эмбеддинга.
            dropout_rate (float): Вероятность обнуления элементов для регуляризации.
        """
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

    def forward(self, x, return_projection=False):
        """
        Прямой проход через модель.
        
        Args:
            x (torch.Tensor): Входное изображение (batch, 3, 224, 224).
            return_projection (bool): Вернуть ли output проекционной головки.
        
        Returns:
            torch.Tensor: 
                Если return_projection=False: Нормализованный вектор эмбеддинга (batch, 512).
                Если return_projection=True: Кортеж (embeddings, projections), где projections (batch, 128).
        """
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_projection:
            projections = self.projection(embeddings)
            projections = F.normalize(projections, p=2, dim=1)
            return embeddings, projections

        return embeddings


# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ОБРАБОТКИ
# ============================================================================

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Загрузить модель ViT для идентификации.
    
    Args:
        model_path (str): Путь к файлу весов (.pt).
        device (torch.device): Устройство для вычислений (cuda/cpu).
    
    Returns:
        torch.nn.Module: Загруженная модель в режиме eval.
    """
    model = EnhancedTripletNet(base_model_name='vit_base_patch16_224', embedding_dim=512)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v
        else:
            print(f"Пропущен ключ {k} (несовпадение формы или имени)")

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path: str, model: torch.nn.Module, transform, device: torch.device) -> Optional[np.ndarray]:
    """
    Получить эмбеддинг изображения через ViT модель.
    
    Args:
        image_path (str): Путь к изображению.
        model (torch.nn.Module): Загруженная модель.
        transform (torchvision.transforms): Трансформы для предобработки.
        device (torch.device): Устройство для вычислений.
    
    Returns:
        Optional[np.ndarray]: Вектор размерности (512,), L2 нормализован, или None при ошибке.
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


def compute_distances(embeddings1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
    """
    Вычислить расстояния между векторами (cosine distance).
    
    Args:
        embeddings1 (np.ndarray): Массив векторов (n, 512).
        embedding2 (np.ndarray): Единичный вектор (512,).
    
    Returns:
        np.ndarray: Массив расстояний (n,), где расстояние = 1 - cosine_similarity.
    """
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    return 1 - similarities.flatten()


def _collect_database_image_paths(database_dir: str) -> List[str]:
    """
    Собирает все пути к изображениям в базе данных рекурсивно.
    
    Игнорирует системные папки (pycache, .git, results) и фильтрует по расширениям.
    
    Args:
        database_dir (str): Корневая директория базы изображений.
    
    Returns:
        List[str]: Отсортированный список полных путей к изображениям.
    """
    image_paths = []
    for root, _, files in os.walk(database_dir):
        if any(x in root.lower() for x in ['pycache', '.git', 'results']):
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                full_path = os.path.join(root, file)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
    image_paths.sort()
    return image_paths


def _save_embeddings(embeddings: np.ndarray, paths: List[str], save_path: str) -> None:
    """
    Сохраняет эмбеддинги и соответствующие пути к файлам в бинарный файл (pickle).
    
    Args:
        embeddings (np.ndarray): Массив эмбеддингов.
        paths (List[str]): Список путей к изображениям.
        save_path (str): Путь для сохранения файла .pkl.
    
    Returns:
        None
    """
    with open(save_path, 'wb') as file_obj:
        pickle.dump({
            'embeddings': embeddings,
            'paths': paths,
        }, file_obj)


def _load_embeddings(save_path: str) -> tuple:
    """
    Загружает эмбеддинги и пути из бинарного файла (pickle).
    
    Args:
        save_path (str): Путь к файлу .pkl.
    
    Returns:
        tuple: Кортеж (embeddings, paths).
    """
    with open(save_path, 'rb') as file_obj:
        data = pickle.load(file_obj)
    return data['embeddings'], data['paths']


def extract_metadata_from_path(image_path: str) -> tuple:
    """
    Извлекает имя класса и особи из пути к файлу на основе структуры директорий.
    
    Ожидается структура: .../class_name/individual_name/filename.ext
    
    Args:
        image_path (str): Полный путь к изображению.
    
    Returns:
        tuple: (class_name, individual).
    """
    class_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    individual = os.path.basename(os.path.dirname(image_path))
    return class_name, individual


def _l2_normalize_np(vector: np.ndarray) -> np.ndarray:
    """
    Выполняет L2 нормализацию для numpy вектора.
    
    Args:
        vector (np.ndarray): Входной вектор.
    
    Returns:
        np.ndarray: Нормализованный вектор. Если норма близка к нулю, возвращается исходный вектор.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def _build_individual_prototypes(database_embeddings: np.ndarray, database_image_paths: List[str]) -> Dict:
    """
    Формирует прототип (средний L2-нормированный эмбеддинг) для каждой особи.
    
    Группирует эмбеддинги по особи, вычисляет среднее значение и находит
    наиболее репрезентативное изображение (ближайшее к прототипу).
    
    Args:
        database_embeddings (np.ndarray): Массив всех эмбеддингов базы.
        database_image_paths (List[str]): Список путей к изображениям базы.
    
    Returns:
        Dict: Словарь, содержащий:
            - 'keys': Список кортежей (class_name, individual).
            - 'embeddings': Массив эмбеддингов прототипов.
            - 'representative_indices': Индексы лучших изображений для каждого прототипа.
            - 'member_indices': Словарь соответствия особей и индексов их изображений.
    """
    groups = {}
    for idx, path in enumerate(database_image_paths):
        class_name, individual = extract_metadata_from_path(path)
        key = (class_name, individual)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    prototype_keys = []
    prototype_embeddings = []
    representative_indices = []

    for key, indices in groups.items():
        emb_stack = database_embeddings[indices]
        # Усредняем и нормализуем
        prototype = _l2_normalize_np(np.mean(emb_stack, axis=0))

        # Находим репрезентативный кадр (ближайший к прототипу)
        dists_to_proto = compute_distances(emb_stack, prototype)
        rep_local_idx = int(np.argmin(dists_to_proto))
        rep_global_idx = indices[rep_local_idx]

        prototype_keys.append(key)
        prototype_embeddings.append(prototype)
        representative_indices.append(rep_global_idx)

    return {
        'keys': prototype_keys,
        'embeddings': np.array(prototype_embeddings),
        'representative_indices': representative_indices,
        'member_indices': groups,
    }


def save_vit_debug_report(query_image_path: str, database_image_paths: List[str], 
                          distances: np.ndarray, output_dir: str, top_k: int = 20) -> None:
    """
    Сохраняет отладочный отчет о поиске в форматах CSV и TXT.
    
    Args:
        query_image_path (str): Путь к изображению запроса.
        database_image_paths (List[str]): Список путей к изображениям базы.
        distances (np.ndarray): Массив расстояний от запроса до базы.
        output_dir (str): Директория для сохранения отчетов.
        top_k (int): Количество лучших совпадений для записи в отчет.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    sorted_idx = np.argsort(distances)
    top_k = min(top_k, len(sorted_idx))

    csv_path = os.path.join(output_dir, 'vit_debug_topk.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['rank', 'similarity_percent', 'distance', 'class_name', 'individual', 'image_path'])

        for rank, idx in enumerate(sorted_idx[:top_k], 1):
            src_path = database_image_paths[idx]
            class_name, individual = extract_metadata_from_path(src_path)
            similarity = (1 - distances[idx]) * 100
            writer.writerow([
                rank,
                round(float(similarity), 3),
                round(float(distances[idx]), 6),
                class_name,
                individual,
                src_path,
            ])

    summary_path = os.path.join(output_dir, 'vit_debug_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as summary:
        summary.write(f"Query: {query_image_path}\n")
        summary.write(f"Candidates in DB: {len(database_image_paths)}\n")
        summary.write(f"Distance min/mean/max: {distances.min():.6f} / {distances.mean():.6f} / {distances.max():.6f}\n")
        summary.write(f"Top-K report: {csv_path}\n")


def compute_database_embeddings_with_paths(model: torch.nn.Module, database_image_paths: List[str], 
                                           transform, device: torch.device) -> tuple:
    """
    Вычисляет эмбеддинги для всех изображений базы данных.
    
    Args:
        model (torch.nn.Module): Модель для извлечения признаков.
        database_image_paths (List[str]): Список путей к изображениям.
        transform (torchvision.transforms): Трансформы для предобработки.
        device (torch.device): Устройство для вычислений.
    
    Returns:
        tuple: (database_embeddings, valid_paths) - массив эмбеддингов и список успешно обработанных путей.
    
    Raises:
        ValueError: Если не удалось обработать ни одно изображение.
    """
    database_embeddings = []
    valid_paths = []
    for path in tqdm(database_image_paths, desc='Обработка базы'):
        emb = get_embedding(path, model, transform, device)
        if emb is not None:
            database_embeddings.append(emb)
            valid_paths.append(path)

    if not valid_paths:
        raise ValueError('Не удалось обработать ни одно изображение из базы')

    return np.array(database_embeddings), valid_paths


def find_similar_images(
    model_path: str,
    database_dir: str,
    query_image_path: str,
    output_dir: str,
    transform,
    device: torch.device,
    size_answer: int,
    search_mode: str = 'by_individual',
    force_recompute_cache: bool = False,
) -> None:
    """
    Основная функция поиска похожих изображений.
    
    Выполняет поиск наиболее похожих особей или изображений в базе данных относительно запроса.
    Поддерживает кэширование эмбеддингов базы данных.
    
    Args:
        model_path (str): Путь к файлу весов модели.
        database_dir (str): Директория с базой изображений для поиска.
        query_image_path (str): Путь к изображению запроса.
        output_dir (str): Директория для сохранения результатов (копии изображений, отчеты).
        transform (torchvision.transforms): Трансформы для предобработки изображений.
        device (torch.device): Устройство для вычислений (cuda/cpu).
        size_answer (int): Количество результатов для возврата.
        search_mode (str): Режим поиска ('by_individual' - по прототипам особей, 'by_image' - по картинкам).
        force_recompute_cache (bool): Принудительно пересчитать кэш эмбеддингов базы.
    
    Returns:
        None: Результаты сохраняются в файлы в output_dir.
    """
    embeddings_save_path = os.path.join(database_dir, 'database_embeddings.pkl')
    database_image_paths = _collect_database_image_paths(database_dir)
    
    use_cache = False
    
    # Управление кэшем (упрощённое — только по существованию файла)
    if force_recompute_cache and os.path.exists(embeddings_save_path):
        try:
            os.remove(embeddings_save_path)
            print('Принудительный пересчет: старый кэш эмбеддингов удален')
        except OSError as error:
            print(f'Не удалось удалить кэш эмбеддингов: {error}')

    if os.path.exists(embeddings_save_path):
        print('Найден кэш эмбеддингов, используем его')
        try:
            database_embeddings, cached_paths = _load_embeddings(embeddings_save_path)
            database_image_paths = cached_paths
            use_cache = True
        except Exception as error:
            print(f'Не удалось прочитать кэш эмбеддингов: {error}. Пересчитываем...')

    if not use_cache:
        print('Вычисление эмбеддингов базы...')
        print(f'Найдено {len(database_image_paths)} изображений в базе')
        model = load_model(model_path, device)
         # 1. ВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ БАЗЫ И СОХРАНЕНИЕ СООТВЕТСТВУЮЩИХ ПУТЕЙ!
        database_embeddings, valid_paths = compute_database_embeddings_with_paths(
            model,
            database_image_paths,
            transform,
            device,
        )
         # 2. СОХРАНЕНИЕ (ВНЕСЕНИЕ В "БАЗУ")
        _save_embeddings(
            database_embeddings,
            valid_paths,
            embeddings_save_path,
        )
        print(f'Эмбеддинги сохранены в {embeddings_save_path}')
        database_image_paths = valid_paths

    # Загрузка модели для запроса
    model = load_model(model_path, device)
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print('Не удалось обработать запросное изображение')
        return

    # Вычисляем расстояния до всех изображений (для отладки и fallback)
    image_distances = compute_distances(np.array(database_embeddings), query_embedding)
    save_vit_debug_report(query_image_path, database_image_paths, image_distances, output_dir, top_k=20)

    # Выбор стратегии поиска
    if search_mode not in ['by_individual', 'by_image']:
        print(f"Неизвестный search_mode={search_mode}, используем by_individual")
        search_mode = 'by_individual'

    result_candidates = []

    if search_mode == 'by_individual':
        print("Режим поиска: by_individual (по прототипам особей)")
        prototypes = _build_individual_prototypes(np.array(database_embeddings), database_image_paths)
        prototype_distances = compute_distances(prototypes['embeddings'], query_embedding)
        top_proto_idx = np.argsort(prototype_distances)[:size_answer]

        for proto_idx in top_proto_idx:
            class_name, individual = prototypes['keys'][proto_idx]
            member_indices = prototypes['member_indices'][(class_name, individual)]
            # Показываем лучший кадр внутри выбранной особи относительно query
            best_member_idx = min(member_indices, key=lambda idx: image_distances[idx])
            result_candidates.append({
                'src_path': database_image_paths[best_member_idx],
                'class_name': class_name,
                'individual': individual,
                'distance': float(prototype_distances[proto_idx]),
            })
    else:
        print("Режим поиска: by_image (по отдельным изображениям)")
        top_idx = np.argsort(image_distances)[:size_answer]
        for idx in top_idx:
            src_path = database_image_paths[idx]
            class_name, individual = extract_metadata_from_path(src_path)
            result_candidates.append({
                'src_path': src_path,
                'class_name': class_name,
                'individual': individual,
                'distance': float(image_distances[idx]),
            })

    # Сохранение результатов
    print('\n=== Топ результатов ===')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'res.txt'), 'w', encoding='utf-8') as file:
        for i, candidate in enumerate(result_candidates, 1):
            src_path = candidate['src_path']
            dst_filename = f'top{i}.jpg'
            dst_path = os.path.join(output_dir, dst_filename)

            try:
                shutil.copy(src_path, dst_path)
                class_name = candidate['class_name']
                individual = candidate['individual']
                similarity = 1 - candidate['distance']
                similarity_percent = similarity * 100

                class_string = 'Ребристый' if class_name.startswith('ribbed') else 'Карелина'
                res_str = (
                    f'{i}. Класс: {class_string} | Особь: {individual} | '
                    f'Схожесть: {similarity_percent:.1f}%\n'
                )
                file.write(res_str)
                print(
                    f'{i}. Класс: {class_name} | Особь: {individual} | '
                    f'Схожесть: {similarity_percent:.1f}% | Путь: {src_path}'
                )
            except Exception as e:
                print(f'Ошибка копирования файла {src_path}: {str(e)}')

    print(f'\nРезультаты сохранены в: {output_dir}')
    