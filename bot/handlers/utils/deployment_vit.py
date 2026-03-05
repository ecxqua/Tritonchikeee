import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv
import hashlib
from tqdm import tqdm
import shutil
import torch.nn as nn
import torch.nn.functional as F
import timm
import pickle
from torchvision import transforms


class EnhancedTripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=512, dropout_rate=0.4):
        super().__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        # Размораживаем слои
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

            nn.Dropout(dropout_rate/2),
            nn.Linear(512, embedding_dim),
        )

        # Проекционная головка
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128)
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
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if return_projection:
            projections = self.projection(embeddings)
            projections = F.normalize(projections, p=2, dim=1)
            return embeddings, projections

        return embeddings


def load_model(model_path, device):
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
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Пропущен ключ {k}")
        else:
            print(f"Ключ {k} не найден в модели")

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path, model, transform, device):
    """
    получение эмбедингов
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def compute_distances(embeddings1, embedding2):
    """
    вычисление косинусного расстояния
    """
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    return 1 - similarities.flatten()


def save_embeddings(embeddings, paths, save_path):
    """сохранение эмбедингов в бызу данных database_embeddings.pkl"""
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'paths': paths}, f)


def load_embeddings(save_path):
    """загрузка эмбеддингов из базы данных"""
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['paths']


def _file_sha256(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect_database_image_paths(database_dir):
    image_paths = []
    for root, _, files in os.walk(database_dir):
        if any(x in root.lower() for x in ['__pycache__', '.git', 'results']):
            continue

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                full_path = os.path.join(root, file)
                if os.path.exists(full_path):
                    image_paths.append(full_path)

    image_paths.sort()
    return image_paths


def _database_signature(image_paths):
    hasher = hashlib.sha256()
    hasher.update(str(len(image_paths)).encode('utf-8'))

    for path in image_paths:
        hasher.update(path.encode('utf-8'))
        try:
            mtime = os.path.getmtime(path)
            hasher.update(str(mtime).encode('utf-8'))
        except OSError:
            continue

    return hasher.hexdigest()


def _transform_signature(transform):
    return hashlib.sha256(str(transform).encode('utf-8')).hexdigest()


def _build_cache_metadata(model_path, database_paths, transform):
    return {
        'model_hash': _file_sha256(model_path),
        'database_signature': _database_signature(database_paths),
        'transform_signature': _transform_signature(transform),
        'num_images': len(database_paths),
    }


def _save_embeddings_with_metadata(embeddings, paths, save_path, metadata):
    with open(save_path, 'wb') as file_obj:
        pickle.dump(
            {
                'embeddings': embeddings,
                'paths': paths,
                'metadata': metadata,
            },
            file_obj,
        )


def _load_embeddings_with_metadata(save_path):
    with open(save_path, 'rb') as file_obj:
        data = pickle.load(file_obj)

    embeddings = data['embeddings']
    paths = data['paths']
    metadata = data.get('metadata')
    return embeddings, paths, metadata


def extract_metadata_from_path(image_path):
    class_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    individual = os.path.basename(os.path.dirname(image_path))
    return class_name, individual


def save_vit_debug_report(query_image_path, database_image_paths, distances, output_dir, top_k=20):
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


def compute_database_embeddings(model, database_dir, transform, device, embeddings_save_path):
    """
    Вычисление эмбеддингов для всей базы данных
    """
    database_image_paths = []
    for root, _, files in os.walk(database_dir):
        # Пропускаем системные файлы
        if any(x in root.lower() for x in ['__pycache__', '.git', 'results']):
            continue

        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                full_path = os.path.join(root, file)
                if os.path.exists(full_path):
                    database_image_paths.append(full_path)

    print(f"Найдено {len(database_image_paths)} изображений в базе")

    # Вычисление эмбеддингов
    database_embeddings = []
    valid_paths = []

    for path in tqdm(database_image_paths, desc="Обработка базы"):
        try:
            image = Image.open(path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(image)

            database_embeddings.append(embedding.cpu().numpy().flatten())
            valid_paths.append(path)

        except Exception as e:
            print(f"⚠️ Ошибка обработки {path}: {str(e)}")

    if not valid_paths:
        raise ValueError("Не удалось обработать ни одно изображение из базы")

    database_embeddings = np.array(database_embeddings)

    # Сохраняем эмбеддинги
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    save_embeddings(database_embeddings, valid_paths, embeddings_save_path)

    return database_embeddings, valid_paths


def compute_database_embeddings_with_paths(model, database_image_paths, transform, device):
    database_embeddings = []
    valid_paths = []

    for path in tqdm(database_image_paths, desc="Обработка базы"):
        emb = get_embedding(path, model, transform, device)
        if emb is not None:
            database_embeddings.append(emb)
            valid_paths.append(path)

    if not valid_paths:
        raise ValueError("Не удалось обработать ни одно изображение из базы")

    return np.array(database_embeddings), valid_paths


def find_similar_images(model_path, database_dir, query_image_path, output_dir, transform, device, bot):
    """
    основная функция поиска похожих изображений по косинусной схожести

    Процесс:
    1)Загрузка/вычисление эмбеддингов базы данных
    2)Получение эмбеддинга запросного изображения
    3)Вычисление схожести и выбор топ-5 результатов
    4.Сохранение результатов и метаинформации

    """
    # путь для эмбеддингов базы данных
    embeddings_save_path = 'bot/dataset_crop/database_embeddings.pkl'
    database_image_paths = _collect_database_image_paths(database_dir)
    current_metadata = _build_cache_metadata(model_path, database_image_paths, transform)
    use_cache = False

    # проверка существования эмбеддингов
    if os.path.exists(embeddings_save_path):
        print("Найден кэш эмбеддингов, проверяем актуальность...")
        try:
            database_embeddings, cached_paths, cached_metadata = _load_embeddings_with_metadata(
                embeddings_save_path
            )
            if cached_metadata == current_metadata and len(cached_paths) == len(database_image_paths):
                print("Кэш эмбеддингов актуален, используем его")
                database_image_paths = cached_paths
                use_cache = True
            else:
                print("Кэш эмбеддингов устарел, пересчитываем...")
        except Exception as error:
            print(f"Не удалось прочитать кэш эмбеддингов: {error}. Пересчитываем...")

    if not use_cache:
        print("Вычисление эмбеддингов базы...")
        print(f"Найдено {len(database_image_paths)} изображений в базе")
        model = load_model(model_path, device)
        database_embeddings, valid_paths = compute_database_embeddings_with_paths(
            model,
            database_image_paths,
            transform,
            device,
        )
        current_metadata = _build_cache_metadata(model_path, valid_paths, transform)
        _save_embeddings_with_metadata(
            database_embeddings,
            valid_paths,
            embeddings_save_path,
            current_metadata,
        )
        print(f"Эмбеддинги сохранены в {embeddings_save_path}")
        database_image_paths = valid_paths

    # загрузка модели и получение эмбеддинга запросного изображения
    model = load_model(model_path, device)
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print("Не удалось обработать запросное изображение")
        return

    # вычисление расстояний и выбор топ-5 результатов
    distances = compute_distances(np.array(database_embeddings), query_embedding)
    save_vit_debug_report(query_image_path, database_image_paths, distances, output_dir, top_k=20)
    top5_idx = np.argsort(distances)[:bot.size_answer]  # bot.size_answer обычно = 5

    print("\n=== Топ-5 результатов ===")
    os.makedirs(output_dir, exist_ok=True)

    # сохранение результатов и копирование изображений
    with open(output_dir + "/res.txt", 'w', encoding='utf-8') as file:
        for i, idx in enumerate(top5_idx, 1):
            src_path = database_image_paths[idx]
            dst_filename = f"top{i}.jpg"
            dst_path = os.path.join(output_dir, dst_filename)

            try:
                # копирование изображения в директорию результатов
                shutil.copy(src_path, dst_path)

                # извлечение метаинформации из пути
                class_name, individual = extract_metadata_from_path(src_path)

                # расчет процента схожести
                similarity = 1 - distances[idx]  # Косинусная схожесть
                similarity_percent = similarity * 100

                # преобразование имени класса в читаемый формат
                class_string = 'Ребристый' if class_name.startswith('ribbed')  else "Карелина"

                # формирование строки результата
                res_str = f"{i}. Класс: {class_string} | Особь: {individual} | Схожесть: {similarity_percent:.1f}%\n"
                file.write(res_str)
                print(
                    f"{i}. Класс: {class_name} | Особь: {individual} | Схожесть: {similarity_percent:.1f}% | Путь: {src_path}")
            except Exception as e:
                print(f"Ошибка копирования файла {src_path}: {str(e)}")
        print(f"\nРезультаты сохранены в: {output_dir}")