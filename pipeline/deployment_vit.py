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

class EnhancedTripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=512, dropout_rate=0.4):
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
            filtered_state_dict[k] = v
        else:
            print(f"Пропущен ключ {k} (несовпадение формы или имени)")

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def compute_distances(embeddings1, embedding2):
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    return 1 - similarities.flatten()


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
        if any(x in root.lower() for x in ['pycache', '.git', 'results']):
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
        pickle.dump({
            'embeddings': embeddings,
            'paths': paths,
            'metadata': metadata,
        }, file_obj)


def _load_embeddings_with_metadata(save_path):
    with open(save_path, 'rb') as file_obj:
        data = pickle.load(file_obj)
    return data['embeddings'], data['paths'], data.get('metadata')


def extract_metadata_from_path(image_path):
    """Извлекает имя класса и особи из пути к файлу."""
    class_name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    individual = os.path.basename(os.path.dirname(image_path))
    return class_name, individual


def _l2_normalize_np(vector):
    """L2 нормализация для numpy вектора."""
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def _build_individual_prototypes(database_embeddings, database_image_paths):
    """Формирует прототип (средний L2-нормированный эмбеддинг) для каждой особи."""
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


def save_vit_debug_report(query_image_path, database_image_paths, distances, output_dir, top_k=20):
    """Сохраняет отладочный отчет в CSV и TXT."""
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


def compute_database_embeddings_with_paths(model, database_image_paths, transform, device):
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
    model_path,
    database_dir,
    query_image_path,
    output_dir,
    transform,
    device,
    size_answer,
    search_mode='by_individual',  # Новый параметр: 'by_individual' или 'by_image'
    force_recompute_cache=False,
):
    embeddings_save_path = os.path.join(database_dir, 'database_embeddings.pkl')
    database_image_paths = _collect_database_image_paths(database_dir)
    current_metadata = _build_cache_metadata(model_path, database_image_paths, transform)
    
    use_cache = False
    
    # Управление кэшем
    if force_recompute_cache and os.path.exists(embeddings_save_path):
        try:
            os.remove(embeddings_save_path)
            print('Принудительный пересчет: старый кэш эмбеддингов удален')
        except OSError as error:
            print(f'Не удалось удалить кэш эмбеддингов: {error}')

    if os.path.exists(embeddings_save_path):
        print('Найден кэш эмбеддингов, проверяем актуальность...')
        try:
            database_embeddings, cached_paths, cached_metadata = _load_embeddings_with_metadata(
                embeddings_save_path
            )
            if cached_metadata == current_metadata and len(cached_paths) == len(database_image_paths):
                print('Кэш эмбеддингов актуален, используем его')
                database_image_paths = cached_paths
                use_cache = True
            else:
                print('Кэш эмбеддингов устарел, пересчитываем...')
        except Exception as error:
            print(f'Не удалось прочитать кэш эмбеддингов: {error}. Пересчитываем...')

    if not use_cache:
        print('Вычисление эмбеддингов базы...')
        print(f'Найдено {len(database_image_paths)} изображений в базе')
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