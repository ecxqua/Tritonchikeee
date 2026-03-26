import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import shutil
import torch.nn as nn
import timm
import pickle
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
        super(EnhancedTripletNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        self._setup_progressive_unfreezing()

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

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

        self._init_weights()

    def _setup_progressive_unfreezing(self):
        """Замораживает大部分 слоёв ViT, размораживает последние 6 блоков"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        if hasattr(self.base_model, 'blocks'):
            num_blocks = len(self.base_model.blocks)
            blocks_to_unfreeze = min(6, num_blocks)
            for i in range(num_blocks - blocks_to_unfreeze, num_blocks):
                for param in self.base_model.blocks[i].parameters():
                    param.requires_grad = True

    def _init_weights(self):
        """Инициализация весов линейных слоёв"""
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
            x: Входное изображение (batch, 3, 224, 224)
            return_projection: Вернуть ли projection head output
        
        Returns:
            embedding: Нормализованный вектор (batch, 512)
            projection: (опционально) вектор (batch, 128)
        """
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if return_projection:
            projections = self.projection(embeddings)
            projections = nn.functional.normalize(projections, p=2, dim=1)
            return embeddings, projections

        return embeddings


# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ И ОБРАБОТКИ
# ============================================================================

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Загрузить модель ViT для идентификации.
    
    Args:
        model_path: Путь к файлу весов (.pt)
        device: Устройство для вычислений (cuda/cpu)
    
    Returns:
        model: Загруженная модель в режиме eval
    """
    model = EnhancedTripletNet(base_model_name='vit_base_patch16_224', embedding_dim=512)
    state_dict = torch.load(model_path, map_location=device)

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path: str, model: torch.nn.Module, transform, device: torch.device) -> Optional[np.ndarray]:
    """
    Получить эмбеддинг изображения через ViT модель.
    
    Args:
        image_path: Путь к изображению
        model: Загруженная модель
        transform: Трансформы для предобработки
        device: Устройство для вычислений
    
    Returns:
        embedding: Вектор размерности (512,), L2 нормализован, или None при ошибке
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


def compute_distances(embeddings1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
    """
    Вычислить расстояния между векторами (cosine distance).
    
    Args:
        embeddings1: Массив векторов (n, 512)
        embedding2: Единичный вектор (512,)
    
    Returns:
        distances: Массив расстояний (n,)
    """
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    return 1 - similarities.flatten()


def save_embeddings(embeddings: np.ndarray, paths: List[str], save_path: str):
    """
    Сохранить эмбеддинги в pickle (устаревший метод).
    
    Args:
        embeddings: Массив векторов (n, 512)
        paths: Список путей к изображениям
        save_path: Путь для сохранения
    """
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'paths': paths}, f)


def load_embeddings(save_path: str) -> tuple[np.ndarray, List[str]]:
    """
    Загрузить эмбеддинги из pickle (устаревший метод).
    
    Args:
        save_path: Путь к файлу pickle
    
    Returns:
        embeddings: Массив векторов (n, 512)
        paths: Список путей к изображениям
    """
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['paths']


# ============================================================================
# ПОИСК (ОБНОВЛЁННЫЙ: FAISS + SQLite)
# ============================================================================

def get_db_connection(db_path: str = "database/cards.db"):
    """
    Получить соединение с базой данных.
    
    Args:
        db_path: Путь к файлу БД
    
    Returns:
        conn: SQLite соединение с row_factory
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_individual_info(cursor, embedding_index: int) -> Optional[Dict]:
    """
    Получить информацию об особи по embedding_index из БД.
    
    Args:
        cursor: Курсор SQLite
        embedding_index: Позиция вектора в FAISS индексе
    
    Returns:
        dict: Информация об особи (individual_id, species, photo_path) или None
    """
    cursor.execute('''
        SELECT p.photo_id, p.individual_id, p.photo_path,
               i.species, i.template_type, i.project_name
        FROM photos p
        JOIN individuals i ON p.individual_id = i.individual_id
        WHERE p.embedding_index = ?
    ''', (embedding_index,))
    
    row = cursor.fetchone()
    if row:
        return {
            'photo_id': row['photo_id'],
            'individual_id': row['individual_id'],
            'photo_path': row['photo_path'],
            'species': row['species'],
            'template_type': row['template_type'],
            'project_name': row['project_name']
        }
    return None


def find_similar_images(
    model_path: str,
    database_dir: str,
    query_image_path: str,
    output_dir: str,
    transform,  # ← Теперь обязательно используется
    device: torch.device,
    size_answer: int = 5
):
    """
    Найти похожих особей по фотографии брюшка.
    """
    FAISS_INDEX_PATH = "data/embeddings/database_embeddings.pkl"
    DB_PATH = "database/cards.db"
    
    # 1. Загрузка модели
    print(f"Загрузка модели: {model_path}")
    model = load_model(model_path, device)
    
    # 2. Загрузка FAISS индекса
    print(f"Загрузка FAISS индекса: {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Ошибка: FAISS индекс не найден: {FAISS_INDEX_PATH}")
        return []
    
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"Векторов в индексе: {faiss_index.ntotal}")
    
    # 3. Получение query embedding
    print(f"Обработка запроса: {query_image_path}")
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print("Не удалось обработать запросное изображение")
        return []
    
    # 4. Поиск в FAISS
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, size_answer)
    
    # 5. Получение метаданных из БД
    conn = get_db_connection(DB_PATH)
    cursor = conn.cursor()
    
    results = []
    print(f"\nТоп-{size_answer} результатов:")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "res.txt"), 'w', encoding='utf-8') as file:
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            if idx == -1:
                continue
            
            info = get_individual_info(cursor, int(idx))
            if not info:
                print(f"Предупреждение: embedding_index {idx} не найден в БД")
                continue
            
            src_path = info['photo_path']
            dst_filename = f"top{i}.jpg"
            dst_path = os.path.join(output_dir, dst_filename)
            
            try:
                shutil.copy(src_path, dst_path)
                
                class_string = 'Ребристый' if info['species'] == 'Гребенчатый' else "Карелина"
                similarity_percent = dist * 100
                
                res_str = f"{i}. Класс: {class_string} | Особь: {info['individual_id']} | Схожесть: {similarity_percent:.1f}%\n"
                file.write(res_str)
                print(f"{i}. Класс: {class_string} | Особь: {info['individual_id']} | Схожесть: {similarity_percent:.1f}%")
                
                results.append({
                    'rank': i,
                    'individual_id': info['individual_id'],
                    'species': info['species'],
                    'photo_path': src_path,
                    'similarity': similarity_percent
                })
            except Exception as e:
                print(f"Ошибка копирования файла {src_path}: {str(e)}")
    
    conn.close()
    print(f"\nРезультаты сохранены в: {output_dir}")
    return results


# ============================================================================
# УСТАРЕВШАЯ ВЕРСИЯ (для отладки/сравнения)
# ============================================================================

def find_similar_images_legacy(
    model_path: str,
    database_dir: str,
    query_image_path: str,
    output_dir: str,
    transform,
    device: torch.device,
    size_answer: int = 5
):
    """
    Устаревшая версия поиска через pickle + sklearn.
    
    Используется только для отладки или если FAISS индекс недоступен.
    Сканирует папки с изображениями вместо использования БД.
    
    Args:
        model_path: Путь к модели ViT
        database_dir: Папка с изображениями для поиска
        query_image_path: Путь к запросному изображению
        output_dir: Папка для результатов
        transform: Трансформы для предобработки
        device: Устройство для вычислений
        size_answer: Количество результатов
    """
    embeddings_save_path = os.path.join(database_dir, 'database_embeddings.pkl')
    if os.path.exists(embeddings_save_path):
        print("Загрузка сохраненных эмбеддингов...")
        database_embeddings, database_image_paths = load_embeddings(embeddings_save_path)
    else:
        print("Вычисление эмбеддингов базы...")
        model = load_model(model_path, device)

        database_image_paths = []
        for root, _, files in os.walk(database_dir):
            if os.path.normpath(root).startswith(os.path.normpath(output_dir)):
                continue

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        database_image_paths.append(full_path)

        print(f"Найдено {len(database_image_paths)} изображений в базе")

        database_embeddings = []
        valid_paths = []
        for path in tqdm(database_image_paths, desc="Обработка базы"):
            emb = get_embedding(path, model, transform, device)
            if emb is not None:
                database_embeddings.append(emb)
                valid_paths.append(path)

        database_embeddings = np.array(database_embeddings)
        save_embeddings(database_embeddings, valid_paths, embeddings_save_path)
        print(f"Эмбеддинги сохранены в {embeddings_save_path}")
        database_image_paths = valid_paths

    model = load_model(model_path, device)
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print("Не удалось обработать запросное изображение")
        return

    distances = compute_distances(np.array(database_embeddings), query_embedding)
    top5_idx = np.argsort(distances)[:size_answer]

    print("\n=== Топ-5 результатов ===")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + "/res.txt", 'w', encoding='utf-8') as file:
        for i, idx in enumerate(top5_idx, 1):
            src_path = database_image_paths[idx]
            dst_filename = f"top{i}.jpg"
            dst_path = os.path.join(output_dir, dst_filename)

            try:
                shutil.copy(src_path, dst_path)
                class_name = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
                individual = os.path.basename(os.path.dirname(src_path))
                similarity = 1 - distances[idx]
                similarity_percent = similarity * 100

                class_string = 'Ребристый' if class_name == 'ribbed' else "Карелина"
                res_str = f"{i}. Класс: {class_string} | Особь: {individual} | Схожесть: {similarity_percent:.1f}%\n"
                file.write(res_str)
                print(f"{i}. Класс: {class_name} | Особь: {individual} | Схожесть: {similarity_percent:.1f}% | Путь: {src_path}")
            except Exception as e:
                print(f"Ошибка копирования файла {src_path}: {str(e)}")
    print(f"\nРезультаты сохранены в: {output_dir}")


# ============================================================================
# MAIN (для тестирования)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Поиск особей тритонов по кропу брюшка")
    parser.add_argument("query_image", type=str, help="Путь к изображению")
    parser.add_argument("--output", type=str, default="data/output/search_results")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default="models/best_id.pt")
    parser.add_argument("--legacy", action="store_true", help="Использовать legacy версию (pickle)")
    
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.legacy:
        find_similar_images_legacy(
            model_path=args.model,
            database_dir="data/dataset_crop/dataset_crop_24",
            query_image_path=args.query_image,
            output_dir=args.output,
            transform=TRANSFORMS,
            device=DEVICE,
            size_answer=args.top_k
        )
    else:
        find_similar_images(
            model_path=args.model,
            database_dir="data/dataset_crop/dataset_crop_24",
            query_image_path=args.query_image,
            output_dir=args.output,
            transform=TRANSFORMS,
            device=DEVICE,
            size_answer=args.top_k
        )