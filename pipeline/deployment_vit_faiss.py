"""
pipeline/deployment_vit_faiss.py — Ядро идентификации (ViT инференс)
Версия: 3.0 (Clean Architecture)

АРХИТЕКТУРНЫЕ ПРИНЦИПЫ:
    1. НЕТ импорта faiss → pipeline не знает про индекс
    2. НЕТ импорта sqlite3 → pipeline не знает про БД
    3. НЕТ файловых операций → только работа с памятью
    4. Детерминированность → одинаковый вход = одинаковый выход

Зависимости:
    - models/best_id.pt — веса модели EnhancedTripletNet
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# =============================================================================
# КОНСТАНТЫ (БЕЗ ИЗМЕНЕНИЙ)
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
# ФУНКЦИИ ЗАГРУЗКИ МОДЕЛИ И ЭМБЕДДИНГОВ (БЕЗ ИЗМЕНЕНИЙ)
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
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
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
    logger.info(f"Модель загружена: {model_path}")
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
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Изображение не найдено: {image_path}")
            return None
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(image_tensor)
        
        result = embedding.cpu().numpy().flatten()
        logger.debug(f"Эмбеддинг получен: shape={result.shape}")
        return result
    
    except Exception as e:
        logger.error(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def get_embedding_from_array(
    crop_array: np.ndarray,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device
) -> Optional[np.ndarray]:
    """
    Получить эмбеддинг из numpy array (без чтения с диска).
    
    КЛЮЧЕВАЯ ФУНКЦИЯ для in-memory пайплайна:
        YOLO → crop_array (в памяти) → ViT → embedding
    
    Преимущества:
        - Нет записи на диск → быстрее на ~50-100ms
        - Нет чтения с диска → меньше I/O операций
        - Нет временных файлов → чище архитектура
    
    Args:
        crop_array: numpy array изображения (H, W, 3) в формате BGR или RGB
        model: Загруженная модель ViT
        transform: Трансформы для предобработки
        device: Устройство для вычислений
    
    Returns:
        Вектор размерности (512,), L2 нормализован, или None при ошибке
    
    Raises:
        ValueError: Если array имеет неверный формат
    """
    try:
        # Валидация входных данных
        if crop_array is None:
            logger.error("crop_array не может быть None")
            return None
        
        if not isinstance(crop_array, np.ndarray):
            logger.error(f"crop_array должен быть numpy array, получено {type(crop_array)}")
            return None
        
        if len(crop_array.shape) != 3:
            logger.error(f"Неверная размерность crop_array: {crop_array.shape}, ожидалось (H, W, 3)")
            return None
        
        # Конвертация BGR → RGB (если пришёл из OpenCV)
        # OpenCV загружает в BGR, PIL ожидает RGB
        if crop_array.shape[2] == 3:
            # Проверяем порядок каналов по значениям
            # Если среднее значение синего канала значительно больше красного → BGR
            if np.mean(crop_array[:, :, 0]) > np.mean(crop_array[:, :, 2]):
                crop_array = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)
        
        # Конвертация numpy → PIL Image
        image = Image.fromarray(crop_array)
        
        # Применение трансформов и инференс
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(image_tensor)
        
        result = embedding.cpu().numpy().flatten()
        logger.debug(f"Эмбеддинг получен из array: shape={result.shape}")
        return result
    
    except Exception as e:
        logger.error(f"Ошибка обработки crop_array: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# УТИЛИТЫ (БЕЗ ИЗМЕНЕНИЙ)
# =============================================================================

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 нормализация эмбеддинга."""
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        return embedding / norm
    return embedding

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Вычислить косинусное сходство между двумя эмбеддингами."""
    emb1 = normalize_embedding(embedding1)
    emb2 = normalize_embedding(embedding2)
    return float(np.dot(emb1, emb2))

# =============================================================================
# ПОИСК ВЕКТОРОВ (ЧИСТАЯ МАТЕМАТИКА для усреднённых эмбеддингов)
# =============================================================================

def search_vectors(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Поиск похожих векторов через косинусное сходство (чистая numpy математика).
    
    ИСПОЛЬЗУЕТСЯ ДЛЯ:
        - Поиска по прототипам (усреднённые эмбеддинги особей)
        - Поиска по временным массивам (не в FAISS)
        - Тестирования без FAISS индекса
    
    АРХИТЕКТУРА:
        - НЕТ импорта faiss → pipeline не зависит от библиотеки
        - НЕТ доступа к БД → только векторы
        - Возвращает [(индекс_в_массиве, similarity), ...]
    
    Args:
        query_embedding: Вектор запроса (512,), L2 нормализован
        reference_embeddings: Массив референсных векторов (n, 512)
        top_k: Количество результатов
    
    Returns:
        Список кортежей: [(индекс_в_массиве, similarity), ...]
        similarity — косинусное сходство (0.0 - 1.0)
    """
    # Валидация
    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("query_embedding не может быть пустым")
    
    if len(reference_embeddings) == 0:
        logger.warning("reference_embeddings пуст — поиск невозможен")
        return []
    
    # Подготовка векторов
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    reference_embeddings = reference_embeddings.astype('float32')
    
    # Косинусное сходство через dot product (векторы L2 нормализованы)
    similarities = np.dot(reference_embeddings, query_vector.T).flatten()
    
    # Топ-K индексов по убыванию схожести
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Формирование результатов
    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim > 0:  # Отсекаем отрицательные схожести
            results.append((int(idx), sim))
    
    logger.debug(f"Поиск векторов завершён: найдено {len(results)} результатов")
    return results