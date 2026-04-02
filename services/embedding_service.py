"""
services/embedding_service.py — Единственный владелец FAISS индекса.

Архитектурные принципы:
    1. Нет вычисления эмбеддингов → получает извне (от pipeline)
    2. Нет доступа к БД → только FAISS операции
    3. Нет метаданных в search() → только embedding_index + similarity
    4. Есть буферизация → commit/rollback для синхронизации с БД

Зависимости:
    - data/embeddings/database_embeddings.pkl — FAISS индекс (512 dim)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

DEFAULT_EMBEDDING_DIM = 512
DEFAULT_INDEX_TYPE = faiss.IndexFlatIP  # Inner Product для косинусного сходства

# =============================================================================
# ТИПЫ ДАННЫХ
# =============================================================================

class SearchResult:
    """Результат поиска (без метаданных)."""
    def __init__(self, embedding_index: int, similarity: float, rank: int):
        self.embedding_index = embedding_index
        self.similarity = similarity
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding_index': self.embedding_index,
            'similarity': float(self.similarity),
            'similarity_percent': float(self.similarity * 100),
            'rank': self.rank
        }

# =============================================================================
# EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    """
    Единственный владелец FAISS индекса.
    
    Отвечает за:
        - Добавление векторов (с буферизацией)
        - Поиск похожих векторов
        - Сохранение/загрузку индекса
        - Откат операций (rollback)
    
    НЕ отвечает за:
        - Вычисление эмбеддингов (это pipeline)
        - Доступ к БД (это CardService/IdentificationService)
        - Метаданные (это оркестратор)
    """
    
    def __init__(
        self, 
        index_path: str, 
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        index_type: Any = DEFAULT_INDEX_TYPE
    ):
        """
        Args:
            index_path: Путь к файлу FAISS индекса (.pkl)
            embedding_dim: Размерность вектора (по умолчанию 512)
            index_type: Тип индекса FAISS (по умолчанию IndexFlatIP)
        """
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Буфер для отложенных добавлений (Unit of Work)
        self._pending_additions: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        
        # Загружаем индекс при инициализации
        self.index = self._load_or_create_index()
        
        logger.info(f"EmbeddingService инициализирован: {index_path} (векторов: {self.index.ntotal})")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Загрузить существующий индекс или создать новый."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                logger.info(f"FAISS индекс загружен: {self.index_path} (векторов: {index.ntotal})")
                return index
            except Exception as e:
                logger.warning(f"Ошибка загрузки FAISS: {e}. Создаём новый индекс.")
        
        # Создаём новый индекс
        index = self.index_type(self.embedding_dim)
        logger.info(f"Создан новый FAISS индекс: {self.index_path}")
        return index
    
    # -------------------------------------------------------------------------
    # ADD (с буферизацией)
    # -------------------------------------------------------------------------
    
    def add(self, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Добавить вектор в БУФЕР (не на диск сразу).
        
        Args:
            embedding: Вектор размерности (512,), L2 нормализован
            metadata: Метаданные для логгирования (не сохраняются в FAISS)
        
        Returns:
            int: Временный ID (будет подтверждён при commit)
        
        Raises:
            ValueError: Если embedding имеет неверный формат
        """
        # Валидация embedding
        embedding = self._validate_embedding(embedding)
        
        # Вычисляем временный ID
        temp_id = self.index.ntotal + len(self._pending_additions)
        
        # Добавляем в буфер
        self._pending_additions.append((embedding, metadata or {}))
        
        logger.debug(f"Добавлено в буфер: временный ID {temp_id}")
        return temp_id
    
    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Проверить и нормализовать embedding."""
        if embedding is None:
            raise ValueError("Embedding не может быть None")
        
        # Конвертация torch tensor → numpy
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().numpy()
        
        # Конвертация в numpy
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Проверка размерности
        if embedding.shape == (self.embedding_dim,):
            embedding = embedding.reshape(1, -1)
        elif embedding.shape != (1, self.embedding_dim):
            raise ValueError(
                f"Неверный размер embedding: {embedding.shape}, "
                f"ожидалось ({self.embedding_dim},) или (1, {self.embedding_dim})"
            )
        
        # Конвертация в float32 (требование FAISS)
        embedding = embedding.astype('float32')
        
        return embedding
    
    def commit(self) -> int:
        """
        Сохранить все отложенные добавления в FAISS и на диск.
        
        Вызывается ПОСЛЕ успешного commit БД (гарантия синхронизации).
        
        Returns:
            int: Количество добавленных векторов
        """
        if not self._pending_additions:
            logger.debug("Нет отложенных добавлений для commit")
            return 0
        
        added_count = 0
        for embedding, metadata in self._pending_additions:
            self.index.add(embedding)
            added_count += 1
            
            # Логгирование метаданных (если есть)
            if metadata:
                ind_id = metadata.get('individual_id', 'Unknown')
                logger.debug(f"Добавлено в FAISS: {ind_id}")
        
        # Сохранение на диск
        self._save_index()
        
        # Очистка буфера
        self._pending_additions = []
        
        logger.info(f"FAISS commit: добавлено {added_count} векторов")
        return added_count
    
    def rollback(self) -> int:
        """
        Откатить все отложенные добавления (при ошибке БД).
        
        Вызывается при rollback БД (гарантия синхронизации).
        
        Returns:
            int: Количество отменённых добавлений
        """
        rolled_back_count = len(self._pending_additions)
        self._pending_additions = []
        
        logger.info(f"FAISS rollback: отменено {rolled_back_count} добавлений")
        return rolled_back_count
    
    def _save_index(self):
        """Сохранить индекс на диск."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.debug(f"FAISS индекс сохранён: {self.index_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения FAISS: {e}")
            raise
    
    # -------------------------------------------------------------------------
    # SEARCH (без метаданных)
    # -------------------------------------------------------------------------
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Поиск похожих векторов в FAISS.
        
        Args:
            query_embedding: Вектор запроса (512,), L2 нормализован
            top_k: Количество результатов
        
        Returns:
            List[SearchResult]: Результаты поиска (только embedding_index + similarity)
        
        Raises:
            ValueError: Если query_embedding имеет неверный формат
        """
        # Валидация query_embedding
        query_embedding = self._validate_embedding(query_embedding)
        
        # Поиск
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Формирование результатов
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS возвращает -1 если не хватает результатов
                continue
            
            results.append(SearchResult(
                embedding_index=int(idx),
                similarity=float(dist),
                rank=i + 1
            ))
        
        logger.debug(f"FAISS search: найдено {len(results)} результатов")
        return results
    
    def search_with_threshold(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 20,
        threshold: float = 0.75
    ) -> List[SearchResult]:
        """
        Поиск с порогом отсечения.
        
        Args:
            query_embedding: Вектор запроса
            top_k: Максимальное количество результатов
            threshold: Минимальная similarity (0.0 - 1.0)
        
        Returns:
            List[SearchResult]: Результаты выше порога
        """
        all_results = self.search(query_embedding, top_k)
        return [r for r in all_results if r.similarity >= threshold]
    
    # -------------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------------
    
    def reload_index(self):
        """Перезагрузить индекс с диска (после внешних изменений)."""
        self.index = self._load_or_create_index()
        logger.info("FAISS индекс перезапущен")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику индекса."""
        return {
            'vector_count': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type.__name__,
            'pending_additions': len(self._pending_additions),
            'index_path': str(self.index_path)
        }
    
    def get_embedding_by_index(self, embedding_index: int) -> Optional[np.ndarray]:
        """
        Восстановить вектор из FAISS по индексу.
        
        Args:
            embedding_index: Позиция вектора в индексе
        
        Returns:
            np.ndarray: Вектор (512,) или None если индекс не найден
        """
        try:
            if embedding_index >= self.index.ntotal:
                return None
            return self.index.reconstruct(embedding_index)
        except Exception as e:
            logger.error(f"Ошибка восстановления вектора {embedding_index}: {e}")
            return None