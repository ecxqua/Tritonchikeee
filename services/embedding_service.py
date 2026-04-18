"""
services/embedding_service.py — Единственный владелец FAISS индекса.

Архитектурные принципы:
    1. Нет вычисления эмбеддингов → получает извне (от pipeline)
    2. Нет доступа к БД → только FAISS операции + локальный кэш
    3. Нет метаданных в search() → только embedding_index + similarity
    4. Есть буферизация → commit/rollback для синхронизации с БД
    5. Кэш восстанавливается автоматически при старте, если отсутствует

Зависимости:
    - data/embeddings/database_embeddings.pkl — FAISS индекс (512 dim)
    - data/embeddings/database_embeddings.cache.pkl — кэш векторов (обход бага FAISS)
"""

import logging
import pickle
import tempfile
import os
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
        - Удаление векторов по ID (через IndexIDMap)
        - Кэширование эмбеддингов (обход бага FAISS reconstruct)
        - Автоматическое восстановление кэша при старте
    
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
        self.cache_path = self.index_path.with_suffix('.cache.pkl')
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Буфер для отложенных добавлений (Unit of Work)
        self._pending_additions: List[Tuple[np.ndarray, Dict[str, Any], int]] = []
        
        # 🔥 Кэш эмбеддингов: photo_id -> np.ndarray (512,)
        self._embedding_cache: Dict[int, np.ndarray] = {}
        
        # Загружаем индекс и кэш при инициализации
        self.index = self._load_or_create_index()
        self._load_cache()
        
        logger.info(f"EmbeddingService инициализирован: {index_path} "
                    f"(векторов: {self.index.ntotal}, в кэше: {len(self._embedding_cache)})")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Загрузить существующий индекс или создать новый.
        
        Возвращает IndexIDMap, оборачивающий базовый индекс.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            try:
                raw_index = faiss.read_index(str(self.index_path))
                if not isinstance(raw_index, faiss.IndexIDMap):
                    logger.info("Обнаружен старый формат индекса. Оборачиваем в IndexIDMap.")
                    index = faiss.IndexIDMap(raw_index)
                else:
                    index = raw_index
                logger.info(f"FAISS индекс загружен: {self.index_path} (векторов: {index.ntotal})")
                return index
            except Exception as e:
                logger.warning(f"Ошибка загрузки FAISS: {e}. Создаём новый индекс.")
        
        base_index = self.index_type(self.embedding_dim)
        index = faiss.IndexIDMap(base_index)
        logger.info(f"Создан новый FAISS индекс (IndexIDMap): {self.index_path}")
        return index
    
    def _load_cache(self):
        """Загрузить кэш с диска. Если отсутствует → автоматически восстановить из FAISS."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Кэш эмбеддингов загружен: {len(self._embedding_cache)} векторов")
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша: {e}. Будет создан новый.")
        
        # 🔥 АВТОМАТИЧЕСКОЕ ВОССТАНОВЛЕНИЕ
        logger.info("Кэш не найден. Запускаем автоматическое восстановление из FAISS...")
        self._rebuild_cache_from_faiss()

    def _rebuild_cache_from_faiss(self):
        """Восстановить кэш напрямую из базового индекса FAISS.
        
        Обходит баг IndexIDMap.reconstruct(id), используя позиционную реконструкцию
        базового IndexFlatIP + маппинг id_map. Не требует доступа к БД или моделям.
        """
        self._embedding_cache.clear()
        n = self.index.ntotal
        if n == 0:
            logger.info("Индекс пуст, кэш не создан.")
            return

        try:
            # IndexIDMap хранит underlying index и маппинг позиций -> user_id
            base_index = self.index.index
            id_map = self.index.id_map
            
            for i in range(n):
                photo_id = int(id_map.at(i))
                vec = base_index.reconstruct(i)  # reconstruct по позиции (0..ntotal-1)
                self._embedding_cache[photo_id] = vec.flatten()
                
            self._save_cache()
            logger.info(f"✅ Кэш автоматически восстановлен: {len(self._embedding_cache)} векторов")
        except Exception as e:
            logger.error(f"❌ Не удалось восстановить кэш из FAISS: {e}")
            logger.warning("Кэш останется пустым. Добавьте новые фото для автоматического заполнения.")

    def _save_cache(self):
        """Сохранить кэш атомарно (защита от краха во время записи)."""
        temp_path = self.cache_path.with_suffix('.cache.pkl.tmp')
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(self._embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(str(temp_path), str(self.cache_path))
            logger.debug("Кэш сохранён атомарно")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Ошибка сохранения кэша: {e}")
            raise

    # -------------------------------------------------------------------------
    # ADD (с буферизацией)
    # -------------------------------------------------------------------------
    
    def add(self, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None, photo_id: Optional[int] = None) -> int:
        """
        Добавить вектор в БУФЕР (не на диск сразу).
        
        Args:
            embedding: Вектор размерности (512,), L2 нормализован
            metadata: Метаданные для логгирования
            photo_id: Стабильный идентификатор фото из БД
        
        Returns:
            int: ID, который будет использован в индексе
        """
        embedding = self._validate_embedding(embedding)
        
        effective_id = photo_id if photo_id is not None else (self.index.ntotal + len(self._pending_additions))
        self._pending_additions.append((embedding, metadata or {}, effective_id))
        
        logger.debug(f"Добавлено в буфер: ID={effective_id}")
        return effective_id
    
    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Проверить и нормализовать embedding."""
        if embedding is None:
            raise ValueError("Embedding не может быть None")
        if hasattr(embedding, 'cpu'):
            embedding = embedding.cpu().numpy()
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        if embedding.shape == (self.embedding_dim,):
            embedding = embedding.reshape(1, -1)
        elif embedding.shape != (1, self.embedding_dim):
            raise ValueError(f"Неверный размер embedding: {embedding.shape}")
        return embedding.astype('float32')
    
    def commit(self) -> int:
        """Сохранить все отложенные добавления в FAISS и кэш."""
        if not self._pending_additions:
            logger.debug("Нет отложенных добавлений для commit")
            return 0
        
        embeddings_batch = [e for e, _, _ in self._pending_additions]
        ids_batch = [pid for _, _, pid in self._pending_additions]
        
        # Добавляем в FAISS
        embeddings_array = np.vstack(embeddings_batch)
        ids_array = np.array(ids_batch, dtype=np.int64)
        self.index.add_with_ids(embeddings_array, ids_array)
        
        # 🔥 Обновляем кэш
        for emb, _, pid in self._pending_additions:
            self._embedding_cache[pid] = emb.flatten()
        
        self._save_index()
        self._save_cache()
        
        count = len(self._pending_additions)
        self._pending_additions = []
        logger.info(f"FAISS commit: добавлено {count} векторов")
        return count
    
    def rollback(self) -> int:
        """Откатить все отложенные добавления."""
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
    # DELETE
    # -------------------------------------------------------------------------
    
    def delete(self, photo_id: int) -> bool:
        """Удалить вектор из индекса и кэша по photo_id."""
        try:
            removed = self.index.remove_ids(np.array([np.int64(photo_id)], dtype=np.int64))
            if removed > 0:
                self._embedding_cache.pop(photo_id, None)
                self._save_index()
                self._save_cache()
                logger.info(f"Вектор удалён: photo_id={photo_id}")
                return True
            else:
                logger.warning(f"Вектор не найден для удаления: photo_id={photo_id}")
                return False
        except Exception as e:
            logger.error(f"Ошибка удаления вектора {photo_id}: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------------
    
    def reload_index(self):
        """Перезагрузить индекс и кэш с диска."""
        self.index = self._load_or_create_index()
        self._load_cache()
        logger.info("FAISS индекс и кэш перезагружены")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику индекса."""
        return {
            'vector_count': self.index.ntotal,
            'cached_count': len(self._embedding_cache),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'pending_additions': len(self._pending_additions),
            'index_path': str(self.index_path)
        }
    
    def get_embedding_by_index(self, embedding_index: int) -> Optional[np.ndarray]:
        """
        Получить эмбеддинг из кэша по photo_id.
        🔥 Обходит баг FAISS reconstruct для IndexIDMap.
        
        Args:
            embedding_index: photo_id из БД
        
        Returns:
            np.ndarray: Вектор (512,) или None если не найден
        """
        emb = self._embedding_cache.get(embedding_index)
        if emb is None:
            logger.debug(f"Эмбеддинг не найден в кэше: embedding_index={embedding_index}")
        return emb