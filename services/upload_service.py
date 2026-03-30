"""
services/upload_service.py — Управление временными загрузками (Two-Phase Commit)

АРХИТЕКТУРНЫЕ ПРИНЦИПЫ:
    1. Хранит временные данные между запросами API (анализ → подтверждение)
    2. Embedding хранится в БД (JSON), НЕ в FAISS (до подтверждения)
    3. Нет доступа к pipeline → только БД операции
    4. Автоочистка старых загрузок (expired)

Зависимости:
    - database/cards.db — SQLite база (таблица uploads)
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from database.card_database import DB_PATH

logger = logging.getLogger(__name__)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

UPLOAD_EXPIRY_HOURS = 24  # Время жизни временной загрузки

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Получить соединение с SQLite базой данных."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def serialize_embedding(embedding: Any) -> str:
    """
    Сериализовать embedding в JSON строку для хранения в БД.
    
    Args:
        embedding: np.ndarray или list[float]
    
    Returns:
        str: JSON строка
    """
    # Конвертация numpy → list
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    
    return json.dumps(embedding)

def deserialize_embedding(embedding_str: str) -> List[float]:
    """
    Десериализовать embedding из JSON строки.
    
    Args:
        embedding_str: JSON строка из БД
    
    Returns:
        list[float]: Вектор эмбеддинга
    """
    return json.loads(embedding_str)

# =============================================================================
# UPLOAD SERVICE
# =============================================================================

class UploadService:
    """
    Управление временными загрузками для Two-Phase Commit.
    
    Поток данных:
        1. POST /analyze → create_upload() → upload_id
        2. POST /confirm → get_upload(upload_id) → finalize (в card_service)
        3. POST /cancel → cancel_upload(upload_id)
    
    Таблица uploads:
        - id: PRIMARY KEY
        - project_id: INTEGER (для изоляции проектов)
        - file_path: TEXT (путь к кропу)
        - embedding: TEXT (JSON строка)
        - status: TEXT (pending, completed, cancelled)
        - card_id: TEXT (ссылка на созданную карточку)
        - created_at: TIMESTAMP
        - expires_at: TIMESTAMP (для автоочистки)
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Args:
            db_path: Путь к SQLite базе данных
        """
        self.db_path = db_path
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Создать таблицу uploads если не существует."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                card_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Индекс для быстрой очистки просроченных записей
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_uploads_status_expires 
            ON uploads(status, expires_at)
        ''')
        
        # Индекс для поиска по project_id
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_uploads_project 
            ON uploads(project_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.debug("Таблица uploads проверена/создана")
    
    def create_upload(
        self,
        project_id: int,
        file_path: str,
        embedding: Any,
        expiry_hours: int = UPLOAD_EXPIRY_HOURS
    ) -> int:
        """
        Создать временную загрузку (Шаг 1 Two-Phase Commit).
        
        Args:
            project_id: ID проекта (для изоляции)
            file_path: Путь к файлу кропа брюшка
            embedding: Вектор эмбеддинга (np.ndarray или list)
            expiry_hours: Время жизни загрузки (часы)
        
        Returns:
            int: upload_id для использования в Шаге 2
        
        Raises:
            ValueError: Если embedding пустой
        """
        # Валидация embedding
        if embedding is None or len(embedding) == 0:
            raise ValueError("Embedding не может быть пустым")
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        expires_at = now + timedelta(hours=expiry_hours)
        embedding_json = serialize_embedding(embedding)
        
        try:
            cursor.execute('''
                INSERT INTO uploads (project_id, file_path, embedding, status, created_at, expires_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
            ''', (project_id, file_path, embedding_json, now.isoformat(), expires_at.isoformat()))
            
            upload_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Создана загрузка {upload_id} (проект {project_id}, истекает {expires_at})")
            return upload_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка создания загрузки: {e}")
            raise e
        finally:
            conn.close()
    
    def get_upload(self, upload_id: int) -> Optional[Dict[str, Any]]:
        """
        Получить данные временной загрузки по ID.
        
        Args:
            upload_id: ID загрузки
        
        Returns:
            Dict с данными загрузки или None если не найдена
            {
                'id': int,
                'project_id': int,
                'file_path': str,
                'embedding': list[float],  # Уже десериализован
                'status': str,
                'card_id': str | None,
                'created_at': str,
                'expires_at': str
            }
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, project_id, file_path, embedding, status, card_id, created_at, expires_at
            FROM uploads
            WHERE id = ?
        ''', (upload_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            logger.warning(f"Загрузка {upload_id} не найдена")
            return None
        
        # Десериализация embedding
        result = dict(row)
        result['embedding'] = deserialize_embedding(row['embedding'])
        
        return result
    
    def complete_upload(self, upload_id: int, card_id: str) -> bool:
        """
        Пометить загрузку как завершённую (после успешного создания карточки).
        
        Args:
            upload_id: ID загрузки
            card_id: ID созданной карточки (individual_id)
        
        Returns:
            bool: True если успешно
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE uploads
                SET status = 'completed', card_id = ?
                WHERE id = ? AND status = 'pending'
            ''', (card_id, upload_id))
            
            if cursor.rowcount == 0:
                logger.warning(f"Загрузка {upload_id} не найдена или уже обработана")
                conn.close()
                return False
            
            conn.commit()
            logger.info(f"Загрузка {upload_id} завершена: карточка {card_id}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка завершения загрузки: {e}")
            raise e
        finally:
            conn.close()
    
    def cancel_upload(self, upload_id: int) -> bool:
        """
        Отменить временную загрузку (пользователь отменил решение).
        
        Args:
            upload_id: ID загрузки
        
        Returns:
            bool: True если успешно
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE uploads
                SET status = 'cancelled'
                WHERE id = ? AND status = 'pending'
            ''', (upload_id,))
            
            if cursor.rowcount == 0:
                logger.warning(f"Загрузка {upload_id} не найдена или уже обработана")
                conn.close()
                return False
            
            conn.commit()
            logger.info(f"Загрузка {upload_id} отменена")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка отмены загрузки: {e}")
            raise e
        finally:
            conn.close()
    
    def cleanup_expired(self) -> int:
        """
        Удалить просроченные загрузки (фоновая задача).
        
        Вызывайте периодически (например, раз в час) для очистки базы.
        
        Returns:
            int: Количество удалённых записей
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            now = datetime.now().isoformat()
            
            cursor.execute('''
                DELETE FROM uploads
                WHERE status = 'pending' AND expires_at < ?
            ''', (now,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Очистка: удалено {deleted_count} просроченных загрузок")
            
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка очистки загрузок: {e}")
            raise e
        finally:
            conn.close()
    
    def get_pending_uploads(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получить все активные (pending) загрузки.
        
        Args:
            project_id: Фильтр по проекту (если None, все проекты)
        
        Returns:
            List[Dict]: Список загрузок
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        if project_id:
            cursor.execute('''
                SELECT id, project_id, file_path, embedding, status, created_at, expires_at
                FROM uploads
                WHERE status = 'pending' AND project_id = ?
                ORDER BY created_at DESC
            ''', (project_id,))
        else:
            cursor.execute('''
                SELECT id, project_id, file_path, embedding, status, created_at, expires_at
                FROM uploads
                WHERE status = 'pending'
                ORDER BY created_at DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        # Десериализация embedding
        results = []
        for row in rows:
            result = dict(row)
            result['embedding'] = deserialize_embedding(row['embedding'])
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику по загрузкам.
        
        Returns:
            Dict со статистикой
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Общее количество
        cursor.execute('SELECT COUNT(*) FROM uploads')
        stats['total'] = cursor.fetchone()[0]
        
        # По статусам
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM uploads 
            GROUP BY status
        ''')
        stats['by_status'] = {row['status']: row[1] for row in cursor.fetchall()}
        
        # Просроченные (ещё не удалённые)
        cursor.execute('''
            SELECT COUNT(*) 
            FROM uploads 
            WHERE status = 'pending' AND expires_at < ?
        ''', (datetime.now().isoformat(),))
        stats['expired_pending'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats