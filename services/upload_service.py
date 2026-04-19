"""
services/upload_service.py — CRUD для временных загрузок (Two-Phase Commit).

Архитектурные принципы:
    1. ✅ CRUD для uploads только здесь
    2. ✅ Нет доступа к pipeline
    3. ✅ Очистка старых загрузок (expired)

Зависимости:
    - database/cards.db — SQLite база
    - services/card_service.py — валидация проектов
"""

import logging
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from database.card_database import DB_PATH

logger = logging.getLogger(__name__)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

UPLOAD_EXPIRY_HOURS = 24

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Получить соединение с SQLite базой данных."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def serialize_embedding(embedding: Any) -> str:
    """Сериализовать embedding в JSON строку для хранения в БД."""
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    return json.dumps(embedding)

def deserialize_embedding(embedding_str: str) -> List[float]:
    """Десериализовать embedding из JSON строки."""
    return json.loads(embedding_str)

# =============================================================================
# UPLOAD SERVICE
# =============================================================================

class UploadService:
    """CRUD для временных загрузок (Two-Phase Commit)."""
    
    def __init__(self, db_path: str = DB_PATH):
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
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_uploads_status_expires 
            ON uploads(status, expires_at)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_uploads_project 
            ON uploads(project_id)
        ''')
        
        conn.commit()
        conn.close()
        logger.debug("Таблица uploads проверена/создана")
    
    def create_upload(
        self,
        file_path: str,
        embedding: Any,
        expiry_hours: int = UPLOAD_EXPIRY_HOURS
    ) -> int:
        """CREATE: Создать временную загрузку."""
        if embedding is None or len(embedding) == 0:
            raise ValueError("Embedding не может быть пустым")
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        expires_at = now + timedelta(hours=expiry_hours)
        embedding_json = serialize_embedding(embedding)

        # Смена под upload_id
        upload_id = self.get_stats()["total"] + 1
        file_suffix = Path(file_path).suffix
        file_parent = str(Path(file_path).parent)
        logger.info("Родительская папка сохранённого кропа: " + file_parent)
        file_path = str(Path(file_path).rename(
            f"{file_parent}\{upload_id}{file_suffix}"
        ))
        
        try:
            cursor.execute('''
                INSERT INTO uploads (project_id, file_path, embedding, status, created_at, expires_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
            ''', (-1, file_path, embedding_json, now.isoformat(), expires_at.isoformat()))
            
            upload_id = cursor.lastrowid
            conn.commit()
            
            return upload_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка создания загрузки: {e}")
            raise e
        finally:
            conn.close()
    
    def get_upload(self, upload_id: int) -> Optional[Dict[str, Any]]:
        """READ: Получить загрузку по ID."""
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
        
        result = dict(row)
        result['embedding'] = deserialize_embedding(row['embedding'])
        
        return result
    
    def complete_upload(self, upload_id: int, card_id: str) -> bool:
        """UPDATE: Завершить загрузку."""
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
        """UPDATE: Отменить загрузку."""
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
        """DELETE: Удалить просроченные загрузки."""
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
        """READ: Получить все активные (pending) загрузки."""
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
        
        results = []
        for row in rows:
            result = dict(row)
            result['embedding'] = deserialize_embedding(row['embedding'])
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """READ: Получить статистику по загрузкам."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute('SELECT COUNT(*) FROM uploads')
        stats['total'] = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT status, COUNT(*) 
            FROM uploads 
            GROUP BY status
        ''')
        stats['by_status'] = {row['status']: row[1] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT COUNT(*) 
            FROM uploads 
            WHERE status = 'pending' AND expires_at < ?
        ''', (datetime.now().isoformat(),))
        stats['expired_pending'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats