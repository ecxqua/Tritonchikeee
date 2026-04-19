"""
services/project_service.py — CRUD операции для проектов.

Архитектурные принципы:
    1. ✅ ВЕСЬ CRUD для проектов здесь
    2. ✅ Нет прямого доступа к FAISS или pipeline

Зависимости:
    - database/cards.db — SQLite база
"""

import logging
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

from database.card_database import DB_PATH

logger = logging.getLogger(__name__)

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Получить соединение с SQLite базой данных."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# =============================================================================
# PROJECT SERVICE
# =============================================================================

class ProjectService:
    """
    CRUD операции для управления проектами.
    
    Таблица projects:
        - id: PRIMARY KEY
        - name: TEXT UNIQUE (название проекта)
        - description: TEXT (описание)
        - species_filter: TEXT (JSON: ["Карелина"])
        - territory_filter: TEXT (JSON: ["Пруд №1"])
        - created_at, updated_at: TIMESTAMP
        - is_active: BOOLEAN (флаг активности)
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Args:
            db_path: Путь к SQLite базе данных
        """
        self.db_path = db_path
    
    def get_or_create_project(
        self,
        name: str,
        description: Optional[str] = None,
        species_filter: Optional[List[str]] = None,
        territory_filter: Optional[List[str]] = None
    ) -> int:
        """
        Получить ID проекта по имени или создать новый.
        
        Args:
            name: Уникальное название проекта
            description: Описание (опционально)
            species_filter: Список видов для фильтрации (опционально)
            territory_filter: Список территорий для фильтрации (опционально)
        
        Returns:
            int: project_id
        """
        import json
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Проверить существует ли проект
            cursor.execute('SELECT id FROM projects WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            if row:
                return row['id']
            
            # Создать новый проект
            cursor.execute('''
                INSERT INTO projects (name, description, species_filter, territory_filter, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                description,
                json.dumps(species_filter) if species_filter else None,
                json.dumps(territory_filter) if territory_filter else None,
                datetime.now().isoformat()
            ))
            
            project_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Создан проект: '{name}' (ID={project_id})")
            return project_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка создания проекта: {e}")
            raise e
        finally:
            conn.close()
    
    def get_project_by_id(self, project_id: int) -> Optional[Dict[str, Any]]:
        """
        Получить данные проекта по ID.
        
        Args:
            project_id: ID проекта
        
        Returns:
            dict: Данные проекта или None
        """
        import json
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, name, description, species_filter, territory_filter, 
                       created_at, updated_at, is_active
                FROM projects
                WHERE id = ?
            ''', (project_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            result = dict(row)
            
            # Десериализация JSON-полей
            if result['species_filter']:
                result['species_filter'] = json.loads(result['species_filter'])
            if result['territory_filter']:
                result['territory_filter'] = json.loads(result['territory_filter'])
            
            return result
            
        finally:
            conn.close()
    
    def get_project_id_by_name(self, project_name: str) -> Optional[int]:
        """
        Получить ID проекта по имени.
        
        Args:
            project_name: Название проекта
        
        Returns:
            int: project_id или None
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id FROM projects WHERE name = ?', (project_name,))
            row = cursor.fetchone()
            return row['id'] if row else None
        finally:
            conn.close()
    
    def list_projects(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Получить список всех проектов.
        
        Args:
            active_only: Возвращать только активные проекты
        
        Returns:
            list[dict]: Список проектов
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            if active_only:
                cursor.execute('''
                    SELECT id, name, description, created_at, species_filter, territory_filter
                    FROM projects
                    WHERE is_active = 1
                    ORDER BY name
                ''')
            else:
                cursor.execute('''
                    SELECT id, name, description, is_active, created_at, species_filter, territory_filter
                    FROM projects
                    ORDER BY name
                ''')
            
            return [dict(row) for row in cursor.fetchall()]
            
        finally:
            conn.close()

    def get_unique_filters(self) -> Dict[str, List[str]]:
        """
        Возвращает уникальные значения territory_filter и species_filter.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT DISTINCT territory_filter 
                FROM projects 
                WHERE territory_filter IS NOT NULL AND territory_filter != ''
            ''')
            territories = [row[0] for row in cursor.fetchall()]

            cursor.execute('''
                SELECT DISTINCT species_filter 
                FROM projects 
                WHERE species_filter IS NOT NULL AND species_filter != ''
            ''')
            species = [row[0] for row in cursor.fetchall()]

            return {"species": species, "territories": territories}
        finally:
            conn.close()

    def update_project(
        self,
        project_id: int,
        **kwargs
    ) -> bool:
        """
        Обновить данные проекта.
        
        Args:
            project_id: ID проекта
            **kwargs: Поля для обновления (description, species_filter, etc.)
        
        Returns:
            bool: True если успешно
        """
        import json
        
        if not kwargs:
            logger.warning("Нет полей для обновления проекта")
            return False
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Обработка JSON-полей
            fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['species_filter', 'territory_filter'] and value is not None:
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value))
                elif key not in ['id', 'created_at', 'updated_at']:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            if not fields:
                return False
            
            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(project_id)
            
            query = f"UPDATE projects SET {', '.join(fields)} WHERE id = ?"
            cursor.execute(query, values)
            
            if cursor.rowcount == 0:
                logger.warning(f"Проект {project_id} не найден для обновления")
                return False
            
            conn.commit()
            logger.info(f"Проект {project_id} обновлён")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка обновления проекта: {e}")
            raise e
        finally:
            conn.close()
    
    def delete_project(self, project_id: int, confirm: bool = False) -> bool:
        """
        Удалить проект (мягкое удаление: is_active = 0).
        
        Args:
            project_id: ID проекта
            confirm: Подтверждение удаления
        
        Returns:
            bool: True если успешно
        """
        if not confirm:
            raise ValueError(
                f"ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ!\n"
                f"Вы уверены, что хотите удалить проект {project_id}?\n"
                f"Передайте confirm=True для подтверждения."
            )
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Мягкое удаление: просто деактивируем
            cursor.execute('''
                UPDATE projects
                SET is_active = 0, updated_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), project_id))
            
            if cursor.rowcount == 0:
                logger.warning(f"Проект {project_id} не найден")
                return False
            
            conn.commit()
            logger.info(f"Проект {project_id} деактивирован (мягкое удаление)")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка удаления проекта: {e}")
            raise e
        finally:
            conn.close()