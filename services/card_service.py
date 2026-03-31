"""
services/card_service.py — CRUD операции для карточек особей.

Архитектурные принципы:
    1. ✅ Только CRUD для individuals и photos (проекты — в project_service.py)
    2. ✅ Нет прямого доступа к FAISS → вызываем EmbeddingService
    3. ✅ Нет print() → используем logging
    4. ✅ Нет хардкода путей → передаются через параметры
    5. ✅ Типизация → для поддержки в IDE и API

Зависимости:
    - database/cards.db — SQLite база
    - services/embedding_service.py — для работы с FAISS
    - services/project_service.py — для валидации проектов (опционально)
"""

import logging
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import numpy as np
from typing import Optional, Dict, List, Any

from database.card_database import DB_PATH
from services.project_service import ProjectService  # Для валидации, если нужно

logger = logging.getLogger(__name__)

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

REQUIRED_FIELDS = {
    'ИК-1': ['length_body', 'weight', 'sex'],
    'ИК-2': ['parent_male_id', 'parent_female_id', 'water_body_name', 'release_date'],
    'КВ-1': ['status', 'water_body_number', 'length_body', 'length_tail'],
    'КВ-2': ['status', 'water_body_name']
}

SPECIES_PREFIX = {
    'Карелина': 'K',
    'Гребенчатый': 'R',
    'Ребристый': 'R'
}

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Получить соединение с SQLite базой данных."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def _validate_template_fields(template_type: str, card_data: dict) -> None:
    """Проверяет наличие обязательных полей для выбранного шаблона."""
    required = REQUIRED_FIELDS.get(template_type, [])
    missing = [field for field in required if card_data.get(field) is None]
    if missing:
        raise ValueError(
            f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
            f"Переданные данные: {list(card_data.keys())}"
        )

def get_next_animal_number(cursor: sqlite3.Cursor, species: str) -> int:
    """Возвращает следующий порядковый номер для животного данного вида."""
    prefix = SPECIES_PREFIX.get(species, 'X')
    cursor.execute('''
        SELECT COUNT(DISTINCT CAST(
            SUBSTR(
                individual_id,
                6,
                INSTR(SUBSTR(individual_id, 6), '-') - 1
            ) AS INTEGER)
        )
        FROM individuals
        WHERE individual_id LIKE ?
    ''', (f"NT-{prefix}-%",))
    count = cursor.fetchone()[0]
    return (count or 0) + 1

def generate_card_id(
    cursor: sqlite3.Cursor, 
    species: str, 
    template_type: str, 
    animal_id: Optional[str] = None
) -> str:
    """Генерирует ID карточки. Ищет СВОБОДНЫЙ ID."""
    prefix = SPECIES_PREFIX.get(species, 'X')
    template_short = template_type.replace("-", " ")
    
    if animal_id is None:
        animal_id = f"NT-{prefix}-{get_next_animal_number(cursor, species)}"
    
    attempts = 0
    while attempts < 1000:
        card_id = f"{animal_id}-{template_short}"
        cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (card_id,))
        if not cursor.fetchone():
            return card_id
        attempts += 1
        current_num = int(animal_id.split('-')[2])
        animal_id = f"NT-{prefix}-{current_num + 1}"
    
    raise ValueError(f"Не удалось сгенерировать уникальный ID после {attempts} попыток")

def _get_next_photo_number(cursor: sqlite3.Cursor, individual_id: str) -> str:
    """Автоматически генерирует порядковый номер фото (01, 02, 03...)."""
    cursor.execute("SELECT COUNT(*) FROM photos WHERE individual_id = ?", (individual_id,))
    count = cursor.fetchone()[0]
    return f"{count + 1:02d}"

# =============================================================================
# CARD SERVICE — Основная бизнес-логика
# =============================================================================

class CardService:
    """
    Универсальный CRUD для управления карточками особей.
    
    ВАЖНО: Для работы с FAISS использует EmbeddingService (не напрямую).
    Это обеспечивает синхронизацию БД и индекса.
    """
    
    def __init__(
        self, 
        db_path: str = DB_PATH,
        embedding_service: Optional[Any] = None,
        project_service: Optional[ProjectService] = None
    ):
        """
        Args:
            db_path: Путь к SQLite базе
            embedding_service: Экземпляр EmbeddingService для работы с FAISS
            project_service: Экземпляр ProjectService (опционально, для валидации)
        """
        self.db_path = db_path
        self.embedding_service = embedding_service
        self.project_service = project_service
    
    def set_embedding_service(self, embedding_service: Any) -> None:
        """Установить сервис для работы с FAISS (dependency injection)."""
        self.embedding_service = embedding_service
    
    def set_project_service(self, project_service: ProjectService) -> None:
        """Установить сервис для работы с проектами (dependency injection)."""
        self.project_service = project_service
    
    # -------------------------------------------------------------------------
    # CREATE
    # -------------------------------------------------------------------------
    
    def save_new_individual(
        self,
        photo_path_full: Optional[str] = None,  # Обычно не исп.
        photo_path_cropped: Optional[str] = None,
        species: str = "Карелина",
        project_id: Optional[int] = None,
        template_type: str = "ИК-1",
        individual_id: Optional[str] = None,
        photo_number: Optional[str] = None,
        is_legacy: bool = False,
        **card_data
    ) -> str:
        """Сохраняет новую особь в базу данных (карточка + фотографии)."""
        _validate_template_fields(template_type, card_data)
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        if individual_id is None:
            individual_id = generate_card_id(cursor, species, template_type)
        file_suffix = Path(photo_path_cropped).suffix
        file_parent = str(Path(photo_path_cropped).parent)
        logger.info("Родительская папка сохранённого кропа: " + file_parent)
        photo_path_cropped = str(Path(photo_path_cropped).rename(
            f"{file_parent}\{individual_id}{file_suffix}"
        ))
        logger.info("Сохранённый кроп: " + photo_path_cropped)
        
        if photo_number is None:
            photo_number = _get_next_photo_number(cursor, individual_id)
        
        embedding_index = None
        
        try:
            # === ТАБЛИЦА 1: individuals ===
            cursor.execute('''
                INSERT INTO individuals (
                    individual_id, template_type, species, project_id,
                    created_at, date, notes,
                    length_body, length_tail, length_total, weight, sex,
                    birth_year_exact, birth_year_approx, origin_region,
                    length_device, weight_device,
                    parent_male_id, parent_female_id, release_date, water_body_name,
                    meeting_time, status, water_body_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, template_type, species, project_id,
                datetime.now().isoformat(),
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('notes'),
                card_data.get('length_body'), card_data.get('length_tail'),
                card_data.get('length_total'), card_data.get('weight'),
                card_data.get('sex'), card_data.get('birth_year_exact'),
                card_data.get('birth_year_approx'), card_data.get('origin_region'),
                card_data.get('length_device'), card_data.get('weight_device'),
                card_data.get('parent_male_id'), card_data.get('parent_female_id'),
                card_data.get('release_date'), card_data.get('water_body_name'),
                card_data.get('meeting_time'), card_data.get('status'),
                card_data.get('water_body_number')
            ))
            
            # === ТАБЛИЦА 2: photos (полное фото) ===
            # P.S. Мы не используем полное фото обычно.
            if photo_path_full and not is_legacy:
                cursor.execute('''
                    INSERT INTO photos (
                        individual_id, photo_type, photo_number, photo_path,
                        date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    individual_id, 'full', photo_number, photo_path_full,
                    card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                    card_data.get('meeting_time'), 1, 0, None, 0
                ))
            
            # === ТАБЛИЦА 2: photos (кроп брюшка) ===
            if photo_path_cropped:
                cursor.execute('''
                    INSERT INTO photos (
                        individual_id, photo_type, photo_number, photo_path,
                        date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    individual_id, 'cropped', photo_number, photo_path_cropped,
                    card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                    card_data.get('meeting_time'), 
                    1 if not photo_path_full else 0,
                    1,
                    -1,
                    1 if is_legacy else 0
                ))
            
            conn.commit()

            logger.info(f"Особь сохранена: {individual_id} ({template_type})")
            return individual_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка сохранения особи: {e}")
            raise e
        finally:
            conn.close()
    
    def finalize_upload(
        self,
        upload_id: int,
        template_type: str,
        species: str = "Карелина",
        **card_data
    ) -> str:
        """Завершает временную загрузку (из uploads) → создаёт постоянную карточку."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # 1. Читаем временную загрузку
        cursor.execute('''
            SELECT file_path, embedding, project_id, status
            FROM uploads
            WHERE id = ?
        ''', (upload_id,))
        
        upload = cursor.fetchone()
        if not upload:
            raise ValueError(f"Загрузка {upload_id} не найдена")
        
        if upload['status'] != 'pending':
            raise ValueError(f"Загрузка {upload_id} уже обработана (статус: {upload['status']})")
        
        # 2. Десериализуем embedding
        embedding = json.loads(upload['embedding'])
        embedding = np.array(embedding, dtype='float32')
        
        photo_path_cropped = upload['file_path']
        project_id = upload['project_id']
        
        # 3. Генерируем ID
        individual_id = generate_card_id(cursor, species, template_type)
        photo_number = _get_next_photo_number(cursor, individual_id)
        
        try:
            # === Создаём карточку ===
            cursor.execute('''
                INSERT INTO individuals (
                    individual_id, template_type, species, project_id,
                    created_at, date
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, template_type, species, project_id,
                datetime.now().isoformat(),
                card_data.get('date', datetime.now().strftime("%d.%m.%Y"))
            ))
            
            cursor.execute('''
                INSERT INTO photos (
                    individual_id, photo_type, photo_number, photo_path,
                    date_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, 'cropped', photo_number, photo_path_cropped,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                1, 1, -1, 0
            ))
            
            # === FAISS: Добавляем embedding ===
            if self.embedding_service:
                embedding_index = self.embedding_service.add(embedding, {
                    'individual_id': individual_id,
                    'photo_path': photo_path_cropped,
                    'template_type': template_type,
                    'species': species
                })
                self._update_photo_embedding_index(cursor, photo_path_cropped, embedding_index)
            
            # === Обновляем статус загрузки ===
            cursor.execute('''
                UPDATE uploads
                SET status = 'completed', card_id = ?
                WHERE id = ?
            ''', (individual_id, upload_id))
            
            conn.commit()
            
            if self.embedding_service:
                self.embedding_service.commit()
            
            logger.info(f"Загрузка {upload_id} завершена: {individual_id}")
            return individual_id
            
        except Exception as e:
            conn.rollback()
            if self.embedding_service:
                self.embedding_service.rollback()
            logger.error(f"Ошибка завершения загрузки: {e}")
            raise e
        finally:
            conn.close()
    
    def _update_photo_embedding_index(
        self, 
        cursor: sqlite3.Cursor, 
        photo_path: str, 
        embedding_index: int
    ):
        """Обновить embedding_index для фотографии в БД."""
        cursor.execute('''
            UPDATE photos 
            SET embedding_index = ?, is_processed = 1
            WHERE photo_path = ?
        ''', (embedding_index, photo_path))
    
    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------
    
    def add_encounter(
        self,
        individual_id: str,
        template_type: str,
        photo_path_full: Optional[str] = None,
        photo_path_cropped: Optional[str] = None,
        **card_data
    ) -> str:
        """Добавляет НОВУЮ ВСТРЕЧУ (КВ-1/КВ-2) для существующей особи."""
        if template_type not in ['КВ-1', 'КВ-2']:
            raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")
        
        _validate_template_fields(template_type, card_data)
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        photo_number = _get_next_photo_number(cursor, individual_id)
        
        # Получаем project_id из существующей особи
        cursor.execute('SELECT project_id FROM individuals WHERE individual_id = ?', (individual_id,))
        row = cursor.fetchone()
        project_id = row['project_id'] if row else 1
        
        try:
            cursor.execute('''
                INSERT INTO individuals (
                    individual_id, template_type, species, project_id,
                    date, meeting_time, status, water_body_number, water_body_name,
                    length_body, length_tail, length_total, weight, sex, notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                individual_id, template_type, "Карелина", project_id,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('time'), card_data.get('status'),
                card_data.get('water_body_number'), card_data.get('water_body_name'),
                card_data.get('length_body'), card_data.get('length_tail'),
                card_data.get('length_total'), card_data.get('weight'),
                card_data.get('sex'), card_data.get('notes'),
                datetime.now().isoformat()
            ))
            
            if photo_path_full:
                cursor.execute('''
                    INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (individual_id, 'full', photo_number, photo_path_full, card_data.get('date'), 0, 0, None, 0))
            
            if photo_path_cropped:
                cursor.execute('''
                    INSERT INTO photos (individual_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (individual_id, 'cropped', photo_number, photo_path_cropped, card_data.get('date'), 0, 1, -1, 0))
            
            conn.commit()
            
            logger.info(f"Встреча {template_type} добавлена к особи {individual_id}")
            return photo_number
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка добавления встречи: {e}")
            raise e
        finally:
            conn.close()
    
    def update_individual(self, individual_id: str, **kwargs) -> bool:
        """Обновляет данные существующей особи."""
        if not kwargs:
            logger.warning("Нет полей для обновления")
            return False
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        fields = [f"{key} = ?" for key in kwargs.keys()]
        values = list(kwargs.values()) + [individual_id]
        
        query = f"UPDATE individuals SET {', '.join(fields)} WHERE individual_id = ?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        
        logger.info(f"Особь {individual_id} обновлена")
        return True
    
    # -------------------------------------------------------------------------
    # DELETE
    # -------------------------------------------------------------------------
    
    def delete_individual(self, individual_id: str, delete_photos: bool = True, confirm: bool = False) -> bool:
        """Полностью удаляет особь и все её фото (hard delete)."""
        if not confirm:
            raise ValueError(
                f"ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ!\n"
                f"Вы уверены, что хотите удалить {individual_id}?\n"
                f"Передайте confirm=True для подтверждения."
            )
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT individual_id FROM individuals WHERE individual_id = ?', (individual_id,))
            if not cursor.fetchone():
                raise ValueError(f"Особь {individual_id} не найдена в базе.")
            
            photo_paths = []
            if delete_photos:
                cursor.execute('SELECT photo_path FROM photos WHERE individual_id = ?', (individual_id,))
                photo_paths = [row['photo_path'] for row in cursor.fetchall()]
                cursor.execute('DELETE FROM photos WHERE individual_id = ?', (individual_id,))
            
            cursor.execute('DELETE FROM individuals WHERE individual_id = ?', (individual_id,))
            
            conn.commit()
            
            if delete_photos:
                for photo_path in photo_paths:
                    try:
                        Path(photo_path).unlink()
                        logger.info(f"Удалён файл: {photo_path}")
                    except FileNotFoundError:
                        pass
            
            logger.info(f"Особь {individual_id} удалена")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка удаления особи: {e}")
            raise e
        finally:
            conn.close()
    
    # -------------------------------------------------------------------------
    # READ
    # -------------------------------------------------------------------------
    
    def get_individual_photos(self, individual_id: str) -> List[Dict[str, Any]]:
        """Получает все фотографии особи из базы данных."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT photo_id, photo_type, photo_number, photo_path, 
                   date_taken, is_main, is_legacy
            FROM photos
            WHERE individual_id = ?
            ORDER BY photo_number ASC, photo_type DESC
        ''', (individual_id,))
        
        photos = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return photos
    
    def get_individual(self, individual_id: str) -> Optional[Dict[str, Any]]:
        """Получает данные особи по ID (с информацией о проекте)."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # JOIN с projects для получения названия проекта
        cursor.execute('''
            SELECT i.*, p.name as project_name
            FROM individuals i
            LEFT JOIN projects p ON i.project_id = p.id
            WHERE i.individual_id = ?
        ''', (individual_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_individuals_by_project(self, project_id: int) -> List[Dict[str, Any]]:
        """Получает список особей по проекту."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT individual_id, template_type, species, created_at, date
            FROM individuals
            WHERE project_id = ?
            ORDER BY created_at DESC
        ''', (project_id,))
        
        individuals = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return individuals