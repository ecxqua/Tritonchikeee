"""
services/card_service.py — CRUD операции для карточек особей.

Архитектурные принципы:
    1. ✅ Только CRUD для cards и photos (проекты — в project_service.py)
    2. ✅ Нет прямого доступа к FAISS → вызываем EmbeddingService
    3. Prototypes - особи, cards - карточки особей

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
import re  # Для извлечения id особи из id карточки
import uuid
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

def get_next_prototype_number(cursor: sqlite3.Cursor, species: str) -> int:
    """Возвращает следующий порядковый номер для животного данного вида."""
    prefix = SPECIES_PREFIX.get(species, 'X')
    cursor.execute('''
        SELECT COUNT(DISTINCT CAST(
            SUBSTR(
                card_id,
                6,
                INSTR(SUBSTR(card_id, 6), '-') - 1
            ) AS INTEGER)
        )
        FROM cards
        WHERE card_id LIKE ?
    ''', (f"NT-{prefix}-%",))
    count = cursor.fetchone()[0]
    return (count or 0) + 1

def generate_card_id(
    cursor: sqlite3.Cursor, 
    species: str, 
    template_type: str, 
    prototype_id: Optional[str] = None
) -> str:
    """Генерирует ID карточки. Ищет СВОБОДНЫЙ ID."""
    prefix = SPECIES_PREFIX.get(species, 'X')
    template_short = template_type.replace("-", " ")
    
    if prototype_id is None:
        prototype_id = f"NT-{prefix}-{get_next_prototype_number(cursor, species)}"
    
    attempts = 0
    while attempts < 1000:
        card_id = f"{prototype_id}-{template_short}"
        cursor.execute('SELECT card_id FROM cards WHERE card_id = ?', (card_id,))
        if not cursor.fetchone():
            return card_id
        attempts += 1
        current_num = int(prototype_id.split('-')[2])
        prototype_id = f"NT-{prefix}-{current_num + 1}"
    
    raise ValueError(f"Не удалось сгенерировать уникальный ID после {attempts} попыток")

def _get_next_photo_number(cursor: sqlite3.Cursor, card_id: str) -> str:
    """Автоматически генерирует порядковый номер фото (01, 02, 03...)."""
    cursor.execute("SELECT COUNT(*) FROM photos WHERE card_id = ?", (card_id,))
    count = cursor.fetchone()[0]
    return f"{count + 1:02d}"

def rename_photo(card_id: str, photo_path: str, suffix: str):
    """
        Переименовывае фотографию с уникальным названием,
        которая прикрепляется к записи в photos.

        Формат: 'card_id' + 'suffix' + 'uuid для фото'
    """
    # Генерируем название фото.
    photo_name = card_id + "_" + suffix + "_" + str(uuid.uuid4())
    # Меняем название.
    file_suffix = Path(photo_path).suffix
    file_parent = str(Path(photo_path).parent)
    logger.info("Родительская папка сохранённого кропа: " + file_parent)
    photo_path = str(Path(photo_path).rename(
        f"{file_parent}\{photo_name}{file_suffix}"
    ))
    return photo_path

def extract_prototype_id(card_id: str) -> str:
    """
    Извлекает ID прототипа из card_id.
    Формат: NT-К-1-ИК1 -> NT-К-1
    Использует последнее вхождение '-' как разделитель типа карточки.
    """
    if not card_id:
        return ""
    parts = card_id.rsplit('-', 1)
    return parts[0] if len(parts) > 1 else card_id

def form_card_id(prototype_id: str, template_type: str):
    """
    Формируем пару id+template_type

    Args:
        prototype_id (str): номер id особи (NT-K-13)
        template_type (str): шаблон карточки (КВ-1/ИК-1)
    """
    return prototype_id + "-" + template_type.replace("-", "")

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
        photo_path_cropped: Optional[str],
        species: str = "Карелина",
        project_id: Optional[int] = None,
        template_type: str = "ИК-1",
        card_id: Optional[str] = None,
        photo_number: Optional[str] = None,
        is_legacy: bool = False,
        **card_data
    ) -> str:
        """
        Сохраняет новую особь в базу данных (основная карточка + фотографии).

        Переименовывает введённое cropped изображение в уникальное id по
        схеме в rename_photo.

        Args:
            photo_path_cropped (str): путь к кропу брюшка.
            species (str): тип тритона для добавления.
            project_id (str): id проекта, куда добавится тритон (не рекомендуется оставлять пустым).
            card_id: номер карточки (рекомендуется оставить пустым!)
            photo_number: номер фото (рекомендуется оставить пустым!)
            is_legacy: поле в таблице photos (рекомендуется оставить)
        """
        _validate_template_fields(template_type, card_data)
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        if card_id is None:
            # Генерируем card_id с учётом template_type
            card_id = generate_card_id(cursor, species, template_type)
        # Сохраняем фотографию.
        if photo_path_cropped:
            photo_path_cropped = rename_photo(card_id, photo_path_cropped, suffix="cropped")
        logger.info("Сохранённый кроп: " + photo_path_cropped)
        
        if photo_number is None:
            photo_number = _get_next_photo_number(cursor, card_id)
        
        embedding_index = None
        
        try:
            # === ТАБЛИЦА 1: cards ===
            cursor.execute('''
                INSERT INTO cards (
                    card_id, template_type, species, project_id,
                    created_at, date, notes,
                    length_body, length_tail, length_total, weight, sex,
                    birth_year_exact, birth_year_approx, origin_region,
                    length_device, weight_device,
                    parent_male_id, parent_female_id, release_date, water_body_name,
                    meeting_time, status, water_body_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                card_id, template_type, species, project_id,
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
            
            # === ТАБЛИЦА 2: photos (кроп брюшка) ===
            if photo_path_cropped:
                cursor.execute('''
                    INSERT INTO photos (
                        card_id, photo_type, photo_number, photo_path,
                        date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    card_id, 'cropped', photo_number, photo_path_cropped,
                    card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                    card_data.get('meeting_time'), 
                    1,
                    1,
                    -1,
                    1 if is_legacy else 0
                ))
            
            conn.commit()

            logger.info(f"Особь сохранена: {card_id} ({template_type})")
            return card_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка сохранения особи: {e}")
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
        prototype_id: str,
        template_type: str,
        species: str = "Карелина",
        photo_path_cropped: Optional[str] = None,
        **card_data
    ) -> str:
        """
        Добавляет НОВУЮ ВСТРЕЧУ (КВ-1/КВ-2) для существующей особи (карточка повторной встречи)

        Args:
            prototype_id (str): id особи (не карточка) вида NT-K-13 (без типа карточки)
            template_type (str): тип карточки (КВ-1/КВ-2)
            photo_path_full (str): путь к полной фотографии особи (исходник)
            photo_path_cropped (str): путь к вырезанному брюшку
            **card_data (dict или аргументы): данные для заполнения карточки.

        Returns (str): номер фотографии в БД.
        """
        if template_type not in ['КВ-1', 'КВ-2']:
            raise ValueError("Для добавления встречи используйте шаблоны КВ-1 или КВ-2")
        
        _validate_template_fields(template_type, card_data)

        # Правильно генерируем ID новой особи.
        # NT-K-13 -> NT-K-13-КВ1
        card_id = form_card_id(prototype_id, template_type)
        # Сохраняем фотографию
        if photo_path_cropped:
            photo_path_cropped = rename_photo(card_id, photo_path_cropped, suffix="cropped")

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        photo_number = _get_next_photo_number(cursor, card_id)
        
        # Получаем project_id из существующей особи
        cursor.execute('SELECT project_id FROM cards WHERE card_id = ?', (card_id,))
        row = cursor.fetchone()
        project_id = row['project_id'] if row else 1
        
        try:
            cursor.execute('''
                INSERT INTO cards (
                    card_id, template_type, species, project_id,
                    date, meeting_time, status, water_body_number, water_body_name,
                    length_body, length_tail, length_total, weight, sex, notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                card_id, template_type, species, project_id,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('time'), card_data.get('status'),
                card_data.get('water_body_number'), card_data.get('water_body_name'),
                card_data.get('length_body'), card_data.get('length_tail'),
                card_data.get('length_total'), card_data.get('weight'),
                card_data.get('sex'), card_data.get('notes'),
                datetime.now().isoformat()
            ))
            
            if photo_path_cropped:
                cursor.execute('''
                    INSERT INTO photos (card_id, photo_type, photo_number, photo_path, date_taken, is_main, is_processed, embedding_index, is_legacy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (card_id, 'cropped', photo_number, photo_path_cropped, card_data.get('date'), 0, 1, -1, 0))
            
            conn.commit()
            
            logger.info(f"Встреча {template_type} добавлена к особи {card_id}")
            return photo_number
        except sqlite3.IntegrityError as e:
            conn.rollback()
            logger.error(f"Ошибка указания ID. Возможно, запись с таким ID уже есть!")
            raise sqlite3.IntegrityError("Ошибка указания ID. Возможно, запись с таким ID уже есть!")
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка добавления встречи: {e}")
            raise e
        finally:
            conn.close()
    
    def update_individual(self, card_id: str, **kwargs) -> bool:
        """Обновляет данные существующей карточки."""
        if not kwargs:
            logger.warning("Нет полей для обновления")
            return False
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        fields = [f"{key} = ?" for key in kwargs.keys()]
        values = list(kwargs.values()) + [card_id]
        
        query = f"UPDATE cards SET {', '.join(fields)} WHERE card_id = ?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        
        logger.info(f"Особь {card_id} обновлена")
        return True
    
    # -------------------------------------------------------------------------
    # DELETE
    # -------------------------------------------------------------------------
    
    def delete_individual(self, card_id: str, delete_photos: bool = True, confirm: bool = False) -> bool:
        """Полностью удаляет карточку и все её фото (hard delete)."""
        if not confirm:
            raise ValueError(
                f"ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ!\n"
                f"Вы уверены, что хотите удалить {card_id}?\n"
                f"Передайте confirm=True для подтверждения."
            )
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT card_id FROM cards WHERE card_id = ?', (card_id,))
            if not cursor.fetchone():
                raise ValueError(f"Особь {card_id} не найдена в базе.")
            
            photo_paths = []
            if delete_photos:
                cursor.execute('SELECT photo_path FROM photos WHERE card_id = ?', (card_id,))
                photo_paths = [row['photo_path'] for row in cursor.fetchall()]
                cursor.execute('DELETE FROM photos WHERE card_id = ?', (card_id,))
            
            cursor.execute('DELETE FROM cards WHERE card_id = ?', (card_id,))
            
            conn.commit()
            
            if delete_photos:
                for photo_path in photo_paths:
                    try:
                        Path(photo_path).unlink()
                        logger.info(f"Удалён файл: {photo_path}")
                    except FileNotFoundError:
                        pass
            
            logger.info(f"Особь {card_id} удалена")
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
    
    def get_individual_photos(self, card_id: str) -> List[Dict[str, Any]]:
        """Получает все фотографии карточки из базы данных."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT photo_id, photo_type, photo_number, photo_path, 
                   date_taken, is_main, is_legacy
            FROM photos
            WHERE card_id = ?
            ORDER BY photo_number ASC, photo_type DESC
        ''', (card_id,))
        
        photos = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return photos
    
    def get_individual(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Получает данные карточки по ID (с информацией о проекте)."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # JOIN с projects для получения названия проекта
        cursor.execute('''
            SELECT i.*, p.name as project_name
            FROM cards i
            LEFT JOIN projects p ON i.project_id = p.id
            WHERE i.card_id = ?
        ''', (card_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_cards_by_project(self, project_id: int) -> List[Dict[str, Any]]:
        """Получает список карточек по проекту."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT card_id, template_type, species, created_at, date
            FROM cards
            WHERE project_id = ?
            ORDER BY created_at DESC
        ''', (project_id,))
        
        cards = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return cards

    def get_matching_card_ids(self, prototype_id: str) -> List[str]:
        """Внутренний метод: находит все card_id, относящиеся к прототипу."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        # Prefix match + exact fallback (работает быстро благодаря PK индексу)
        cursor.execute('''
            SELECT card_id FROM cards 
            WHERE card_id LIKE ? || '-%' OR card_id = ?
        ''', (prototype_id, prototype_id))
        
        ids = [row['card_id'] for row in cursor.fetchall()]
        conn.close()
        return ids

    def get_prototype(self, prototype_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает агрегированные данные по прототипу.
        Возвращает мета-информацию + список всех привязанных карточек.
        """
        card_ids = self.get_matching_card_ids(prototype_id)
        if not card_ids:
            return None

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join('?' for _ in card_ids)
        cursor.execute(f'''
            SELECT i.*, p.name as project_name
            FROM cards i
            LEFT JOIN projects p ON i.project_id = p.id
            WHERE i.card_id IN ({placeholders})
            ORDER BY i.template_type, i.created_at
        ''', card_ids)
        
        cards = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Формируем агрегированный ответ
        # Берем общие поля из первой карточки (ожидается консистентность данных)
        base = cards[0]
        return {
            'prototype_id': prototype_id,
            'species': base.get('species'),
            'project_id': base.get('project_id'),
            'project_name': base.get('project_name'),
            'total_cards': len(cards),
            'cards': cards  # Полный список карточек (ИК-1, ИК-2, КВ-1, КВ-2)
        }

    def get_prototype_photos(self, prototype_id: str) -> List[Dict[str, Any]]:
        """
        Получает ВСЕ фотографии всех карточек, относящихся к прототипу.
        Включает поле template_type для понимания, к какой карточке относится фото.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT ph.*, ind.template_type
            FROM photos ph
            JOIN cards ind ON ph.card_id = ind.card_id
            WHERE ind.card_id LIKE ? || '-%' OR ind.card_id = ?
            ORDER BY ind.template_type, ph.photo_type DESC, ph.photo_number ASC
        ''', (prototype_id, prototype_id))
        
        photos = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return photos

    def get_prototype_by_card_id(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Удобный враппер: ищет прототип по ID любой из его карточек."""
        prototype_id = extract_prototype_id(card_id)
        return self.get_prototype(prototype_id) if prototype_id else None

    def get_prototype_photos_by_card_id(self, card_id: str) -> List[Dict[str, Any]]:
        """Удобный враппер для фото по ID карточки."""
        prototype_id = extract_prototype_id(card_id)
        return self.get_prototype_photos(prototype_id) if prototype_id else []

    def search_prototypes(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Поиск прототипов по частичному совпадению ID или виду.
        Полезно для автодополнения в REST-клиенте.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # Ищем по префиксу ID или виду
        pattern = f"%{query}%"
        cursor.execute('''
            SELECT card_id, species, project_id, template_type
            FROM cards
            WHERE card_id LIKE ? OR species LIKE ?
            GROUP BY card_id
            LIMIT ?
        ''', (pattern, pattern, limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Группируем по прототипам, убирая дубликаты типов карточек
        prototype_map = {}
        for row in results:
            pid = extract_prototype_id(row['card_id'])
            if pid not in prototype_map:
                prototype_map[pid] = {
                    'prototype_id': pid,
                    'species': row['species'],
                    'project_id': row['project_id'],
                    'matched_cards': []
                }
            prototype_map[pid]['matched_cards'].append(row['card_id'])
            
        return list(prototype_map.values())

    def get_prototypes_by_project(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Возвращает список всех прототипов (биологических особей) в рамках проекта.
        Группирует карточки по prototype_id (NT-К-1).
        
        Проверяет целостность архитектуры:
        Если хотя бы одна карточка прототипа привязана к другому проекту, 
        выбрасывает исключение DB_INTEGRITY_ERROR.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # 1. Забираем все карточки проекта
        cursor.execute('''
            SELECT card_id, template_type, species, created_at, date
            FROM cards
            WHERE project_id = ?
            ORDER BY card_id ASC
        ''', (project_id,))
        
        project_rows = cursor.fetchall()
        if not project_rows:
            conn.close()
            return []
            
        # 2. Группируем по prototype_id
        prototype_map: Dict[str, Dict[str, Any]] = {}
        for row in project_rows:
            proto_id = extract_prototype_id(row['card_id'])
            
            if proto_id not in prototype_map:
                prototype_map[proto_id] = {
                    'prototype_id': proto_id,
                    'species': row['species'],
                    'project_id': project_id,
                    'cards': []
                }
                
            prototype_map[proto_id]['cards'].append({
                'card_id': row['card_id'],
                'template_type': row['template_type'],
                'created_at': row['created_at'],
                'date': row['date']
            })
            
        # 3. ВАЛИДАЦИЯ ЦЕЛОСТНОСТИ
        # Проверяем, не "размазаны" ли эти особи по другим проектам в БД
        all_ind_ids = [r['card_id'] for r in project_rows]
        placeholders = ','.join('?' for _ in all_ind_ids)
        
        cursor.execute(f'''
            SELECT card_id, project_id 
            FROM cards 
            WHERE card_id IN ({placeholders}) 
            AND project_id != ?
        ''', all_ind_ids + [project_id])
        
        cross_project_conflicts = cursor.fetchall()
        conn.close()
        
        if cross_project_conflicts:
            # Извлекаем уникальные ID прототипов с нарушенной целостностью
            broken_prototypes = list(set(
                extract_prototype_id(row['card_id']) 
                for row in cross_project_conflicts
            ))
            raise ValueError(
                f"DB_INTEGRITY_ERROR: Карточки следующих особей привязаны к разным проектам: {broken_prototypes}. "
                "По архитектуре проекта все карточки одной особи (NT-К-1) должны находиться строго в одном проекте. "
                "Проверьте логику импорта или миграции данных."
            )
            
        return list(prototype_map.values())

    def get_all_prototypes(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех прототипов (биологических особей) во всей базе данных.
        Группирует карточки по prototype_id.
        Проверяет глобальную целостность архитектуры.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # 1. Забираем ВСЕ карточки из базы
        cursor.execute('''
            SELECT card_id, template_type, species, project_id, created_at, date
            FROM cards
            ORDER BY card_id ASC
        ''')
        
        all_rows = cursor.fetchall()
        if not all_rows:
            conn.close()
            return []
            
        # 2. Группировка по prototype_id
        prototype_map: Dict[str, Dict[str, Any]] = {}
        for row in all_rows:
            proto_id = extract_prototype_id(row['card_id'])
            
            if proto_id not in prototype_map:
                prototype_map[proto_id] = {
                    'prototype_id': proto_id,
                    'species': row['species'],
                    'projects': set(),
                    'cards': []
                }
                
            prototype_map[proto_id]['projects'].add(row['project_id'])
            prototype_map[proto_id]['cards'].append({
                'card_id': row['card_id'],
                'template_type': row['template_type'],
                'created_at': row['created_at'],
                'date': row['date']
            })
            
        # 3. ВАЛИДАЦИЯ ГЛОБАЛЬНОЙ ЦЕЛОСТНОСТИ
        # Проверяем, не "размазаны" ли особи по разным проектам
        broken_prototypes = [
            pid for pid, data in prototype_map.items() if len(data['projects']) > 1
        ]
        
        if broken_prototypes:
            conn.close()
            raise ValueError(
                f"DB_INTEGRITY_ERROR: Карточки следующих особей привязаны к разным проектам: {broken_prototypes}. "
                "По архитектуре все карточки одной особи должны находиться строго в одном проекте."
            )
            
        # 4. Формирование чистого результата (убираем служебный set проектов)
        result = []
        for proto_id, data in prototype_map.items():
            result.append({
                'prototype_id': proto_id,
                'species': data['species'],
                'project_id': next(iter(data['projects'])),
                'cards': data['cards']
            })
            
        conn.close()
        return result