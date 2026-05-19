"""
services/card_service.py — CRUD операции для карточек особей.

Архитектурные принципы:
    1. CRUD для cards и photos (проекты — в project_service.py)
    2. Нет прямого доступа к FAISS → вызываем EmbeddingService
    3. cards - уникальные карточки особей
    4. Валидация полей карточек

Зависимости:
    - database/cards.db — SQLite база
    - services/embedding_service.py — для работы с FAISS
    - services/project_service.py — для валидации проектов (опционально)
"""

import logging
from warnings import deprecated
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

BASE_FIELDS = {
    'card_id',
    'prototype_id',
    'template_type',
    'species',
    'territory',
    'project_id',
    'project_name',
    'created_at'
}

ALLOWED_FIELDS = {
    'ИК-1': [
        'date', 'length_body', 'weight', 'sex', 'length_tail',
        'birth_year_exact', 'birth_year_approx', 
        'origin_region', 'length_device', 'weight_device', 'notes',
	'species',
    ],
    'ИК-2': [
        'date', 'release_date', 'parent_male_id', 'parent_female_id',
        'length_total', 'weight', 'water_body_name', 'notes',
	'species',
    ],
    'КВ-1': [
        'date', 'meeting_time', 'length_body', 'length_tail',
        'weight', 'sex', 'status', 'water_body_number',
        'length_device', 'weight_device', 'notes',
	'species',
    ],
    'КВ-2': [
        'date', 'meeting_time', 'length_total',
        'status', 'water_body_name', 'notes',
	'species',
    ]
}

ALLOWED_COMMIT_FIELDS = {
    'species', 'date', 'notes', 'length_body', 'length_tail', 'length_total',
    'weight', 'sex', 'birth_year_exact', 'birth_year_approx', 'origin_region',
    'length_device', 'weight_device', 'parent_male_id', 'parent_female_id',
    'release_date', 'water_body_name', 'meeting_time', 'status', 'water_body_number'
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

# ВАЛИДАТОР
def validate_template_fields(
        template_type: str,
        card_data: dict,
        require: bool = True
    ) -> dict:
    """
    Валидатор. Проверяет наличие обязательных полей и отсутствие лишних для выбранного шаблона.
    Возвращает очищенный dict card_data (содержит только допустимые поля).
    """
    allowed = ALLOWED_FIELDS.get(template_type)
    if allowed is None:
        raise ValueError(f"Неизвестный тип шаблона: {template_type}")

    # 1. Проверка обязательных полей
    if require:
        required = REQUIRED_FIELDS.get(template_type, [])
        missing = [f for f in required if f not in card_data or card_data.get(f) is None]
        if missing:
            raise ValueError(
                f"Для шаблона '{template_type}' обязательны поля: {', '.join(missing)}\n"
                f"Переданные данные: {list(card_data.keys())}"
            )

    # 2. Проверка на лишние поля
    extra = [f for f in card_data.keys() if f not in allowed]
    #if extra:
    #    raise ValueError(
    #        f"Шаблон '{template_type}' не поддерживает следующие поля: {', '.join(extra)}\n"
    #        f"Допустимые поля: {allowed}"
    #    )

    # 3. Возвращаем только разрешённые поля (защита от SQL-инъекций/мусора)
    return {k: v for k, v in card_data.items() if k in allowed and k not in extra}

# ВАЛИДАТОР НА ЧТЕНИЕ
def filter_card_by_template(
        card_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
    """Валидтор на чтениею Фильтрует dict карточки, оставляя только системные
    и разрешённые для шаблона поля."""
    if not card_data:
        return None
        
    template_type = card_data.get('template_type')
    
    # Безопасный fallback: если шаблон неизвестен, отдаём всё (например, для легаси-данных)
    if template_type not in ALLOWED_FIELDS:
        return card_data
        
    allowed = BASE_FIELDS | set(ALLOWED_FIELDS[template_type])
    return {k: v for k, v in card_data.items() if k in allowed}

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
) -> str:
    """Генерирует ID карточки. Ищет СВОБОДНЫЙ ID."""
    prefix = SPECIES_PREFIX.get(species, 'X')
    
    prototype_id = f"NT-{prefix}-{get_next_prototype_number(cursor, species)}"
    
    attempts = 0
    while attempts < 1000:
        card_id = f"{prototype_id}"
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
        Переименовывает фотографию с уникальным названием,
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
        f"{file_parent}/{photo_name}{file_suffix}"
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
    (LEGACY)
    Формируем пару id+template_type

    Args:
        prototype_id (str): номер id особи (NT-K-13)
        template_type (str): шаблон карточки (КВ-1/ИК-1)
    """
    return prototype_id

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
    
    def _save_new_individual(
        self,
        photo_path_cropped: Optional[str],
        photo_path_full: Optional[str],
        card_data: Dict[str, Any],
        species: str = "Карелина",
        project_id: Optional[int] = None,
        template_type: str = "ИК-1",
        photo_number: Optional[str] = None,
        is_legacy: bool = False,
    ) -> Dict[str, Any]:
        """
        Сохраняет новую особь в базу данных (основная карточка + фотографии).

        Переименовывает введённое cropped изображение в уникальное id по
        схеме в rename_photo.

        Args:
            photo_path_cropped (str): путь к кропу брюшка.
            photo_path_full (str): путь к полному фото.
            species (str): тип тритона для добавления.
            project_id (str): id проекта, куда добавится тритон (не рекомендуется оставлять пустым).
            card_id: номер карточки (рекомендуется оставить пустым!)
            photo_number: номер фото (рекомендуется оставить пустым!)
            is_legacy: поле в таблице photos (рекомендуется оставить)
        
        Returns Dict[str, Any]:
            crop_path: путь к вырезанному брюшку
            success: успешность операции
            photo_id: id сохранённой фотографии в photos
            card_id: id сохранённой карточки
            error: сообщение об ошибке
        """
        result = {
            "crop_path": None,
            "full_path": None,
            "card_id": None,
            "photo_id": None,
            "full_photo_id": None,
            "success": False,
            "error": None
        }
        card_data = validate_template_fields(template_type, card_data)
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # Генерируем card_id с учётом template_type
        card_id = generate_card_id(cursor, species)
        result['card_id'] = card_id
        # Сохраняем фотографию.
        if photo_path_cropped:
            photo_path_cropped = rename_photo(card_id, photo_path_cropped, suffix="cropped")
            result['crop_path'] = photo_path_cropped
        if photo_path_full:
            photo_path_full = rename_photo(card_id, photo_path_full, suffix="full")
            result['full_path'] = photo_path_full
        logger.info(f"Сохранённый кроп: {photo_path_cropped}")
        
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

            # Создание коммита инициализации.
            commit_id = str(uuid.uuid4())
            commit_fields = {
                'commit_id': commit_id,
                'card_id': card_id,
                'species': card_data.get('species'),
                'date': card_data.get('date'),
                'notes': card_data.get('notes'),
                'length_body': card_data.get('length_body'),
                'length_tail': card_data.get('length_tail'),
                'length_total': card_data.get('length_total'),
                'weight': card_data.get('weight'),
                'sex': card_data.get('sex'),
                'birth_year_exact': card_data.get('birth_year_exact'),
                'birth_year_approx': card_data.get('birth_year_approx'),
                'origin_region': card_data.get('origin_region'),
                'length_device': card_data.get('length_device'),
                'weight_device': card_data.get('weight_device'),
                'parent_male_id': card_data.get('parent_male_id'),
                'parent_female_id': card_data.get('parent_female_id'),
                'release_date': card_data.get('release_date'),
                'water_body_name': card_data.get('water_body_name'),
                'meeting_time': card_data.get('meeting_time'),
                'status': card_data.get('status'),
                'water_body_number': card_data.get('water_body_number'),
                'created_at': datetime.now().isoformat(),
            }

            # Динамический INSERT
            cols = list(commit_fields.keys())
            placeholders = ', '.join(['?' for _ in cols])
            query = f"INSERT INTO commits ({', '.join(cols)}) VALUES ({placeholders})"
            cursor.execute(query, [commit_fields[c] for c in cols])
            logger.info(f"Коммит к особи {card_id} создан")
            
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
                # Сохраняем photo_id
                result['photo_id'] = cursor.lastrowid

            if photo_path_full:
                cursor.execute('''
                    INSERT INTO photos (
                        card_id, photo_type, photo_number, photo_path,
                        date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    card_id, 'full', photo_number, photo_path_full,
                    card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                    card_data.get('meeting_time'), 
                    1,
                    1,
                    -1,
                    1 if is_legacy else 0
                ))
                # Сохраняем photo_id
                result['full_photo_id'] = cursor.lastrowid
            
            conn.commit()

            logger.info(f"Особь сохранена: {card_id} ({template_type})")
            result['success'] = True
            return result
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка сохранения особи: {e}")
            raise e
        finally:
            conn.close()

    def _add_photo_to_card(
        self,
        photo_path: str,
        prefix: str,
        card_id: str
    ) -> Dict[str, Any]:
        """
        Добавляет фото к карточке по card_id.
        
        Returns Dict[str, Any]:
            crop_path: путь к вырезанному брюшку
            full_path: путь к полному фото
            success: успешность операции
            card_id: id сохранённой карточки
            photo_id: id добавленного вырезанного бюршка в photos
            error: сообщение об ошибке
        """
        result = {
            "crop_path": None,
            "full_path": None,
            "card_id": card_id,
            "photo_id": None,
            "success": False,
            "error": None
        }
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        photo_number = _get_next_photo_number(cursor, card_id)
        card_data = self.get_card(card_id=card_id)
        # Переименование файла
        photo_path_cropped = rename_photo(card_id, photo_path, suffix=prefix)
        result['crop_path'] = photo_path_cropped
        if not card_data:
            logger.error(f"При добавлении фотографии к карточке не вышло получить card_data")
            result["error"] = "При добавлении фотографии к карточке не вышло получить card_data"
            return result
        try:
            cursor.execute('''
                INSERT INTO photos (
                    card_id, photo_type, photo_number, photo_path,
                    date_taken, time_taken, is_main, is_processed, embedding_index, is_legacy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                card_id, prefix, photo_number, photo_path_cropped,
                card_data.get('date', datetime.now().strftime("%d.%m.%Y")),
                card_data.get('meeting_time'), 
                1,
                1,
                -1,
                0
            ))
            # Сохраняем photo_id
            result['photo_id'] = cursor.lastrowid
            conn.commit()
            logger.info(f"Фото к карточке добавлено: {photo_path_cropped} ({card_id})")
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка добавления фотографии к карточке: {e}")
            raise e
        finally:
            conn.close()
            result['card_id'] = card_id
            result['success'] = True
            return result
    
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
    def _commit_card(
        self,
        card_id: str,
        template_type: Optional[str] = None,
        added_photo_ids: Optional[list[int]] = None,
        card_data: Optional[dict] = None
    ) -> bool:
        """Обновляет данные существующей карточки коммитом (история)."""
        # if not card_data:
        #     logger.warning("Нет полей для обновления")
        #     return False
        # Валидация полей (если нужна).
        if template_type and card_data:
            card_data = validate_template_fields(template_type, card_data, False)

        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        if card_data:
            fields = [f"{key} = ?" for key in card_data.keys()]
            values = list(card_data.values()) + [card_id]
            query = f"UPDATE cards SET {', '.join(fields)} WHERE card_id = ?"
            cursor.execute(query, values)
        # 2. Создаём коммит — снапшот состояния ПОСЛЕ обновления
        commit_id = str(uuid.uuid4())
        
        # Формируем список полей для INSERT в commits
        # Берём новые значения из card_data, остальные — None (можно доработать под чтение старых)
        if not card_data:
            card_data = dict()
        commit_fields = {
            'commit_id': commit_id,
            'card_id': card_id,
            'species': card_data.get('species'),
            'date': card_data.get('date'),
            'notes': card_data.get('notes'),
            'length_body': card_data.get('length_body'),
            'length_tail': card_data.get('length_tail'),
            'length_total': card_data.get('length_total'),
            'weight': card_data.get('weight'),
            'sex': card_data.get('sex'),
            'birth_year_exact': card_data.get('birth_year_exact'),
            'birth_year_approx': card_data.get('birth_year_approx'),
            'origin_region': card_data.get('origin_region'),
            'length_device': card_data.get('length_device'),
            'weight_device': card_data.get('weight_device'),
            'parent_male_id': card_data.get('parent_male_id'),
            'parent_female_id': card_data.get('parent_female_id'),
            'release_date': card_data.get('release_date'),
            'water_body_name': card_data.get('water_body_name'),
            'meeting_time': card_data.get('meeting_time'),
            'status': card_data.get('status'),
            'water_body_number': card_data.get('water_body_number'),
            'created_at': datetime.now().isoformat(),
        }

        # Динамический INSERT
        cols = list(commit_fields.keys())
        placeholders = ', '.join(['?' for _ in cols])
        query = f"INSERT INTO commits ({', '.join(cols)}) VALUES ({placeholders})"
        cursor.execute(query, [commit_fields[c] for c in cols])
        logger.info(f"Коммит к особи {card_id} создан")

        # 3. Связываем коммит с добавленными фото (если есть)
        if added_photo_ids:
            for photo_id in added_photo_ids:
                cursor.execute(
                    "INSERT OR IGNORE INTO commits_photos (commit_id, photo_id) VALUES (?, ?)",
                    (commit_id, photo_id)
                )

        conn.commit()
        conn.close()
        
        logger.info(f"Особь {card_id} обновлена")
        return True
    
    # -------------------------------------------------------------------------
    # DELETE
    # -------------------------------------------------------------------------
    
    def _delete_card(
            self,
            card_id: str,
            delete_photos: bool = True,
            confirm: bool = False
        ) -> Dict[str, Any]:
        """
        Полностью удаляет карточку и все её фото (hard delete).

        Args:
            card_id (str): id карточки вида NT-K-1-ИК1
            delete_photos (bool): удалить фото (по умолчанию True)
            confirm (bool): обязательное подтверждение удаления

        Returns:
            Dict:
                - success: bool
                - error: str
                - photo_ids: list[int]
        """
        result = {
            "success": False,
            "error": None,
            "photo_ids": list()
        }
        if not confirm:
            result['error'] = str(
                f"ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ!\n"
                f"Вы уверены, что хотите удалить {card_id}?\n"
                f"Передайте confirm=True для подтверждения."
            )
            return result
        
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT card_id FROM cards WHERE card_id = ?', (card_id,))
            if not cursor.fetchone():
                result['error'] = f"Особь {card_id} не найдена в базе."
                return result

            cursor.execute('DELETE FROM commits WHERE card_id = ?', (card_id,))
            
            photo_paths = []
            if delete_photos:
                # 1. Получаем photo_id ДО удаления (важно сохранить перед очисткой)
                cursor.execute('SELECT photo_id FROM photos WHERE card_id = ?', (card_id,))
                result['photo_ids'] = [row['photo_id'] for row in cursor.fetchall()]

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
                        # Не смогли удалить файл, но всё равно выполняем операции
                        result['error'] = result['error'] + f"Файл {photo_path} не был найден\n"
            
            logger.info(f"Карточка {card_id} удалена")
            result['success'] = True
            return result
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка удаления карточк: {e}")
            raise e
        finally:
            conn.close()

    def _delete_photo(
        self,
        photo_id: int,
        delete_file: bool = True
    ):
        """
        Удаляет фото, привязанное к карточке.

        Args:
            photo_id (str): id фото из таблицы photos, можно получить по card_service.get_card_photos()
            delete_file (bool): удалить файл, связанный с записью о фото (по умолчанию True)

        Returns:
            Dict:
                - success: bool
                - error:
        """
        result = {
            "success": False,
            "error": None
        }
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Удаление из таблицы
            cursor.execute('SELECT photo_path FROM photos WHERE photo_id = ?', (photo_id,))
            row = cursor.fetchone()
            photo_path = str()
            if row:
                photo_path = str(row['photo_path'])
            cursor.execute('DELETE FROM photos WHERE photo_id = ?', (photo_id,))
            conn.commit()

            # Удаление файла
            if delete_file and photo_path:
                try:
                    Path(photo_path).unlink()
                    logger.info(f"Удалён файл: {photo_path}")
                except FileNotFoundError:
                    # Не смогли удалить файл, но всё равно выполняем операции
                    result['error'] = f"Файл {photo_path} не был найден\n"
            
            result['success'] = True
            return result
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка удаления фото: {e}")
            raise e
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # READ (История)
    # -------------------------------------------------------------------------
    def get_commit_history(self, card_id: str, limit: int = None) -> list[dict]:
        """Возвращает историю изменений особи как список коммитов (снапшотов)."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            # Коммиты в обратном хронологическом порядке (новые сверху)
            query = "SELECT * FROM commits WHERE card_id = ? ORDER BY created_at DESC"
            params = [card_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            commits = []
            
            for row in cursor.fetchall():
                commit = dict(zip(columns, row))
                
                # Добавляем привязанные фото к каждому коммиту
                cursor.execute(
                    "SELECT photo_id FROM commits_photos WHERE commit_id = ?",
                    (commit['commit_id'],)
                )
                commit['photo_ids'] = [p[0] for p in cursor.fetchall()]
                commits.append(commit)
                
            return commits
        finally:
            conn.close()


    def get_field_history(self, card_id: str, field_name: str) -> list[dict]:
        """Возвращает хронологию изменений конкретного поля."""
        if field_name not in ALLOWED_COMMIT_FIELDS:
            raise ValueError(f"Поле '{field_name}' не отслеживается в истории")
            
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            # Запрашиваем только те коммиты, где поле было явно обновлено
            query = f"""
                SELECT commit_id, created_at, {field_name} 
                FROM commits 
                WHERE card_id = ? AND {field_name} IS NOT NULL 
                ORDER BY created_at ASC
            """
            cursor.execute(query, (card_id,))
            
            history = []
            prev_value = None
            for commit_id, timestamp, new_value in cursor.fetchall():
                history.append({
                    'commit_id': commit_id,
                    'timestamp': timestamp,
                    'old_value': prev_value,
                    'new_value': new_value
                })
                prev_value = new_value
                
            return history
        finally:
            conn.close()
    # -------------------------------------------------------------------------
    # READ
    # -------------------------------------------------------------------------
    
    def get_card_photos(self, card_id: str) -> List[Dict[str, Any]]:
        """Получает все фотографии карточки из базы данных."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT photo_id, photo_type, photo_number, photo_path, 
                   date_taken, is_main, is_legacy, embedding_index
            FROM photos
            WHERE card_id = ?
            ORDER BY photo_number ASC, photo_type DESC
        ''', (card_id,))
        
        photos = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return photos
    
    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Получает данные карточки по ID, автоматически фильтруя поля по шаблону."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT i.*, p.name as project_name
            FROM cards i
            LEFT JOIN projects p ON i.project_id = p.id
            WHERE i.card_id = ?
        ''', (card_id,))
        
        row = cursor.fetchone()
        row = dict(row)
        row['prototype_id'] = extract_prototype_id(card_id)
        conn.close()
        
        return filter_card_by_template(dict(row) if row else None)
    
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

    def search_cards(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Поиск карточек по частичному совпадению ID или виду.
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
        
        return results

    def get_all_cards(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех карточек (биологических особей) во всей базе данных.
        """
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        # 1. Забираем ВСЕ карточки из базы
        cursor.execute('''
            SELECT card_id, template_type, species, project_id, created_at, date
            FROM cards
            ORDER BY card_id ASC
        ''')
        
        all_rows = [dict(row) for row in cursor.fetchall()]
        if not all_rows:
            conn.close()
            return []
        conn.close()
        return all_rows
