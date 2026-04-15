"""
services/identification_service.py — Оркестратор идентификации тритонов.

Это главный вход в приложение для идентификации.

АРХИТЕКТУРНЫЕ ПРИНЦИПЫ:
    1. Единый вход для анализа (YOLO + ViT + FAISS + Upload)
    2. Two-Phase Commit: identify_and_prepare() → confirm_decision()
    3. Прототипы (усреднённые эмбеддинги) вычисляются здесь
    4. EmbeddingService — только хранение/поиск векторов

ИСПОЛЬЗОВАНИЕ:
    См. API.md

Зависимости:
    - pipeline/deployment_yolo_new.py — сегментация
    - pipeline/deployment_vit_faiss.py — ViT модель
    - services/embedding_service.py — FAISS операции
    - services/card_service.py — CRUD карточек
    - services/upload_service.py — временные загрузки
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from torchvision import transforms
import os
import shutil
import cv2
import sqlite3

from pipeline.deployment_yolo_new import process_single_image_sync
from pipeline.deployment_vit_faiss import load_model, get_embedding, EnhancedTripletNet, DEFAULT_TRANSFORM, search_vectors
from services.embedding_service import EmbeddingService
from services.card_service import CardService
from services.upload_service import UploadService
from services.project_service import ProjectService
from database.card_database import DB_PATH

# =============================================================================
# ЛОГГЕР
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s – %(name)s – %(levelname)s – %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# КОНСТАНТЫ
# =============================================================================

SUPPORTED_TEMPLATES = ['ИК-1', 'ИК-2', 'КВ-1', 'КВ-2']
SUPPORTED_SPECIES = ['Карелина', 'Гребенчатый']

# =============================================================================
# IDENTIFICATION SERVICE
# =============================================================================

class IdentificationService:
    """
    Оркестратор идентификации тритонов.
    
    Поток данных (Two-Phase Commit):
        1. identify_and_prepare() → анализ, создание uploads, поиск
        2. Пользователь принимает решение (NEW / MATCH / CANCEL)
        3. confirm_decision() → завершение (карточка + FAISS)
    
    Работа с проектами:
        - project_id (INTEGER, FK)
        - Фильтрация прототипов по project_id
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        embedding_service: EmbeddingService,
        card_service: CardService,
        upload_service: UploadService,
        project_service: ProjectService,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            config: Конфигурация (dict)
            embedding_service: Сервис для работы с FAISS
            card_service: Сервис для работы с карточками
            upload_service: Сервис для временных загрузок
            device: Устройство для вычислений (cuda/cpu)
        """
        self.config = config
        self.embedding_service = embedding_service
        self.card_service = card_service
        self.upload_service = upload_service
        self.project_service = project_service
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загрузка ViT модели (один раз при инициализации)
        model_path = config.get('id-model', {}).get('path', 'models/best_id.pt')
        self.vit_model = load_model(model_path, self.device)
        self.transform = DEFAULT_TRANSFORM
        
        logger.info(f"IdentificationService инициализирован (device={self.device})")
    
    # ==========================================================================
    # ШАГ 1: АНАЛИЗ + ПОДГОТОВКА
    # ==========================================================================
    
    def identify_and_prepare(
        self,
        image_path: str,
        project_id: int,
        top_k: int = 20,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Единый вход для анализа фотографии.
        
        Выполняет:
            1. YOLO сегментация → кроп брюшка
            2. ViT инференс → эмбеддинг
            3. Создание временной загрузки (uploads)
            4. Поиск похожих особей (по прототипам, фильтр по project_id)
            5. Сохранение кропа на диск (для архива)
        
        Args:
            image_path: Путь к исходной фотографии
            project_id: ID проекта (для изоляции поиска) 🔥 FK
            top_k: Количество кандидатов для возврата
            debug: Сохранять ли debug-артефакты YOLO
        
        Returns:
            Dict:
                - upload_id: int (для confirm_decision)
                - embedding: np.ndarray (вектор)
                - crop_path: str (путь к кропу)
                - candidates: List[Dict] (топ-K похожих особей)
                - success: bool
                - error: str | None
        """
        result: Dict[str, Any] = {
            'upload_id': None,
            'embedding': None,
            'crop_path': None,
            'candidates': [],
            'success': False,
            'error': None
        }
        
        try:
            # Валидация проекта
            project = self.project_service.get_project_by_id(project_id)
            if not project:
                raise ValueError(f"Проект с ID={project_id} не найден")

            # === 1. YOLO СЕГМЕНТАЦИЯ ===
            logger.info(f"Сегментация: {Path(image_path).name}")
            
            output_dir = self.config.get('db', {}).get('cropped_folder', 'cropped/temp')
            
            yolo_result = process_single_image_sync(
                img_path=image_path,
                output_dir=output_dir,
                trim_top_pct=self.config.get('seg-model', {}).get('trim_top_pct', 0.15),
                trim_bottom_pct=self.config.get('seg-model', {}).get('trim_bottom_pct', 0.3),
                final_size=self.config.get('seg-model', {}).get('final_size', 244),
                seg_model_path=self.config.get('seg-model', {}).get('path', 'models/best_seg.pt'),
                debug=debug,
                return_array=True
            )
            
            # --- НОВАЯ ЛОГИКА: работа с crop_array вместо crop_path ---
            
            # 1. Сохранение кропа из массива в config["cropped_folder"]
            crop_array = yolo_result.get('crop_array')
            if crop_array is not None:
                cropped_folder = self.config.get("cropped_folder", "data/cropped")
                os.makedirs(cropped_folder, exist_ok=True)
                
                # Формируем имя файла на основе оригинала
                original_name = Path(image_path).stem
                crop_filename = f"{original_name}_cropped.jpg"
                crop_save_path = os.path.join(cropped_folder, crop_filename)
                
                cv2.imwrite(crop_save_path, crop_array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                result['crop_path'] = crop_save_path
            else:
                result['error'] = "crop_array не получен из process_single_image_sync"
                return result
            
            # 2. Копирование оригинала в config["full_folder"]
            full_folder = self.config.get("full_folder", "data/full")
            os.makedirs(full_folder, exist_ok=True)
            
            original_filename = Path(image_path).name
            full_save_path = os.path.join(full_folder, original_filename)
            shutil.copy2(image_path, full_save_path)
            result['full_path'] = full_save_path
            
            # === 2. ViT ЭМБЕДДИНГ ===
            logger.info(f"Вычисление эмбеддинга: {Path(crop_path).name}")
            
            embedding = get_embedding(
                crop_path,
                self.vit_model,
                self.transform,
                self.device
            )
            
            if embedding is None:
                result['error'] = "Не удалось вычислить эмбеддинг"
                return result
            
            result['embedding'] = embedding
            
            # === 3. СОЗДАНИЕ ВРЕМЕННОЙ ЗАГРУЗКИ ===
            logger.info(f"Создание загрузки (проект ID={project_id})")
            
            upload_id = self.upload_service.create_upload(
                project_id=project_id,  # 🔥 FK, валидация внутри upload_service
                file_path=crop_path,
                embedding=embedding,
                expiry_hours=self.config.get('db', {}).get('expiry_hours', 24)
            )
            
            result['upload_id'] = upload_id
            
            # === 4. ПОИСК ПОХОЖИХ (по прототипам, фильтр по project_id) ===
            logger.info(f"Поиск похожих (top_k={top_k}, project_id={project_id})")
            
            prototypes = self._load_prototypes(project_id)
            
            candidates = []
            if prototypes['individual_ids']:
                logger.info(f"Найдено {len(prototypes['individual_ids'])} особей для поиска")
                candidates = self._search_similar(embedding, prototypes, top_k)
                result['candidates'] = candidates
            else:
                logger.info("В базе нет особей для поиска (новая база или пустой проект)")
                result['candidates'] = []
            
            result['success'] = True
            logger.info(f"Анализ завершён: upload_id={upload_id}, кандидатов={len(candidates)}")
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
            return result
    
    # ==========================================================================
    # ШАГ 2: ПОДТВЕРЖДЕНИЕ РЕШЕНИЯ
    # ==========================================================================
    
    def confirm_decision(
        self,
        upload_id: int,
        decision: str,
        card_data: Optional[Dict[str, Any]] = None,
        existing_card_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Подтвердить решение пользователя (Two-Phase Commit Шаг 2).
        
        Args:
            upload_id: ID временной загрузки (из identify_and_prepare)
            decision: Решение пользователя ('NEW', 'MATCH', 'CANCEL')
            card_data: Данные карточки (для NEW)
            existing_card_id: ID существующей особи (для MATCH)
        
        Returns:
            Dict:
                - success: bool
                - card_id: str | None (ID созданной/обновленной карточки)
                - message: str
        """
        result: Dict[str, Any] = {
            'success': False,
            'card_id': None,
            'message': None
        }
        
        # === 1. Получить загрузку ===
        upload = self.upload_service.get_upload(upload_id)
        
        if not upload:
            result['message'] = f"Загрузка {upload_id} не найдена"
            return result
        
        if upload['status'] != 'pending':
            result['message'] = f"Загрузка уже обработана (статус: {upload['status']})"
            return result
        
        try:
            if decision == 'NEW':
                # === НОВАЯ ОСОБЬ ===
                card_id = self._handle_new_individual(upload, card_data or {})
                result['card_id'] = card_id
                result['message'] = f"Создана новая особь: {card_id}"
                
            elif decision == 'MATCH':
                # === ПОВТОРНАЯ ВСТРЕЧА ===
                if not existing_card_id:
                    result['message'] = "Не указан existing_card_id для MATCH"
                    return result
                
                card_id = self._handle_encounter(upload, existing_card_id, card_data or {})
                result['card_id'] = card_id
                result['message'] = f"Добавлена встреча к особи: {card_id}"
                
            elif decision == 'CANCEL':
                # === ОТМЕНА ===
                self.upload_service.cancel_upload(upload_id)
                result['message'] = "Загрузка отменена"
                
            else:
                result['message'] = f"Неизвестное решение: {decision}"
                return result
            
            result['success'] = True
            logger.info(f"Решение подтверждено: upload_id={upload_id}, decision={decision}")
            return result
            
        except Exception as e:
            # Откат загрузки при ошибке
            self.upload_service.cancel_upload(upload_id)
            result['message'] = f"Ошибка подтверждения: {str(e)}"
            logger.error(f"Ошибка подтверждения решения: {e}")
            import traceback
            traceback.print_exc()
            return result

    # ==========================================================================
    # Проекты
    # ==========================================================================

    def get_or_create_project(self, project_name: str, description: str) -> int:
        """Получает название проекта или создаёт новый.
        
        Args:
            project_name: название проекта (существующее или новое).
            description: описание проекта.

        Returns:
            int: существующий или созданный project_id проекта.
        """
        return self.project_service.get_or_create_project(
            name=project_name,
            description=description
        )

    def get_project_by_id(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Получает метаданные проекта по project_id в таблице projects."""
        return self.project_service.get_project_by_id(project_id=project_id)

    def get_project_id_by_name(self, project_name: str) -> Optional[int]:
        """Получает метаданные проекта по project_name в таблице projects."""
        return self.project_service.get_project_id_by_name(project_name=project_name)

    def update_project(
        self,
        project_id: int,
        **kwargs
    ) -> bool:
        """Обновляет поля проекта в таблице projects.
        
        Args:
            project_id: id проекта в таблице projects.
        """
        return self.project_service.update_project(project_id=project_id, kwargs=kwargs)

    def delete_project(self, project_id: int, confirm: bool = False) -> bool:
        """Удаляет проект по project_id из таблицы projects.
        
        Args:
            project_id: id проекта в таблице.
            confirm: подтверждение удаления, по умолчанию False.

        Returns:
            bool: успешное выполнение операции.
        """
        return self.project_service.delete_project(
            project_id=project_id,
            confirm=confirm
        )

    def list_projects(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Список проектов с метаданными из таблицы projects.

        Args:
            active_only: передать только проекты с меткой active.

        Returns:
            list[dict]: список проектов с доступом к полям (чтение)."""
        return self.project_service.list_projects(active_only=active_only)

    # ==========================================================================
    # Временные загрузки
    # ==========================================================================
    # def cancel_upload(self, upload_id: int) -> bool:
    #     """Отменяет загрузку по upload_id."""
    #     return self.upload_service.cancel_upload(upload_id)

    def cleanup_expired(self) -> int:
        """
        Очищает просроченные загрузки в uploads.
        Для изменения expiry_hours см. `config.yaml`.

        Returns:
            int: число удалённых загрузок.
        """
        return self.upload_service.cleanup_expired()

    # ==========================================================================
    # Входы для внесения карточек о новой особи и повторной встречи.
    # Объединение и управление card_service.py, upload_service.py
    # и embedding_service.py
    # ==========================================================================

    def _handle_new_individual(
        self,
        upload: Dict[str, Any],
        card_data: Dict[str, Any]
    ) -> str:
        """
        Обработать решение NEW (создать новую особь).
        
        Args:
            upload: Данные загрузки
            card_data: Данные карточки от пользователя
        
        Returns:
            str: individual_id созданной карточки
        """
        # Извлечь embedding из загрузки
        embedding = np.array(upload['embedding'], dtype='float32')
        crop_path = upload['file_path']
        project_id = upload['project_id']  # 🔥 Уже есть project_id
        
        # Создать карточку через card_service (БЕЗ FAISS)
        # 🔥 Передаём project_id, card_service сам получит project_name если нужно
        # Внутри card_service СОХРАНЯЕТС ФОТОГРАФИЯ НА ДИСКЕ
        individual_id = self.card_service.save_new_individual(
            photo_path_cropped=crop_path,
            project_id=project_id,  # 🔥 FK
            **card_data
        )
        
        # Добавить embedding в FAISS через embedding_service
        embedding_index = self.embedding_service.add(embedding, {
            'individual_id': individual_id,
            'photo_path': crop_path
        })
        self.embedding_service.commit()
        
        # Обновить photos.embedding_index в БД
        self._update_photo_embedding_index(crop_path, embedding_index)
        
        # Завершить загрузку
        self.upload_service.complete_upload(upload_id=upload['id'], card_id=individual_id)
        
        return individual_id
    
    def _handle_encounter(
        self,
        upload: Dict[str, Any],
        existing_card_id: str,
        card_data: Dict[str, Any]
    ) -> str:
        """
        Обработать решение MATCH (добавить встречу).
        
        Args:
            upload: Данные загрузки
            existing_card_id: ID существующей особи
            card_data: Данные встречи
        
        Returns:
            str: individual_id (тот же)
        """
        embedding = np.array(upload['embedding'], dtype='float32')
        crop_path = upload['file_path']
        
        # Добавить встречу через card_service (БЕЗ FAISS)
        # Внутри card_service СОХРАНЯЕТС ФОТОГРАФИЯ НА ДИСКЕ
        self.card_service.add_encounter(
            individual_id=existing_card_id,
            template_type=card_data.get('template_type', 'КВ-1'),
            photo_path_cropped=crop_path,
            **card_data
        )
        
        # Добавить embedding в FAISS
        embedding_index = self.embedding_service.add(embedding, {
            'individual_id': existing_card_id,
            'photo_path': crop_path
        })
        self.embedding_service.commit()
        
        # Обновить photos.embedding_index в БД
        self._update_photo_embedding_index(crop_path, embedding_index)
        
        # Завершить загрузку
        self.upload_service.complete_upload(upload_id=upload['id'], card_id=existing_card_id)
        
        return existing_card_id
    
    # ==========================================================================
    # ВНУТРЕННИЕ МЕТОДЫ
    # Многие из них выенесены сюда, потому что они не совсем имеют место
    # в CRUD для бд или faiss. Это внутренняя логика идентификации типа построения
    # усреднённых эмбеддингов, обновления embedding_index для фото в photos
    # ==========================================================================
    
    def _update_photo_embedding_index(self, photo_path: str, embedding_index: int):
        """Обновить embedding_index для фотографии в БД."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE photos 
            SET embedding_index = ?, is_processed = 1
            WHERE photo_path = ?
        ''', (embedding_index, photo_path))
        conn.commit()
        conn.close()
    
    def _load_prototypes(self, project_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Загрузить прототипы особей (средние эмбеддинги) из БД + FAISS.
        Важно: метод обрабатывает усреднённые эмбеддинги по особи,
        не по карточкам.
        
        Это значит, что NT-K-1-КВ1 и NT-K-1-ИК1 образуют один усреднённый эмбеддинг.
        Усредняем ПО ОСОБИ.

        В этом разница между prototype_id и individual_id.
        Первое - id особи, второе - id карточки.
        У одной особи может быть множество карточек.
        
        Args:
            project_id: Фильтр по проекту (если None, все проекты)
        
        Returns:
            Dict:
                - prototype_ids: List[str]
                - embeddings: np.ndarray (n_individuals, 512)
                - metadata: Dict[prototype, Dict]
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 🔥 ИЗМЕНЕНО: Добавлен JOIN с projects для получения project_name
        # individual_id это NT-K-1-КВ1
        if project_id:
            cursor.execute('''
                SELECT p.photo_id, p.individual_id, p.embedding_index, 
                    i.species, i.template_type, i.project_id,
                    pr.name as project_name  -- 🔥 Получаем имя проекта
                FROM photos p
                JOIN individuals i ON p.individual_id = i.individual_id
                LEFT JOIN projects pr ON i.project_id = pr.id  -- 🔥 JOIN с projects
                WHERE p.embedding_index != -1
                AND p.photo_type = 'cropped'
                AND i.project_id = ?
                ORDER BY p.individual_id, p.photo_id
            ''', (project_id,))
        else:
            cursor.execute('''
                SELECT p.photo_id, p.individual_id, p.embedding_index, 
                    i.species, i.template_type, i.project_id,
                    pr.name as project_name
                FROM photos p
                JOIN individuals i ON p.individual_id = i.individual_id
                LEFT JOIN projects pr ON i.project_id = pr.id
                WHERE p.embedding_index != -1
                AND p.photo_type = 'cropped'
                ORDER BY p.individual_id, p.photo_id
            ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return {
                'individual_ids': [],
                'embeddings': np.array([]),
                'metadata': {}
            }
        
        # Группировка по особям
        groups: Dict[str, List[Dict]] = {}
        metadata: Dict[str, Dict] = {}
        
        for row in rows:
            ind_id = row['individual_id']
            if ind_id not in groups:
                groups[ind_id] = []
                metadata[ind_id] = {
                    'species': row['species'],
                    'template_type': row['template_type'],
                    'project_id': row['project_id'],  # 🔥 project_id
                    'project_name': row['project_name']  # 🔥 Теперь это работает
                }
            groups[ind_id].append({
                'embedding_index': row['embedding_index']
            })
        
        conn.close()
        
        # Вычисление прототипов (средний эмбеддинг на особь)
        prototype_ids = []
        prototype_embeddings = []
        
        for ind_id, photos in groups.items():
            indices = [p['embedding_index'] for p in photos]
            
            # Извлечь векторы из FAISS
            embeddings_list = []
            for idx in indices:
                emb = self.embedding_service.get_embedding_by_index(idx)
                if emb is not None:
                    embeddings_list.append(emb)
            
            if embeddings_list:
                # Средний эмбеддинг + L2 нормализация
                prototype = np.mean(embeddings_list, axis=0)
                norm = np.linalg.norm(prototype)
                if norm > 1e-12:
                    prototype = prototype / norm
                
                prototype__ids.append(ind_id)
                prototype_embeddings.append(prototype)
        
        # Диагностика
        embeddings_array = np.array(prototype_embeddings) if prototype_embeddings else np.array([])
        logger.info(f"Загружено прототипов: {len(prototype__ids)}, форма: {embeddings_array.shape}")
        
        return {
            'prototype_ids': prototype_ids,
            'embeddings': embeddings_array,
            'metadata': metadata
        }
    
    def _search_similar(
        self,
        query_embedding: np.ndarray,
        prototypes: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих особей по прототипам.
        
        Args:
            query_embedding: Вектор запроса
            prototypes: Прототипы из _load_prototypes()
            top_k: Количество результатов
        
        Returns:
            List[Dict]: Кандидаты с метаданными
        """
        if not prototypes['individual_ids']:
            return []
        
        # Поиск через pipeline (чистая математика)
        results = search_vectors(
            query_embedding=query_embedding,
            reference_embeddings=prototypes['embeddings'],
            top_k=top_k
        )
        
        # Обогатить метаданными
        candidates = []
        for idx, similarity in results:
            ind_id = prototypes['individual_ids'][idx]
            meta = prototypes['metadata'].get(ind_id, {})
            
            candidates.append({
                'individual_id': ind_id,
                'species': meta.get('species', 'Unknown'),
                'template_type': meta.get('template_type', 'Unknown'),
                'project_name': meta.get('project_name', 'Unknown'),  # Для отображения
                'similarity': similarity,
                'similarity_percent': similarity * 100
            })
        
        return candidates

# =============================================================================
# FACTORY FUNCTION (для удобной инициализации)
# Используйте фабрику для работы с идентификацией в целом.
# =============================================================================

def create_identification_service(config: Optional[Dict] = None) -> IdentificationService:
    """
    Создать IdentificationService со всеми зависимостями.
    
    Args:
        config: Конфигурация (если None, загружается из config.yaml)
    
    Returns:
        IdentificationService: Готовый к использованию сервис
    """
    from config import load_config
    from services.embedding_service import EmbeddingService
    from services.card_service import CardService
    from services.upload_service import UploadService
    from services.project_service import ProjectService
    
    if config is None:
        config = load_config()

    DB_PATH = config.get('db', {}).get('db_path', 'database/cards.db')
    INDEX_PATH = config.get('db', {}).get('faiss_index_path', 'data/embeddings/database_embeddings.pkl')
    
    # Инициализация сервисов
    embedding_service = EmbeddingService(
        index_path=INDEX_PATH
    )

    project_service = ProjectService(
        db_path=DB_PATH
    )
    
    card_service = CardService(
        db_path=DB_PATH,
        embedding_service=embedding_service,
        project_service=project_service  # Опционально
    )
    
    upload_service = UploadService(
        db_path=DB_PATH
    )
    
    return IdentificationService(
        config=config,
        embedding_service=embedding_service,
        card_service=card_service,
        upload_service=upload_service,
        project_service=project_service
    )