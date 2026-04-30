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
from services.card_service import CardService, extract_prototype_id, form_card_id
from services.upload_service import UploadService
from services.project_service import ProjectService

from database.card_database import DB_PATH, init_database
from database.build_faiss_index import build_faiss_index
from database.migrate_dataset import migrate_dataset

from config import load_config
from services.embedding_service import EmbeddingService
from services.card_service import CardService
from services.upload_service import UploadService
from services.project_service import ProjectService

from utils.download_models import download_models_folder
from utils.dir_utils import delete_file, clear_directory

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
CROPPED_NAME = "yolo_cropped.jpg"

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

    def refresh(self, confirm: bool = False, remigrate: bool = False):
        """Жёсткий перезапуск всех баз данных."""
        if not confirm:
            raise PermissionError("Перезапуск бд без подтверждения запрещён!")
        DB_PATH = self.config.get('db', {}).get('db_path', 'database/cards.db')
        INDEX_DIR = self.config.get('db', {}).get(
            'faiss_index_dir', 'data/embeddings/'
        )
        CROPPED_DIR = self.config.get('db', {}).get(
            'cropped_folder', 'data/embeddings/database_embeddings.pkl'
        )
        FULL_DIR = self.config.get('db', {}).get(
            'full_folder', 'data/embeddings/database_embeddings.pkl'
        )
        delete_file(DB_PATH)
        clear_directory(INDEX_DIR)
        clear_directory(CROPPED_DIR)
        clear_directory(FULL_DIR)

        setup(migrate=remigrate)
    
    # ==========================================================================
    # ШАГ 1: АНАЛИЗ + ПОДГОТОВКА
    # ==========================================================================
    
    
    def identify_and_prepare(
        self,
        image_path: str,
        project_ids: Optional[list[int]] = None,
        territory: Optional[str] = None,
        species: Optional[str] = None,
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
            project_ids: ID проектов (для изоляции поиска) 🔥 FucK it
            territory: фильтр проектов по территории
            species: фильтр проектов по видам
            top_k: Количество кандидатов для возврата
            debug: Сохранять ли debug-артефакты YOLO
        
        Returns:
            Dict:
                - upload_id: int (для confirm_decision)
                - embedding: np.ndarray (вектор)
                - crop_path: str (путь к кропу, ОСТОРОЖНО: временный путь)
                - candidates: List[Dict] (топ-K похожих особей)
                - success: bool
                - error: str | None
        """
        result: Dict[str, Any] = {
            'upload_id': None,
            'embedding': None,
            'crop_path': None,
            'full_path': None,
            'candidates': [],
            'success': False,
            'error': None
        }
        
        try:
            logger.info("СТАРТ ОБРАБОТКИ")
            # Валидация проектов
            if project_ids:
                for project_id in project_ids:
                    project = self.project_service.get_project_by_id(project_id)
                    if not project:
                        raise ValueError(f"Проект с ID={project_id} не найден")
            elif territory or species:
                projects = self.project_service.search_projects(
                    territory=territory,
                    species=species
                )
                project_ids = [project["id"] for project in projects]
                if not project_ids:
                    result['error'] = "Фильтр не нашёл проекты. Либо удалите фильтры, либо измените их."
                    logger.error(result['error'])
                    return result
                else:
                    logger.info(f"Найденные проекты: {project_ids}")
            else:
                # list_projects возвращает List[Dict] с ключами: id, name, description, created_at...
                projects_to_process = self.project_service.list_projects(active_only=False)
                project_ids = [project["id"] for project in projects_to_process]
                logger.info(f"Включён сквозной поиск по всей базе! Найденные проекты: {project_ids}")
                

            # Обработка
            process_result = self.get_crop_and_embedding(image_path, debug)
            if process_result['error']:
                result['error'] = process_result['error']
                return result
            
            result['embedding'] = process_result['embedding']
            result['crop_path'] = process_result['crop_path']
            result['full_path'] = process_result['crop_path']
            
            # === 3. СОЗДАНИЕ ВРЕМЕННОЙ ЗАГРУЗКИ ===
            logger.info(f"Создание загрузки")
            
            upload_id = self.upload_service.create_upload(
                file_path=process_result['crop_path'],
                embedding=process_result['embedding'],
                expiry_hours=self.config.get('db', {}).get('expiry_hours', 24)
            )
            
            result['upload_id'] = upload_id
            
            # === 4. ПОИСК ПОХОЖИХ (по прототипам, фильтр по project_id) ===
            logger.info(f"Поиск похожих (top_k={top_k}, project_ids={project_ids})")
            
            prototypes = self._load_prototypes(project_ids)
            
            candidates = []
            if prototypes['prototype_ids']:
                logger.info(f"Найдено {len(prototypes['prototype_ids'])} особей для поиска")
                candidates = self._search_similar(process_result['embedding'], prototypes, top_k)
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
        project_id: Optional[int] = None,
        species: Optional[str] = "Карелина",
        prototype_id: Optional[str] = None,
        template_type: Optional[str] = None,
        **card_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Подтвердить решение пользователя (Two-Phase Commit Шаг 2).
        
        Args:
            upload_id: ID временной загрузки (из identify_and_prepare)
            decision: Решение пользователя ('NEW', 'MATCH', 'CANCEL')
            project_id: Проект, куда сохранить тритона ('NEW')
            species: вид тритона: "Карелина", "Ребристый" (для NEW, MATCH)
            card_data: Данные карточки (для NEW и MATCH)
            prototype_id: ID существующей особи (для MATCH)
            template_type: тип шаблона для карточки (для NEW, MATCH)
        
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
            process_result: Dict[str, Any] = {
                'embedding': upload['embedding'],
                'crop_path': upload['file_path'],
                'full_path': None,
            }
            if decision == 'NEW':
                # === НОВАЯ ОСОБЬ ===
                add_result = self.add_new_individual(
                    species=species,
                    project_id=project_id,
                    template_type=template_type,
                    process_result=process_result,
                    **card_data
                )
                card_id = add_result['card_id']
                self.upload_service.complete_upload(upload_id=upload['id'], card_id=card_id)
                result['card_id'] = card_id
                result['message'] = f"Создана новая особь: {card_id}"
                
            elif decision == 'MATCH':
                # === ПОВТОРНАЯ ВСТРЕЧА ===
                if not prototype_id or not template_type:
                    result['message'] = "Не указан card_id или template_type для MATCH"
                    return result
                
                add_result = self.add_encounter(
                    prototype_id=prototype_id,
                    template_type=template_type,
                    species=species,
                    process_result=process_result,
                    **card_data
                )
                card_id = add_result['card_id']
                self.upload_service.complete_upload(upload_id=upload['id'], card_id=card_id)
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
    # Вспомогательные функции анализа
    # =========================================================================
    def get_crop_and_embedding(
        self,
        image_path: str,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Универсальный обработчик фото с выдачей кропов и эмбеддингов.
        Важно! Выданные пути к кропу и полному фото - временные.
        Учтите, что внутри пайплайна эти названия зарезервированы для быстрого удаления.
        
        Args:
            image_path: путь к полному фото для обработки.
        
        Returns Dict[str, Any]:
            embedding: полученный по фото эмбеддинг
            crop_path: путь к вырезанному брюшку
            full_path: путь к полному фото
            success: успешность операции
            error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'embedding': None,
            'crop_path': None,
            'full_path': None,
            'success': False,
            'error': None
        }
        # Сегментация  
        output_dir = self.config.get('db', {}).get('cropped_folder', 'cropped/temp') 
        yolo_result = process_single_image_sync(
            img_path=image_path,
            output_dir=output_dir,
            crop_name=CROPPED_NAME,
            trim_top_pct=self.config.get('seg-model', {}).get('trim_top_pct', 0.15),
            trim_bottom_pct=self.config.get('seg-model', {}).get('trim_bottom_pct', 0.3),
            final_size=self.config.get('seg-model', {}).get('final_size', 244),
            seg_model_path=self.config.get('seg-model', {}).get('path', 'models/best_seg.pt'),
            pose_align_enabled=self.config.get('seg-model', {}).get('pose_align_enabled', False),
            pose_model_path=self.config.get('seg-model', {}).get('pose_model_path', 'models/best_pose.pt'),
            pose_head_kpt_index=self.config.get('seg-model', {}).get('pose_head_kpt_index', 0),
            pose_tail_kpt_index=self.config.get('seg-model', {}).get('pose_tail_kpt_index', 1),
            pose_min_kpt_conf=self.config.get('seg-model', {}).get('pose_min_kpt_conf', 0.25),
            pose_rotation_mode=self.config.get('seg-model', {}).get('pose_rotation_mode', 'flip-only'),
            pose_flip_vertical_ratio=self.config.get('seg-model', {}).get('pose_flip_vertical_ratio', 1.15),
            pose_upright_skip_threshold_deg=self.config.get('seg-model', {}).get('pose_upright_skip_threshold_deg', 20.0),
            pose_min_rotation_deg=self.config.get('seg-model', {}).get('pose_min_rotation_deg', 5.0),
            pose_swap_penalty_deg=self.config.get('seg-model', {}).get('pose_swap_penalty_deg', 35.0),
            pose_rotation_direction=self.config.get('seg-model', {}).get('pose_rotation_direction', 1.0),
            debug=debug,
            return_array=True
        )
    
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

        crop_path = result['crop_path']
        # 2. Копирование оригинала в config["full_folder"]
        full_folder = self.config.get("full_folder", "data/full")
        os.makedirs(full_folder, exist_ok=True)

        original_filename = Path(image_path).name
        full_save_path = os.path.join(full_folder, original_filename)
        shutil.copy2(image_path, full_save_path)
        result['full_path'] = full_save_path

        # Извлечение embedding
        embedding = get_embedding(
            crop_path,
            self.vit_model,
            self.transform,
            self.device
        )
        if embedding is None:
            result['error'] = "Не удалось вычислить эмбеддинг"
            return result
        result["embedding"] = embedding
        result['success'] = True
        return result

    # ==========================================================================
    # CREATE
    # ==========================================================================

    def add_photo_to_card(
        self,
        card_id: str,
        image_path: Optional[str] = None,
        process_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Обработать фото и связать его с существующей карточкой (кроп + эмбеддинг + сохранение + индекс).
        
        Modes:
            image_path: обработка полного фото (ещё не вырезано)
            process_result: обработка с уже полученным вырезанным брюшком и эмбеддингом
        
        Args:
            image_path: полное изображение обработки (не указывать только при наличии process_result).
            card_id: id карточки для добавления фото.
            process_result: Если обработка уже была совершена, то можно подгрузить данные:
            {
                embedding: эмбеддинг фото
                crop_path: путь к вырезанному брюшку
                full_path: путь к полному фото
            }
            По умолчанию обработка совершается.
        
        Returns Dict[str, Any]:
            crop_path: путь к вырезанному брюшку
            photo_id: id добавленного фото в photos
            success: успешность операции
            error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'crop_path': None,
            'photo_id': None,
            'success': False,
            'error': None
        }
        # Обработка
        if not process_result:
            process_result = self.get_crop_and_embedding(image_path)
            if process_result['error']:
                result['error'] = process_result['error']
                return result
        
        result['card_id'] = card_id
        
        # Добавление фото в БД
        save_result = self.card_service._add_photo_to_card(process_result['crop_path'], card_id=card_id)
        
        if save_result['success']:
            # Добавить embedding в FAISS через embedding_service
            embedding_index = self.embedding_service.add(
                process_result['embedding'],
                {
                    'card_id': card_id,
                    'photo_path': save_result['crop_path'],
                },
                photo_id=save_result['photo_id']
            )
            self.embedding_service.commit()
            
            # Обновить photos.embedding_index в БД
            self._update_photo_embedding_index(save_result['crop_path'], embedding_index)

            result['success'] = True
            result['crop_path'] = save_result['crop_path']
            return result
        else:
            result['error'] = save_result['error']
            return result
        
    def add_new_individual(
        self,
        species: str,
        project_id: Optional[int] = None,
        template_type: str = "ИК-1",
        image_path: Optional[str] = None,
        process_result: Optional[Dict[str, Any]] = None,
        **card_data
    ):
        """
        Обработать фото и создать запись в базах данных (кроп + эмбеддинг + сохранение + индекс).
        Не включает анализ. Создаёт новую карточку, не повторную (ИК-1/ИК-2)

        Modes:
            image_path: обработка полного фото (ещё не вырезано)
            process_result: обработка с уже полученным вырезанным брюшком и эмбеддингом
        
        Args:
            image_path: Полное изображение обработки (не указывать только при наличии process_result).
            species: вид особи: "Карелина", "Ребристый".
            project_id: проект, куда сохранить особь.
            template_type: тип карточки: "ИК-1", "ИК-2"
            card_data: Данные карточки от пользователя
            process_result: Если обработка уже была совершена, то можно подгрузить данные:
                {
                    embedding: эмбеддинг фото
                    crop_path: путь к вырезанному брюшку
                    full_path: путь к полному фото
                }
            По умолчанию обработка совершается.
        
        Returns Dict[str, Any]:
            crop_path: путь к вырезанному брюшку
            full_path: путь к полному фото
            success: успешность операции
            card_id: id сохранённой карточки
            error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'crop_path': None,
            'full_path': None,
            'card_id': None,
            'success': False,
            'error': None
        }
        # Обработка
        if not process_result:
            if not image_path:
                raise ValueError(
                    "Нет фото или результатов"
                    " для обработки и добавления особи."
                )
            process_result = self.get_crop_and_embedding(image_path)
            if process_result['error']:
                result['error'] = process_result['error']
                return result
        
        # Создать карточку через card_service (БЕЗ FAISS)
        # 🔥 Передаём project_id, card_service сам получит project_name если нужно
        # Внутри card_service СОХРАНЯЕТСЯ ФОТОГРАФИЯ НА ДИСКЕ
        save_result = self.card_service._save_new_individual(
            photo_path_cropped=process_result['crop_path'],
            template_type=template_type,
            species=species,
            project_id=project_id,  # 🔥 FK
            **card_data
        )
        card_id = save_result['card_id']
        result['card_id'] = card_id
        
        # Добавить embedding в FAISS через embedding_service
        embedding_index = self.embedding_service.add(
            process_result['embedding'],
            {
            'card_id': card_id,
            'photo_path': save_result['crop_path']
            },
            photo_id=save_result['photo_id']
        )
        self.embedding_service.commit()
        
        # Обновить photos.embedding_index в БД
        self._update_photo_embedding_index(save_result['crop_path'], embedding_index)

        result['success'] = True
        result['crop_path'] = save_result['crop_path']
        return result

    def add_encounter(
        self,
        prototype_id: str,
        template_type: str,
        species: str,
        image_path: Optional[str] = None,
        process_result: Optional[Dict[str, Any]] = None,
        **card_data
    ) -> Dict[str, Any]:
        """
        Обработать фото и создать запись о повторной встрече в базах данных
        (кроп + эмбеддинг + сохранение + индекс).
        Не включает вывод кандидатов. Создаёт повторную картчоку (КВ-1/КВ-2)

        Args:
            image_path (str): фото для кропа и добавления (не указывать только при наличии process_result)
            prototype_id (str): id особи (не карточка) вида NT-K-13 (без типа карточки)
            template_type (str): тип карточки (КВ-1/КВ-2)
            species (str): вид особи: "Карелина", "Ребристый"
            process_result: Если обработка уже была совершена, то можно подгрузить данные:
                {
                    embedding: эмбеддинг фото
                    crop_path: путь к вырезанному брюшку
                    full_path: путь к полному фото
                }
            **card_data (dict или аргументы): данные для заполнения карточки.

        Returns Dict[str, Any]:
            crop_path: путь к вырезанному брюшку
            full_path: путь к полному фото
            success: успешность операции
            card_id: id сохранённой карточки
            error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'crop_path': None,
            'full_path': None,
            'card_id': None,
            'success': False,
            'error': None
        }
        # Обработка
        if not process_result:
            if not image_path:
                raise ValueError(
                    "Нет фото или результатов"
                    " для обработки и добавления особи."
                )
            process_result = self.get_crop_and_embedding(image_path)
            if process_result['error']:
                result['error'] = process_result['error']
                return result

        card_id = form_card_id(prototype_id, template_type)
        
        # Добавить встречу через card_service (БЕЗ FAISS)
        # Внутри card_service СОХРАНЯЕТС ФОТОГРАФИЯ НА ДИСКЕ
        save_result = self.card_service._add_encounter(
            prototype_id=prototype_id,
            template_type=card_data.get('template_type', 'КВ-1'),
            species=species,
            photo_path_cropped=process_result['crop_path'],
            **card_data
        )
        
        # Добавить embedding в FAISS
        embedding_index = self.embedding_service.add(
            process_result['embedding'],
            {
            'card_id': card_id,
            'photo_path': save_result['crop_path']
            },
            photo_id=save_result['photo_id']
        )
        self.embedding_service.commit()
        
        # Обновить photos.embedding_index в БД
        self._update_photo_embedding_index(save_result['crop_path'], embedding_index)

        result['success'] = True
        result['crop_path'] = save_result['crop_path']
        result['card_id'] = card_id
        return result

    # ==========================================================================
    # DELETE
    # ==========================================================================

    def delete_card(
        self,
        card_id: str,
        delete_photos: bool = True,
        confirm: bool = False
    ):
        """Удалить карточку особи (не все карточки особи, а конкретную) (+ FAISS)

        Args:
            card_id (str): id карточки вида NT-K-1-ИК1
            delete_photos (bool): удалить фото (по умолчанию True)
            confirm (bool): обязательное подтверждение удаления

        Returns:
            Dict:
                - success: bool
                - error: str
        """
        result = {
            "success": False,
            "error": None
        }
        if confirm:
            delete_result = self.card_service._delete_card(
                card_id=card_id,
                delete_photos=delete_photos,
                confirm=confirm
            )
            
            if not delete_result['success']:
                result['error'] = delete_result['error']
                return result

            # Удаляем все связанные фото в photos из FAISS
            for photo_id in delete_result['photo_ids']:
                if self.embedding_service.delete(photo_id=photo_id):
                    self.embedding_service.commit()
                else:
                    result['error'] = "Ошибка удаления из FAISS. Вектор не найден"
                    return result

            result['success'] = True
            return result
        else:
            result['error'] = "Необходимо подтверждение операции"
            return result

    def delete_prototype(
        self,
        prototype_id: str,
        confirm: bool = False
    ):
        """Удалить особь со всеми карточками и фотографиями (+ FAISS)

        Args:
            prototype_id (str): id особи вида NT-K-25 (без уточнения типа карточки)
            confirm (bool): обязательное подтверждение удаления

        Returns:
            Dict:
                - success: bool
                - error: str
        """
        result = {
            "success": False,
            "error": None
        }
        if confirm:
            card_ids = self.card_service.get_matching_card_ids(prototype_id=prototype_id)
            for card_id in card_ids:
                delete_result = self.delete_card(
                    card_id=card_id,
                    delete_photos=True,
                    confirm=confirm
                )
                if not delete_result['success']:
                    result['error'] = delete_result['error']
                    return result

            logger.info(f"Особь {prototype_id} удалена")
            logger.info(f"Удалённые карточки особи: {card_ids}")
            result['success'] = True
            return result
        else:
            result['error'] = "Необходимо подтверждение операции"
            return result

    def delete_photo(
        self,
        photo_id: int,
        delete_file: bool = True
    ):
        """
        Удаляет фото, привязанное к карточке (+ удаление эмбеддинга).

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

        delete_result = self.card_service._delete_photo(
            photo_id=photo_id,
            delete_file=delete_file
        )
        # Удаление фото из FAISS
        if self.embedding_service.delete(photo_id=photo_id):
            self.embedding_service.commit()
        else:
            result['error'] = "Ошибка удаления из FAISS. Вектор не найден"
            return result

        result['success'] = True
        return result

    # ==========================================================================
    # UPDATE
    # ==========================================================================
    def update_card(self, card_id: str, **card_data) -> Dict[str, Any]:
        """Обновляет данные существующей карточки

        Args:
            card_id (str): id карточки, которую нужно изменить (у одной особи много карточек)
            **kwargs: словарь с полями для изменения

        Returns:
            Dict:
                - success: bool
                - error:
        """
        result = {
            "success": False,
            "error": None
        }

        update_result = self.card_service._update_card(
            card_id=card_id,
            **card_data
        )
        # Удаление фото из FAISS
        if update_result:
            result['success'] = True
            return result
        else:
            result['error'] = "Не удалось обновить карточку особи"
            return result

    def cleanup_expired_uploads(self) -> int:
        """Очищает просроченные загрузки."""
        return self.upload_service.cleanup(True)

    def cleanup_uploads(self) -> int:
        """Очищает все загрузки (полезно для защиты от рассинхрона)."""
        return self.upload_service.cleanup(False)

    # ==========================================================================
    # ВНУТРЕННИЕ МЕТОДЫ
    # Многие из них вынеесены сюда, потому что они не совсем имеют место
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
    
    def _load_prototypes(self, project_ids: Optional[list[int]] = None) -> Dict[str, Any]:
        """
        Загрузить прототипы особей (средние эмбеддинги) из БД + FAISS.
        Группировка и усреднение — по биологической особи (прототипу), не по карточке.
        """
        # 1. Определяем список проектов для итерации
        if project_ids is not None:
            projects_to_process: list[dict] = list()
            for project_id in project_ids:
                projects_to_process.append({'id': project_id})
        else:
            # list_projects возвращает List[Dict] с ключами: id, name, description, created_at...
            projects_to_process = self.project_service.list_projects(active_only=False)
            
        # Кэш имён проектов для быстрого доступа в метаданных
        project_names = {p['id']: p.get('name', f'Project_{p["id"]}') for p in projects_to_process}
        
        all_prototypes = []
        for proj in projects_to_process:
            pid = proj['id']
            try:
                protos = self.card_service.get_prototypes_by_project(pid)
                all_prototypes.extend(protos)
            except ValueError:
                logger.warning(f"Пропущен проект {pid}: нарушение целостности данных")
                continue
                
        if not all_prototypes:
            return {'prototype_ids': [], 'embeddings': np.array([]), 'metadata': {}}
            
        prototype_ids = []
        prototype_embeddings = []
        metadata = {}

        for proto in all_prototypes:
            proto_id = proto['prototype_id']
            
            # Получаем все фото всех карточек прототипа
            photos = self.card_service.get_prototype_photos(proto_id)
            
            # Фильтр: только кропы с валидным embedding_index
            valid_indices = [
                p['embedding_index'] for p in photos
                if p.get('photo_type') == 'cropped' and p.get('embedding_index') not in (None, -1)
            ]
            
            if not valid_indices:
                continue
                
            # Загрузка эмбеддингов из FAISS-сервиса
            embeddings_list = []
            for idx in valid_indices:
                emb = self.embedding_service.get_embedding_by_index(idx)
                if emb is not None:
                    embeddings_list.append(emb)

            if not embeddings_list:
                continue
                
            # Усреднение + L2-нормализация
            avg_emb = np.mean(embeddings_list, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 1e-12:
                avg_emb = avg_emb / norm
                
            prototype_ids.append(proto_id)
            prototype_embeddings.append(avg_emb)
            
            # Метаданные по всем карточкам особи
            metadata[proto_id] = {
                'species': proto.get('species'),
                'project_id': proto.get('project_id'),
                'project_name': project_names.get(proto.get('project_id'), 'Unknown'),
                'cards': [
                    {
                        'card_id': c['card_id'],
                        'template_type': c['template_type'],
                        'photo_count': sum(1 for p in photos if p['card_id'] == c['card_id'] and p.get('embedding_index') not in (None, -1))
                    }
                    for c in proto.get('cards', [])
                ]
            }
            
        embeddings_array = np.array(prototype_embeddings) if prototype_embeddings else np.array([])
        logger.info(f"Загружено прототипов: {len(prototype_ids)}, форма: {embeddings_array.shape}")
        
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
        if not prototypes['prototype_ids']:
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
            ind_id = prototypes['prototype_ids'][idx]
            meta = prototypes['metadata'].get(ind_id, {})
            
            candidates.append({
                'prototype_id': ind_id,
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

def setup(migrate: bool = True):
    """
    Скачать модели и поднять базы данных.

    Args:
        migrate (bool): произвести миграцию датасета по умолчанию.
    """
    download_models_folder()
    init_database()
    if migrate:
        migrate_dataset()
    build_faiss_index()

def create_identification_service() -> IdentificationService:
    """
    Создать IdentificationService со всеми зависимостями.
    
    Args:
        checkout (bool): проверяет веса моделей, базу данных и индекс
        перед запуском. Создаёт их, если они отсутствуют.
    
    Returns:
        IdentificationService: Готовый к использованию сервис
    """
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
