"""
services/identification_service.py — Оркестратор идентификации тритонов.

Это главный вход в приложение для идентификации.

АРХИТЕКТУРНЫЕ ПРИНЦИПЫ:
    1. Единый вход для анализа (YOLO + DINOv2 + FAISS + Upload)
    2. Two-Phase Commit: identify_and_prepare() → confirm_decision()
    3. Прототипы (усреднённые эмбеддинги) вычисляются здесь
    4. EmbeddingService — только хранение/поиск векторов

ИСПОЛЬЗОВАНИЕ:
    См. API.md

Зависимости:
    - pipeline/deployment_yolo_new.py — сегментация
    - pipeline/deployment_dinov2_faiss.py — DINOv2 модель (замена старого ViT)
    - services/embedding_service.py — FAISS операции
    - services/card_service.py — CRUD карточек
    - services/upload_service.py — временные загрузки
"""

import logging
from pathlib import Path
import uuid
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
from pipeline.deployment_dinov2_faiss import load_model, get_embedding, get_embedding_from_array, DEFAULT_TRANSFORM, search_vectors, get_attention_heatmap
from services.embedding_service import EmbeddingService
from services.card_service import CardService, REQUIRED_FIELDS, validate_template_fields
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
        self.embedding_service.reload_index()
    
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
        debug: bool = False,
        heatmap: bool = False
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
            heatmap: Генерировать тепловую карту.
        
        Returns:
            Dict:
                - upload_id: int (для confirm_decision)
                - embedding: np.ndarray (вектор)
                - crop_path: str (путь к кропу, ОСТОРОЖНО: временный путь)
                - full_path: str (путь к полному фото, ОСТОРОЖНО: временный путь)
                - heatmap_path: str (путь к тепловой карте, ОСТОРОЖНО: временный путь)
                - candidates: List[Dict] (топ-K похожих особей)
                - success: bool
                - error: str | None
        """
        result: Dict[str, Any] = {
            'upload_id': None,
            'embedding': None,
            'crop_path': None,
            'full_path': None,
            'heatmap_path': None,
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
            process_result = self._get_crop_and_embedding(image_path, heatmap, debug)
            if process_result['error']:
                result['error'] = process_result['error']
                return result
            
            result['embedding'] = process_result['embedding']
            result['crop_path'] = process_result['crop_path']
            result['full_path'] = process_result['full_path']
            result['heatmap_path'] = process_result.get('heatmap_path', None)
            
            # === 3. СОЗДАНИЕ ВРЕМЕННОЙ ЗАГРУЗКИ ===
            logger.info(f"Создание загрузки")
            
            upload_id = self.upload_service.create_upload(
                crop_path=process_result['crop_path'],
                full_path=process_result['full_path'],
                embedding=process_result['embedding'],
                expiry_hours=self.config.get('db', {}).get('expiry_hours', 24),
                heatmap_path=process_result.get('heatmap_path', None)
            )
            
            result['upload_id'] = upload_id
            
            # === 4. ПОИСК ПОХОЖИХ (по прототипам, фильтр по project_id) ===
            logger.info(f"Поиск похожих (top_k={top_k}, project_ids={project_ids})")
            
            prototypes = self._load_prototypes(project_ids)
            
            candidates = []
            if prototypes['card_ids']:
                logger.info(f"Найдено {len(prototypes['card_ids'])} особей для поиска")
                candidates = self._search_similar(process_result['embedding'], prototypes, top_k)
                result['candidates'] = candidates
            else:
                logger.info("В базе нет особей для поиска (новая база или пустой проект)")
                result['candidates'] = []

            # Статистика распознаваний в бд.
            self.card_service._add_reid_count()
            
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
        card_id: Optional[str] = None,
        template_type: Optional[str] = None,
        card_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Подтвердить решение пользователя (Two-Phase Commit Шаг 2).
        
        Args:
            upload_id: ID временной загрузки (из identify_and_prepare)
            decision: Решение пользователя ('NEW', 'MATCH', 'CANCEL')
            project_id: Проект, куда сохранить тритона ('NEW')
            species: вид тритона: "Карелина", "Ребристый" (для NEW, MATCH)
            card_data: Данные карточки (для NEW и MATCH)
            card_id: ID существующей карточки особи (для MATCH)
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
                'crop_path': upload['crop_path'],
                'full_path': upload['full_path'],
                'heatmap_path': upload.get('heatmap_path', None)
            }
            if decision == 'NEW':
                # === НОВАЯ ОСОБЬ ===
                add_result = self.add_new_individual(
                    species=species,
                    project_id=project_id,
                    template_type=template_type,
                    process_result=process_result,
                    card_data=card_data
                )
                card_id = add_result['card_id']
                self.upload_service.complete_upload(upload_id=upload['id'], card_id=card_id)
                result['card_id'] = card_id
                result['message'] = f"Создана новая особь: {card_id}"
                
            elif decision == 'MATCH':
                # === СОЗДАТЬ КАРТОЧКУ К СУЩЕСТВУЮЩЕЙ ОСОБИ ===
                if not card_id or not template_type:
                    result['message'] = "Не указан card_id или template_type для MATCH"
                    return result
                
                add_result = self.commit_card(
                    card_id=card_id,
                    template_type=template_type,
                    process_results=[process_result],
                    card_data=card_data
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
    def _get_crop_and_embedding(
        self,
        image_path: str,
        heatmap: bool,
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
            heatmap_path: путь к тепловой карте
            success: успешность операции
            error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'embedding': None,
            'crop_path': None,
            'heatmap_path': None,
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
            crop_filename = f"{original_name}_{str(uuid.uuid4())}_cropped.jpg"
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

        original_filename = Path(image_path).stem
        full_filename = f"{original_filename}_{str(uuid.uuid4())}_full.jpg"
        
        full_save_path = os.path.join(full_folder, full_filename)
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

        # Генерация и сохранение overlay "Where Model Focuses" рядом с кропом
        if heatmap:
            try:
                logger.info(f"Обработка тепловой карты...")
                heatmap_name = f"{Path(crop_path).stem}_where_model_focuses.png"
                heatmap_path = os.path.join(Path(crop_path).parent, heatmap_name)

                if get_attention_heatmap(
                    crop_array=crop_array,
                    model=self.vit_model,
                    transform=self.transform,
                    device=self.device,
                    output_path=heatmap_path,
                ):
                    result["heatmap_path"] = heatmap_path
            except Exception as e:
                logger.warning(f"Не удалось сохранить heatmap: {e}")
        if not heatmap:
            result["heatmap_path"] = None

        result['success'] = True
        return result

    # ==========================================================================
    # CREATE
    # ========================================================================== 
    def add_new_individual(
        self,
        species: str,
        card_data: Dict[str, Any],
        project_id: Optional[int] = None,
        template_type: str = "ИК-1",
        image_path: Optional[str] = None,
        heatmap: bool = False,
        process_result: Optional[Dict[str, Any]] = None,
    ):
        """
        Обработать фото и создать запись в базах данных (кроп + эмбеддинг + сохранение + индекс).
        Не включает анализ. Создаёт новую карточку, не повторную (ИК-1/ИК-2)

        Modes:
            image_path: обработка полного фото (ещё не вырезано)
            process_result: обработка с уже полученным вырезанным брюшком и эмбеддингом
        
        Args:
            image_path: Полное изображение обработки (не указывать только при наличии process_result).
            heatmap: Нужна ли тепловая карта (по умолчанию - нет, не указывать при наличии process_result).
            species: вид особи: "Карелина", "Ребристый".
            project_id: проект, куда сохранить особь.
            template_type: тип карточки: "ИК-1", "ИК-2"
            card_data: Данные карточки от пользователя
            process_result: Если обработка уже была совершена, то можно подгрузить данные:
                {
                    embedding: эмбеддинг фото
                    crop_path: путь к вырезанному брюшку
                    full_path: путь к полному фото
                    heatmap_path: путь к тепловой карте
                }
            По умолчанию обработка совершается.
        
        Returns Dict[str, Any]:
            - crop_path: путь к вырезанному брюшку
            - full_path: путь к полному фото
            - heatmap_path: путь к тепловой карте
            - success: успешность операции
            - card_id: id сохранённой карточки
            - error: сообщение об ошибке
        """
        result: Dict[str, Any] = {
            'crop_path': None,
            'full_path': None,
            'heatmap_path': None,
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
            process_result = self._get_crop_and_embedding(image_path, heatmap)
            if process_result['error']:
                result['error'] = process_result['error']
                return result
        
        # Создать карточку через card_service (БЕЗ FAISS)
        # 🔥 Передаём project_id, card_service сам получит project_name если нужно
        # Внутри card_service СОХРАНЯЕТСЯ ФОТОГРАФИЯ НА ДИСКЕ
        save_result = self.card_service._save_new_individual(
            photo_path_cropped=process_result['crop_path'],
            photo_path_full=process_result['full_path'],
            heatmap_path=process_result.get('heatmap_path', None),
            template_type=template_type,
            species=species,
            project_id=project_id,  # 🔥 FK
            card_data=card_data
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
        result['full_path'] = save_result['full_path']
        result['heatmap_path'] = save_result['heatmap_path']
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
        """Удалить карточку особи (+ FAISS) с историей.

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
        ОПАСНАЯ ОПЕРАЦИЯ.
        Удаляет фото, привязанное к карточке (+ удаление эмбеддинга).
        Удаление НЕ СОХРАНЯЕТСЯ КАК ФАКТ В ИСТОРИЮ.
        Удаление фотографии приводит к удалению факта добавления фотографии из истории.

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
        if delete_result['photo_type'] == 'cropped':
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
    def commit_card(
        self,
        card_id: str,
        card_data: Optional[dict] = {},
        template_type: Optional[str] = None,
        image_paths: Optional[list[str]] = None,
        heatmap: bool = False,
        process_results: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Изменение данных уникальной карточки особи коммитом (история)
        (+ обработка добавления фотографии)

        см. issues История изменений тритонов #23
        Для редактирования полей особи используется только эта функция

        Args:
            card_id (str): id карточки, которую нужно изменить.
            template_type (str): тип заполненной карточки (None, если микроизменение).
                Микроизменение позволяет вносить изменения без ограничений
                валидации типа карточки (полезно для редактирования случайных полей).
            image_paths: изображения для анализа и добавления в коммите.
            process_results: проанализированный пакет изображенный.
                [{
                    embedding: эмбеддинг фото
                    crop_path: путь к вырезанному брюшку
                    full_path: путь к полному фото
                    heatmap_path: путь к тепловой карте
                }, ... ]
                Пакет собирается из результатов обработки отдельных фото.
                Словарь обработки отдельного фото можно подать в виде
                ответа функции обработки
                identify_and_prepare().
            **kwargs: словарь с полями для изменения в коммите.

        Returns:
            Dict:
                - success: bool
                - error:
        """
        result = {
            "success": False,
            "error": None
        }
        if not self.card_service.is_card_exist(card_id):
            result["error"] = f"commit_card: Карточки {card_id} существует"
            return result
        # Валидатор работает заранее
        if template_type and card_data:
            card_data = validate_template_fields(
                template_type,
                card_data,
                False
            )

        # Обработка фотографий (если есть).
        if not process_results:
            process_results = list()
            if image_paths:
                for image_path in image_paths:
                    process_result = self._get_crop_and_embedding(image_path, heatmap)
                    if process_result['error']:
                        result['error'] = process_result['error']
                        return result
                    process_results.append(process_result)

        # 1. Добавление фотографий в индекс эмбеддингов.
        added_photo_ids = list()
        if process_results:
            for process_result in process_results:
                save_result = self.card_service._add_photo_to_card(
                    process_result['crop_path'],
                    prefix='cropped',
                    card_id=card_id
                )

                if save_result['success']:
                    # Добавить embedding в FAISS через embedding_service
                    photo_id = save_result['photo_id']
                    embedding_index = self.embedding_service.add(
                        process_result['embedding'],
                        {
                            'card_id': card_id,
                            'photo_path': save_result['photo_path'],
                        },
                        photo_id=photo_id
                    )
                    self.embedding_service.commit()
                    
                    # Обновить photos.embedding_index в БД
                    self._update_photo_embedding_index(save_result['photo_path'], embedding_index)

                    result['success'] = True
                    added_photo_ids.append(photo_id)

                    # Добавление полного фото отдельно.
                    save_full_result = self.card_service._add_photo_to_card(
                        process_result['full_path'],
                        prefix='full',
                        card_id=card_id,
                        photo_group=save_result['photo_group']
                    )
                    added_photo_ids.append(save_full_result['photo_id'])
                    # Добавление тепловой карты отдельно.
                    if process_result['heatmap_path']:
                        save_heatmap_result = self.card_service._add_photo_to_card(
                            process_result['heatmap_path'],
                            prefix='heatmap',
                            card_id=card_id,
                            photo_group=save_result['photo_group']
                        )
                        added_photo_ids.append(save_heatmap_result['photo_id'])
                    logger.info(f"Фото к добавлению: {added_photo_ids}")
                else:
                    result['error'] = save_result['error']
                    return result
        # 2. Обновление полей карточки с занесением в историю.
        update_result = self.card_service._commit_card(
            card_id=card_id,
            template_type=template_type,
            added_photo_ids=added_photo_ids,
            card_data=card_data
        )
        if update_result:
            result['success'] = True
            return result
        else:
            result['error'] = "Не удалось обновить карточку особи"
            return result

    def get_required_card_fields(self):
        """Возвращает словарь обязательных полей для каждого шаблона карты."""
        return REQUIRED_FIELDS

    def get_species(self):
        """Возвращает актуальную статистику по всем видам особей."""
        return self.card_service.get_species()

    def refresh_reid_count(self):
        """Обнуляет счётчик количества распознаваний."""
        self.card_service._refresh_reid_count()

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
                protos = self.card_service.get_cards_by_project(pid)
                all_prototypes.extend(protos)
            except ValueError:
                logger.warning(f"Пропущен проект {pid}: нарушение целостности данных")
                continue
                
        if not all_prototypes:
            return {'card_ids': [], 'embeddings': np.array([]), 'metadata': {}}
            
        card_ids = []
        prototype_embeddings = []
        metadata = {}

        for proto in all_prototypes:
            card_id = proto['card_id']
            
            # Получаем все фото всех карточки прототипа
            photos = self.card_service.get_card_photos(card_id)
            
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
                
            card_ids.append(card_id)
            prototype_embeddings.append(avg_emb)
            
        embeddings_array = np.array(prototype_embeddings) if prototype_embeddings else np.array([])
        logger.info(f"Загружено прототипов: {len(card_ids)}, форма: {embeddings_array.shape}")
        
        return {
            'card_ids': card_ids,
            'embeddings': embeddings_array,
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
        if not prototypes['card_ids']:
            return []
        
        # Поиск через pipeline (чистая математика)
        top_indices, top_distances = search_vectors(
            query_embedding=query_embedding,
            index_embeddings=prototypes['embeddings'],
            top_k=top_k
        )
        
        # Обогатить метаданными
        candidates = []
        for idx, distance in zip(top_indices, top_distances):
            similarity = float(-distance)
            ind_id = prototypes['card_ids'][idx]
            
            candidates.append({
                'card_id': ind_id,
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
    Если подняты, то не поднимает повторно!

    Args:
        migrate (bool): произвести миграцию датасета по умолчанию.
    """
    config = load_config()
    DB_PATH = config.get('db', {}).get('db_path', 'database/cards.db')
    INDEX_PATH = config.get('db', {}).get(
        'faiss_index_path', 'data/embeddings/database_embeddings.pkl'
    )
    db_exists = os.path.exists(DB_PATH)
    index_exists = os.path.exists(INDEX_PATH)

    download_models_folder()
    if not db_exists and not index_exists:
        init_database()
        if migrate:
            migrate_dataset()
        build_faiss_index()
    elif db_exists and not index_exists:
        build_faiss_index()
        print("✅ Инициализация завершена.")
    else:
        print(f"⏭️ {DB_PATH} и {INDEX_PATH} уже существуют. Повторная инициализация пропущена.")

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
