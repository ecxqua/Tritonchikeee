"""
analyse.py — Точка входа в пайплайн идентификации тритонов

Предоставляет интерфейс для REST API:
    - Обработка фото (YOLO + ViT)
    - Добавление новых особей
    - Поиск похожих особей
    - Пакетная обработка

Все функции возвращают структурированные dict для JSON-сериализации.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import asyncio
from torchvision import transforms

from pipeline.deployment_yolo_new import process_single_image
from pipeline.deployment_vit_faiss import find_similar_images, load_model, get_embedding
from pipeline.save_new import save_new_individual
from config import load_config


# =============================================================================
# КОНСТАНТЫ
# =============================================================================

DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

SUPPORTED_TEMPLATES = ['ИК-1', 'ИК-2', 'КВ-1', 'КВ-2']
SUPPORTED_SPECIES = ['Карелина', 'Гребенчатый']


# =============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# =============================================================================

async def process_photo(
    input_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Обработать фотографию тритона: сегментация + идентификация.
    
    Args:
        input_path: Путь к исходной фотографии
        output_dir: Директория для результатов
        config: Конфигурация (если None, загружается из config.yaml)
        top_k: Количество результатов поиска

    Returns:
        Dict с результатами:
            - success: bool
            - cropped_path: str | None
            - results: List[Dict] | None
            - error: str | None
    """
    if config is None:
        config = load_config()
    
    result = {
        'success': False,
        'input_path': input_path,
        'cropped_path': None,
        'results': [],
        'error': None,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # 1. YOLO-сегментация
        print(f"🔍 Сегментация: {Path(input_path).name}")
        
        success = await process_single_image(
            input_path,
            output_dir,
            config['seg-model']['trim_top_pct'],
            config['seg-model']['trim_bottom_pct'],
            config['seg-model']['final_size'],
            config['seg-model']['path'],
        )
        
        if not success:
            result['error'] = 'YOLO сегментация не удалась'
            return result
        
        cropped_path = f'{output_dir}/image_cropped.jpg'
        
        if not Path(cropped_path).exists():
            result['error'] = 'Файл кропа не найден после сегментации'
            return result
        
        # 2. ViT-идентификация
        print(f"🧠 Идентификация: {Path(cropped_path).name}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = find_similar_images(
            model_path=config['id-model']['path'],
            db_path=config['db']['db_path'],
            faiss_index_path=config['db']['faiss_index_path'],
            query_image_path=cropped_path,
            output_dir=output_dir,
            transform=DEFAULT_TRANSFORMS,
            device=device,
            size_answer=top_k,
        )
        
        result['success'] = True
        result['cropped_path'] = cropped_path
        result['results'] = results
        result['output_dir'] = output_dir
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Ошибка при обработке: {str(e)}")
    
    return result


def process_photo_sync(
    input_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Синхронная обёртка для process_photo()."""
    return asyncio.run(process_photo(input_path, output_dir, config, top_k))


# =============================================================================
# УПРАВЛЕНИЕ ОСОБЯМИ (ТЗ: п.1.3 — Интеллектуальное управление данными)
# =============================================================================

async def add_individual(
    photo_path_full: str,
    species: str = 'Карелина',
    template_type: str = 'ИК-1',
    project_name: str = 'Основной',
    individual_id: Optional[str] = None,
    config: Optional[Dict] = None,
    photo_path_cropped: Optional[str] = None,  # ← Если уже есть кроп
    output_dir: Optional[str] = None,
    **card_data
) -> Dict[str, Any]:
    """
    Добавить новую особь в базу данных.
    
    Поток данных (согласно техническому отчёту):
        1. Полное фото → YOLO → Кроп брюшка
        2. Кроп → ViT → Эмбеддинг (512 dim, L2 норм)
        3. Эмбеддинг + метаданные → SQLite + FAISS
    
    ТЗ требования:
        - 4 шаблона карточек (ИК-1, ИК-2, КВ-1, КВ-2)
        - Поддержка проектов
        - Автогенерация ID если не указан
        - Сегментация перед вычислением эмбеддинга

    Args:
        photo_path_full: Путь к полному фото (от пользователя)
        species: Вид тритона ('Карелина' или 'Гребенчатый')
        template_type: Шаблон карточки ('ИК-1', 'ИК-2', 'КВ-1', 'КВ-2')
        project_name: Название проекта
        individual_id: ID особи (если None, генерируется автоматически)
        config: Конфигурация
        photo_path_cropped: Путь к кропу (если None, выполняется YOLO сегментация)
        output_dir: Директория для сохранения кропа (если сегментация нужна)
        **card_ Дополнительные поля карточки

    Returns:
        Dict с результатами:
            - success: bool
            - individual_id: str | None
            - cropped_path: str | None (путь к кропу брюшка)
            - error: str | None
    """
    if config is None:
        config = load_config()
    
    result = {
        'success': False,
        'individual_id': None,
        'cropped_path': None,
        'error': None
    }
    
    try:
        # 1. Валидация входных данных
        if species not in SUPPORTED_SPECIES:
            result['error'] = f"Неподдерживаемый вид: {species}"
            return result
        
        if template_type not in SUPPORTED_TEMPLATES:
            result['error'] = f"Неподдерживаемый шаблон: {template_type}"
            return result
        
        if not Path(photo_path_full).exists():
            result['error'] = f"Файл не найден: {photo_path_full}"
            return result
        
        # 2. Сегментация (YOLO) если кроп не предоставлен
        if photo_path_cropped is None:
            if output_dir is None:
                output_dir = config['io']['output_folder']
            
            print(f"🔍 Сегментация: {Path(photo_path_full).name}")
            
            success = await process_single_image(
                photo_path_full,
                output_dir,
                config['seg-model']['trim_top_pct'],
                config['seg-model']['trim_bottom_pct'],
                config['seg-model']['final_size'],
                config['seg-model']['path'],
            )
            
            if not success:
                result['error'] = 'YOLO сегментация не удалась'
                return result
            
            photo_path_cropped = f'{output_dir}/image_cropped.jpg'
            
            if not Path(photo_path_cropped).exists():
                result['error'] = 'Файл кропа не найден после сегментации'
                return result
            
            print(f"✅ Кроп сохранён: {photo_path_cropped}")
        
        # 3. Генерация ID если не предоставлен
        if individual_id is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            prefix = 'K' if species == 'Карелина' else 'R'
            individual_id = f"NT-{prefix}-NEW-{timestamp}"
        
        # 4. Вычисление эмбеддинга из КРОПА (не полного фото!)
        print(f"🧠 Вычисление эмбеддинга: {Path(photo_path_cropped).name}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(config['id-model']['path'], device)
        embedding = get_embedding(photo_path_cropped, model, DEFAULT_TRANSFORMS, device)
        
        if embedding is None:
            result['error'] = "Не удалось вычислить эмбеддинг"
            return result
        
        # 5. Сохранение в БД + FAISS
        print(f"💾 Сохранение особи: {individual_id}")
        
        individual_id = save_new_individual(
            embedding=embedding,
            photo_path_full=photo_path_full,
            photo_path_cropped=photo_path_cropped,
            species=species,
            project_name=project_name,
            template_type=template_type,
            individual_id=individual_id,
            date=datetime.now().strftime('%d.%m.%Y'),
            # notes='Добавлено через API',
            **card_data
        )
        
        result['success'] = True
        result['individual_id'] = individual_id
        result['cropped_path'] = photo_path_cropped
        
        print(f"✅ Особь добавлена: {individual_id}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Ошибка добавления особи: {str(e)}")
    
    return result


def add_individual_sync(
    photo_path_full: str,
    species: str = 'Карелина',
    template_type: str = 'ИК-1',
    project_name: str = 'Основной',
    individual_id: Optional[str] = None,
    config: Optional[Dict] = None,
    photo_path_cropped: Optional[str] = None,
    output_dir: Optional[str] = None,
    **card_data
) -> Dict[str, Any]:
    """Синхронная обёртка для add_individual()."""
    return asyncio.run(add_individual(
        photo_path_full=photo_path_full,
        species=species,
        template_type=template_type,
        project_name=project_name,
        individual_id=individual_id,
        config=config,
        photo_path_cropped=photo_path_cropped,
        output_dir=output_dir,
        **card_data
    ))


def search_individuals(
    query_image_path: str,
    output_dir: str,
    top_k: int = 5,
    threshold: float = 0.6,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Поиск похожих особей в базе.
    
    ТЗ требования:
        - Ранжированный список Top-N с вероятностью совпадения
        - Статус "Новый тритон" если совпадений нет
    
    Args:
        query_image_path: Путь к фото запроса (кроп брюшка)
        output_dir: Директория для результатов
        top_k: Количество результатов
        threshold: Порог уверенности (результаты ниже = новая особь)
        config: Конфигурация

    Returns:
        Dict с результатами:
            - success: bool
            - results: List[Dict]
            - is_new_individual: bool (True если схожесть ниже порога)
            - error: str | None
    """
    if config is None:
        config = load_config()
    
    result = {
        'success': False,
        'results': [],
        'is_new_individual': False,
        'error': None
    }
    
    try:
        if not Path(query_image_path).exists():
            result['error'] = f"Файл не найден: {query_image_path}"
            return result
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        results = find_similar_images(
            model_path=config['id-model']['path'],
            db_path=config['db']['db_path'],
            faiss_index_path=config['db']['faiss_index_path'],
            query_image_path=query_image_path,
            output_dir=output_dir,
            transform=DEFAULT_TRANSFORMS,
            device=device,
            size_answer=top_k,
        )
        
        # Фильтрация по порогу
        filtered_results = [r for r in results if r.get('similarity', 0) >= threshold]
        
        result['success'] = True
        result['results'] = filtered_results
        result['is_new_individual'] = len(filtered_results) == 0
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Ошибка поиска: {str(e)}")
    
    return result


# =============================================================================
# ПАКЕТНАЯ ОБРАБОТКА (ТЗ: п.2 — Пакетная обработка и скорость)
# =============================================================================

async def batch_process(
    input_paths: List[str],
    output_base_dir: str,
    config: Optional[Dict] = None,
    max_concurrent: int = 3,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Пакетная обработка множества фотографий.
    
    ТЗ требования:
        - Загрузка десятков/сотен фотографий за один раз
        - Параллельный анализ
        - Время обработки < 10 секунд на фото
    
    Args:
        input_paths: Список путей к фотографиям
        output_base_dir: Базовая директория для результатов
        config: Конфигурация
        max_concurrent: Максимальное количество одновременных задач
        top_k: Количество результатов для каждого фото

    Returns:
        Dict с результатами:
            - success: bool
            - processed: int
            - failed: int
            - results: List[Dict]
    """
    if config is None:
        config = load_config()
    
    result = {
        'success': False,
        'processed': 0,
        'failed': 0,
        'results': [],
        'timestamp': datetime.now().isoformat()
    }
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(input_path: str) -> Dict:
        async with semaphore:
            try:
                output_dir = f"{output_base_dir}/{Path(input_path).stem}"
                res = await process_photo(input_path, output_dir, config, top_k)
                res['input_path'] = input_path
                return res
            except Exception as e:
                return {
                    'input_path': input_path,
                    'success': False,
                    'error': str(e)
                }
    
    try:
        tasks = [process_with_semaphore(path) for path in input_paths]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in batch_results:
            if isinstance(res, Exception):
                result['results'].append({'error': str(res)})
                result['failed'] += 1
            elif res.get('success'):
                result['results'].append(res)
                result['processed'] += 1
            else:
                result['results'].append(res)
                result['failed'] += 1
        
        result['success'] = result['processed'] > 0
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_process_sync(
    input_paths: List[str],
    output_base_dir: str,
    config: Optional[Dict] = None,
    max_concurrent: int = 3,
    top_k: int = 5
) -> Dict[str, Any]:
    """Синхронная обёртка для batch_process()."""
    return asyncio.run(batch_process(input_paths, output_base_dir, config, max_concurrent, top_k))


# =============================================================================
# УПРАВЛЕНИЕ ПРОЕКТАМИ (ТЗ: п.1.3 — Поддержка множества проектов)
# =============================================================================

def list_projects(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Получить список всех проектов.
    
    ТЗ требования:
        - Отдельные проекты по видам/территориям
        - Изолированный или сквозной поиск
    
    Returns:
        Dict со списком проектов
    """
    if config is None:
        config = load_config()
    
    # TODO: Реализовать получение из БД
    return {
        'success': True,
        'projects': [
            {'name': 'Основной', 'individuals_count': 0},
        ]
    }


def create_project(
    project_name: str,
    species: Optional[str] = None,
    description: str = ''
) -> Dict[str, Any]:
    """
    Создать новый проект.
    
    Returns:
        Dict с результатами создания проекта
    """
    # TODO: Реализовать создание в БД
    return {
        'success': True,
        'project_name': project_name,
        'message': 'Функционал проектов в разработке'
    }


# =============================================================================
# УПРАВЛЕНИЕ КАРТОЧКАМИ (ТЗ: п.1.3 — Система атрибутов, 4 шаблона)
# =============================================================================

def get_individual_card(
    individual_id: str,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Получить карточку особи по ID.
    
    ТЗ требования:
        - 4 шаблона (ИК-1, ИК-2, КВ-1, КВ-2)
        - Возможность редактирования
        - Связи между карточками (родители)
    
    Returns:
        Dict с данными карточки
    """
    if config is None:
        config = load_config()
    
    # TODO: Реализовать получение из БД
    return {
        'success': True,
        'individual_id': individual_id,
        'data': {},
        'message': 'Функционал карточек в разработке'
    }


def update_individual_card(
    individual_id: str,
    template_type: str,
    **fields
) -> Dict[str, Any]:
    """
    Обновить поля карточки особи.
    
    Returns:
        Dict с результатами обновления
    """
    # TODO: Реализовать обновление в БД
    return {
        'success': True,
        'individual_id': individual_id,
        'updated_fields': list(fields.keys()),
        'message': 'Функционал обновления в разработке'
    }


def list_individuals(
    project_name: Optional[str] = None,
    species: Optional[str] = None,
    template_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Получить список особей с фильтрацией.
    
    Returns:
        Dict со списком особей
    """
    if config is None:
        config = load_config()
    
    # TODO: Реализовать получение из БД
    return {
        'success': True,
        'individuals': [],
        'total': 0,
        'limit': limit,
        'offset': offset
    }


# =============================================================================
# СИСТЕМНЫЕ ФУНКЦИИ
# =============================================================================

def health_check(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Проверка работоспособности системы.
    
    Returns:
        Dict со статусом компонентов
    """
    if config is None:
        config = load_config()
    
    status = {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    
    components = {
        'seg_model': config.get('seg-model', {}).get('path'),
        'id_model': config.get('id-model', {}).get('path'),
        'database': config.get('db', {}).get('db_path'),
        'faiss_index': config.get('db', {}).get('faiss_index_path'),
    }
    
    for name, path in components.items():
        exists = Path(path).exists() if path else False
        status['components'][name] = 'ok' if exists else 'missing'
    
    return status


# =============================================================================
# ТОЧКА ВХОДА (ДЛЯ ТЕСТИРОВАНИЯ)
# =============================================================================

if __name__ == '__main__':
    print("🦎 Тестирование пайплайна...")
    
    config = load_config()
    
    # Health check
    health = health_check(config)
    print(f"\n📊 Статус компонентов: {health['components']}")
    
    # Пример обработки (если есть тестовое фото)
    test_image = Path(config['io']['input_folder'], config['io']['image_name'])
    if test_image.exists():
        print(f"\n🔍 Обработка тестового фото: {test_image}")
        result = asyncio.run(process_photo(
            str(test_image),
            config['io']['output_folder'],
            config,
            top_k=5
        ))
        print(f"✅ Результат: {result['success']}")
    else:
        print(f"\n⚠️ Тестовое фото не найдено: {test_image}")
