# Отчёт по рефакторингу: Архитектура системы идентификации тритонов

**Дата:** 30 марта 2026

**Цель:** Подготовить код к интеграции с REST API, обеспечить масштабируемость и понятность архитектуры для команды.


## Общая архитектура (3 слоя)

```
┌─────────────────────────────────────────┐
│  API / CLI (Interface Layer)            │
│  └─> FastAPI endpoints / CLI скрипты    │
├─────────────────────────────────────────┤
│  Services (Business Logic Layer) ⭐     │
│  ├── IdentificationService — оркестратор│
│  ├── CardService — CRUD карточек        │
│  ├── EmbeddingService — FAISS операции  │
│  └── UploadService — Two-Phase Commit   │
├─────────────────────────────────────────┤
│  Pipeline (Technical Core)              │
│  ├── deployment_vit_faiss.py — ViT      │
│  └── deployment_yolo_new.py — YOLO      │
├─────────────────────────────────────────┤
│  Database (Schema Only)                 │
│  └── card_database.py — init_database() │
└─────────────────────────────────────────┘
```

**Ключевой принцип:**  
✅ `pipeline` не знает про БД и FAISS  
✅ `services` оркестрируют логику  
✅ `database` только создаёт схему базы (без операций)


## Структура файлов

```
Tritonchikeee/
├── database/
│   ├── card_database.py    # init_database() — ТОЛЬКО схема
│   └── migrate_dataset.py  # Скрипт миграции данных
├── pipeline/
│   ├── deployment_vit_faiss.py  # ViT: load_model, get_embedding, search_vectors
│   └── deployment_yolo_new.py   # YOLO: процессинг → numpy array
├── services/
│   ├── card_service.py          # CRUD: individuals, photos, projects
│   ├── embedding_service.py     # FAISS: add/commit/rollback + search
│   ├── upload_service.py        # Uploads: Two-Phase Commit
│   └── identification_service.py # Оркестратор: identify_and_prepare, confirm_decision
├── data/
│   ├── embeddings/              # FAISS индекс
│   ├── dataset_crop/            # Исходные данные
│   └── output/                  # Результаты
└── models/
    ├── best_seg.pt              # YOLO веса
    └── best_id.pt               # ViT веса
```


## Схема базы данных (SQLite)

### Таблицы

| Таблица | Назначение | Ключевые поля |
|---------|-----------|--------------|
| `projects` | Метаданные проектов | `id (PK)`, `name`, `description`, `species_filter`, `is_active` |
| `individuals` | Карточки особей | `individual_id (PK)`, `project_name`, `species`, `template_type`, биометрия... |
| `photos` | Фотографии | `photo_id (PK)`, `individual_id (FK)`, `embedding_index`, `photo_type` |
| `uploads` | Временные загрузки | `id (PK)`, `project_id`, `embedding (JSON)`, `status`, `expires_at` |

### Индексы (для производительности)

```sql
-- individuals
CREATE INDEX idx_individuals_project ON individuals(project_name);
CREATE INDEX idx_individuals_species ON individuals(species);

-- photos
CREATE INDEX idx_photos_embedding ON photos(embedding_index);  -- 🔥 Критично для FAISS
CREATE INDEX idx_photos_individual ON photos(individual_id);

-- uploads
CREATE INDEX idx_uploads_status_expires ON uploads(status, expires_at);  -- Для автоочистки
```

## Ключевые паттерны

### 1. Two-Phase Commit (анализ → подтверждение)

```
Шаг 1: POST /analyze
├─> YOLO сегментация → кроп
├─> ViT → эмбеддинг
├─> Создать запись в uploads (status='pending')
├─> Поиск похожих через _load_prototypes()
└─> Вернуть: {upload_id, candidates, embedding}

Шаг 2: POST /confirm
├─> Пользователь выбирает: NEW / MATCH / CANCEL
├─> NEW: CardService.save_new_individual() + FAISS.add()
├─> MATCH: CardService.add_encounter() + FAISS.add()
├─> CANCEL: UploadService.cancel_upload()
└─> Обновить статус загрузки
```

То есть сначала пользователь отправляет фото для анализа (это первый запрос), а вторым
запросом подтверждает отправку, выбрав действие:
* NEW: новый тритон, заносим карточку новой особи
* MATCH: повторная встреча, заполняем карточку о повторной встрече.
* CANCEL: отмена операции.

Реализует Two-Phase Commit оркестратор анализа в лице `services/identification_service.py`.

**Преимущество архитектуры:** Нет состояния на сервере, можно масштабировать горизонтально.

### 2. FAISS: Буферизация и транзакционность

```python
# EmbeddingService
embedding_service.add(embedding, metadata)  # Добавляет в буфер
# ... после успешного commit БД ...
embedding_service.commit()  # Сохраняет в индекс на диск
# ... при ошибке ...
embedding_service.rollback()  # Очищает буфер
```

**Зачем:** Гарантия синхронизации БД и векторного индекса.

### 3. Прототипы (усреднённые эмбеддинги) на лету (on-the-fly prototypes)

```python
# В IdentificationService._load_prototypes():
# 1. Получить все фото особи из БД (по project_id фильтру)
# 2. Извлечь векторы из FAISS по embedding_index
# 3. Вычислить средний эмбеддинг + L2 нормализация
# 4. Использовать для поиска через search_vectors()
```

**Почему так:** Одна особь → много фото → один усреднённый вектор для точного поиска.

**TODO**: задуматься о буферизации усреднённых эмбеддингов.

### 4. In-Memory Pipeline (YOLO → ViT без диска)

```python
# deployment_yolo_new.py
result = process_single_image(..., return_array=True)
crop_array = result['crop_array']  # numpy array в памяти

# deployment_vit_faiss.py
embedding = get_embedding_from_array(crop_array, model, transform, device)
```

**Выигрыш:** ~50-100ms на фото за счёт отсутствия записи/чтения с диска.
**TODO:** сейчас эта система не подключена полноценно.


## 🚀 Как запустить и тестировать

### 1. Инициализация БД

```bash
bash launch.sh
# Создаст таблицы: projects, individuals, photos, uploads
```

### 2. Тест оркестратора (`tests/test_identification_service.py`)

```python
from services.identification_service import create_identification_service

service = create_identification_service()

# Шаг 1: Анализ
result = service.identify_and_prepare(
    image_path="data/input/test.jpg",
    project_id=1,  # или project_name="Основной"
    top_k=20
)

if result['success']:
    print(f"Кандидатов: {len(result['candidates'])}")
    
    # Шаг 2: Подтверждение (пример: новая особь)
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        decision='NEW',
        card_data={
            'species': 'Карелина',
            'template_type': 'ИК-1',
            'length_body': 42.5,
            'weight': 3.2,
            'sex': 'М'
        }
    )
    print(confirm['message'])
```


## 🛠️ Для разработчиков: как расширять

### Добавить новое поле в карточку

1.  Обновить `card_database.py`: добавить колонку в `individuals`, `photos`, `uploads` или `projects`.
2.  Обновить `services/card_service.py`: добавить поле в `save_new_individual()` и `add_encounter()` (не в аргументы, так как там они учитываются по умолчанию, а в SQL-запрос)
3.  Обновить `REQUIRED_FIELDS` валидацию если поле обязательное

### Добавить новый шаблон карточки (например, ИК-3)

1.  Добавить в `REQUIRED_FIELDS` в `card_service.py`:
    ```python
    REQUIRED_FIELDS['ИК-3'] = ['field1', 'field2', ...]
    ```
2.  Обновить `SUPPORTED_TEMPLATES` в `identification_service.py`
3.  Протестировать валидацию

### Подключить новый проект с фильтрами

```python
from services.card_service import get_or_create_project

project_id = get_or_create_project(
    name="Уральские тритоны",
    description="Популяция Свердловской области",
    species_filter=["Карелина"]  # Автоматическая фильтрация поиска
)
```

### Заменить ViT модель

1.  Положить новый `.pt` файл в `models/`
2.  Обновить путь в `config.yaml`: `id-model.path: models/new_model.pt`
3.  Убедиться, что выходной размер эмбеддинга = 512 (или обновить `EMBEDDING_DIM`)


## Известные ограничения и будущие улучшения

| Ограничение | Решение / План |
|------------|---------------|
| Фильтрация по проекту после поиска | Оптимизировать `_load_prototypes()` для pre-filtering |
| FAISS не поддерживает удаление | Периодическая перестройка индекса скриптом |
| Нет асинхронности в сервисах | Добавить `async/await` при подключении к высоконагруженному API |


## Контакты и документация

- **ТЗ проекта:** `Идентификация тритонов.pdf`
- **Конфигурация:** `config.yaml` (пути к моделям, БД, параметрам сегментации)
- **Тесты:** Создать `tests/` с юнит-тестами для сервисов
- **API спецификация:** `services/API.md`


## **Совет новому разработчику:**  
Начинайте с `services/identification_service.py` — это точка входа в бизнес-логику.

Не меняйте `pipeline/` без необходимости — это стабильное ядро инференса моделей (сравнение и схожесть частично в `pipeline/deployment_yolo_faiss.py` для усреднённых эмбеддингов и частично просто в FAISS-индексе для всех эмбеддингов).  

Все изменения в БД делайте через `card_service.py`, не напрямую (CRUD-операции).

**Удачи в разработке! 🦎✨**