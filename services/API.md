# Ключевые точки для связки API

# `service/identification_service.py`
Единственный сервис, с которым можно полностью безопасно взаимодействовать
сторонним системам без ущерба логике приложения.
### Инициализация сервиса.

```python
from services.identification_service import create_identification_service

# 1. Инициализация
service = create_identification_service()
```
## Анализ
### Старт анализа фотографии
Начинает операцию анализа фото, отдаёт информацию о схожести пути к кропам брюшек тритонов в топе.

Выдаёт `dict` с `upload_id` для отправки подтверждения следующей операции: сохранить новую особь, сохранить запись о повторной встрече или отменить операцию

Как получить id проекта? С помощью `service.get_or_create_project()`.
```python
result = service.identify_and_prepare(
    image_path="data/input/image.png",
    project_id=1,
    top_k=5,
    ...  # Будут параметры для фильтрации!
    debug=True
)
```

### Подтверждение операции
Подтверждает операцию выше.

`upload_id`: незакоченная операция

```python
upload_id = result['upload_id']  # Получаем upload_id
```

`decision`: решение биолога
* 'NEW': новая особь, заполняем карточку особи.
* 'MATCH': известная особь, заполняем карточку повторной встречи.
* 'CANCEL': отмена операции.

`prototype_id`: id особи (NT-K-1)

У каждой особи может быть несколько карточек различных типов (КВ-1/ИК-1).

`template_type`: тип карточки (КВ-1)

`card_id` (здесь не используется, но для справки): id карточки особи (NT-K-1-КВ1)

`**card_data`: аргументы для заполнения карточки, у каждого типа
карточки есть свои обязательные поля, которые высветятся в ошибке, если их не достаёт.

```python
confirm = service.confirm_decision(
    upload_id=result['upload_id'],
    decision='NEW',
    template_type="ИК-1",
    species="Карелина",
    **{
        'length_body': 55,
        'weight': 3.22,
        'sex': 'М'
    }
)
```

```python
Returns:
    success: bool
    card_id: str | None (ID созданной/обновленной карточки)
    message: str
```

## Карточки

```python
card_service = service.project_service
# Получаем подсерсив работы с проектами
# Криво я знаю Т_Т
```

### Получение всех особей (не карточек) в базе

Возвращает список всех прототипов (биологических особей) во всей базе данных. Группирует карточки по prototype_id. Проверяет глобальную целостность архитектуры.

```python
prototypes: List[Dict[str, Any]] = card_service.get_all_prototypes()
```

### Получение всех карточек проекта

```python
all_cards: List[Dict[str, Any]] = card_service.get_cards_by_project(
    project_id: int
)
```

### ВАЖНО! Для CREATE используйте только методы identification_service!
### Добавление карточки новой особи (ИК-1/ИК-2)
```python
save_result = service.add_new_individual(
    species="Карелина",
    image_path="data/input/image.png",
    project_id=1,
    template_type="ИК-1",
    **{
        'length_body': 55,
        'weight': 3.22,
        'sex': 'М'
    }
)
```
```python
Modes:
    image_path: обработка полного фото (ещё не вырезано)
    process_result: обработка с уже полученным вырезанным брюшком и эмбеддингом
```
```python
Returns Dict[str, Any]:
    crop_path: путь к вырезанному брюшку
    full_path: путь к полному фото
    success: успешность операции
    card_id: id сохранённой карточки
    error: сообщение об ошибке
```

### Добавление карточки повторной встречи с особью (КВ-1/КВ-2)
```python
save_result = service.add_encounter(
    prototype_id="NT-K-2",
    template_type="КВ-1",
    species="Карелина",
    image_path="data/input/image.png",
    **{
        "status": "жив",
        "water_body_number": 0.5,
        "length_body": 0.4,
        "length_tail": 0.1
    }
)
```
```python
Modes:
    image_path: обработка полного фото (ещё не вырезано)
    process_result: обработка с уже полученным вырезанным брюшком и эмбеддингом
```
```python
Returns Dict[str, Any]:
    crop_path: путь к вырезанному брюшку
    full_path: путь к полному фото
    success: успешность операции
    card_id: id сохранённой карточки
    error: сообщение об ошибке
```

### Добавление нового фото к карточке
```python
save_result = service.add_photo_to_card(
    card_id="NT-K-1-ИК1",
    image_path="data/input/image.png"
)
```
```python
Returns Dict[str, Any]:
    crop_path: путь к вырезанному брюшку
    success: успешность операции
    error: сообщение об ошибке
```
### ВАЖНО! Для UPDATE используйте только методы из indentification_service
### Обновление существующей карточки

`**kwargs`: заполнение полей для обновления

`card_id`: id карточки (NT-K-1-КВ1, `prototype_id`-`template_type`)
```python
result = service.update_card(
    card_id="NT-R-9-ИК1"
    **{"weight": 1}
)
```

### ВАЖНО! Для DELETE используйте только методы из indentification_service
### Удаление карточки

`card_id`: id карточки (NT-K-1-КВ1, `prototype_id`-`template_type`)

`delete_photos`: удалять фото, связанные с карточкой (по умолчанию True, рекомендую оставить)

`confirm`: подтверждение операции

```python
result = service.delete_card(
    card_id="NT-K-88-ИК1",
    confirm=True
)
```

### Удаление особи (всех карточек особи)
`card_id`: id особи (NT-K-1)

`confirm`: подтверждение операции
```python
result = service.delete_prototype(
    prototype_id="NT-K-2",
    confirm=True
)
```

### Удаление фотографии, привязнной к карточке
`photo_id`: id фото в таблице photos, можно получить GET методами (см. в card_service)

`delete_file`: удалить файл фотографии (в файловой системе) (по умолчанию True)
```python
result = service.delete_photo(
    photo_id=610
)
```

Больше методов в `services/card_service.py`
## Проекты

```python
project_service = service.project_service
# Получаем подсерсив работы с проектами
# Криво я знаю Т_Т
```

### Создание нового проекта или получение `project_id`
Создаёт проект по названию и описанию или получает существующий из таблицы `projects`.
```python
project_id: int = project_service.get_or_create_project(
    project_name="Название",
    description=None,
)
```

### Получение `project_id` по `project_name`
```python
id: int = project_service.get_project_id_by_name(
    project_name="Название проекта"
)
```

### Получение метаданных проекта по `project_id`
```python
metadata: dict[str, Any] = project_service.get_project_by_id(
    project_id=1
)
```

### Изменение проекта
```python
result = project_service.update_project(
    project_id=1,
    # Изменяем поля из таблицы projects
    name="Новое название",
    is_active=False,
    ...
)
```

### Удаление проекта
```python
result = project_service.delete_project(
    project_id=1,
    confirm=True
)
```

Больше методов в `services/project_service.py`

## Управление загрузками
```python
upload_service = service.upload_service
# Получаем подсерсив работы с проектами
# Криво я знаю Т_Т
```
Между шагами 1 и 2 анализа в таблице `uploads` лежат незавершённые операции - загрузки.

### Очистка просроченных загрузок
У каждой записи в `uploads` есть `expires_at` - дата просрочки загрузки.
Можно настроить промежуток до просрочки в часах в `config.yaml`.

Удаляюся только `pending` загрузки в "промежуточном" состоянии.
Записи о завершении и отмене не затрагиваются (логгирование?)

```python
expired_count = upload_service.cleanup_expired()
```
