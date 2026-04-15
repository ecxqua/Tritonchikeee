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
    image_path="data/input/image3.jpg",
    project_id=1,
    top_k=5,
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
    upload_id: int,
    decision: str,
    prototype_id: Optional[str] = None,
    template_type: Optional[str] = None,
    **card_data
) -> Dict[str, Any]:
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

### Обновление существующей карточки

`**kwargs`: заполнение полей для обновления

`card_id`: id карточки (NT-K-1-КВ1, `prototype_id`-`template_type`)
```python
result: bool = card_service.update_individual(
    card_id: str,
    **kwargs
)
```

### Удаление карточки

`card_id`: id карточки (NT-K-1-КВ1, `prototype_id`-`template_type`)

`delete_photos`: удалять фото, связанные с карточкой?

`confirm`: подтверждение операции

```python
result: bool = card_service.delete_individual(
    card_id: str,
    delete_photos: bool = True,
    confirm: bool = False
)
```

### Получение всех особей (не карточек) в базе

Возвращает список всех прототипов (биологических особей) во всей базе данных. Группирует карточки по prototype_id. Проверяет глобальную целостность архитектуры.

```python
prototypes: List[Dict[str, Any]] = card_service.get_all_prototypes()
```

### Получение всех карточек проекта

```python
all_cards: List[Dict[str, Any]] = get_cards_by_project(
    project_id: int
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
result = service.delete_project(
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