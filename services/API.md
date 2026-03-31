# Ключевые точки для связки API

Котенька, это твой главный мануал для связки с API. Если хочешь
сделать что-то, то читай сперва здесь.

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

`decision`: решение биолога
* 'NEW': новая особь, заполняем карточку особи.
* 'MATCH': известная особь, заполняем карточку повторной встречи.
* 'CANCEL': отмена операции.
```python
confirm = service.confirm_decision(
    upload_id=result['upload_id'],
    decision='NEW',
    card_data ={
        'species': 'Карелина',
        'template_type': 'ИК-1',
        'length_body': 42.5,
        'weight': 3.2,
        'sex': 'М'
    }
)
```

## Проекты
### Создание нового проекта или получение `project_id`
Создаёт проект по названию и описанию или получает существующий из таблицы `projects`.
```python
project_id: int = service.get_or_create_project(
    project_name="Название",
    description=None,
)
```

### Получение `project_id` по `project_name`
```python
id: int = service.get_project_id_by_name(
    project_name="Название проекта"
)
```

### Получение метаданных проекта по `project_id`
```python
metadata: dict[str, Any] = service.get_project_by_id(
    project_id=1
)
```

### Изменение проекта
```python
result = service.update_project(
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

## Управление загрузками
Между шагами 1 и 2 анализа в таблице `uploads` лежат незавершённые операции - загрузки.

### Очистка просроченных загрузок
У каждой записи в `uploads` есть `expires_at` - дата просрочки загрузки.
Можно настроить промежуток до просрочки в часах в `config.yaml`.

Удаляюся только `pending` загрузки в "промежуточном" состоянии.
Записи о завершении и отмене не затрагиваются (логгирование?)

```python
expired_count = service.cleanup_expired()
```