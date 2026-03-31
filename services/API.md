# Ключевые точки для связки API

## `service/identification_service.py`
### Инициализация сервиса.

```python
from services.identification_service import create_identification_service

# 1. Инициализация
service = create_identification_service()
```

### Старт анализа фотографии
Начинает операцию анализа фото, отдаёт информацию о схожести пути к кропам брюшек тритонов в топе.

Выдаёт `upload_id` для отправки подтверждения следующей операции: сохранить новую особь, сохранить запись о повторной встрече или отменить операцию
```python
result = service.identify_and_prepare(
    image_path="data/input/image3.jpg",
    project_id=1,
    top_k=5,
    debug=True
)
```

### Подтверждение операции (`upload_id`=незакоченная операция)
Подтверждает операцию выше.
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
### Создание нового проекта или получение `project_id`
Создаёт проект по названию и описанию или получает существующий из таблицы `projects`.
```python
project_id: int = service.get_or_create_project(
    project_name: str,
    description: str = None,
)
```
