# Ключевые точки для связки API

# `service/identification_service.py`
Единственный сервис, с которым можно полностью безопасно взаимодействовать
сторонним системам без ущерба логике приложения.
### Инициализация сервиса.

```python
from services.identification_service import create_identification_service, setup

# 1. Инициализация
setup(migrate=False)  # Подгрузка моделей, поднятие баз данных, идемпотентно
service = create_identification_service()

# Обнуление баз данных и перезагрузка
service.refresh(confirm=True)
```
## Анализ
### Старт анализа фотографии
Начинает операцию анализа фото, отдаёт информацию о схожести пути к кропам брюшек тритонов в топе.

Выдаёт `dict` с `upload_id` для отправки подтверждения следующей операции: сохранить новую особь, сохранить запись о повторной встрече или отменить операцию

Как получить id проекта? С помощью `service.get_or_create_project()`.
```python
result = service.identify_and_prepare(
    image_path="data/input/image.png",
    project_id=1,  # пустое поле = сковзной поиск
    top_k=5,
    territory=... (str),  # опционально
    species=... (str),  # опционально
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
* 'MATCH': известная особь, заполняем карточку (коммит для фиксации состояния особи, изменяет поля карточки особи и отражается в её истории, по факту просто вариант редактирования особи).
* 'CANCEL': отмена операции.

`card_id`: id особи (NT-K-1)

У каждой особи ЛИШЬ ОДНА КАРТОЧКА, но есть история коммитов. Заполнение картчоки для сущетсвующей особи (MATCH) это коммит.

`template_type`: тип карточки (КВ-1)

`card_data`: аргументы для заполнения карточки, у каждого типа
карточки есть свои обязательные поля, которые высветятся в ошибке, если их не достаёт.

```python
confirm = service.confirm_decision(
    upload_id=result['upload_id'],
    decision='NEW',
    template_type="ИК-1",
    species="Карелина",
    card_data = {
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
```

### Получение всех карточек (особей) в базе

Возвращает список всех биологических особей во всей базе данных. 

```python
all_cards: List[Dict[str, Any]] = card_service.get_all_cards()
```

### Получение всех карточек проекта

```python
all_cards: List[Dict[str, Any]] = card_service.get_cards_by_project(
    project_id: int
)
```

### ВАЖНО! Для CREATE используйте только методы identification_service!
### Добавление карточки новой особи
```python
save_result = service.add_new_individual(
    species="Карелина",
    image_path="data/input/image.png",
    project_id=1,
    template_type="ИК-1",  # По факту влияет лишь на формат ввода данных, все досье особей идентичны по полям.
    card_data = {
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

### Коммит (история изменений).
Редактировать карточку можно только посредством коммита.
Коммит работает так же, как и создание особи. Это заполнение карточки определённого типа, НО С УКАЗАНИЕМ id карточки особи. Опять, у особи только одна карточка.

Коммит может принимать на обработку пакет фотографий особи.
```python
result = service.commit_card(
    card_id=card_id,
    template_type="ИК-1",
    image_paths=[...],  # опционально
    card_data = {
        'weight': 67  # Новые значения полей.
    }
)

```
```python
Modes:
    image_paths: обработка полных фото (ещё не вырезано)
    process_results: обработка с уже полученным вырезанными брюшками и эмбеддингами. Лист словарей.
```
```python
Returns:
    Dict:
        - success: bool
        - error:
```

### ВАЖНО! Для UPDATE используйте только методы из indentification_service.
Конкретно для обновелния данных особей используется ТОЛЬКО СИСТЕМА КОММИТОВ. Заметьте, что тип шаблона для коммита можно не указывать. Тогда отключится система валидации полей по шаблонам. Это полезно, если коммит просто изменяет определённое поле или сразу множество полей из различных шаблонов. У особи в бд все поля активны ВСЕГДА. Шаблоны это надстрйока для удобства и бизнес-логики заполнения.
### Удаление карточки (особи).
 
`card_id`: id карточки (NT-K-1-КВ1, `prototype_id`-`template_type`)

`delete_photos`: удалять фото, связанные с карточкой (по умолчанию True, рекомендую оставить)

`confirm`: подтверждение операции

```python
result = service.delete_card(
    card_id="NT-K-88",
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
Между шагами 1 и 2 анализа в таблице `uploads` лежат незавершённые операции - загрузки.

### Очистка просроченных загрузок
У каждой записи в `uploads` есть `expires_at` - дата просрочки загрузки.
Можно настроить промежуток до просрочки в часах в `config.yaml`.

Удаляюся только `pending` загрузки в "промежуточном" состоянии.
Записи о завершении и отмене не затрагиваются (логгирование?)

```python
expired_count = service.cleanup_expired_uploads()
deleted_count = service.cleanup_uploads()
```

## Информация
### Обязательные поля для заполнения карточки по шаблону
```python
result = service.get_required_card_fields()
```

Возвращает ответ вида:
```
{
    'ИК-1': ['length_body', 'weight', 'sex', ...],
    'ИК-2': ['parent_male_id', 'parent_female_id', 'water_body_name', 'release_date', ...],
    'КВ-1': ['status', 'water_body_number', 'length_body', 'length_tail', ...],
    'КВ-2': ['status', 'water_body_name', ...]
}
```