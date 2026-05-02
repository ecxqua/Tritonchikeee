# Система идентификации индивидуальных особей тритонов

## Запуск приложения (в режиме скрипта)

```identification_service.py``` - это вход в приложение. В ```test_identification_service.py``` описан пример работы со входом, который вы можете использовать.

1. Подтяните зависимости из `requirements.txt`.
2. Заполните `config/config.yaml`.

Пример скрипта взаимодействия с приложением.
```python
from services.identification_service import create_identification_service, setup

setup(migrate=True)  # Запуск приложения
service = create_identification_service()  # Запуск сервиса идентификации
result = service.identify_and_prepare(
    image_path="data/input/image.png",
    top_k=5,
    debug=True
)  # Анализ

print(result["candidates"])  # Кандидаты (самые похожие особи)
if result['success']:
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        decision='CANCEL'
    )  # Решение биолога
```
Если хотите запустить быстрый тест анализа, то можно использовать `tests/`.

В выходных файлах в папке `data/cropped` появляются артефакты обработки (дебаг и сохранённые в базу кропы брюшек).

Подробные возможности приложения описаны в `API.md`

ВНИМАНИЕ: обучение YOLO и ViT обычно проходит не в данной ветке, а в ```test_version```.

ВНИМАНИЕ: Если вы хотите обнулить все базы, то выполните следующие шаги: Удалите `database/cards.sqlite3` и `data/embeddings/database_embeddings.pkl`, очистите папки `data/cropped`.

## Запуск API-сервиса
...

## Docker-контейнеризация
Контейнеры Docker не настроены.
