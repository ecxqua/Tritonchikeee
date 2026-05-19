# Система идентификации индивидуальных особей тритонов

Система реидентификации тритонов по узору брюшка.

## Отличительные особенности
* Автоматическое выравнивание и вырезание брюшка с поворотом для обработки фотографии.
* Модель DinoV2 для реидентификации.
* Анализ схожести особи с существующими по базе с фильтрами по проектам и иным характеристикам.
* Система карточек с историей изменений и редактированием.
* Система деления карточек на проекты.
* Серверное решение с API-сервисом и клиентским приложением.

## Запуск API-сервиса
```bash
git clone https://github.com/ecxqua/Tritonchikeee.git
cd Tritonchikeee
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Поднятие приложения, баз, подгрузка моделей происходит внутри приложения.
uvicorn api.entrypoint:app --port 3002
```

## Запуск веб-приложение
```bash
git clone https://github.com/ecxqua/Tritonchikeee.git
cd Tritonchikapp
cd web
npm install
npm run start
```

## Использование ядра приложения.

```identification_service.py``` - это вход в приложение. В ```test_identification_service.py``` описан пример работы со входом, который вы можете использовать.

1. Подтяните зависимости из `requirements.txt`.
2. Заполните `config/config.yaml`.

Пример скрипта взаимодействия с приложением.
```python
from services.identification_service import create_identification_service, setup

setup(migrate=True)  # Запуск приложения (поднятие баз)
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

## Навигация
Для ручной проверки функционала ядра можно использовать `tests/`.

В выходных файлах в папке `data/cropped` появляются артефакты обработки (дебаг и сохранённые в базу кропы брюшек).

Подробные возможности ядра приложения описаны в `API.md`.

API-сервис в `api/`.

Фронтенд в `web/`.

## Docker-контейнеризация
Контейнеры Docker не настроены.
