# Ключевые точки для связки API

`service/identification_service.py` - единственный сервис, с которым можно полностью безопасно взаимодействовать
сторонним системам без ущерба логике приложения.
## Содержание
1. Инициализация сервиса
2. Анализ
3. Карточки
4. История изменений карточки
5. Проекты
6. Управление загрузками
7. Статистика и информация
## Инициализация сервиса.

```python
from services.identification_service import create_identification_service, setup

# 1. Инициализация
setup(migrate=False)  # Подгрузка моделей, поднятие баз данных, идемпотентно
service = create_identification_service()

# Обнуление баз данных и перезагрузка
service.refresh(confirm=True)
```
