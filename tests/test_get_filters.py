from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Тестирование проектов
service.project_service.get_or_create_project("Test", "test", "Карелина", "Урал")
print(service.project_service.get_unique_filters())