from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Тестирование проектов
service.project_service.get_or_create_project("Test2", "test2", ["Карелина", "Ребристый"], ["Урал", "Сибирь"])
# print(service.project_service.get_unique_filters())