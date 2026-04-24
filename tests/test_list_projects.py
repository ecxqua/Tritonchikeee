from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Удаление одной карточки
print(service.project_service.list_projects())