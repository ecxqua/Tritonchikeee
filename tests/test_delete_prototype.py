from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Удаление всех карточек особи
print(service.delete_prototype(
    prototype_id="NT-K-2",
    confirm=True
))