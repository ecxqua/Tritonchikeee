from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

print(service.delete_photo(612, delete_file=True))