from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)

# Выводим сгрупированные наборы фото.
print(service.card_service.get_card_photos_grouped("NT-K-85").keys())