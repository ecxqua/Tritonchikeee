from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

print(service.add_photo_to_card(
    card_id="NT-K-1-ИК111",
    image_path="data/input/image.png"
))