from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

print(service.card_service.get_card("NT-K-1-ИК1"))