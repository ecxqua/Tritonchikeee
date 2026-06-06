from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

results = service.card_service.search_cards(query="NT-K-1")
print(results)