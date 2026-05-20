from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
print(service.get_species())
print(service.get_required_card_fields())