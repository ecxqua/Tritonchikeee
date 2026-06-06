from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)

print(service.card_service.get_commit_history("NT-K-90"))