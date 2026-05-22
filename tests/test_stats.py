from services.identification_service import create_identification_service, setup
import time


# 1. Инициализация

# setup()
service = create_identification_service()
print(service.card_service.get_last_commits(2))
print(service.card_service.get_reid_count())
service.refresh_reid_count()