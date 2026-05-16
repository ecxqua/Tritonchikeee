from services.identification_service import create_identification_service, setup
import time


# 1. Инициализация
service = create_identification_service()

# 2. Удаление карточки (она же особь)
print(service.delete_card(
    card_id="NT-X-2",
    confirm=True
))