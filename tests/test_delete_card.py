from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Удаление одной карточки
print(service.delete_card(
    card_id="NT-K-88-ИК1",
    confirm=True
))