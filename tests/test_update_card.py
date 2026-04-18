from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# 2. Удаление всех карточек особи
print(service.update_card("NT-R-9-ИК1", **{"weight": 1}))