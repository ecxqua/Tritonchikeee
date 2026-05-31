from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)

# Коммит, редактируем особь. Процесс идентичен заполнению карточки
# При создании, но с указанием id картчоки
# if service.card_service.is_card_exist("NT-K-85"):
#     service.commit_card(
#         card_id="NT-K-85",
#         # Добавляем коммитом фотографий!
#         image_paths=["data/input/image.png"]
#     )

print(service.card_service.get_field_history("NT-K-85", "weight"))
print(service.card_service.get_commit_history("NT-K-85"))