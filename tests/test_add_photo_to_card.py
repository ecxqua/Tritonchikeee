from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

# result = service.add_photo_to_card(
#     card_id="NT-K-1-ИК1",
#     image_path="data/input/image.png"
# )
# print(result)

print(service.delete_photo(612, delete_file=True))