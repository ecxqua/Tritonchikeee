from services.identification_service import create_identification_service, setup
import time


# 1. Инициализация
service = create_identification_service()
# print(service.add_new_individual(
#     template_type="ИК-1",
#     species="Карелина",
#     image_path="data/input/image.png",
#     project_id=1,
#     **{
#         'weight': 3.22,
#         'sex': 'М',
#         'length_body': 100
#     }
# ))
# 2. Удаление одной карточки
print(service.delete_prototype(
    prototype_id="NT-X-2-ИК1",
    confirm=True
))