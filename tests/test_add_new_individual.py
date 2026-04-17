from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

print(service.add_new_individual(
    species="Карелина",
    image_path="data/input/image.png",
    project_id=1,
    template_type="ИК-1",
    **{
        'length_body': 55,
        'weight': 3.22,
        'sex': 'М'
    }
))