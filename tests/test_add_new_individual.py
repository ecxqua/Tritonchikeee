from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

card_id = service.add_new_individual(
    project_id=1,
    template_type="ИК-1",
    species="Карелина",
    image_path="data/input/image.png",
    card_data={
        'length_body': 50,
        'weight': 3.22,
        'sex': 'М',
    }
)["card_id"]