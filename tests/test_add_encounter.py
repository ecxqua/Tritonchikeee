from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

print(service.add_encounter(
    prototype_id="NT-K-2",
    template_type="КВ-1",
    species="Карелина",
    image_path="data/input/image.png",
    **{
        "status": "жив",
        "water_body_number": 0.5,
        "length_body": 0.4,
        "length_tail": 0.1
    }
))