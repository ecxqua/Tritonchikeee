from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)

# Добавление особи (у особи только одна уникальная карточка).
card_id = service.add_new_individual(
    project_id=1,
    template_type="ИК-1",
    species="Карелина",
    image_path="data/input/image.png",
    **{
        'length_body': 50,
        'weight': 3.22,
        'sex': 'М',
    }
)["card_id"]

# Коммит, редактируем особь. Процесс идентичен заполнению карточки
# При создании, но с указанием id картчоки
service.commit_card(
    card_id=card_id,
    template_type="ИК-1",
    card_data = {
        'weight': 67
    }
)

print(service.card_service.get_field_history(card_id, "weight"))