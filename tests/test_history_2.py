from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)

# Коммит, редактируем особь. Процесс идентичен заполнению карточки
# При создании, но с указанием id картчоки
service.commit_card(
    card_id="NT-K-91",
    # Без шаблона можно редактировать все поля без ограничений валидации шаблонов.
    # template_type="ИК-1",
    card_data = {
        'weight': 27
    }
)

print(service.card_service.get_field_history("NT-K-91", "weight"))