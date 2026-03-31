from services.upload_service import UploadService
import numpy as np

# 1. Инициализация
upload_service = UploadService(db_path="database/cards.db")

# 2. Шаг 1: Анализ → создание загрузки
embedding = np.random.rand(512).astype('float32')
upload_id = upload_service.create_upload(
    project_id=1,
    file_path="data/crop/triton_001.jpg",
    embedding=embedding
)
print(f"Upload ID: {upload_id}")  # → 55

# 3. Шаг 2: Подтверждение → завершение
upload = upload_service.get_upload(upload_id)
if upload and upload['status'] == 'pending':
    # ... создаём карточку в card_service ...
    upload_service.complete_upload(upload_id, card_id="NT-K-1-ИК 1")

# 4. Отмена (если пользователь передумал)
# upload_service.cancel_upload(upload_id)

# 5. Очистка (фоновая задача)
# deleted = upload_service.cleanup_expired()