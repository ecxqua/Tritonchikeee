from services.card_service import CardService
from services.embedding_service import EmbeddingService

# 1. Создаём сервисы
embedding_service = EmbeddingService("data/embeddings/database_embeddings.pkl")
card_service = CardService(
    db_path="database/cards.db",
    embedding_service=embedding_service
)

# 2. Сохраняем особь (FAISS добавится через embedding_service)
individual_id = card_service.save_new_individual(
    photo_path_cropped="data/crop/test.jpg",
    template_type="ИК-1",
    length_body=42.5,
    weight=3.2,
    sex="М"
)

# 3. Two-Phase: завершаем загрузку
card_service.finalize_upload(
    upload_id=55,
    template_type="ИК-1",
    length_body=45.0
)