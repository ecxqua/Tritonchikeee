from services.card_service import CardService
from services.project_service import ProjectService
from services.embedding_service import EmbeddingService

# 1. Инициализация сервисов
project_service = ProjectService()
embedding_service = EmbeddingService(index_path="data/embeddings/database_embeddings.pkl")
card_service = CardService(
    db_path="database/cards.db",
    embedding_service=embedding_service,
    project_service=project_service
)

# 2. Создать проект (если нужно)
project_id = project_service.get_or_create_project(
    name="Уральские тритоны",
    description="Исследование популяции",
    species_filter=["Карелина"]
)

# 3. Сохранить особь (через project_id)
individual_id = card_service.save_new_individual(
    photo_path_cropped="data/cropped/NT-K-89-ИК 1.jpg",
    species="Карелина",
    project_id=project_id,  # ← FK
    template_type="ИК-1",
    length_body=42.5,
    weight=3.2,
    sex="М"
)

# 4. Получить список проектов
projects = project_service.list_projects()
print([p['name'] for p in projects])