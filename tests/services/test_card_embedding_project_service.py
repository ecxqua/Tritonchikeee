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

# # 2. Создать проект (если нужно)
# project_id = project_service.get_or_create_project(
#     name="Уральские тритоны",
#     description="Исследование популяции",
#     species_filter=["Карелина"]
# )
project_id = 1
card_service.add_encounter(
    "NT-K-1",
    "КВ-1",
    "data/output_old/NT-K-1-КВ1_full_0526202d-263c-412b-9f32-736c5e36f872.jpg",
    "data/output_old/top5.jpg",
    status = 'мертв',
    water_body_number = 4,
    length_body = 0.2,
    length_tail = 0.1
)

# 3. Получить список проектов
projects = project_service.list_projects()
print([p['name'] for p in projects])

# 4. Тестирование метода get_prototypes_by_project
prototypes = card_service.get_prototypes_by_project(project_id)
print(f"Найдено прототипов в проекте {project_id}: {len(prototypes)}")

for proto in prototypes:
    print(f"  {proto['prototype_id']} | {proto['species']} | карточек: {len(proto['cards'])}")