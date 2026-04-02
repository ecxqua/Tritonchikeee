from services.identification_service import create_identification_service
import time

# 1. Инициализация
service = create_identification_service()

# 2. Шаг 1: Анализ
result = service.identify_and_prepare(
    image_path="data/input/image3.jpg",
    project_id=1,
    top_k=5,
    debug=True
)

time.sleep(1)

service.cleanup_expired()