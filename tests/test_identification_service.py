from services.identification_service import create_identification_service, setup
import time


# 1. Инициализация

# setup()
service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)
print("Старт обработки")
start_time = time.time()

# 2. Анализ
result = service.identify_and_prepare(
    image_path="data/input/image.png",
    top_k=5,
    debug=True
)

print(result["candidates"])

if result['success']:
    print(f"Upload ID: {result['upload_id']}")
    print(f"Кандидатов: {len(result['candidates'])}")
    # 3. Пользователь видит кандидатов и принимает решение
    
    # 4. Отмена операции
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        decision='CANCEL'
    )

print("Финальное время обработки: ", time.time() - start_time)