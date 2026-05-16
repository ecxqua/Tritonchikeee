from services.identification_service import create_identification_service, setup
import time


# 1. Инициализация

service = create_identification_service()
# service.refresh(confirm=True, remigrate=True)
print("Старт обработки")
start_time = time.time()

# 2. Шаг 1: Анализ
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
    
    # 4. Шаг 2: Подтверждение (НОВАЯ ОСОБЬ)
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        project_id=service.project_service.get_or_create_project("Новый"),
        decision='NEW',
        template_type="ИК-1",
        species="Карелина",
        card_data = {
            'length_body': 55,
            'weight': 3.22,
            'sex': 'М'
        }
    )

print("Финальное время обработки: ", time.time() - start_time)