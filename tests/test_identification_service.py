from services.identification_service import create_identification_service

# 1. Инициализация
service = create_identification_service()

# 2. Шаг 1: Анализ
result = service.identify_and_prepare(
    image_path="data/input/image.jpg",
    project_id=1,
    top_k=5
)

if result['success']:
    print(f"Upload ID: {result['upload_id']}")
    print(f"Кандидатов: {len(result['candidates'])}")
    
    # 3. Пользователь видит кандидатов и принимает решение
    
    # 4. Шаг 2: Подтверждение (НОВАЯ)
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        decision='NEW',
        card_data ={
            'species': 'Карелина',
            'template_type': 'ИК-1',
            'length_body': 42.5,
            'weight': 3.2,
            'sex': 'М'
        }
    )
    
    # Или (ПОВТОРНАЯ)
    confirm = service.confirm_decision(
        upload_id=result['upload_id'],
        decision='MATCH',
        existing_card_id='NT-K-1-ИК 1',
        card_data={'status': 'жив'}
    )