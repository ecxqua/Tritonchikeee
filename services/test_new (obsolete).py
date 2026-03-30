from config import load_config
from pipeline.analyse import add_individual_sync, process_photo
import asyncio

if __name__ == '__main__':
    print("🦎 Тестирование пайплайна...")
    
    config = load_config()

    print("\n Добавление новой особи...")
    
    result = add_individual_sync(
        photo_path_full="data/input/01.jpg",
        species='Карелина',
        template_type='ИК-1',
        project_name='Тест_2026',
        individual_id='NT-K-TEST-001',  # Фиксированный ID для теста
        output_dir=config['db']['cropped_folder'],
        config=config,
        notes='Тестовая особь для проверки пайплайна',
        length_body=50.0,
        weight=4.5,
        sex="Ж"
    )

    test_image = "data/input/01_1.jpg"
    print(f"\n🔍 Обработка тестового фото: {test_image}")
    result = asyncio.run(process_photo(
            str(test_image),
            config['io']['output_folder'],
            config,
            top_k=5
    ))