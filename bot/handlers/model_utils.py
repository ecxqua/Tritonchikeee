import torch
from torchvision import transforms
from bot.handlers.utils.deployment_yolo import process_single_image
from bot.handlers.utils.deployment_vit import find_similar_images


async def photo_processing(bot, input_image_path):
    """
    Основная функция обработки фотографии

    Процесс состоит из двух этапов:
    1)Обработка YOLO моделью для детекции и обрезки брюшка тритона
    2) Поиск похожих изображений с помощью ViT модели

    """
    try:
        # Вызываем YOLO модель для детекции и обрезки брюшка тритона
        # Функция сохраняет обрезанное изображение как image_cropped.jpg
        success = await process_single_image(input_image_path, bot.save_dir)

        # Если YOLO успешно обработал изображение, переходим к поиску похожих
        if success:
            #поиск схлжих фото с помощью ViT

            #Путь к обученной ViT модели
            MODEL_PATH = "bot/models/best_model.pth"

            # Директория с базой данных изображений для поиска
            DATABASE_DIR = "bot/dataset_crop"

            # Путь к обрезанному изображению от YOLO
            QUERY_IMAGE = f'{bot.save_dir}/image_cropped.jpg'

            # Директория для сохранения результатов
            OUTPUT_DIR = bot.result_dir

            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Трансформации для предобработки изображений
            TRANSFORMS = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),

            ])

            # Вызываем функцию поиска похожих изображений
            find_similar_images(
                model_path=MODEL_PATH,  # Путь к модели
                database_dir=DATABASE_DIR,  # База данных для поиска
                query_image_path=QUERY_IMAGE,  # Запрашиваемое изображение
                output_dir=OUTPUT_DIR,  # Куда сохранять результаты
                transform=TRANSFORMS,
                device=DEVICE,
                bot=bot

            )
            return True

        # Если YOLO не смог обработать изображение
        return False

    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")

        return False
