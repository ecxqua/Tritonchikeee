from datetime import datetime
import torch
import yaml
import asyncio
from torchvision import transforms
from .deployment_yolo import process_single_image
from .deployment_vit import find_similar_images
from .save_new import save_new_individual


# Загрузка конфигураций
def load_config(config_path="config/config.yaml"):
    """Загрузка файла конфигураций 'config.yaml'"""
    try:
        with open(config_path, "r", encoding="UTF-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Добавьте файл конфигурации для анализа: config/config.yaml"
        )


async def photo_processing(config, filepath):
    """
    Основная функция обработки фотографии

    Процесс состоит из двух этапов:
    1) Обработка YOLO моделью для детекции и обрезки брюшка тритона.
    2) Поиск похожих изображений с помощью ViT модели.
    """
    try:
        # Вызываем YOLO модель для детекции и обрезки брюшка тритона
        # Функция сохраняет обрезанное изображение как image_cropped.jpg
        success = await process_single_image(
            filepath,
            config["io"]["output_folder"],
            config["seg-model"]["trim_top_pct"],
            config["seg-model"]["trim_bottom_pct"],
            config["seg-model"]["final_size"],
            config["seg-model"]["path"],
        )

        # Если YOLO успешно обработал изображение, переходим к поиску похожих
        if success:
            # Поиск схлжих фото с помощью ViT

            # Путь к обученной ViT модели
            MODEL_PATH = config["id-model"]["path"]

            # Директория с базой данных изображений для поиска
            DATABASE_DIR = config["db"]["path"]

            # Путь к обрезанному изображению от YOLO
            QUERY_IMAGE = f'{config["io"]["output_folder"]}/image_cropped.jpg'

            # Директория для сохранения результатов
            OUTPUT_DIR = config["io"]["output_folder"]

            # Размер ответа для топ-5 (legacy тг-бота)
            SIZE_ANSWER = config["view"]["size_answer"]

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Трансформации для предобработки изображений
            TRANSFORMS = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Вызываем функцию поиска похожих изображений
            find_similar_images(
                model_path=MODEL_PATH,  # Путь к модели
                database_dir=DATABASE_DIR,  # База данных для поиска
                query_image_path=QUERY_IMAGE,  # Запрашиваемое изображение
                output_dir=OUTPUT_DIR,  # Куда сохранять результаты
                transform=TRANSFORMS,
                device=DEVICE,
                size_answer=SIZE_ANSWER,
            )
            return True

        # Если YOLO не смог обработать изображение
        return False

    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")

        return False


def save_new_person(
    embedding, 
    photo_path_full: str,           # ← полное фото от пользователя
    photo_path_cropped: str = None, # ← кроп от YOLO (может быть None)
    species: str = "Карелина",
    project_name: str = "Основной", # ← Добавлено
    template_type: str = "ИК-1",
    individual_id: str = None,      # ← Добавлено
    **card_data
):
    """
    Вызывается, когда бот хочет сохранить новую особь
    """
    individual_id = save_new_individual(
        embedding=embedding,
        photo_path_full=photo_path_full,      # ← Исправлено: правильное имя
        photo_path_cropped=photo_path_cropped, # ← Добавлено: путь к кропу
        species=species,
        project_name=project_name,             # ← Добавлено
        template_type=template_type,
        individual_id=individual_id,           # ← Добавлено
        date=datetime.now().strftime("%d.%m.%Y"),
        notes="Добавлено через бота",
        **card_data
    )
    return individual_id


if __name__ == "__main__":
    asyncio.run(photo_processing(load_config(), "data/input/image.png"))
