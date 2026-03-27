from datetime import datetime
import torch
import yaml
import asyncio
from torchvision import transforms
from pipeline.deployment_vit import load_model, get_embedding, find_similar_in_database
from pipeline.save_new import save_new_individual, add_encounter, generate_card_id
from pipeline.deployment_yolo_new import process_single_image
from database.card_database import DB_PATH
import sqlite3
import numpy as np


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


# ============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Трансформы для ViT
# ============================================================================
def get_vit_transforms():
    """Создаёт трансформы для предобработки изображений (как в deployment_vit.py)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ============================================================================
# ОБРАБОТКА ФОТО (исправленная версия)
# ============================================================================
async def photo_processing(config, filepath):
    """
    Обрабатывает фото и возвращает результат для РУЧНОЙ проверки биологом.
    
    ⚠️ ВАЖНО: Никогда не принимает решения автоматически!
    """
    try:
        # === 1. YOLO: Обработка изображения ===
        success = await process_single_image(
            filepath,
            config["io"]["output_folder"],
            config["seg-model"]["trim_top_pct"],
            config["seg-model"]["trim_bottom_pct"],
            config["seg-model"]["final_size"],
            config["seg-model"]["path"],
        )
        if not success:
            return {"status": "error", "message": "YOLO не смог обработать изображение"}
        
        # === 2. ViT: Генерация эмбеддинга ===
        cropped_path = f'{config["io"]["output_folder"]}/image_cropped.jpg'
        model_path = config["id-model"]["path"]
        
        # 🔥 ОПРЕДЕЛЯЕМ device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🔥 ОПРЕДЕЛЯЕМ трансформы
        TRANSFORMS = get_vit_transforms()
        
        # 🔥 ЗАГРУЖАЕМ модель
        model = load_model(model_path, device)
        
        # 🔥 ГЕНЕРИРУЕМ эмбеддинг
        embedding = get_embedding(cropped_path, model, TRANSFORMS, device)
        if embedding is None:
            return {"status": "error", "message": "Не удалось сгенерировать эмбеддинг"}
        
        # === 3. 🔥 НОВЫЙ ПОИСК: FAISS + SQLite ===
        matches = find_similar_in_database(
            query_embedding=embedding,
            db_path="database/cards.db",
            faiss_index_path="data/embeddings/database_embeddings.pkl",
            top_k=5
        )
        
        # === 4. 🔥 ВСЕГДА возвращаем на ручную проверку ===
        return {
            "status": "review_required",  # ← НИКОГДА не "auto_add"
            "matches": matches,           # ← Топ-5 совпадений из БД
            "embedding": embedding,       # ← Для сохранения
            "cropped_path": cropped_path, # ← Путь к кропу
            "full_path": filepath,        # ← Путь к полному фото
            "message": "Требуется ручная проверка биологом"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

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

def add_repeat_encounter(
    individual_id: str,
    embedding: np.ndarray,
    photo_path_full: str,
    photo_path_cropped: str,
    template_type: str = "КВ-1",
    **card_data
):
    """
    Вызывается, когда биолог подтвердил: "это та же особь, новая встреча".
    """
    photo_number = add_encounter(
        individual_id=individual_id,
        template_type=template_type,
        photo_path_full=photo_path_full,
        photo_path_cropped=photo_path_cropped,
        embedding=embedding,
        date=datetime.now().strftime("%d.%m.%Y"),
        **card_data
    )
    return photo_number

if __name__ == "__main__":
    asyncio.run(photo_processing(load_config(), "data/input/image.png"))
