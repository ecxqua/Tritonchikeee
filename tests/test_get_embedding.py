# 1. Проверьте, что модель загрузилась корректно
from pipeline.deployment_vit_faiss import load_model
import torch
import numpy as np
from pathlib import Path
MODEL_PATH = Path("models/best_model.pth")
FAISS_INDEX_PATH = Path("data/embeddings/database_embeddings.pkl")

# Параметры
BATCH_SIZE = 32
EMBEDDING_DIM = 512  # Размер вектора из EnhancedTripletNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(MODEL_PATH, DEVICE)
print(f"Модель в eval: {not model.training}")
print(f"Первый параметр: {next(model.parameters())[0, :5].detach().cpu().numpy()}")

# 2. Сравните эмбеддинги одного фото через обе функции
from pipeline.deployment_vit_faiss import get_embedding, get_embedding_from_array, DEFAULT_TRANSFORM
import cv2

img_path = "data/cropped/7.jpg"

# Через get_embedding (как в запросе)
emb1 = get_embedding(img_path, model, DEFAULT_TRANSFORM, DEVICE)

# Через get_embedding_from_array (как в миграции)
img_array = cv2.imread(img_path)
emb2 = get_embedding_from_array(img_array, model, DEFAULT_TRANSFORM, DEVICE)

if emb1 is not None and emb2 is not None:
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Косинус между эмбеддингами: {cos_sim:.4f} (должно быть >0.999)")
    print(f"Макс. разница: {np.max(np.abs(emb1 - emb2)):.6f}")
else:
    print(f"emb1={emb1 is not None}, emb2={emb2 is not None}")
