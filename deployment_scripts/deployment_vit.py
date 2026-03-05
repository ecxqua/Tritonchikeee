import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm
import shutil
import torch.nn as nn
import timm
import pickle
from torchvision import transforms


class EnhancedTripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=512, dropout_rate=0.4):
        super(EnhancedTripletNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        self._setup_progressive_unfreezing()

        self.embedding = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, embedding_dim),
        )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

        self._init_weights()

    def _setup_progressive_unfreezing(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        if hasattr(self.base_model, 'blocks'):
            num_blocks = len(self.base_model.blocks)
            blocks_to_unfreeze = min(6, num_blocks)
            for i in range(num_blocks - blocks_to_unfreeze, num_blocks):
                for param in self.base_model.blocks[i].parameters():
                    param.requires_grad = True

    def _init_weights(self):
        for module in self.embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, return_projection=False):
        features = self.base_model(x)
        embeddings = self.embedding(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if return_projection:
            projections = self.projection(embeddings)
            projections = nn.functional.normalize(projections, p=2, dim=1)
            return embeddings, projections

        return embeddings


def load_model(model_path, device):
    model = EnhancedTripletNet(base_model_name='vit_base_patch16_224', embedding_dim=512)
    state_dict = torch.load(model_path, map_location=device)

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            filtered_state_dict[k] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_embedding(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def compute_distances(embeddings1, embedding2):
    similarities = cosine_similarity(embeddings1, embedding2.reshape(1, -1))
    return 1 - similarities.flatten()


def save_embeddings(embeddings, paths, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'paths': paths}, f)


def load_embeddings(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['paths']


def find_similar_images(model_path, database_dir, query_image_path, output_dir, transform, device, bot):
    embeddings_save_path = os.path.join(database_dir, 'database_embeddings.pkl')
    if os.path.exists(embeddings_save_path):
        print("Загрузка сохраненных эмбеддингов...")
        database_embeddings, database_image_paths = load_embeddings(embeddings_save_path)
    else:
        print("Вычисление эмбеддингов базы...")
        model = load_model(model_path, device)

        database_image_paths = []
        for root, _, files in os.walk(database_dir):
            if os.path.normpath(root).startswith(os.path.normpath(output_dir)):
                continue

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        database_image_paths.append(full_path)

        print(f"Найдено {len(database_image_paths)} изображений в базе")

        database_embeddings = []
        valid_paths = []
        for path in tqdm(database_image_paths, desc="Обработка базы"):
            emb = get_embedding(path, model, transform, device)
            if emb is not None:
                database_embeddings.append(emb)
                valid_paths.append(path)

        database_embeddings = np.array(database_embeddings)
        save_embeddings(database_embeddings, valid_paths, embeddings_save_path)
        print(f"Эмбеддинги сохранены в {embeddings_save_path}")
        database_image_paths = valid_paths

    model = load_model(model_path, device)
    query_embedding = get_embedding(query_image_path, model, transform, device)
    if query_embedding is None:
        print("Не удалось обработать запросное изображение")
        return

    distances = compute_distances(np.array(database_embeddings), query_embedding)
    top5_idx = np.argsort(distances)[:bot.size_answer]

    print("\n=== Топ-5 результатов ===")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + "/res.txt", 'w', encoding='utf-8') as file:
        for i, idx in enumerate(top5_idx, 1):
            src_path = database_image_paths[idx]
            dst_filename = f"top{i}.jpg"
            dst_path = os.path.join(output_dir, dst_filename)

            try:
                shutil.copy(src_path, dst_path)
                class_name = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
                individual = os.path.basename(os.path.dirname(src_path))
                similarity = 1 - distances[idx]
                similarity_percent = similarity * 100

                class_string = 'Ребристый' if class_name == 'ribbed' else "Карелина"
                res_str = f"{i}. Класс: {class_string} | Особь: {individual} | Схожесть: {similarity_percent:.1f}%\n"
                file.write(res_str)
                print(
                    f"{i}. Класс: {class_name} | Особь: {individual} | Схожесть: {similarity_percent:.1f}% | Путь: {src_path}")
            except Exception as e:
                print(f"Ошибка копирования файла {src_path}: {str(e)}")
        print(f"\nРезультаты сохранены в: {output_dir}")