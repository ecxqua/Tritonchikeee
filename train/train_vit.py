import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm
from sklearn.metrics.pairwise import pairwise_distances
from transformers import get_cosine_schedule_with_warmup
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random


# было несоотвествие расчета расстояний на этапе обучения и оценке метркики (евклидовое и косинусное - заменить на косинусное) + архивирован ресайз и рандом кроп, флипов нет

class EnhancedTripletNet(nn.Module):
    def __init__(self, base_model_name='vit_base_patch16_224', embedding_dim=512, dropout_rate=0.4):
        super(EnhancedTripletNet, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        # Прогрессивное размораживание слоев
        self._setup_progressive_unfreezing()

        # Улучшенная сеть эмбеддингов
        self.embedding = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Dropout(dropout_rate/2),
            nn.Linear(512, embedding_dim),
        )

        # Проекционная головка для контрастивного обучения
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )

        self._init_weights()

    def _setup_progressive_unfreezing(self):
        """Поэтапное размораживание слоев модели"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        if hasattr(self.base_model, 'blocks'):
            num_blocks = len(self.base_model.blocks)
            blocks_to_unfreeze = min(6, num_blocks)
            for i in range(num_blocks - blocks_to_unfreeze, num_blocks):
                for param in self.base_model.blocks[i].parameters():
                    param.requires_grad = True

    def _init_weights(self):
        """Инициализация весов"""
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

class IndividualCenterLoss(nn.Module):
    """Center Loss для отдельных особей"""
    def __init__(self, num_unique_individuals, feat_dim=512, lambda_ic=0.2):
        super(IndividualCenterLoss, self).__init__()
        self.num_individuals = num_unique_individuals
        self.feat_dim = feat_dim
        self.lambda_ic = lambda_ic
        centers = torch.randn(num_unique_individuals, feat_dim)
        self.individual_centers = nn.Parameter(F.normalize(centers, p=2, dim=1))

    def forward(self, x, individual_labels):
        batch_size = x.size(0)
        if self.individual_centers.device != x.device:
            self.individual_centers.data = self.individual_centers.data.to(x.device)

        centers_batch = self.individual_centers[individual_labels]

        cos_sim = F.cosine_similarity(x, centers_batch, dim=1)
        loss = 1 - cos_sim.mean()
        return self.lambda_ic * loss

    def update_centers(self, x, individual_labels, alpha=0.5):
        """Обновление центров особей с нормализацией"""
        with torch.no_grad():
            if self.individual_centers.device != x.device:
                self.individual_centers.data = self.individual_centers.data.to(x.device)

            for i in range(self.num_individuals):
                mask = (individual_labels == i)
                if mask.sum() > 0:
                    new_center = alpha * self.individual_centers.data[i] + \
                                 (1 - alpha) * x[mask].mean(dim=0)
                    # Нормализуем новый центр
                    self.individual_centers.data[i] = F.normalize(new_center.unsqueeze(0), p=2, dim=1).squeeze(0)

class ProgressiveTripletDataset(Dataset): #train(прогрессивная сложность, обновление кэша эмбедингов)
    def __init__(self, df, species_mapping, transform=None, hard_mining=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.hard_mining = hard_mining
        self.species_mapping = species_mapping
        self.class_to_indices = {}

        # Кэш для вычисления трудных примеров
        self.embeddings_cache = None
        self.distance_cache = None

        for idx, row in self.df.iterrows():
            class_id = row['class_id']
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)

        # Прогрессивные параметры
        self.current_phase = 0
        self.phases = [
            {'intra_species_prob': 0.2, 'description': 'Легкие: 80% разных видов'},
            {'intra_species_prob': 0.5, 'description': 'Средние: 50/50'},
            {'intra_species_prob': 0.8, 'description': 'Сложные: 80% один вид'}
        ]

    def update_phase(self, epoch, total_epochs):
        """Обновляет фазу сложности в зависимости от эпохи"""
        phase_boundaries = [total_epochs * 0.2, total_epochs * 0.5]

        if epoch < phase_boundaries[0]:
            new_phase = 0
        elif epoch < phase_boundaries[1]:
            new_phase = 1
        else:
            new_phase = 2

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            print(f"Переход к фазе {new_phase + 1}: {self.phases[new_phase]['description']}")

    def update_embeddings_cache(self, model, device, transform):
        """Обновление кэша эмбеддингов для hard mining"""
        model.eval()
        embeddings = []

        basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for idx in tqdm(range(len(self.df)), desc="Вычисление эмбеддингов для hard mining"):
                row = self.df.iloc[idx]
                image = Image.open(row['path']).convert('RGB')
                image = basic_transforms(image).unsqueeze(0).to(device)
                embed = model(image)
                embeddings.append(embed.cpu().numpy())

        self.embeddings_cache = np.vstack(embeddings)
        self.distance_cache = pairwise_distances(self.embeddings_cache, metric='cosine')
        print(f"Кэш эмбеддингов обновлен: {self.embeddings_cache.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]
        anchor_image = Image.open(anchor_row['path']).convert('RGB')
        anchor_class = anchor_row['class_id']
        anchor_species = self.species_mapping[anchor_class]

        # Позитивный пример
        positive_indices = [i for i in self.class_to_indices[anchor_class] if i != idx]
        if not positive_indices:
            positive_image = anchor_image.copy()
        else:
            positive_idx = random.choice(positive_indices)
            positive_image = Image.open(self.df.iloc[positive_idx]['path']).convert('RGB')

        # Стратегия выбора негативного примера
        current_phase_settings = self.phases[self.current_phase]

        if random.random() < current_phase_settings['intra_species_prob']:
            negative_species = anchor_species
            difficulty = "hard"
        else:
            negative_species = 1 - anchor_species
            difficulty = "easy"

        # Выбираем конкретную особь
        negative_candidates = []
        for individual_id, species_id in self.species_mapping.items():
            if species_id == negative_species and individual_id in self.class_to_indices:
                negative_candidates.extend(self.class_to_indices[individual_id])

        if not negative_candidates:
            negative_classes = [c for c in self.class_to_indices.keys() if c != anchor_class]
            if negative_classes:
                negative_class = random.choice(negative_classes)
                negative_candidates = self.class_to_indices[negative_class]
            else:
                negative_image = anchor_image.copy()
                negative_candidates = [idx]

        # Hard negative mining для сложных примеров
        if difficulty == "hard" and self.hard_mining and self.distance_cache is not None and negative_candidates:
            hardest_negative_idx = None
            hardest_distance = float('inf')

            for neg_idx in negative_candidates:
                if neg_idx < len(self.distance_cache) and idx < len(self.distance_cache):
                    distance = self.distance_cache[idx, neg_idx]
                    if distance < hardest_distance:
                        hardest_distance = distance
                        hardest_negative_idx = neg_idx

            negative_idx = hardest_negative_idx if hardest_negative_idx is not None else random.choice(negative_candidates)
        else:
            negative_idx = random.choice(negative_candidates)

        negative_image = Image.open(self.df.iloc[negative_idx]['path']).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_class

class AdaptiveMultiLossFunction(nn.Module):
    def __init__(self, species_mapping, num_epochs, num_individuals,
                 easy_margin=0.05, medium_margin=0.15, hard_margin=0.25):
        super().__init__()

        self.species_mapping = species_mapping
        self.num_epochs = num_epochs

        self.margins = {
            'easy': easy_margin,
            'medium': medium_margin,
            'hard': hard_margin
        }

        self.triplet_loss = nn.TripletMarginLoss(margin=medium_margin, p=2)
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.individual_center_loss = IndividualCenterLoss(
            num_unique_individuals=num_individuals, feat_dim=512, lambda_ic=0.2
        )

        self.phase_weights = {
            0: {'triplet': 1.0, 'contrast': 0.1, 'center': 0.3},
            1: {'triplet': 1.2, 'contrast': 0.2, 'center': 0.5},
            2: {'triplet': 1.5, 'contrast': 0.3, 'center': 0.8}
        }

    def get_current_phase(self, epoch):
        """Определяет текущую фазу обучения"""
        phase_boundaries = [self.num_epochs * 0.2, self.num_epochs * 0.5]

        if epoch < phase_boundaries[0]:
            return 0, 'easy'
        elif epoch < phase_boundaries[1]:
            return 1, 'medium'
        else:
            return 2, 'hard'

    def forward(self, anchor, positive, negative, projections_anchor=None,
                projections_positive=None, labels=None, current_epoch=0):

        # Определяем сложность триплета
        if labels is not None and len(labels) > 2:
            anchor_species = self.species_mapping[labels[0].item()]
            negative_species = self.species_mapping[labels[2].item()]

            if anchor_species == negative_species:
                triplet_difficulty = 'hard'
            else:
                triplet_difficulty = 'easy'
        else:
            triplet_difficulty = 'medium'

        # Динамически обновляем margin
        current_margin = self.margins[triplet_difficulty]
        self.triplet_loss.margin = current_margin

        triplet_loss = self.triplet_loss(anchor, positive, negative)

        # Получаем веса для текущей фазы
        current_phase, phase_name = self.get_current_phase(current_epoch)
        weights = self.phase_weights[current_phase]

        total_loss = weights['triplet'] * triplet_loss
        contrast_loss = torch.tensor(0.0).to(anchor.device)
        individual_center_loss_value = torch.tensor(0.0).to(anchor.device)

        if projections_anchor is not None and labels is not None:
            all_projections = torch.cat([projections_anchor, projections_positive], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            contrast_loss = self.supervised_contrastive_loss(all_projections, all_labels)

            individual_center_loss_value = self.individual_center_loss(anchor, labels)

            total_loss = (weights['triplet'] * triplet_loss +
                         weights['contrast'] * contrast_loss +
                         weights['center'] * individual_center_loss_value)

            if self.training:
                self.individual_center_loss.update_centers(anchor.detach(), labels.detach())

        return (total_loss, triplet_loss, contrast_loss, individual_center_loss_value,
                current_phase, triplet_difficulty)

    def supervised_contrastive_loss(self, projections, labels):
        batch_size = projections.shape[0]
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0))
        similarity_matrix = torch.matmul(projections, projections.T) / 0.1

        mask = torch.eye(batch_size, device=projections.device).bool()
        label_matrix_clean = torch.logical_and(label_matrix, torch.logical_not(mask))

        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim * torch.logical_not(mask).float(), dim=1)
        pos_sim = torch.sum(exp_sim * label_matrix_clean.float(), dim=1)
        loss = -torch.log(pos_sim / (sum_exp_sim + 1e-8) + 1e-8)

        return loss.mean()

class EnhancedTripletDataset(Dataset): #для валидации (фиксированная сложность,без апдейта фаз)
    def __init__(self, df, species_mapping, transform=None, hard_mining=True, intra_species_neg_prob=0.7):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.hard_mining = hard_mining
        self.species_mapping = species_mapping
        self.intra_species_neg_prob = intra_species_neg_prob
        self.class_to_indices = {}
        self.embeddings_cache = None
        self.distance_cache = None

        for idx, row in self.df.iterrows():
            class_id = row['class_id']
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)

    def update_embeddings_cache(self, model, device, transform):
        """Обновление кэша эмбеддингов для hard mining"""
        model.eval()
        embeddings = []

        basic_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            for idx in tqdm(range(len(self.df)), desc="Вычисление эмбеддингов для hard mining"):
                row = self.df.iloc[idx]
                image = Image.open(row['path']).convert('RGB')
                image = basic_transforms(image).unsqueeze(0).to(device)
                embed = model(image)
                embeddings.append(embed.cpu().numpy())

        self.embeddings_cache = np.vstack(embeddings)
        self.distance_cache = pairwise_distances(self.embeddings_cache, metric='cosine')
        print(f"Кэш эмбеддингов обновлен: {self.embeddings_cache.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_row = self.df.iloc[idx]
        anchor_image = Image.open(anchor_row['path']).convert('RGB')
        anchor_class = anchor_row['class_id']
        anchor_species = self.species_mapping[anchor_class]

        positive_indices = [i for i in self.class_to_indices[anchor_class] if i != idx]
        if not positive_indices:
            positive_image = anchor_image.copy()
        else:
            positive_idx = random.choice(positive_indices)
            positive_image = Image.open(self.df.iloc[positive_idx]['path']).convert('RGB')

        if random.random() < self.intra_species_neg_prob:
            negative_species = anchor_species
        else:
            negative_species = 1 - anchor_species

        negative_candidates = []
        for individual_id, species_id in self.species_mapping.items():
            if species_id == negative_species and individual_id in self.class_to_indices:
                negative_candidates.extend(self.class_to_indices[individual_id])

        if not negative_candidates:
            negative_classes = [c for c in self.class_to_indices.keys() if c != anchor_class]
            if negative_classes:
                negative_class = random.choice(negative_classes)
                negative_candidates = self.class_to_indices[negative_class]
            else:
                negative_image = anchor_image.copy()
                negative_candidates = [idx]

        if self.hard_mining and self.distance_cache is not None and negative_candidates:
            hardest_negative_idx = None
            hardest_distance = float('inf')

            for neg_idx in negative_candidates:
                if neg_idx < len(self.distance_cache) and idx < len(self.distance_cache):
                    distance = self.distance_cache[idx, neg_idx]
                    if distance < hardest_distance:
                        hardest_distance = distance
                        hardest_negative_idx = neg_idx

            negative_idx = hardest_negative_idx if hardest_negative_idx is not None else random.choice(negative_candidates)
        else:
            negative_idx = random.choice(negative_candidates)

        negative_image = Image.open(self.df.iloc[negative_idx]['path']).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_class

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, row['class_id']

def get_enhanced_transforms():
    """Аугментации для тренировки"""
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms

def create_unified_species_mapping(df_train, df_val, df_test):
    """Создает маппинг для числовых class_id -> species_id"""
    species_mapping = {}

    for df in [df_train, df_val, df_test]:
        df['unique_individual'] = df['class_name'] + '_' + df['individual_name']

    # Объединяем все данные
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Создаем маппинг числовых class_id -> species_id (class_id оригинального вида)
    all_individuals = df_all['unique_individual'].unique()
    individual_to_idx = {ind: idx for idx, ind in enumerate(all_individuals)}

    # Применяем маппинг ко всем датасетам
    for df in [df_train, df_val, df_test]:
        df['class_id_numeric'] = df['unique_individual'].map(individual_to_idx)

    # Создаем маппинг для числовых ID
    for unique_ind in all_individuals:
        numeric_id = individual_to_idx[unique_ind]
        # species_id берем из class_id оригинального вида
        species_id = df_all[df_all['unique_individual'] == unique_ind]['class_id'].iloc[0]
        species_mapping[numeric_id] = species_id

    print(f"Создан маппинг для {len(species_mapping)} уникальных особей")
    print(f"Пример маппинга: {dict(list(species_mapping.items())[:5])}")
    return species_mapping

def visualize_embeddings_with_individuals(model, dataloader, device, species_mapping, method='tsne', n_samples=1000):
    """Визуализация эмбеддингов с цветами для видов и особей"""
    model.eval()
    embeddings = []
    species_labels = []
    individual_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Извлечение эмбеддингов"):
            if len(batch) == 4:
                anchor_imgs, _, _, class_ids = batch
                images = anchor_imgs
            else:
                images, class_ids = batch

            images = images.to(device)
            embed = model(images)

            embeddings.append(embed.cpu().numpy())
            individual_labels.extend(class_ids.numpy())

            for ind_id in class_ids.numpy():
                if ind_id in species_mapping:
                    species_labels.append(species_mapping[ind_id])
                else:
                    fallback_species = 0 if ind_id < 100 else 1
                    species_labels.append(fallback_species)
                    species_mapping[ind_id] = fallback_species

    embeddings = np.vstack(embeddings)
    species_labels = np.array(species_labels)
    individual_labels = np.array(individual_labels)

    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        species_labels = species_labels[indices]
        individual_labels = individual_labels[indices]

    print(f"Визуализация {len(embeddings)} точек")
    print(f"Количество видов: {len(np.unique(species_labels))}")
    print(f"Количество особей: {len(np.unique(individual_labels))}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. t-SNE по ВИДАМ
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)

    scatter1 = axes[0,0].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                                c=species_labels, cmap='tab10', alpha=0.7, s=30, marker='o')
    axes[0,0].set_title('t-SNE по ВИДАМ', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('t-SNE 1')
    axes[0,0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0,0])

    # 2. t-SNE по ОСОБЯМ
    scatter2 = axes[0,1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                                c=individual_labels, cmap='tab20', alpha=0.7, s=30, marker='o')
    axes[0,1].set_title(f't-SNE по ОСОБЯМ ({len(np.unique(individual_labels))} особей)',
                       fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('t-SNE 1')
    axes[0,1].set_ylabel('t-SNE 2')

    # 3. UMAP по ВИДАМ
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_umap = reducer.fit_transform(embeddings)

    scatter3 = axes[1,0].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                                c=species_labels, cmap='tab10', alpha=0.7, s=30, marker='o')
    axes[1,0].set_title('UMAP по ВИДАМ', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('UMAP 1')
    axes[1,0].set_ylabel('UMAP 2')
    plt.colorbar(scatter3, ax=axes[1,0])

    # 4. UMAP по ОСОБЯМ
    scatter4 = axes[1,1].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                                c=individual_labels, cmap='tab20', alpha=0.7, s=30, marker='o')
    axes[1,1].set_title(f'UMAP по ОСОБЯМ', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('UMAP 1')
    axes[1,1].set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig('enhanced_embedding_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return embeddings, species_labels, individual_labels


def evaluate_model(model, dataloader, device, epoch=None, mode='test'):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluation {mode} epoch {epoch}" if epoch else f"Evaluation {mode}"):
            if mode == 'test':
                images, class_ids = batch
                images = images.to(device)
                embed = model(images)
            else:
                anchor_imgs, _, _, class_ids = batch
                images = anchor_imgs.to(device)
                embed = model(images)

            embeddings.append(embed.cpu().numpy())
            labels.extend(class_ids.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Вычисление метрик
    dist_matrix = pairwise_distances(embeddings, metric='cosine')

    avg_pos_dist = []
    avg_neg_dist = []
    precision_at_1 = 0
    precision_at_5 = 0
    total = 0

    for i in range(len(labels)):
        pos_mask = labels == labels[i]
        pos_mask[i] = False
        if np.any(pos_mask):
            avg_pos_dist.append(np.mean(dist_matrix[i, pos_mask]))

        neg_mask = labels != labels[i]
        if np.any(neg_mask):
            avg_neg_dist.append(np.mean(dist_matrix[i, neg_mask]))

        if i < len(labels) - 1:
            sorted_indices = np.argsort(dist_matrix[i, :])
            sorted_indices = sorted_indices[sorted_indices != i]

            # Precision@1
            nearest = sorted_indices[0]
            precision_at_1 += (labels[nearest] == labels[i])

            # Precision@5
            top5 = sorted_indices[:5]
            precision_at_5 += np.sum(labels[top5] == labels[i]) / 5

            total += 1

    metrics = {
        'avg_pos_distance': np.mean(avg_pos_dist),
        'avg_neg_distance': np.mean(avg_neg_dist),
        'pos_neg_ratio': np.mean(avg_pos_dist) / np.mean(avg_neg_dist),
        'precision@1': precision_at_1 / total,
        'precision@5': precision_at_5 / total,
        'num_samples': len(labels)
    }

    return metrics

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 30


    df_train = pd.read_csv("/content/drive/MyDrive/TRITONS-PROJECT/clean_crop_dataset_train/labels_train.csv")
    df_val = pd.read_csv("/content/drive/MyDrive/TRITONS-PROJECT/clean_crop_dataset_train/labels_val.csv")
    df_test = pd.read_csv("/content/drive/MyDrive/TRITONS-PROJECT/clean_crop_dataset_train/labels_test.csv")

    for df in [df_train, df_val, df_test]:
        df['individual_name'] = df['individual_name'].astype(str)
        df['unique_individual'] = df['class_name'] + '_' + df['individual_name']

    species_mapping = create_unified_species_mapping(df_train, df_val, df_test)

    all_individuals = pd.concat([df_train, df_val, df_test])['unique_individual'].unique()
    individual_to_idx = {ind: idx for idx, ind in enumerate(all_individuals)}

    # Применяем маппинг ко всем датасетам
    df_train_individual = df_train.copy()
    df_train_individual['class_id'] = df_train_individual['unique_individual'].map(individual_to_idx)

    df_val_individual = df_val.copy()
    df_val_individual['class_id'] = df_val_individual['unique_individual'].map(individual_to_idx)

    df_test_individual = df_test.copy()
    df_test_individual['class_id'] = df_test_individual['unique_individual'].map(individual_to_idx)

    num_unique_individuals = len(all_individuals)
    print(f"Всего уникальных особей (вид_особь): {num_unique_individuals}")
    print(f"Примеры особей: {list(all_individuals[:10])}")


    train_dataset = ProgressiveTripletDataset(
        df_train_individual,
        species_mapping=species_mapping,
        transform=get_enhanced_transforms(),
        hard_mining=True
    )

    basic_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = EnhancedTripletDataset(
        df_val_individual,
        species_mapping=species_mapping,
        transform=basic_transforms,
        hard_mining=False
    )

    test_dataset = TestDataset(df_test_individual, transform=basic_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)


    model = EnhancedTripletNet().to(device)

    criterion = AdaptiveMultiLossFunction(
        species_mapping=species_mapping,
        num_epochs=num_epochs,
        num_individuals=num_unique_individuals,
        easy_margin=0.1,      
        medium_margin=0.3,
        hard_margin=0.5
    ).to(device)

    #Упрощенный оптимизатор с единым LR
    optimizer = optim.AdamW(
        model.parameters(),  # Все параметры вместе
        lr=2e-5,            # Стандартный LR для fine-tuning
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=3 * len(train_loader),  # 3 эпохи warmup
        num_training_steps=num_epochs * len(train_loader)
    )

    history = {
        'train_loss': [], 'val_loss': [], 'train_triplet_loss': [],
        'train_contrast_loss': [], 'train_center_loss': [],
        'val_metrics': [], 'learning_rates': [], 'phases': [], 'difficulties': []
    }

    print("Начало прогрессивного обучения...")

    for epoch in range(num_epochs):
        train_dataset.update_phase(epoch, num_epochs)

        # Обновляем hard mining каждые 5 эпох
        if epoch % 5 == 0:
            train_dataset.update_embeddings_cache(model, device, basic_transforms)


        model.train()
        train_loss = 0.0
        train_triplet_loss = 0.0
        train_contrast_loss = 0.0
        train_center_loss = 0.0
        phase_counts = {'easy': 0, 'medium': 0, 'hard': 0}

        for batch in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs}"):
            anchor_imgs, pos_imgs, neg_imgs, class_ids = batch
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)
            class_ids = class_ids.to(device)

            optimizer.zero_grad()

            # Получаем эмбеддинги и проекции
            embed_anchor, proj_anchor = model(anchor_imgs, return_projection=True)
            embed_positive, proj_positive = model(pos_imgs, return_projection=True)
            embed_negative = model(neg_imgs)

            # Вычисляем потерю с учетом текущей эпохи
            (total_loss, triplet_loss, contrast_loss,
             center_loss, current_phase, difficulty) = criterion(
                embed_anchor, embed_positive, embed_negative,
                proj_anchor, proj_positive, class_ids, epoch
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_triplet_loss += triplet_loss.item()
            train_contrast_loss += contrast_loss.item()
            train_center_loss += center_loss.item()
            phase_counts[difficulty] += 1


        num_batches = len(train_loader)
        history['train_loss'].append(train_loss / num_batches)
        history['train_triplet_loss'].append(train_triplet_loss / num_batches)
        history['train_contrast_loss'].append(train_contrast_loss / num_batches)
        history['train_center_loss'].append(train_center_loss / num_batches)
        history['phases'].append(current_phase)
        history['difficulties'].append(phase_counts)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                anchor_imgs, pos_imgs, neg_imgs, class_ids = batch
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)
                class_ids = class_ids.to(device)

                embed_anchor = model(anchor_imgs)
                embed_positive = model(pos_imgs)
                embed_negative = model(neg_imgs)

                loss, _, _, _, _, _ = criterion(embed_anchor, embed_positive, embed_negative, labels=class_ids, current_epoch=epoch)
                val_loss += loss.item()

        history['val_loss'].append(val_loss / len(val_loader))

        # Обновление learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)

        # Оценка на валидации
        val_metrics = evaluate_model(model, val_loader, device, epoch + 1, 'val')
        history['val_metrics'].append(val_metrics)


        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Triplet: {history['train_triplet_loss'][-1]:.4f} | Contrast: {history['train_contrast_loss'][-1]:.4f}")
        print(f"Individual Center: {history['train_center_loss'][-1]:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print("Метрики валидации:")
        print(f"  Pos Distance: {val_metrics['avg_pos_distance']:.4f} | Neg Distance: {val_metrics['avg_neg_distance']:.4f}")
        print(f"  Pos/Neg Ratio: {val_metrics['pos_neg_ratio']:.4f}")
        print(f"  Precision@1: {val_metrics['precision@1']:.4f} | Precision@5: {val_metrics['precision@5']:.4f}")


        total_triplets = sum(phase_counts.values())
        print(f"Сложность триплетов: "
              f"Легкие {phase_counts['easy']/total_triplets*100:.1f}%, "
              f"Сложные {phase_counts['hard']/total_triplets*100:.1f}%")

        # Сохранение лучшей модели
        if epoch == 0 or val_metrics['pos_neg_ratio'] < min([m['pos_neg_ratio'] for m in history['val_metrics'][:epoch+1]]):
            torch.save(model.state_dict(), 'github_model.pth')
            print(" Сохранена новая лучшая модель")


    print("\nФинальное тестирование...")
    model.load_state_dict(torch.load('github_model.pth', map_location=device))

    test_metrics = evaluate_model(model, test_loader, device, mode='test')
    history['test_metrics'] = test_metrics


    print("\nВизуализация эмбеддингов...")
    embeddings, species_labels, individual_labels = visualize_embeddings_with_individuals(
        model, test_loader, device, species_mapping, method='all'
    )


    print("\n" + "="*50)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*50)
    print(f"Тестовые примеры: {test_metrics['num_samples']}")
    print(f"Среднее расстояние позитивных пар: {test_metrics['avg_pos_distance']:.4f}")
    print(f"Среднее расстояние негативных пар: {test_metrics['avg_neg_distance']:.4f}")
    print(f"Соотношение Pos/Neg: {test_metrics['pos_neg_ratio']:.4f}")
    print(f"Precision@1: {test_metrics['precision@1']:.4f}")
    print(f"Precision@5: {test_metrics['precision@5']:.4f}")

    return history, model, species_mapping

