from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.deployment_dinov2_faiss import DEFAULT_IMAGE_SIZE, DinoEmbeddingNet


@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    output_dir: str
    backbone: str = "vit_base_patch14_dinov2"
    embedding_dim: int = 512
    image_size: int = DEFAULT_IMAGE_SIZE
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 1e-4
    margin: float = 0.2
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class TripletCsvDataset(Dataset):
    def __init__(self, csv_path: str, transform: transforms.Compose):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.transform = transform
        self.frame = pd.read_csv(self.csv_path)
        required_columns = {"path", "individual_id"}
        missing = required_columns.difference(self.frame.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} is missing columns: {sorted(missing)}")

        self.frame["individual_id"] = self.frame["individual_id"].astype(int)
        self.records = self.frame.to_dict(orient="records")

        self.indices_by_individual: Dict[int, List[int]] = {}
        for idx, row in enumerate(self.records):
            individual_id = int(row["individual_id"])
            self.indices_by_individual.setdefault(individual_id, []).append(idx)

        self.individual_ids = sorted(self.indices_by_individual.keys())
        if len(self.individual_ids) < 2:
            raise ValueError("Training requires at least 2 unique individuals")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _sample_positive(self, anchor_idx: int, individual_id: int) -> int:
        candidates = self.indices_by_individual[individual_id]
        if len(candidates) == 1:
            return anchor_idx
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = random.choice(candidates)
        return positive_idx

    def _sample_negative(self, individual_id: int) -> int:
        negative_individual = individual_id
        while negative_individual == individual_id:
            negative_individual = random.choice(self.individual_ids)
        return random.choice(self.indices_by_individual[negative_individual])

    def __getitem__(self, index: int):
        anchor_row = self.records[index]
        anchor_individual = int(anchor_row["individual_id"])

        positive_index = self._sample_positive(index, anchor_individual)
        negative_index = self._sample_negative(anchor_individual)

        anchor_image = self.transform(self._load_image(anchor_row["path"]))
        positive_image = self.transform(self._load_image(self.records[positive_index]["path"]))
        negative_image = self.transform(self._load_image(self.records[negative_index]["path"]))

        return anchor_image, positive_image, negative_image, anchor_individual


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    positive_distance = 1.0 - F.cosine_similarity(anchor, positive)
    negative_distance = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(positive_distance - negative_distance + margin).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, margin: float) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for anchor, positive, negative, _ in loader:
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = cosine_triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: TrainConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "metadata": {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": asdict(config),
        },
    }
    torch.save(payload, output_path)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)

    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TripletCsvDataset(config.train_csv, build_transforms(config.image_size, train=True))
    val_dataset = TripletCsvDataset(config.val_csv, build_transforms(config.image_size, train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DinoEmbeddingNet(
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        image_size=config.image_size,
        pretrained=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for anchor, positive, negative, _ in progress:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)
                loss = cosine_triplet_loss(anchor_embedding, positive_embedding, negative_embedding, config.margin)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1
            progress.set_postfix(loss=running_loss / batch_count)

        scheduler.step()

        train_loss = running_loss / max(batch_count, 1)
        val_loss = evaluate(model, val_loader, device, config.margin)

        print(
            f"Epoch {epoch}/{config.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        last_path = output_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, config)
            print(f"Saved best checkpoint to {best_path}")

    summary = {
        "train_csv": config.train_csv,
        "val_csv": config.val_csv,
        "output_dir": config.output_dir,
        "epochs": config.epochs,
        "best_val_loss": best_val_loss,
        "device": str(device),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DINOv2 embedding model for Triton identification")
    parser.add_argument("--train-csv", required=True, help="Path to labels_train.csv")
    parser.add_argument("--val-csv", required=True, help="Path to labels_val.csv")
    parser.add_argument("--output-dir", default="models/dinov2_scratch_10e", help="Directory for checkpoints")
    parser.add_argument("--backbone", default="vit_base_patch14_dinov2", help="timm backbone name")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")

    args = parser.parse_args()
    return TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()