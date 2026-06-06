"""
pipeline/deployment_dinov2_faiss.py — DINOv2 embedding model for Triton identification

This module replaces the old ViT-based EnhancedTripletNet with DINOv2 embedding model
trained on Triton belly photos. The model is trained from scratch for 10 epochs 
and provides better generalization to unseen individuals.

Model checkpoint: models/dinov2_scratch_10e/best.pt
Backbone: vit_base_patch14_dinov2 (pretrained=False)
Embedding dim: 512
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_IMAGE_SIZE = 224
DEFAULT_EMBEDDING_DIM = 512

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class DinoEmbeddingNet(nn.Module):
    """
    DINOv2-based embedding model for Triton identification.
    
    Architecture:
        - Backbone: vit_base_patch14_dinov2 (trained from scratch)
        - Projection head: 768 -> 1024 -> 512 (with L2 normalization)
        - Output: 512-dim normalized embeddings for similarity search
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = 0.2,
        image_size: int = DEFAULT_IMAGE_SIZE,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        # Load backbone (pretrained=False for from-scratch model)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, img_size=image_size
        )
        feature_dim = self.backbone.num_features

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image -> features -> embeddings (L2 normalized)"""
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


# =============================================================================
# LOADING & INFERENCE
# =============================================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Load DINOv2 embedding model from checkpoint.
    
    Compatible interface with deployment_vit_faiss.load_model().
    
    Args:
        checkpoint_path: Path to model checkpoint (best.pt or last.pt)
        device: torch.device object ('cuda' or 'cpu')
    
    Returns:
        model: Loaded DinoEmbeddingNet model on specified device (eval mode)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading DINOv2 model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    metadata = checkpoint.get("metadata", {})
    saved_args = metadata.get("args", {})
    
    backbone = saved_args.get("backbone", "vit_base_patch14_dinov2")
    embedding_dim = saved_args.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
    image_size = saved_args.get("image_size", DEFAULT_IMAGE_SIZE)
    
    # Create and load model
    model = DinoEmbeddingNet(
        backbone_name=backbone,
        embedding_dim=embedding_dim,
        image_size=image_size,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"  Backbone: {backbone}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  Image size: {image_size}x{image_size}")
    
    return model


@torch.no_grad()
def get_embedding(
    image_path: str,
    model: DinoEmbeddingNet,
    transform,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Compute embedding for an image file.
    
    Compatible interface with deployment_vit_faiss.get_embedding().
    
    Args:
        image_path: Path to image file
        model: Loaded DinoEmbeddingNet model
        transform: Transform to apply (default: DEFAULT_TRANSFORM)
        device: torch.device object
    
    Returns:
        512-dim normalized embedding as numpy array, or None on error
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        image = Image.open(image_path).convert("RGB")
        if transform is None:
            transform = DEFAULT_TRANSFORM
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        embedding = model(img_tensor)[0].cpu().numpy()
        return embedding
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


@torch.no_grad()
def get_embedding_from_array(
    crop_array: np.ndarray,
    model: DinoEmbeddingNet,
    transform,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Compute embedding for a numpy array (e.g., from cv2.imread or YOLO).
    
    Compatible interface with deployment_vit_faiss.get_embedding_from_array().
    Pipeline: BGR → RGB → DINOv2 → embedding (512-dim, L2-normalized).
    
    Args:
        crop_array: numpy array (H, W, C) in BGR format (from cv2/YOLO)
        model: Loaded DinoEmbeddingNet model
        transform: Transform to apply (default: DEFAULT_TRANSFORM)
        device: torch.device object
    
    Returns:
        512-dim normalized embedding as numpy array, or None on error
    """
    try:
        if not isinstance(crop_array, np.ndarray) or crop_array.ndim != 3 or crop_array.shape[2] != 3:
            logger.error(f"Invalid array format: expected (H, W, 3), got {getattr(crop_array, 'shape', type(crop_array))}")
            return None
        
        # Convert BGR to RGB (YOLO/OpenCV always returns BGR)
        import cv2
        crop_array = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)
        
        if transform is None:
            transform = DEFAULT_TRANSFORM
        
        image = Image.fromarray(crop_array)
        img_tensor = transform(image).unsqueeze(0).to(device)
        embedding = model(img_tensor)[0].cpu().numpy()
        return embedding
    except Exception as e:
        logger.error(f"Error processing array: {e}")
        return None


def get_attention_heatmap(
    crop_array: np.ndarray,
    model: DinoEmbeddingNet,
    transform,
    device: torch.device,
    output_path: str,
) -> bool:
    """
    Generates an occlusion sensitivity heatmap for identification.

    The image is covered by small patches one by one, then the embedding is
    recomputed and compared to the original embedding. If hiding a patch makes
    the embedding change a lot, that patch is important for recognition.
    """
    try:
        import cv2
        import matplotlib.pyplot as plt

        if not isinstance(crop_array, np.ndarray) or crop_array.ndim != 3:
            logger.error("Invalid array format for heatmap")
            return False

        orig_h, orig_w = crop_array.shape[:2]
        crop_array_rgb = cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_array_rgb)

        if transform is None:
            transform = DEFAULT_TRANSFORM

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            original_embedding = model(img_tensor)

        occlusion_size = 14
        stride = 14
        base_image = img_tensor.clone()
        occlusion_scores = []
        occlusion_positions = []
        batch_images = []
        batch_positions = []
        batch_size = 32

        def flush_batch(images, positions):
            if not images:
                return
            batch = torch.cat(images, dim=0)
            with torch.no_grad():
                occluded_embeddings = model(batch)
                similarities = F.cosine_similarity(
                    occluded_embeddings,
                    original_embedding.expand_as(occluded_embeddings),
                    dim=1,
                )
                scores = (1.0 - similarities).detach().cpu().numpy()
            occlusion_scores.extend(scores.tolist())
            occlusion_positions.extend(positions)

        for y in range(0, DEFAULT_IMAGE_SIZE, stride):
            for x in range(0, DEFAULT_IMAGE_SIZE, stride):
                occluded = base_image.clone()
                y_end = min(y + occlusion_size, DEFAULT_IMAGE_SIZE)
                x_end = min(x + occlusion_size, DEFAULT_IMAGE_SIZE)
                occluded[:, :, y:y_end, x:x_end] = 0.0
                batch_images.append(occluded)
                batch_positions.append((x, y, x_end, y_end))

                if len(batch_images) >= batch_size:
                    flush_batch(batch_images, batch_positions)
                    batch_images = []
                    batch_positions = []

        flush_batch(batch_images, batch_positions)

        if not occlusion_scores:
            logger.warning("No occlusion scores computed")
            return False

        heatmap_224 = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.float32)
        coverage = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.float32)

        for score, (x, y, x_end, y_end) in zip(occlusion_scores, occlusion_positions):
            heatmap_224[y:y_end, x:x_end] += score
            coverage[y:y_end, x:x_end] += 1.0

        heatmap_224 = np.divide(
            heatmap_224,
            np.maximum(coverage, 1.0),
            out=np.zeros_like(heatmap_224),
            where=coverage > 0,
        )
        heatmap_224 = (heatmap_224 - heatmap_224.min()) / (heatmap_224.max() - heatmap_224.min() + 1e-8)
        heatmap_224 = np.power(heatmap_224, 0.85)

        heatmap = cv2.resize(heatmap_224, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        # Save only the final overlay near the crop (no multi-panel collage).
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(crop_array_rgb, alpha=0.82)
        ax.imshow(heatmap, cmap='plasma', alpha=0.38, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title("Where Model Focuses", fontsize=12, fontweight='bold')
        ax.axis('off')

        plt.tight_layout(pad=0.1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Occlusion sensitivity heatmap saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}", exc_info=True)
        return False


def search_vectors(
    query_embedding: np.ndarray,
    index_embeddings: np.ndarray,
    top_k: int = 5,
) -> tuple:
    """
    Simple cosine similarity search (for when FAISS is not available).
    
    Args:
        query_embedding: 512-dim query embedding
        index_embeddings: N x 512 matrix of indexed embeddings
        top_k: Number of top results to return
    
    Returns:
        (indices, distances) — top_k indices and distances
    """
    # Cosine similarity = dot product (since embeddings are L2 normalized)
    similarities = query_embedding @ index_embeddings.T
    top_indices = np.argsort(-similarities)[:top_k]
    top_distances = -similarities[top_indices]
    return top_indices, top_distances


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(checkpoint_path: str = "models/dinov2_scratch_10e/best.pt") -> Dict[str, Any]:
    """Load and display model metadata without loading weights."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    
    return {
        "checkpoint": str(checkpoint_path),
        "backbone": metadata.get("args", {}).get("backbone", "unknown"),
        "embedding_dim": metadata.get("args", {}).get("embedding_dim", "unknown"),
        "image_size": metadata.get("args", {}).get("image_size", "unknown"),
        "epoch": metadata.get("epoch", "unknown"),
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model, metadata = load_model("models/dinov2_scratch_10e/best.pt")
    
    # Load a test image
    test_image_path = Path("unknown/1/01-1.jpg")
    if test_image_path.exists():
        test_image = Image.open(test_image_path).convert("RGB")
        embedding = get_embedding(model, test_image)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
