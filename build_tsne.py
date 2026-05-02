#!/usr/bin/env python3
"""Build t-SNE from embeddings recomputed with current model from config."""

import logging
import os
import sqlite3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from pipeline.deployment_dinov2_faiss import (
    DEFAULT_TRANSFORM,
    get_embedding_from_array,
    load_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_path = config["id-model"]["path"]
db_path = config["db"]["db_path"]

logger.info("Model from config: %s", model_path)
logger.info("Database: %s", db_path)

device = torch.device("cpu")
model = load_model(model_path, device)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute(
    """
    SELECT p.photo_id, p.photo_path, c.species, c.card_id
    FROM photos p
    JOIN cards c ON p.card_id = c.card_id
    WHERE p.photo_type = 'cropped'
    ORDER BY p.photo_id
    """
)
rows = cursor.fetchall()
conn.close()

logger.info("Cropped photos found: %d", len(rows))

# Process entire dataset (no sampling) — may take long on CPU.
# If it's too slow, re-run locally with a reduced sample size.

embeddings = []
species_values = []

for i, (_, photo_path, species, _) in enumerate(rows, start=1):
    if i % 50 == 0:
        logger.info("Embedded %d / %d", i, len(rows))

    image = cv2.imread(photo_path)
    if image is None:
        continue

    emb = get_embedding_from_array(image, model, DEFAULT_TRANSFORM, device)
    if emb is None:
        continue

    embeddings.append(emb)
    species_values.append(species)

embeddings = np.asarray(embeddings, dtype=np.float32)
logger.info("Embeddings computed: %d", len(embeddings))

if len(embeddings) < 10:
    raise RuntimeError("Too few embeddings for t-SNE")

# Optional PCA pre-reduction to speed up t-SNE and reduce memory use.
if embeddings.shape[1] > 50:
    logger.info("Applying PCA reduction to 50 dims before t-SNE")
    pca = PCA(n_components=50, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
else:
    embeddings_reduced = embeddings

colors_map = {"Карелина": 0, "Гребенчатый": 1, "unknown": 2}
species_ids = np.array([colors_map.get(s, 2) for s in species_values], dtype=np.int32)

perplexity = min(30, max(5, len(embeddings) // 20))
logger.info("Running t-SNE (perplexity=%d)...", perplexity)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000, verbose=1)
embeddings_2d = tsne.fit_transform(embeddings_reduced)

fig, ax = plt.subplots(figsize=(14, 10))
palette = {0: "red", 1: "blue", 2: "gray"}
labels_map = {0: "Карелина", 1: "Гребенчатый", 2: "unknown"}

for sid in [0, 1, 2]:
    mask = species_ids == sid
    if mask.sum() == 0:
        continue
    ax.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=palette[sid],
        label=f"{labels_map[sid]} (n={int(mask.sum())})",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.4,
    )

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title(f"t-SNE from config model: {Path(model_path).as_posix()}")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/tsne_visualization_from_config_model.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

logger.info("Saved: %s", output_path)
print("\nDone")
print(output_path)
