from pipeline.deployment_vit_faiss import load_model, get_embedding, search_with_faiss, load_faiss_index

# 1. Модель загружается
model = load_model("models/best_id.pt", torch.device("cpu"))

# 2. Эмбеддинг вычисляется
embedding = get_embedding("data/dataset_crop/test.jpg", model, DEFAULT_TRANSFORM, torch.device("cpu"))

# 3. Поиск работает (но прототипы передаются ИЗВНЕ)
results = search_with_faiss(
    query_embedding=embedding,
    prototype_embeddings=np.random.rand(100, 512),  # Из services
    individual_ids=["NT-K-1", "NT-K-2", ...],       # Из services
    metadata=[{...}, {...}, ...],                   # Из services
    top_k=5
)

# 4. Нет доступа к БД (можно тестировать без базы)