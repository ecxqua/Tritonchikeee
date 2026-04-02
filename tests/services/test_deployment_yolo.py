# 1. In-memory pipeline (быстро, без диска)
result = await process_single_image(
    img_path="data/input/triton.jpg",
    output_dir=None,  # Не сохраняем
    debug=False,
    return_array=True
)

if result['success']:
    crop_array = result['crop_array']  # ← numpy array в памяти
    embedding = get_embedding_from_array(crop_array, model, transform, device)

# 2. С сохранением для архива
result = await process_single_image(
    img_path="data/input/triton.jpg",
    output_dir="data/archive/NT-K-1/",
    debug=True,  # Сохраняем маску для отладки
    return_array=True
)

# 3. Синхронная версия (для тестов)
result = process_single_image_sync(
    img_path="data/input/triton.jpg",
    output_dir="output/",
    debug=False
)