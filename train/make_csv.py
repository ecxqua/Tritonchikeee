import os
from pathlib import Path
import csv

def create_labels(root_dir, output_csv='labels.csv'):
    root_path = Path(root_dir)
    samples = []

    # Проходим по папкам классов
    class_dirs = sorted([d for d in root_path.glob("*") if d.is_dir()])
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    individual_to_idx = {}
    individual_counter = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]

        # Проходим по папкам особей внутри класса
        individual_dirs = sorted([d for d in class_dir.glob("*") if d.is_dir()])

        for individual_dir in individual_dirs:
            individual_name = individual_dir.name
            unique_id = f"{class_name}_{individual_name}"

            if unique_id not in individual_to_idx:
                individual_to_idx[unique_id] = individual_counter
                individual_counter += 1

            individual_idx = individual_to_idx[unique_id]

            # Проходим по изображениям внутри особи
            for img_path in individual_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append({
                        "path": str(img_path),
                        "class_id": class_idx,
                        "individual_id": individual_idx,
                        "class_name": class_name,
                        "individual_name": individual_name
                    })

    # Записываем в CSV файл
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_id", "individual_id", "class_name", "individual_name"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)

    print(f"Создано {len(samples)} меток и сохранено в {output_csv}")
    print(f"Классы ({len(class_to_idx)}): {list(class_to_idx.keys())}")
    print(f"Особи ({len(individual_to_idx)}): {list(individual_to_idx.keys())[:10]} ...")  # первые 10

# Использование:
create_labels("C:/Users/User/Documents/Triton_dataset_new/crop_dataset/val", output_csv='C:/Users/User/Documents/Triton_dataset_new/crop_dataset/labels_val.csv')