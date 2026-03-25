# 📋 ЗАДАНИЕ ДЛЯ КОМАНДЫ: Миграция датасета и FAISS

## 🎯 Цель
Перенести существующие фото тритонов из папок в базу данных и построить векторный индекс для поиска.

---

## 1️⃣ ЧТО УЖЕ ГОТОВО (не трогать)

### 📁 Структура проекта
```
Tritonchikeee/
├── database/
│   ├── card_database.py      # ✅ Схема БД (2 таблицы)
│   └── migrate_dataset.py    # ⏳ Нужно написать (ваша задача)
├── pipeline/
│   ├── save_new.py           # ✅ Функции сохранения в БД
│   └── analyse.py            # ✅ Обработка фото (YOLO + ViT)
├── data/
│   ├── dataset_crop/dataset_crop_24/
│   │   ├── karelin/          # ✅ Исходные данные (Карелина)
│   │   │   ├── 1/, 2/, 3/... # Папки с особями (по 3-5 фото каждая)
│   │   └── ribbed/           # ✅ Исходные данные (Гребенчатый)
│   │       ├── 1/, 2/, 3/...
│   ├── photos/full/          # ✅ Для новых фото (полные)
│   └── output/               # ✅ Результаты обработки
└── models/
    ├── best_seg.pt           # ✅ YOLO сегментация
    └── best_id.pt            # ✅ ViT идентификация
```

### 🗄 База данных (`cards.db`)

**Таблица 1: `individuals`** (паспорт особи)
| Поле | Тип | Описание |
|------|-----|----------|
| `individual_id` | TEXT | PRIMARY KEY (например, `NT-K-1`) |
| `template_type` | TEXT | ИК-1, ИК-2, КВ-1, КВ-2 |
| `species` | TEXT | "Карелина" / "Гребенчатый" |
| `project_name` | TEXT | Название проекта |
| `photo_path` | TEXT | Путь к основному фото |
| `photo_number` | TEXT | Номер фото (01, 02, 03...) |
| `embedding_index` | INTEGER | Ссылка на FAISS (-1 если нет) |
| `created_at` | TIMESTAMP | Дата создания записи |
| ... | ... | + поля биометрии, родители, встречи |

**Таблица 2: `photos`** (все фотографии)
| Поле | Тип | Описание |
|------|-----|----------|
| `photo_id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `individual_id` | TEXT | FOREIGN KEY → individuals |
| `photo_type` | TEXT | 'full' / 'cropped' |
| `photo_number` | TEXT | 01, 02, 03... |
| `photo_path` | TEXT | Путь к файлу |
| `date_taken` | TEXT | Дата съёмки |
| `is_main` | BOOLEAN | Основное фото (1/0) |
| `is_legacy` | BOOLEAN | Из старого датасета (1/0) |
| `is_processed` | BOOLEAN | Обработано ViT (1/0) |
| `embedding_index` | INTEGER | **Позиция в FAISS** (-1 если нет) |

### 💾 Готовые функции (`pipeline/save_new.py`)

```python
save_new_individual(
    embedding,                    # Вектор от ViT
    photo_path_full,              # Полное фото
    photo_path_cropped,           # Кроп брюшка
    species, project_name,
    template_type, individual_id,
    is_legacy=False,              # ⚠️ Важно для миграции!
    **card_data
)

add_encounter(
    individual_id,                # ID существующей особи
    template_type,                # КВ-1 или КВ-2
    photo_path_full,
    photo_path_cropped,
    **card_data
)

update_individual(individual_id, **kwargs)  # Редактирование
get_individual_photos(individual_id)        # Получить все фото
```

---

## 2️⃣ ЧТО НУЖНО СДЕЛАТЬ (ваши задачи)

### 🔹 ЗАДАЧА 1: Скрипт миграции (`database/migrate_dataset.py`)

**Что делает:**
1. Сканирует папки `data/dataset_crop/dataset_crop_24/karelin/` и `ribbed/`
2. Для каждой папки особи создаёт запись в `individuals`
3. Для каждого фото создаёт запись в `photos`
4. Помечает все записи как `is_legacy = 1`
5. Ставит `embedding_index = -1` (заглушка)

**Требования:**
| Параметр | Значение |
|----------|----------|
| `individual_id` | `NT-K-{номер}` для karelin, `NT-R-{номер}` для ribbed |
| `species` | "Карелина" / "Гребенчатый" |
| `project_name` | "Миграция_Датасет_2024" |
| `template_type` | "ИК-1" (по умолчанию) |
| `photo_type` | "cropped" (только кропы!) |
| `photo_number` | "01", "02", "03"... (по порядку файлов) |
| `is_legacy` | 1 (все фото из датасета) |
| `is_main` | 1 для первого фото, 0 для остальных |
| `embedding_index` | -1 (будет обновлено позже) |
| `date_taken` | EXIF → дата файла → NULL |

**Псевдокод:**
```python
for species_folder in ["karelin", "ribbed"]:
    species_name = "Карелина" if species_folder == "karelin" else "Гребенчатый"
    species_prefix = "K" if species_folder == "karelin" else "R"
    
    for individual_folder in species_folder.iterdir():
        individual_id = f"NT-{species_prefix}-{individual_folder.name}"
        
        # Пропустить если уже в БД
        if exists_in_db(individual_id):
            continue
        
        # Создать запись в individuals
        create_individual(individual_id, species_name, "Миграция_Датасет_2024")
        
        # Обработать все фото
        photos = sorted(individual_folder.glob("*.jpg"))
        for idx, photo_path in enumerate(photos, 1):
            create_photo(
                individual_id=individual_id,
                photo_type="cropped",
                photo_number=f"{idx:02d}",
                photo_path=str(photo_path),
                is_main=(idx == 1),
                is_legacy=1,
                embedding_index=-1
            )
```

---

### 🔹 ЗАДАЧА 2: Построение FAISS индекса (`database/build_faiss_index.py`)

**Что делает:**
1. Загружает все кропы из `photos` где `is_legacy = 1` и `embedding_index = -1`
2. Прогоняет каждое фото через ViT модель (`models/best_id.pt`)
3. Добавляет вектор в FAISS индекс
4. Обновляет `embedding_index` в БД с реальным номером позиции

**Требования:**
| Параметр | Значение |
|----------|----------|
| Модель | `models/best_id.pt` (ViT) |
| Трансформы | Resize(224) → ToTensor → Normalize |
| FAISS индекс | `data/embeddings/database_embeddings.pkl` |
| Batch size | 32 (для ускорения) |
| Progress bar | Показывать прогресс (tqdm) |

**Псевдокод:**
```python
# 1. Загрузить модель ViT
model = load_vit_model("models/best_id.pt")
model.eval()

# 2. Получить все необработанные кропы из БД
photos = get_photos_where(embedding_index=-1, is_legacy=1)

# 3. Загрузить или создать FAISS индекс
if exists("data/embeddings/database_embeddings.pkl"):
    faiss_index = faiss.read_index("data/embeddings/database_embeddings.pkl")
else:
    faiss_index = faiss.IndexFlatIP(512)  # 512 = размер эмбеддинга ViT

# 4. Обработать фото батчами
for batch in batches(photos, size=32):
    embeddings = model(batch)  # Получить векторы
    start_index = faiss_index.ntotal
    faiss_index.add(embeddings)  # Добавить в индекс
    
    # Обновить БД
    for i, photo in enumerate(batch):
        embedding_index = start_index + i
        update_photo_embedding(photo.photo_id, embedding_index)

# 5. Сохранить индекс
faiss.write_index(faiss_index, "data/embeddings/database_embeddings.pkl")
```

---

### 🔹 ЗАДАЧА 3: Интеграция с `save_new.py` (опционально)

**Что делает:**
Добавляет автоматическое обновление `embedding_index` при сохранении новой особи через бота.

**Изменения в `save_new_individual()`:**
```python
# После сохранения в БД:
if embedding is not None and photo_path_cropped:
    faiss_index = faiss.read_index("data/embeddings/database_embeddings.pkl")
    embedding_index = faiss_index.add(embedding)
    faiss.write_index(faiss_index, "data/embeddings/database_embeddings.pkl")
    
    # Обновить БД
    cursor.execute(
        "UPDATE photos SET embedding_index = ? WHERE photo_path = ?",
        (embedding_index, photo_path_cropped)
    )
    conn.commit()
```

---

## 3️⃣ СВЯЗЬ МЕЖДУ КОМПОНЕНТАМИ

```
┌─────────────────────────────────────────────────────────────┐
│                    SQLite (cards.db)                        │
│  individuals: NT-K-1, NT-R-47, ... (метаданные)            │
│  photos: photo_path, embedding_index (связь с FAISS)       │
└─────────────────────────────────────────────────────────────┘
                          ↕ связка по embedding_index
┌─────────────────────────────────────────────────────────────┐
│                    FAISS (index.pkl)                        │
│  Позиция 0: вектор для NT-K-1 фото 01                      │
│  Позиция 1: вектор для NT-K-1 фото 02                      │
│  Позиция 2: вектор для NT-K-2 фото 01                      │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Важно:** 
- `embedding_index = -1` → фото есть, но вектора нет (ошибка/не завершено)
- `embedding_index = 0, 1, 2...` → вектор добавлен в FAISS на позиции N

---

## 4️⃣ ПРОВЕРКА РЕЗУЛЬТАТА

После выполнения задач проверить:

```sql
-- 1. Сколько особей мигрировано
SELECT species, COUNT(*) FROM individuals 
WHERE project_name = 'Миграция_Датасет_2024'
GROUP BY species;

-- 2. Сколько фото мигрировано
SELECT photo_type, is_legacy, COUNT(*) FROM photos 
GROUP BY photo_type, is_legacy;

-- 3. Сколько фото без эмбеддинга (должно быть 0 после FAISS)
SELECT COUNT(*) FROM photos 
WHERE embedding_index = -1 AND is_legacy = 1;

-- 4. Проверка связи
SELECT i.individual_id, p.photo_path, p.embedding_index
FROM individuals i
JOIN photos p ON i.individual_id = p.individual_id
WHERE p.embedding_index != -1
LIMIT 10;
```

---

## 5️⃣ СРОКИ И ПРИОРИТЕТЫ

| Задача | Приоритет | Срок |
|--------|-----------|------|
| Миграция датасета | 🔴 Высокий | 1-2 дня |
| Построение FAISS | 🔴 Высокий | 2-3 дня |
| Интеграция с save_new | 🟡 Средний | 1 день |
| Тестирование поиска | 🟡 Средний | 1 день |

---

## 6️⃣ КОНТАКТЫ И ВОПРОСЫ

**Готовый код для начала:**
- `database/card_database.py` — схема БД
- `pipeline/save_new.py` — функции сохранения

**Вопросы решать через:** [ваш контакт]

**Документация:**
- ТЗ: `ТЗ тритоны от биологов (весен сем 2025).docx`
- Отчёт: `Тритоны_УрФУ_OneOme.pdf`

---

## ✅ ЧЕК-ЛИСТ ГОТОВНОСТИ

- [ ] Миграция: все папки `karelin/` и `ribbed/` обработаны
- [ ] Миграция: все записи помечены `is_legacy = 1`
- [ ] FAISS: все кропы прогнаны через ViT
- [ ] FAISS: все `embedding_index` обновлены (нет `-1`)
- [ ] FAISS: индекс сохранён в `data/embeddings/database_embeddings.pkl`
- [ ] Тест: поиск находит мигрированных особей
- [ ] Тест: новые особи через бота добавляются в FAISS автоматически

---

**Удачи! 🦎**