#!/bin/bash

python -m database.card_database
python -m database.migrate_dataset
python -m database.build_faiss_index

echo "Запуск баз данных окончен"