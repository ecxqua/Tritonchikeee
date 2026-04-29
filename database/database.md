# Базы данных

Существует две базы: `cards.sqlite3` и `database_embeddings.pkl` (faiss-index). В первой в таблице `cards` хранятся паспорта индивидуальных особей тритонов, в таблице `photos` карточки к каждой фотографии в базе и `embedding_index`, ссылающийся на индекс эмбеддинга в faiss-index, в таблице `uploads` хранятся незавершённые сохранения в бд, в `projects` - проекты (деление карточек на проекты). `cards` -> `photos` (1 ко многим, одна особь ко многим фото). В faiss-index хранятся только эмбеддинги для каждого фото.

Схема sqlite-бд описана в cards_database.py. В `services/card_service.py` описаны взаимодействия с бд, в `services/embedding_service.py` - взаимодействия с FAISS-индексом.