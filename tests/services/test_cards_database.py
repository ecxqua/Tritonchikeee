from database.cards_database import init_database
import sqlite3

# 1. Инициализация (создаст таблицу uploads если нет)
init_database()

# 2. Проверка структуры
conn = sqlite3.connect("database/cards.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Таблицы: {tables}")
# Ожидаем: ['cards', 'photos', 'uploads']

# 3. Проверка индексов uploads
cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='uploads'")
indexes = [row[0] for row in cursor.fetchall()]
print(f"Индексы uploads: {indexes}")
# Ожидаем: ['idx_uploads_status_expires', 'idx_uploads_project', 'idx_uploads_status']

conn.close()