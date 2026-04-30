import shutil
from pathlib import Path

def clear_directory(dir_path: str | Path) -> bool:
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return True
    
    try:
        # Самый надёжный способ: удалить папку целиком и создать заново
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"⛔ Ошибка при очистке директории {path}: {e}")
        return False

def delete_file(file_path: str | Path) -> bool:
    path = Path(file_path)
    if not path.exists():
        return False  # Файл уже отсутствует
    try:
        path.unlink(missing_ok=True)  # missing_ok доступен с Python 3.8
        return True
    except PermissionError as e:
        # Часто возникает, если файл открыт другой программой (например, SQLite)
        print(f"⛔ Не удалось удалить {path}: {e}")
        return False