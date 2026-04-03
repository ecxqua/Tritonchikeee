import uuid
import shutil
from pathlib import Path
from typing import Optional


class TempStorage:
    TEMP_DIR: Path


    def __init__(self):
        self.TEMP_DIR = Path("data") / "_temp_input"
        self.TEMP_DIR.mkdir(exist_ok=True)
    
    def make_temp_file_name(self, begin_with: Optional[str], end_with: Optional[str]) -> Path:
        begin_with = begin_with or ""
        end_with = end_with or ""
        return self.TEMP_DIR / f"{begin_with}{uuid.uuid4().hex}{end_with}"
    
    def write_temp_file(self, path: Path, data: bytes):
        path.write_bytes(data)
        return path
    
    def cleanup(self):
        if self.TEMP_DIR.exists():
            shutil.rmtree(self.TEMP_DIR)
            self.TEMP_DIR.mkdir(exist_ok=True)