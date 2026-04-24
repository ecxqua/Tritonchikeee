from pydantic import BaseModel


class FileData(BaseModel):
    name: str
    ext: str
    data: bytes
