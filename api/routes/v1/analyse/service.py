from services.identification_service import IdentificationService
from api.services.temp import TempStorage
from api.models.file_data import FileData
from utils.json_utils import make_json_safe

from typing import Any, Dict
import re


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def complete_analyse(
    file_data: FileData,
    id_service: IdentificationService,
    temp: TempStorage
) -> Dict[str, Any]:
    path = temp.write_temp_file(
        path=temp.make_temp_file_name(
            begin_with=sanitize_filename(file_data.name),
            end_with=file_data.ext
        ),
        data=file_data.data
    )

    return make_json_safe(id_service.identify_and_prepare(
        image_path=path,
        project_id=1,
        top_k=5,
        debug=True
    ))