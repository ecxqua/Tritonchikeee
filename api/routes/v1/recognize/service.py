from services.identification_service import IdentificationService
from api.services.temp import TempStorage
from api.error import APIError
from api.models.file_data import FileData
from utils import sanitize_filename

from pathlib import Path
from typing import Any, Dict

import base64
import mimetypes


_allowed_scopes = {"all", "by_species", "by_territory"}


def _build_match(
    match: Dict[str, Any],
    id_service: IdentificationService,
) -> Dict[str, Any]:
    id: str = match["prototype_id"]
    similarity: float = match["similarity_percent"]

    result: dict[str, Any] = {
        "newtId": id,
        "confidence": similarity,
        "photoUrl": "unknown"
    }

    photos = id_service.card_service.get_prototype_photos(id)
    if photos:
        path = photos[0]["photo_path"]
        photo_base64 = None

        if path and isinstance(path, str):
            file_path = Path(path)

            if file_path.exists():
                mime_type, _ = mimetypes.guess_type(file_path)
                mime_type = mime_type or "image/jpeg"

                with open(file_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                photo_base64 = f"data:{mime_type};base64,{encoded}"
                result["photoUrl"] = photo_base64

    return result


def complete_recognize(
    file_data: FileData,
    scope: str,
    project_id: int | None,
    id_service: IdentificationService,
    temp: TempStorage
) -> Dict[str, Any]:
    if scope not in _allowed_scopes:
        raise APIError(status=400, msg=f"Incorrect scope {scope}")
    
    if project_id is not None and \
        id_service.project_service.get_project_by_id(project_id) is None:
        raise APIError(status=400, msg=f"Unknown project ID {project_id}")

    path = temp.write_temp_file(
        path=temp.make_temp_file_name(
            begin_with=sanitize_filename(file_data.name),
            end_with=file_data.ext
        ),
        data=file_data.data
    )

    # add filters later
    # with scope and projectId

    try:
        res = id_service.identify_and_prepare(
            image_path=str(path),
            project_ids=[1],
            top_k=5,
            debug=True
        )

        error = res["error"]
        if error is not None:
            return { "status": "not_found" }
            # raise APIError(status=500, msg=error)

        return {
            "status": "found",
            "matches": [
                _build_match(match, id_service)
                for match in res["candidates"]
            ]
        }
    except ValueError as ex:
        # raise APIError(status=400, msg=str(ex))
        return { "status": "not_found" }
