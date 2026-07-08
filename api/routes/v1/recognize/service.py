from services.identification_service import IdentificationService
from api.services.temp import TempStorage
from api.error import APIError
from api.models.file_data import FileData
from utils import sanitize_filename

from pathlib import Path
from typing import Any, Dict

import base64
import mimetypes

HEATMAP = False
_allowed_scopes = {"all", "by_species", "by_territory"}


def _build_match(
    match: Dict[str, Any],
    id_service: IdentificationService,
) -> Dict[str, Any]:
    id: str = match["card_id"]
    similarity: float = match["similarity_percent"]

    result: dict[str, Any] = {
        "newtId": id,
        "confidence": similarity,
        "photoUrl": "unknown"
    }

    photos = id_service.card_service.get_card_photos(id)
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
    scope: str | None,
    project_id: int | None,
    top_k: int | None,
    id_service: IdentificationService,
    temp: TempStorage
) -> Dict[str, Any]:
    if scope is not None and scope not in _allowed_scopes:
        raise APIError(status=400, msg=f"Incorrect scope {scope}")

    territories = None
    species = None

    if project_id is not None:
        project = id_service.project_service.get_project_by_id(project_id)
        if not project:
            raise APIError(status=404, msg=f"Unknown project {project_id}")

        territories = project.get('territories_filter', None)
        species = project.get('species_filter', None)

    path = temp.write_temp_file(
        path=temp.make_temp_file_name(
            begin_with=sanitize_filename(file_data.name),
            end_with=file_data.ext
        ),
        data=file_data.data
    )

    k = top_k if top_k is not None else 5
    try:
        k = int(k)
    except (ValueError, TypeError):
        k = 5
    
    k = max(1, min(k, 100))

    try:
        res = id_service.identify_and_prepare(
            image_path=str(path),
            project_ids=[project_id] if project_id else None,
            territory=territories,
            species=species,
            top_k=k,
            debug=True,
            heatmap=HEATMAP
        )

        error = res["error"]
        if error is not None:
            return {"status": "not_found"}
            # raise APIError(status=500, msg=error)

        heatmap_b64 = None
        if HEATMAP:
            with open(res["heatmap_path"], 'rb') as heatmap_file:
                heatmap_b64 = base64.b64encode(heatmap_file.read()).decode("utf-8")
                heatmap_b64 = f"data:image/png;base64,{heatmap_b64}"

        return {
            "status": "found",
            "matches": [
                _build_match(match, id_service)
                for match in res["candidates"]
            ],
            "heatmap": heatmap_b64
        }
    except ValueError:
        # raise APIError(status=400, msg=str(ex))
        return {"status": "not_found"}
