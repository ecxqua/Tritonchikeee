from services.identification_service import IdentificationService
from api.services.temp import TempStorage
from api.error import APIError
from api.models.file_data import FileData
from utils import sanitize_filename

from typing import Any, Dict


_allowed_scopes = {"all", "by_species", "by_territory"}


def _build_match(
    match: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "newtId": match["prototype_id"],
        "confidence": match["similarity_percent"],
        "photoUrl": "..."
    }


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
            project_id=1,
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
                _build_match(match)
                for match in res["candidates"]
            ]
        }
    except ValueError as ex:
        # raise APIError(status=400, msg=str(ex))
        return { "status": "not_found" }
