from services.identification_service import IdentificationService
from services.card_service import CardService
from api.error import APIError
from api.models import FileData
from api.services.temp import TempStorage

from utils import sanitize_filename

from typing import Any, Dict


def add_new_card(
    file_data: FileData,
    species: str,
    project_id: str | None,
    template_type: str,
    card_id: str | None,
    params: Dict[str, str],
    id_service: IdentificationService,
    temp: TempStorage
) -> Dict[str, Any]:
    card_service = id_service.card_service

    if card_id and card_service.get_prototype(card_id):
        raise APIError(msg=f"card_id {card_id} already taken", status=409)

    if project_id and not project_id.isnumeric():
        raise APIError(msg="project_id must be an integer", status=400)

    file_name = sanitize_filename(file_data.name)

    path = temp.write_temp_file(
        path=temp.make_temp_file_name(
            begin_with=file_name,
            end_with=file_data.ext
        ),
        data=file_data.data
    ),

    crop_output = temp.make_temp_file_name(
        begin_with=f".{file_name}",
        end_with=".CROP"
    )
"""
    try:
        id_service.get_crop_and_embedding(
            image_path=str(path),
            output_file=str(crop_output),
            crop_name=file_name,
            debug=False
        )


        id = card_service.save_new_individual(
            photo_path_cropped=str(crop_output),
            species=species,
            project_id=int(project_id) if project_id else None,
            template_type=template_type,
            card_id=card_id,
            card_data=params
        )

        return {"card_id": id}
    except Exception as ex:
        raise APIError(msg=str(ex), status=500)
"""