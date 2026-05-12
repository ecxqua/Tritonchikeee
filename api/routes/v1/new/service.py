from services.identification_service import IdentificationService
from api.error import APIError
from api.models import FileData
from api.services.temp import TempStorage

from utils import sanitize_filename

from typing import Any, Dict, List


def add_new_card(
    file_data: List[FileData],
    species: str,
    project_id: int | None,
    template_type: str,
    card_id: str | None,
    params: Dict[str, str],
    id_service: IdentificationService,
    temp: TempStorage
) -> Dict[str, Any]:
    card_service = id_service.card_service

    if card_id and card_service.get_prototype_by_card_id(card_id):
        raise APIError(msg=f"card_id {card_id} already taken", status=409)

    if not file_data:
        raise APIError(msg="No photos were provided", status=400)

    first_photo = file_data[0]
    file_data.pop(0)

    file_name = sanitize_filename(first_photo.name)

    path = temp.write_temp_file(
        path=temp.make_temp_file_name(
            begin_with=file_name,
            end_with=first_photo.ext
        ),
        data=first_photo.data
    )

    try:
        mapped_params = {k: v for k, v in {
            "date": params.get("dateFilled", None),
            "length_body": params.get("bodyLength", None),
            "length_tail": params.get("tailLength", None),
            "weight": params.get("weight", None),
            "sex": params.get("sex", None),
            "birth_year_exact": params.get("exactBirthDate", None),
            "birth_year_approx": params.get("estimatedBirthDate", None),
            "photo_number": params.get("photoNumber", None),
            "origin_region": params.get("regionOfOrigin", None),
            "length_device": params.get("measurementDevice", None),
            "weight_device": params.get("scaleBrand", None),
            "notes": params.get("notes", None),
            "release_date": params.get("releaseDate", None),
            "parent_male_id": params.get("fatherId", None),
            "parent_female_id": params.get("motherId", None),
            "length_total": params.get("totalLength", None),
            "water_body_name": params.get("waterBodyName", None),
            "encounterDate": None,  # not tracked rn
            "meeting_time": params.get("encounterTime", None),
            "photo_id": params.get("bellyPhotoNumber", None),
            "status": params.get("status", None),
            "water_body_number": params.get("waterBodyNumber", None),
        }.items() if v}

        result = id_service.add_new_individual(
            species=mapped_params.pop("species", species),
            project_id=mapped_params.pop("project_id", project_id),
            template_type=mapped_params.pop("template_type", template_type),
            image_path=str(path),
            **mapped_params
        )

        if result["error"] is not None:
            raise APIError(status=500, msg=result["error"])

        card = result["card_id"]

        if file_data:
            for photo in file_data:
                file_name = sanitize_filename(photo.name)

                photo_path = temp.write_temp_file(
                    path=temp.make_temp_file_name(
                        begin_with=file_name,
                        end_with=photo.ext
                    ),
                    data=photo.data
                )

                id_service.add_photo_to_card(card, str(photo_path))

        return {"id": card}
    except Exception as ex:
        print(f"{type(ex)}: {ex}")
        raise APIError(status=500, msg=str(ex))

