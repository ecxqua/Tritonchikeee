from services.identification_service import IdentificationService
from api.error import APIError

from typing import Any, Dict, List

import base64
import mimetypes
from pathlib import Path


def get_newt_by_id(
    id: str,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    card = id_service.card_service.get_card(id)
    if not card:
        raise APIError(status=404, msg=f"No card by ID {id}")

    return {
        "id": card.get("card_id", None),
        "projectId": card.get("project_id", None),
        "cardType": card.get("template_type", None),
        "createdAt": card.get("created_at", None),
        "sex": card.get("sex", None),
        "status": card.get("status", None),
    }


def get_cards_by_newt_id(
    id: str,
    id_service: IdentificationService,
) -> List[Dict[str, Any]]:
    card0 = id_service.card_service.get_card(id)
    if not card0:
        raise APIError(status=404, msg=f"No card by ID {id}")

    result: List[Dict[str, Any]] = []

    for card in [card0]:  # easier legacy comp
        photo_objs = id_service.card_service.get_card_photos(card["card_id"])

        photos: List[str] = []
        for obj in photo_objs:
            path = obj["photo_path"]
            photo_base64 = None

            if path and isinstance(path, str):
                file_path = Path(path)

                if file_path.exists():
                    mime_type, _ = mimetypes.guess_type(file_path)
                    mime_type = mime_type or "image/jpeg"

                    with open(file_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")

                    photo_base64 = f"data:{mime_type};base64,{encoded}"
                    photos.append(photo_base64)

        first_photo: Dict[str, Any] = {}
        if photo_objs:
            first_photo = photo_objs[0]

        result.append({
            "cardType": card["template_type"],
            "data": {k: v for k, v in {
		        "species": card.get("species", None),
                "dateFilled": card.get("date", None),
                "bodyLength": card.get("length_body", None),
                "tailLength": card.get("length_tail", None),
                "weight": card.get("weight", None),
                "sex": card.get("sex", None),
                "exactBirthDate": card.get("birth_year_exact", None),
                "estimatedBirthDate": card.get("birth_year_approx", None),
                "photoNumber": first_photo.get("photo_number", None),
                "regionOfOrigin": card.get("origin_region", None),
                "measurementDevice": card.get("length_device", None),
                "scaleBrand": card.get("weight_device", None),
                "notes": card.get("notes", None),
                "releaseDate": card.get("release_date", None),
                "fatherId": card.get("parent_male_id", None),
                "motherId": card.get("parent_female_id", None),
                "totalLength": card.get("length_total", None),
                "waterBodyName": card.get("water_body_name", None),
                "encounterDate": "",  # not tracked rn
                "encounterTime": card.get("meeting_time", None),
                "bellyPhotoNumber": first_photo.get("photo_id", None),
                "status": card.get("status", None),
                "waterBodyNumber": card.get("water_body_number", None),
            }.items() if v is not None},
            "photos": photos,
        })

    return result


def patch_card_by_newt_id(
    id: str,
    params: Dict[str, Any],
    id_service: IdentificationService,
) -> Dict[str, Any]:
    card = id_service.card_service.get_card(id)
    if not card:
        raise APIError(status=404, msg=f"No card by ID {id}")

    template_type = params["cardType"]
    submission_id = id # f"{id}-{template_type.replace('-', '')}"

    filtered_params = {
	    key: value
	    for key, value in params.items()
	    if key not in card or card[key] != value or not value
    }

    for key in [  # cleanup & extract later
	    "cardType", "photoNumber"
    ]:
        filtered_params.pop(key)

    new_params = {
	    k: v for k, v in {
            "date": filtered_params.get("dateFilled", None),
            "length_body": filtered_params.get("bodyLength", None),
            "length_tail": filtered_params.get("tailLength", None),
            "weight": filtered_params.get("weight", None),
            "sex": filtered_params.get("sex", None),
            "birth_year_exact": filtered_params.get("exactBirthDate", None),
            "birth_year_approx": filtered_params.get("estimatedBirthDate", None),
            "origin_region": filtered_params.get("regionOfOrigin", None),
            "length_device": filtered_params.get("measurementDevice", None),
            "weight_device": filtered_params.get("scaleBrand", None),
            "notes": filtered_params.get("notes", None),
            "release_date": filtered_params.get("releaseDate", None),
            "parent_male_id": filtered_params.get("fatherId", None),
            "parent_female_id": filtered_params.get("motherId", None),
            "length_total": filtered_params.get("totalLength", None),
            "water_body_name": filtered_params.get("waterBodyName", None),
            "meeting_time": filtered_params.get("encounterTime", None),
            "status": filtered_params.get("status", None),
            "water_body_number": filtered_params.get("waterBodyNumber", None),
	        "species": filtered_params.get("species", None),
	    }.items() if v
    }
    print(f"{new_params=}")

    if not id_service.commit_card(
        submission_id,
        card_data=new_params
    ):
        raise APIError(status=500, msg="Something went wrong")

    return {}


def delete_card(
    newt_id: str,
    card_type: str,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    card_id = f"{newt_id}"
    result = id_service.delete_card(card_id, True, True)

    if result['error'] is not None:
        raise APIError(status=400, msg=result['error'])

    return {}
