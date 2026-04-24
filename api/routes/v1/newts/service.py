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
    proto = id_service.card_service.get_prototype(id)
    if not proto:
        raise APIError(status=404, msg=f"No prototype by ID {id}")

    cards = sorted(proto["cards"], key=lambda c: c["created_at"])

    first = cards[0]
    last = cards[-1]

    return {
        "id": proto["prototype_id"],
        "projectId": proto["project_id"],
        "cardType": last["template_type"],
        "createdAt": first["created_at"],
        "sex": first["sex"],
        "status": first["status"]
    }


def get_cards_by_newt_id(
    id: str,
    id_service: IdentificationService,
) -> List[Dict[str, Any]]:
    proto = id_service.card_service.get_prototype(id)
    if not proto:
        raise APIError(status=404, msg=f"No prototype by ID {id}")

    result: List[Dict[str, Any]] = []

    for card in proto["cards"]:
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
    proto = id_service.card_service.get_prototype_by_card_id(id)
    if not proto:
        raise APIError(status=404, msg=f"No prototype by ID {id}")

    card = sorted(proto["cards"], key=lambda c: c["created_at"])[0]
    card_id = card["card_id"]

    if not id_service.card_service._update_card(card_id, **params):
        raise APIError(status=500, msg="Something went wrong")

    return {}
