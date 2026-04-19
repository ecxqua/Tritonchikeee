from services.identification_service import IdentificationService
from api.error import APIError

from typing import Any, Dict

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


def get_card_by_newt_id(
    id: str,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    proto = id_service.card_service.get_prototype(id)
    if not proto:
        raise APIError(status=404, msg=f"No prototype by ID {id}")
    
    card = sorted(proto["cards"], key=lambda c: c["created_at"])[0]

    return {
        "cardType": card["template_type"],
        "data": {k: v for k, v in {
            "dateFilled": card.get("date", None),
            "bodyLength": card.get("length_body", None),
            "tailLength": card.get("length_tail", None),
            "weight": card.get("weight", None),
            "sex": card.get("sex", None),
            "exactBirthDate": card.get("birth_year_exact", None),
            "estimatedBirthDate": card.get("birth_year_approx", None),
            "photoNumber": "",  # ???
            "regionOfOrigin": card.get("origin_region", None),
            "measurementDevice": card.get("length_device", None),
            "scaleBrand": card.get("weight_device", None),
            "notes": card.get("notes", None),
            "releaseDate": card.get("release_date", None),
            "fatherId": card.get("parent_male_id", None),
            "motherId": card.get("parent_female_id", None),
            "totalLength": card.get("length_total", None),
            "waterBodyName": card.get("water_body_name", None),
            "encounterDate": "",  # ???
            "encounterTime": card.get("meeting_time", None),
            "bellyPhotoNumber": "",  # ???
            "status": card.get("status", None),
            "waterBodyNumber": card.get("water_body_number", None),
        } if v is not None}
    }


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
