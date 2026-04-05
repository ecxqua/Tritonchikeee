from services.identification_service import IdentificationService
from api.error import APIError

from utils.json_utils import make_json_safe

from typing import Any, Dict


def complete_confirmation(
    upload_id: int,
    decision: str,
    existing_id: str | None,
    params: Dict[str, str],
    id_service: IdentificationService
) -> Dict[str, Any]:
    up = id_service.upload_service.get_upload(upload_id)
    if not up or up["status"] not in ["pending"]:
        raise APIError(f"Invalid upload code {upload_id}", status=400)

    if decision not in ["NEW", "MATCH", "CANCEL"]:
        raise APIError("Invalid decision choice", status=400)

    return make_json_safe(id_service.confirm_decision(
        upload_id=upload_id,
        decision=decision,
        existing_card_id=existing_id,
        card_data=params
    ))
