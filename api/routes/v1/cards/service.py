from services.card_service import ALLOWED_FIELDS
from services.identification_service import IdentificationService

from typing import Any, Dict


def get_card_history(
    card_id: str,
    id_service: IdentificationService
) -> Dict[str, Any]:
    template_type = id_service.card_service.get_card(card_id)["template_type"]
    template_type = template_type if len(template_type) == 4 else (
        template_type[0:2] + '-' + template_type[-1]
    )

    fields = ALLOWED_FIELDS.get(template_type, None)
    if not fields:
        raise ValueError(f"Could not find allowed fields for {template_type}")

    result = {}
    for field in fields:
        result[field] = id_service.card_service.get_field_history(card_id, field)

    return result
