from services.identification_service import IdentificationService

from typing import List


def fetch_territories(
    id_service: IdentificationService,
) -> List[str]:
    return id_service.project_service.get_unique_filters()["territories"]
