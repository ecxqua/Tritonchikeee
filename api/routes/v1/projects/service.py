from services.identification_service import IdentificationService
from api.error import APIError

from typing import Any, Dict, List


def create_project(
    name: str,
    description: str,
    species: str | None,
    territory: str | None,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    name = name.strip()
    description = description.strip()

    if not name or not description:
        raise APIError(status=400, msg=f"Name/description cannot be blank")

    species_filter = [species] if species else None
    territory_filter = [territory] if territory else None

    return {
        "id": id_service.project_service.get_or_create_project(
            name=name,
            description=description,
            species_filter=species_filter,
            territory_filter=territory_filter
        )
    }


def fetch_projects(
    id_service: IdentificationService,      
) -> List[Dict[str, Any]]:
    proj = id_service.project_service.list_projects()
    # id_service.card_service.get_prototypes_by_project
