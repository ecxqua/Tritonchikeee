from services.identification_service import IdentificationService
from services.card_service import CardService
from api.error import APIError

from typing import Any, Dict, List


def _resolve_project(project: Dict[str, Any], cards: CardService) -> Dict[str, Any]:
    return {
        "id": project["id"],
        "name": project["name"],
        "description": project["description"],
        "species": project["species_filter"],
        "territory": project["territory_filter"],
        "createdAt": project["created_at"],
        "newtCount": len(cards.get_prototypes_by_project(project["id"]))
    }


def create_project(
    name: str,
    description: str,
    species: List[str] | None,
    territory: List[str] | None,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    name = name.strip()
    description = description.strip()

    if not name or not description:
        raise APIError(status=400, msg=f"Name/description cannot be blank")

    return {
        "id": id_service.project_service.get_or_create_project(
            name=name,
            description=description,
            species_filter=species,
            territory_filter=territory
        )
    }


def fetch_projects(
    id_service: IdentificationService,      
) -> List[Dict[str, Any]]:
    cards = id_service.card_service
    projects = id_service.project_service.list_projects()

    return [_resolve_project(pr, cards) for pr in projects]


def fetch_project(
    id: int,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    cards = id_service.card_service
    project = id_service.project_service.get_project_by_id(id)
    if project is None:
        raise APIError(status=404, msg=f"No project with ID {id}")

    return _resolve_project(project, cards)


def update_project(
    id: int,
    name: str | None,
    description: str | None,
    species: List[str] | None,
    territory: List[str] | None,
    id_service: IdentificationService
) -> Dict[str, Any]:
    service = id_service.project_service

    project = service.get_project_by_id(id)
    if project is None:
        raise APIError(status=404, msg=f"No project with ID {id}")

    params: Dict[str, Any] = {
        k: v for k, v in {
            "name": name,
            "description": description,
            "species_filter": species,
            "territory_filter": territory,
        }.items() if v is not None
    }

    if not service.update_project(id, **params):
        raise APIError(status=500, msg="Something went wrong")

    return {}


def delete_project(
    id: int,
    id_service: IdentificationService,
) -> Dict[str, Any]:
    if not id_service.project_service.delete_project(id, confirm=True):
        raise APIError(status=500, msg="Something went wrong")

    return {}


def get_project_newts(
    id: int,
    id_service: IdentificationService,
) -> List[Dict[str, Any]]:
    if id_service.project_service.get_project_by_id(id) is None:
        raise APIError(status=404, msg=f"No project with ID {id}")

    result: List[Dict[str, Any]] = []
    for proto in id_service.card_service.get_prototypes_by_project(id):
        cards = sorted(proto["cards"], key=lambda c: c["created_at"])

        first = cards[0]
        last = cards[-1]

        result.append({
            "id": proto.get("prototype_id", None),
            "projectId": proto.get("project_id", None),
            "cardType": last.get("template_type", None),
            "createdAt": first.get("created_at", None),
            "sex": first.get("sex", None),
            "status": first.get("status", None)
        })

    return sorted(result, key=lambda item: int(item["id"].split("-")[-1]))
