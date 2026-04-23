from services.identification_service import IdentificationService

from collections import Counter
from typing import Any, Dict


def get_stats(
    id_service: IdentificationService,
) -> Dict[str, Any]:
    projects = len(id_service.project_service.list_projects())

    prototypes = id_service.card_service.get_all_prototypes()
    species = [pr["species"] for pr in prototypes if "species" in pr]
    species_breakdown = dict(Counter(species))

    return {
        "totalProjects": projects,
        "totalNewts": len(prototypes),
        "totalRecognitions": 40,  # TODO some link to upload service?..
        "recentActivity": [],  # TODO not yet tracked
        "speciesBreakdown": [
            {"species": name, "count": count}
            for name, count in species_breakdown.items()
        ]
    }
