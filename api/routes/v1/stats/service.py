from services.identification_service import IdentificationService

from collections import Counter
from typing import Any, Dict


def get_stats(
    id_service: IdentificationService,
) -> Dict[str, Any]:
    species = id_service.get_species()

    return {
        "totalProjects": len(id_service.project_service.list_projects()),
        "totalNewts": sum(item["count"] for item in species),
        "totalRecognitions": 40,  # TODO some link to upload service?..
        "recentActivity": [],  # TODO not yet tracked
        "speciesBreakdown": [
            {"species": item["name"], "count": item["count"]}
            for item in species
        ]
    }
