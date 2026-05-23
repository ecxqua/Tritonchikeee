from services.identification_service import IdentificationService

from collections import Counter
from typing import Any, Dict


def get_stats(
    id_service: IdentificationService,
) -> Dict[str, Any]:
    species = id_service.get_species()
    reco = id_service.card_service.get_reid_count()
    recent_commits = id_service.card_service.get_last_commits(10)

    return {
        "totalProjects": len(id_service.project_service.list_projects()),
        "totalNewts": sum(item["count"] for item in species),
        "totalRecognitions": reco,
        "recentActivity": recent_commits,
        "speciesBreakdown": [
            {"species": item["name"], "count": item["count"]}
            for item in species
        ]
    }
