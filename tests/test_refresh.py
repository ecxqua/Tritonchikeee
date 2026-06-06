from services.identification_service import create_identification_service, setup
import time


service = create_identification_service()
service.refresh(confirm=True, remigrate=True)
