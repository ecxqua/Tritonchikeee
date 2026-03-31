from services.identification_service import create_identification_service

# 1. Инициализация
service = create_identification_service()

service.cleanup_expired()