from services.identification_service import create_identification_service
import time


# 1. Инициализация
service = create_identification_service()

# print(service.cleanup_expired_uploads())
print(service.cleanup_uploads())