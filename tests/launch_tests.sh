#!/bin/bash

python -m tests.test_identification_service
python -m tests.test_cleanup_expired

echo "Проверки окончены"