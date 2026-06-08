#!/bin/bash
# SageMaker passes 'serve' as argument — we ignore it
# and always start uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
