# FROM python:3.11-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code
# COPY app/ ./app/
# COPY models/ ./models/

# # Expose the port the app runs on
# EXPOSE 8001

# ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Changes for Sagemaker compatibility
# 1. Using ENTRYPOINT instead of CMD to ensure the application starts correctly in SageMaker.
# 2. Exposing port 8080, which is the default port for SageMaker inference containers

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY models/ ./models/
COPY serve.sh .
RUN chmod +x serve.sh
EXPOSE 8080
ENTRYPOINT ["./serve.sh"]