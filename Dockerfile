# Changes for Sagemaker compatibility
# 1. Using ENTRYPOINT instead of CMD to ensure the application starts correctly in SageMaker.
# 2. Exposing port 8080, which is the default port for SageMaker inference containers

# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY app/ ./app/
# COPY models/ ./models/
# COPY start.py .
# EXPOSE 8080
# ENTRYPOINT ["python", "start.py"]

FROM python:3.11-slim
WORKDIR /opt

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy internal application logic code blocks
COPY app/ ./app/
COPY models/ ./models/

# Expose internal SageMaker hosting port
EXPOSE 8080

# Launch Uvicorn natively without relying on a wrapper shell file
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8080"]
