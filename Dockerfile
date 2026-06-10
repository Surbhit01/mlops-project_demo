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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy internal application logic
COPY app/ ./app/
COPY models/ ./models/

# Generate the script dynamically with explicit paths and execution permissions
RUN echo '#!/bin/sh' > /usr/local/bin/serve && \
    echo 'cd /opt' >> /usr/local/bin/serve && \
    echo 'exec uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port 8080' >> /usr/local/bin/serve && \
    chmod +x /usr/local/bin/serve

EXPOSE 8080

# This matches SageMaker's default expectation exactly
CMD ["serve"]

