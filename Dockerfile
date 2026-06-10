# 1. Force the base image to be AMD64 directly at the root layer
FROM --platform=linux/amd64 python:3.11-slim

# 2. Use the strict default directory SageMaker expects for custom containers
WORKDIR /opt/program

# Install system dependencies
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application directories
COPY app/ ./app/
COPY models/ ./models/

# 3. Create a clean python execution file directly inside the image
# This bypasses the shell entirely and maps Uvicorn straight to python's interpreter
RUN echo '#!/usr/bin/env python3' > /opt/program/serve && \
    echo 'import os' >> /opt/program/serve && \
    echo 'os.system("uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port 8080")' >> /opt/program/serve && \
    chmod 755 /opt/program/serve

# Ensure the SageMaker program directory is actively in the path
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

# Explicitly match SageMaker's direct call expectation
CMD ["serve"]
