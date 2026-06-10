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
COPY requirements.txt .
RUN apt-get update && apt-get install -y dos2unix
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY models/ ./models/
# COPY serve /opt/serve
# RUN chmod +x /opt/serve
COPY serve /usr/local/bin/serve
RUN dos2unix /usr/local/bin/serve && chmod +x /usr/local/bin/serve
EXPOSE 8080
ENTRYPOINT ["serve"]