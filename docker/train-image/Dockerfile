# Accept base image URI from CodeBuild environment variable
ARG BASE_IMAGE

# Use the official AWS DLC PyTorch image as base
FROM ${BASE_IMAGE}

# --- Install System Dependencies ---
USER root
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- Install Python Dependencies ---
# Copy requirements.txt to container
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Copy Project Code ---
COPY scripts/ /opt/ml/code/scripts/

# Set working directory
WORKDIR /opt/ml/code

# --- Fix Permissions (IMPORTANT) ---
# Create 'sagemaker' user if missing, then set ownership
RUN if ! id -u sagemaker >/dev/null 2>&1; then useradd -m sagemaker; fi && \
    chown -R sagemaker:sagemaker /opt/ml/code

# Run as the default SageMaker user
USER sagemaker

# --- Train Script Entrypoint ---
ENTRYPOINT ["python", "/opt/ml/code/scripts/train.py"]
