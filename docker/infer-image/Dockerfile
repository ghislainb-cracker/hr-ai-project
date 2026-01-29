# We use the official vLLM container as a base.
# This image includes vLLM, PyTorch, and CUDA 12.1
FROM vllm/vllm-openai:v0.5.3.post1

# Install SageMaker toolkit and our web server
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy our server logic into the directory SageMaker expects
COPY model/ /opt/ml/code/

WORKDIR /opt/ml/code

# SageMaker will send requests to port 8080
ENV SAGEMAKER_BIND_TO_PORT 8080
ENV PORT 8080

# The entrypoint is our custom FastAPI wrapper script
ENTRYPOINT ["python", "/opt/ml/code/app.py"]