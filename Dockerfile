FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends p7zip-full && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorboard && \
    rm -rf /root/.cache/pip

# Copy project source
COPY src/ src/
COPY configs/ configs/
COPY run.sh .

# sensiAPI.txt is imported at module level in symbolizer.py
COPY data/sensiAPI.txt data/sensiAPI.txt

ENV PYTHONPATH="/workspace"

# Data is mounted here at runtime
VOLUME /workspace/data

ENTRYPOINT ["python"]
CMD ["src/run.py"]
