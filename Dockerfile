FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies (Java for joern, p7zip for data extraction)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-11-jre-headless \
        p7zip-full && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir torch-geometric==2.7.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy project
COPY src/ src/
COPY configs/ configs/
COPY joern/ joern/
COPY run.sh .

# sensiAPI.txt is imported at module level in symbolizer.py
COPY data/sensiAPI.txt data/sensiAPI.txt

RUN chmod +x joern/joern-parse

ENV PYTHONPATH="/workspace"

# Data is mounted here at runtime
VOLUME /workspace/data

ENTRYPOINT ["python"]
CMD ["src/run.py"]
