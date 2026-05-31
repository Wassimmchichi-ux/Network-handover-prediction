FROM python:3.11-slim

WORKDIR /app

# Copy only what we need first (better caching in real builds)
COPY requirements-edge.txt /app/requirements-edge.txt

RUN pip install --no-cache-dir -r /app/requirements-edge.txt

COPY . /app

# Default: show help for training script
CMD ["python", "src/train_model.py", "--help"]
