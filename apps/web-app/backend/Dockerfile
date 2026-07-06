# CAFE platform backend. Built from the repo root so it can install the local cafe-core package.
# Build context in docker-compose is the repo root (../../ from this file).
FROM python:3.11-slim

WORKDIR /app

# System deps: R is optional (only for the ordinal CLMM / logistic GLMM layers). Skipped here to
# keep the image lean — the linear mixed model + descriptive/pareto/reliability all work without it.
RUN pip install --no-cache-dir --upgrade pip

# Install the local cafe-core library first (its deps come via the extras).
COPY packages/cafe-core /src/cafe-core
RUN pip install --no-cache-dir "/src/cafe-core"

# Backend deps.
COPY apps/web-app/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Backend code.
COPY apps/web-app/backend /app

# The techniques folder is mounted at runtime (see docker-compose); default path inside the container.
ENV CAFE_TECHNIQUES_DIR=/techniques

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
