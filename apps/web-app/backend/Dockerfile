# CAFE platform backend. Built from the repo root so it can install the local cafe-core package.
# Build context in docker-compose is the repo root (../../ from this file).
FROM python:3.11-slim

WORKDIR /app

# R + the model packages for the ordinal CLMM and binary logistic GLMM layers, so studies run in the
# app compute the same statistics the library does. (Debian prebuilt packages — no compilation.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base-core r-cran-ordinal r-cran-lme4 \
    && rm -rf /var/lib/apt/lists/*

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
