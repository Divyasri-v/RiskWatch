# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# Spark needs Java
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Long timeout + retries for slow connections.
# The cache mount keeps already-downloaded packages, so a failed build resumes instead of starting over.
ENV PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10

WORKDIR /app

# PySpark is the big download (~300 MB), so it gets its own step
RUN --mount=type=cache,target=/root/.cache/pip pip install pyspark==3.5.3

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# lets scripts do "import config" from any folder
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
