FROM python:3-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN mkdir -p /app/processed /app/output

RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

# Pipeline: cleaning → prestige → enrich (genres) → features (TF-IDF) → run (model)
CMD ["sh", "-c", "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && python cleaning.py && python prestige.py && python enrich.py && python features.py && python run.py"]