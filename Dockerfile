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

# detect JAVA_HOME at runtime so it works on arm64 (Mac) and amd64 (Windows/Linux)
CMD ["sh", "-c", "export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && python cleaning.py && python features.py && python run.py"]
