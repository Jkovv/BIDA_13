FROM python:3-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalacja Javy 21 (zamiast 17)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Aktualizacja ścieżki JAVA_HOME dla wersji 21
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    chown -R appuser /app

USER appuser

CMD ["python", "cleaning.py"]