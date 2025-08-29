# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py /app/agent.py
RUN mkdir -p /app/utils
COPY utils/*.py /app/utils/
RUN touch /app/utils/__init__.py

RUN mkdir -p /app/models /app/logs

ENV PORT=8000 \
    TFIDF_MAX_FEATURES=20000 \
    RL_LR=0.1 \
    DATABASE_URL="sqlite:///models/agent.db" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "agent:app"]
