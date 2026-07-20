
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /HR-Assistance

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 5003
EXPOSE 5002

WORKDIR /HR-Assistance/backend

CMD ["python", "main.py"]
