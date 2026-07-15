FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    LLAMA_SERVER_BINARY=/usr/local/bin/llama-server \
    LLAMA_MODEL_PATH=/app/models/model.gguf \
    AUTO_START_LLAMA=true \
    AUTO_START_FRONTEND=true \
    AUTO_OPEN_BROWSER=false \
    FRONTEND_HOST=0.0.0.0 \
    FRONTEND_PORT=5003 \
    FRONTEND_DIR=/app/frontend \
    RAG_PDF_PATH=/app/data/Employee_Handbook.pdf

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
        pkg-config \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        -DGGML_CUDA=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /tmp/llama.cpp/build --target llama-server -j"$(nproc)" \
    && cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

COPY Requirements.txt ./Requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 8000 5003

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["sh", "-c", "cd /app/backend && python main.py"]
