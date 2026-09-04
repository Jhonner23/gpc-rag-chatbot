# Imagen unica para los servicios "api" y "chat" (ver docker-compose.yml,
# el `command` de cada servicio decide que proceso correr).
FROM python:3.11-slim AS base

# Dependencias de sistema:
#  - ocrmypdf + tesseract: OCR de respaldo para GPC escaneadas
#  - build-essential: por si algun paquete Python necesita compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ocrmypdf \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# uv: mismo gestor de dependencias que se usa en desarrollo local
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copiar solo los archivos de dependencias primero (mejor cache de capas Docker)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group docs

# Copiar el resto del codigo
COPY conf/ ./conf/
COPY src/ ./src/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    RAG_ENV="docker"

EXPOSE 8000 8001

# Por defecto arranca la API; docker-compose.yml sobreescribe `command` para el
# servicio "chat" (chainlit run ...).
CMD ["uvicorn", "gpc_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
