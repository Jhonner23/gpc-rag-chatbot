# GPC RAG Chatbot

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Chatbot RAG (Retrieval-Augmented Generation) **on-premise** para consultar Guias de
Practica Clinica (GPC). Responde preguntas citando siempre la guia, seccion y
pagina de origen. Corre 100% local con [Ollama] (LLM y embeddings) y [Qdrant]
(base vectorial) -- sin llamadas a APIs de pago y sin dependencias en la nube,
pensado para cumplir con la normativa colombiana de proteccion de datos
(Ley 1581/2012, Resolucion 1995/1999).

Generado a partir del [data science project template] de JoseRZapata (uv, ruff,
mypy, pytest, Hydra, MkDocs, pre-commit, GitHub Actions).

> Guia de arquitectura completa (por que cada decision de stack, roadmap por
> fases, cumplimiento normativo): ver `docs/arquitectura.md` en este repo, o el
> documento compartido en el proyecto de Claude.

## Arquitectura (resumen)

```text
PDFs GPC -> extraccion (+OCR si aplica) -> chunking por seccion -> embeddings (Ollama)
   -> Qdrant

Usuario -> Chatbot (Chainlit) -> API (FastAPI) -> Retrieval hibrido (denso + BM25 + rerank)
   -> LLM (Ollama) -> Respuesta con citas
```

| Componente          | Herramienta                                   |
| ------------------- | ---------------------------------------------- |
| LLM de generacion    | Ollama -- `qwen2.5:3b-instruct` (dev) / `qwen2.5:7b-instruct` (prod con mas VRAM) |
| Embeddings           | Ollama -- `bge-m3` (multilingue)               |
| Base vectorial       | Qdrant (self-hosted, gratis)                   |
| Retrieval            | Hibrido (denso + BM25) + reranking (cross-encoder) |
| Orquestacion         | Codigo propio (`src/gpc_rag/pipelines`)        |
| API                  | FastAPI                                        |
| Chatbot              | Chainlit                                       |
| Evaluacion           | RAGAS (con Ollama como juez, no OpenAI)        |
| Config               | Hydra (`conf/`)                                |
| Despliegue           | Docker Compose                                 |

## Requisitos previos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download) instalado y corriendo
- Docker + Docker Compose (solo para el despliegue final, no obligatorio en desarrollo)

## Puesta en marcha local (sin Docker)

1. Instalar dependencias:

    ```bash
    uv sync
    ```

2. Descargar los modelos de Ollama (una sola vez):

    ```bash
    ollama pull bge-m3
    ollama pull qwen2.5:3b-instruct
    ollama pull qwen2.5:7b-instruct  # usado por el coordinador y el evaluador (ver conf/config.yaml -> agents)
    ```

3. Levantar Qdrant (mas simple con Docker, aunque el resto del proyecto corra local):

    ```bash
    docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
    ```

4. Colocar tus PDFs de GPC en `data/01_raw/gpc/` e indexarlos:

    ```bash
    PYTHONPATH=src uv run python -m gpc_rag.pipelines.build_index --input-dir data/01_raw/gpc --recreate
    ```

5. Levantar la API (agrega GPC_USE_AGENTS=true para usar la arquitectura de agentes -- coordinador + RAG + evaluador, ver seccion 9 de docs/arquitectura.md -- en vez del pipeline directo):

    ```bash
    PYTHONPATH=src uv run uvicorn gpc_rag.api.main:app --reload --port 8000
    # o, con la capa de agentes activada:
    GPC_USE_AGENTS=true PYTHONPATH=src uv run uvicorn gpc_rag.api.main:app --reload --port 8000
    ```

6. En otra terminal, levantar el chatbot:

    ```bash
    PYTHONPATH=src uv run chainlit run src/gpc_rag/chat/app.py --port 8001
    ```

7. Abrir <http://localhost:8001> y preguntar.

## Despliegue en Docker (produccion)

```bash
docker compose up -d ollama qdrant
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose up -d --build api chat
docker compose exec api python -m gpc_rag.pipelines.build_index --input-dir data/01_raw/gpc --recreate
```

El chatbot queda disponible en `http://<servidor>:8001`. Para exponerlo con
HTTPS fuera de la red interna, poner un reverse proxy (nginx/traefik) delante
del puerto 8001 -- no expongas los puertos de `ollama` (11434) ni `qdrant`
(6333) directamente a internet.

## Evaluacion (RAGAS)

```bash
PYTHONPATH=src uv run python -m gpc_rag.evaluation.ragas_eval --dataset data/05_model_input/preguntas_evaluacion.jsonl
```

El dataset es un JSONL con `{"question": "...", "ground_truth": "..."}` por
linea -- idealmente validado por un clinico. El juez de RAGAS usa el mismo
Ollama local (no OpenAI), asi que las metricas son gratis pero algo mas
ruidosas que con un modelo grande como juez; util para comparar versiones del
pipeline entre si, no como numero absoluto.

## Desarrollo

```bash
uv run pytest              # tests
uv run ruff check src tests --config .code_quality/ruff.toml   # lint
uv run ruff format src tests                                   # formato
uv run mypy src --config-file .code_quality/mypy.ini           # tipos
```

## Estructura del codigo

```text
src/gpc_rag/
  ingestion/    # extraccion PDF (+OCR) y chunking jerarquico por seccion
  embeddings/   # cliente de embeddings (Ollama)
  vectorstore/  # cliente de Qdrant
  retrieval/    # busqueda hibrida (denso + BM25) + reranking
  generation/   # prompts (con reglas anti-alucinacion) + cliente LLM (Ollama)
  api/          # FastAPI (/chat, /health)
  chat/         # chatbot Chainlit (cliente HTTP de la API)
  evaluation/   # evaluacion RAGAS
  pipelines/    # build_index (indexacion) y query_pipeline (consulta)
  common/       # configuracion (Hydra) y tipos compartidos
conf/           # configuracion Hydra (modelos, chunking, retrieval, ambientes dev/docker)
```

## Cumplimiento y limites

- Todo el procesamiento (LLM, embeddings, base vectorial) corre on-premise;
  no se envian documentos ni preguntas a ningun servicio externo.
- Las respuestas citan siempre la guia/seccion/pagina de origen y el prompt
  del sistema exige responder solo con base en el contexto recuperado -- aun
  asi, esto **apoya pero no reemplaza** el juicio clinico profesional.
- Este repositorio es un punto de partida (POC); antes de un uso clinico real
  se recomienda validacion con personal medico y una revision formal de
  seguridad y privacidad de datos.

[Ollama]: https://ollama.com
[Qdrant]: https://qdrant.tech
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
