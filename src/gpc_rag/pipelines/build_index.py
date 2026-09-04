"""Pipeline de indexacion (feature pipeline): PDFs -> chunks -> embeddings -> Qdrant.

Uso:
    uv run python -m gpc_rag.pipelines.build_index --input-dir data/01_raw/gpc

Correr de nuevo con los mismos archivos actualiza los chunks (upsert por id).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gpc_rag.common.settings import load_config
from gpc_rag.embeddings.ollama_embeddings import OllamaEmbedder
from gpc_rag.ingestion.chunking import chunk_document
from gpc_rag.ingestion.extract import extract_pages_markdown
from gpc_rag.vectorstore.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(input_dir: Path, recreate_collection: bool = False) -> None:
    cfg = load_config()
    embedder = OllamaEmbedder(cfg)

    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning("No se encontraron PDFs en %s", input_dir)
        return

    logger.info("Indexando %d guias desde %s", len(pdf_files), input_dir)

    all_chunks = []
    for pdf_path in pdf_files:
        logger.info("Procesando %s ...", pdf_path.name)
        pages = extract_pages_markdown(pdf_path)
        chunks = chunk_document(
            pages,
            source_file=pdf_path.name,
            chunk_size_tokens=cfg.chunking.chunk_size_tokens,
            chunk_overlap_ratio=cfg.chunking.chunk_overlap_ratio,
            min_chunk_tokens=cfg.chunking.min_chunk_tokens,
        )
        logger.info("  -> %d chunks", len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No se genero ningun chunk. Revisa los PDFs de entrada.")
        return

    logger.info("Generando embeddings para %d chunks (puede tardar)...", len(all_chunks))
    vectors = embedder.embed_texts([c.text for c in all_chunks])

    store = QdrantStore(cfg, vector_size=len(vectors[0]))
    store.ensure_collection(recreate=recreate_collection)
    store.upsert_chunks(all_chunks, vectors)

    logger.info(
        "Listo: %d chunks indexados en la coleccion '%s'.",
        len(all_chunks),
        cfg.app.collection_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa GPC en la base vectorial.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/01_raw/gpc"),
        help="Carpeta con los PDF de las guias a indexar.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Borra y recrea la coleccion antes de indexar (empezar de cero).",
    )
    args = parser.parse_args()
    run(args.input_dir, recreate_collection=args.recreate)


if __name__ == "__main__":
    main()
