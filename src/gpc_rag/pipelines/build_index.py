"""Pipeline de indexacion (feature pipeline): PDFs -> chunks -> embeddings -> Qdrant.

Uso:
    uv run python -m gpc_rag.pipelines.build_index --input-dir data/01_raw/gpc --recreate

Indexa los protocolos institucionales (--input-dir) y, si existe, la carpeta
de fuentes externas (--external-dir, por defecto data/01_raw/externas), cuyos
PDF deben estar en su catalogo fuentes.yaml (ver ingestion/sources.py).

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
from gpc_rag.ingestion.sources import INSTITUTIONAL, load_catalog, source_for
from gpc_rag.ingestion.tables import load_curated_tables, table_chunks
from gpc_rag.vectorstore.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _documents(input_dir: Path, external_dir: Path | None) -> list[tuple[Path, str, str]]:
    """(pdf, tipo de fuente, titulo) de cada documento a indexar. Falla antes
    de procesar nada si un PDF externo no esta en el catalogo."""
    documents = [(pdf, INSTITUTIONAL, "") for pdf in sorted(Path(input_dir).glob("*.pdf"))]
    if external_dir and Path(external_dir).is_dir():
        external = sorted(Path(external_dir).glob("*.pdf"))
        if external:
            catalog = load_catalog(external_dir)
            for pdf in external:
                info = source_for(pdf, catalog)
                documents.append((pdf, info.source_type, info.title))
    return documents


def run(
    input_dir: Path, recreate_collection: bool = False, external_dir: Path | None = None
) -> None:
    cfg = load_config()
    embedder = OllamaEmbedder(cfg)

    documents = _documents(input_dir, external_dir)
    if not documents:
        logger.warning("No se encontraron PDFs en %s", input_dir)
        return

    logger.info(
        "Indexando %d documentos (%d institucionales, %d externos)",
        len(documents),
        sum(1 for _, kind, _ in documents if kind == INSTITUTIONAL),
        sum(1 for _, kind, _ in documents if kind != INSTITUTIONAL),
    )

    all_chunks = []
    for pdf_path, source_type, title in documents:
        logger.info("Procesando %s (%s) ...", pdf_path.name, source_type)
        curated = load_curated_tables(pdf_path)
        pages = extract_pages_markdown(pdf_path)
        chunks = chunk_document(
            pages,
            source_file=pdf_path.name,
            chunk_size_tokens=cfg.chunking.chunk_size_tokens,
            chunk_overlap_ratio=cfg.chunking.chunk_overlap_ratio,
            min_chunk_tokens=cfg.chunking.min_chunk_tokens,
            drop_tables_on_pages=curated.drop_extracted_tables_on_pages if curated else None,
        )
        if curated:
            rows = table_chunks(curated, start_index=len(chunks))
            logger.info("  -> %d filas de tablas revisadas", len(rows))
            chunks += rows
        for chunk in chunks:
            chunk.source_type = source_type
            chunk.source_title = title
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
        help="Carpeta con los PDF de los protocolos institucionales.",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("data/01_raw/externas"),
        help="Carpeta con las guias y revisiones externas (y su fuentes.yaml).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Borra y recrea la coleccion antes de indexar (empezar de cero).",
    )
    args = parser.parse_args()
    run(args.input_dir, recreate_collection=args.recreate, external_dir=args.external_dir)


if __name__ == "__main__":
    main()
