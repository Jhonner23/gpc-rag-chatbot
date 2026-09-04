from gpc_rag.ingestion.chunking import approx_token_count, chunk_document, split_into_sections

SAMPLE_MD = """# Seccion 1
Contenido de prueba sobre diabetes tipo 2, manejo farmacologico y no
farmacologico segun la guia clinica nacional para pacientes adultos con
comorbilidades relevantes.

## Subseccion recomendaciones
Mas texto clinico de ejemplo para validar el chunking jerarquico por
seccion, con suficientes palabras para superar el minimo configurado.
"""


def test_split_into_sections_detects_headers():
    sections = split_into_sections(SAMPLE_MD)
    titles = [s.title for s in sections]
    assert "Seccion 1" in titles
    assert "Subseccion recomendaciones" in titles


def test_split_into_sections_without_headers_returns_single_section():
    sections = split_into_sections("Texto plano sin encabezados.")
    assert len(sections) == 1
    assert sections[0].title == "Sin seccion"


def test_chunk_document_preserves_section_and_source_metadata():
    pages = [{"text": SAMPLE_MD, "page": 3}]
    chunks = chunk_document(pages, source_file="guia_diabetes.pdf", chunk_size_tokens=50, min_chunk_tokens=5)

    expected_chunk_count = 2
    expected_page = 3
    assert len(chunks) == expected_chunk_count
    assert all(c.source_file == "guia_diabetes.pdf" for c in chunks)
    assert all(c.page == expected_page for c in chunks)
    assert {c.section for c in chunks} == {"Seccion 1", "Subseccion recomendaciones"}
    assert all(c.chunk_id for c in chunks)  # cada chunk tiene un id unico


def test_chunk_document_filters_chunks_below_min_tokens():
    pages = [{"text": "# Titulo\nmuy corto", "page": 1}]
    chunks = chunk_document(pages, source_file="x.pdf", min_chunk_tokens=40)
    assert chunks == []


def test_approx_token_count_is_positive_for_nonempty_text():
    assert approx_token_count("hola mundo") > 0
    assert approx_token_count("") >= 1
