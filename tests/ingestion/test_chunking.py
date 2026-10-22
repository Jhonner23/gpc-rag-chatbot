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


def test_split_into_sections_detects_bold_numbered_headers_from_ocr_pages():
    # Paginas OCR (ver ingestion/extract.py) no traen '#', pero suelen marcar
    # preguntas clinicas numeradas en negrita -- debe tratarse como seccion.
    ocr_like_text = (
        "Texto introductorio de la pagina, sin encabezado real.\n\n"
        "**13. En pacientes adultos con diagnostico de NAC grave, "
        "cual es el esquema antibiotico recomendado?**\n\n"
        "**Recomendacion 13** Ampicilina/sulbactam o ceftriaxona mas "
        "claritromicina."
    )
    sections = split_into_sections(ocr_like_text)
    titles = [s.title for s in sections]
    expected_section_count = 2
    assert len(sections) == expected_section_count
    assert titles[0] == "Sin seccion"
    assert titles[1].startswith("13. En pacientes adultos")
    assert "Ampicilina/sulbactam" in sections[1].text


def test_chunk_document_preserves_section_and_source_metadata():
    pages = [{"text": SAMPLE_MD, "page": 3}]
    chunks = chunk_document(
        pages, source_file="guia_diabetes.pdf", chunk_size_tokens=50, min_chunk_tokens=5
    )

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


def test_inline_bold_labels_split_subsections():
    """Etiquetas en negrita al inicio de parrafo (formato de los protocolos de
    la CUB) separan subsecciones: los criterios de hemocultivos no deben
    quedar en el fragmento de los examenes para definir hospitalizacion."""
    text = (
        "## **ESTUDIOS IMAGINOLÓGICOS**\n\n"
        "**Exámenes de laboratorio clínico (IIB):** Para definir la necesidad de "
        "hospitalización, se hará hemoleucograma y nitrógeno ureico en todo paciente.\n\n"
        "**DIAGNÓSTICO MICROBIOLÓGICO:** Los estudios microbiológicos sólo se harán en "
        "paciente con indicación de hospitalización\n\n"
        "**Hemocultivos:** Deben tomarse en los pacientes hospitalizados por neumonía "
        "con BUN mayor a 30 mg/dL.\n\n"
        "- **Indicaciones de hospitalización en pacientes con ITU:**\n\n- Choque séptico\n\n"
        "**Gráfico 1.** Etiología de la neumonía"
    )
    sections = split_into_sections(text)
    titles = [s.title for s in sections]
    assert titles == [
        "Exámenes de laboratorio clínico (IIB)",
        "DIAGNÓSTICO MICROBIOLÓGICO",
        "Hemocultivos",
        "Indicaciones de hospitalización en pacientes con ITU",
    ]
    assert "Gráfico 1" in sections[-1].text  # la leyenda no abre seccion
    assert "BUN" not in sections[0].text


def test_short_titled_section_is_kept_and_page_continuation_keeps_its_title():
    pages = [
        {
            "text": "**DIAGNÓSTICO MICROBIOLÓGICO:** Los estudios microbiológicos sólo se "
            "harán en paciente con indicación de hospitalización\n\n"
            "**Filmarray:** Prueba de ácidos nucleicos que se debe solicitar en todo "
            "paciente adulto con neumonía e indicación de hospitalización.",
            "page": 3,
        },
        {
            "text": "- Panel de neumonía: muestra de esputo, aspirado traqueal o lavado "
            "broncoalveolar en pacientes sin sospecha de SARS-CoV-2.",
            "page": 4,
        },
    ]
    chunks = chunk_document(pages, source_file="x.pdf", min_chunk_tokens=40)
    assert [(c.page, c.section) for c in chunks] == [
        (3, "DIAGNÓSTICO MICROBIOLÓGICO"),
        (3, "Filmarray"),
        (4, "Filmarray"),
    ]
