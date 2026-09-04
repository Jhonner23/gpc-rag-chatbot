from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.generation.prompts import build_messages, format_context


def _sample_retrieved_chunk() -> RetrievedChunk:
    chunk = Chunk(
        text="La metformina es de primera linea en diabetes tipo 2.",
        source_file="guia_diabetes.pdf",
        section="Tratamiento farmacologico",
        page=12,
        chunk_id="abc-123",
    )
    return RetrievedChunk(chunk=chunk, score=0.87)


def test_format_context_includes_source_citation():
    context = format_context([_sample_retrieved_chunk()])
    assert "guia_diabetes.pdf" in context
    assert "Tratamiento farmacologico" in context
    assert "12" in context


def test_format_context_handles_empty_list():
    assert "sin contexto" in format_context([]).lower()


def test_build_messages_has_system_and_user_roles():
    messages = build_messages("Cual es el tratamiento de primera linea?", [_sample_retrieved_chunk()])
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "guia_diabetes.pdf" in messages[1]["content"]
    assert "no inventes" in messages[0]["content"].lower()
    assert "no uses conocimiento medico propio" in messages[0]["content"].lower()
