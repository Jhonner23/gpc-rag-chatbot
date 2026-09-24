"""Todo protocolo indexado en el RAG debe tener su arbol correspondiente.

Este es el amarre explicito entre "agregar un documento al RAG" y "crear su
arbol de decision": si alguien deja un PDF nuevo en data/01_raw/gpc/ (para
que gpc_rag.pipelines.build_index lo indexe) pero no le crea un arbol en
trees/data/ con ese mismo `gpc_source`, la suite de pruebas falla y lo dice
explicitamente -- no depende de que alguien se acuerde de leer un README.

Lo contrario (un arbol sin PDF en el corpus del RAG) no se valida aqui: es
valido tener el arbol de un protocolo cuya guia todavia no se ha indexado
(o cuyo PDF, por la razon que sea, no vive en data/01_raw/gpc/), sobre todo
mientras el bosque se construye antes que el corpus del RAG.

Si data/01_raw/gpc/ esta vacio (clon fresco antes de agregar cualquier GPC),
no hay nada que exigir todavia y la prueba no falla.
"""

from __future__ import annotations

from pathlib import Path

from gpc_rag.trees.registry import list_trees

_RAG_CORPUS_DIR = Path(__file__).parents[2] / "data" / "01_raw" / "gpc"


def test_todo_pdf_del_corpus_rag_tiene_al_menos_un_arbol() -> None:
    pdfs = sorted(_RAG_CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        return  # nada que exigir todavia -- corpus vacio en un clon fresco

    fuentes_con_arbol = {entry.tree.gpc_source for entry in list_trees()}

    sin_arbol = [pdf.name for pdf in pdfs if pdf.name not in fuentes_con_arbol]

    assert not sin_arbol, (
        "Estos PDF estan en el corpus del RAG (data/01_raw/gpc/) pero no tienen "
        f"ningun arbol de decision que los cite como gpc_source: {sin_arbol}. "
        "Cada protocolo agregado al RAG necesita su propio arbol -- ver "
        "src/gpc_rag/trees/README.md, seccion 'Agregar un arbol nuevo'."
    )


def test_todo_arbol_cita_un_gpc_source_no_vacio() -> None:
    # Chequeo de sanidad inverso y barato: si un arbol tiene gpc_source vacio
    # o mal escrito, la prueba de arriba lo dejaria pasar silenciosamente
    # (nunca coincidiria con ningun PDF, pero tampoco se reportaria como
    # huerfano). Esto detecta ese typo antes de que alguien lo note a mano.
    for entry in list_trees():
        assert entry.tree.gpc_source.strip(), f"{entry.tree_id}: gpc_source vacio"
