"""Retrieval hibrido: busqueda densa (Qdrant) + BM25 (lexica), fusionadas con
Reciprocal Rank Fusion, y opcionalmente reordenadas con un cross-encoder.

El hibrido importa especialmente en GPC: muchas recomendaciones comparten
vocabulario muy similar (ej. "manejo farmacologico de X"), y la busqueda
puramente densa a veces no distingue bien terminos clinicos exactos que BM25
si captura (siglas, nombres de farmacos, codigos).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig
from rank_bm25 import BM25Okapi

from gpc_rag.common.models import Chunk, RetrievedChunk
from gpc_rag.embeddings.ollama_embeddings import OllamaEmbedder
from gpc_rag.ingestion.sources import INSTITUTIONAL, _file_key, load_catalog
from gpc_rag.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Palabras en minuscula, sin tildes ni puntuacion.

    Antes se separaba solo por espacios, asi que "asintomática**", "ITU?" o
    "días," no coincidian con "asintomática", "ITU" o "días" de la pregunta:
    BM25 no encontraba, por ejemplo, la fila de bacteriuria asintomatica de la
    Tabla 1 del protocolo de ITU.
    """
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOPWORDS]


# Palabras vacias del español (sin tildes, como quedan tras _tokenize). Sin
# filtrarlas, en una pregunta como "¿En qué situaciones ... debe ser tratada
# y en cuáles no?" pesan "en", "que", "debe", "ser", "no", y BM25 favorece
# fragmentos largos que las repiten. "no" se conserva: en clinica cambia el
# sentido ("no tratar").
_STOPWORDS = frozenset(
    """a al algo algunas algunos ante antes como con contra cual cuales cuando de del
    desde donde durante e el ella ellas ellos en entre era es esa esas ese eso esos
    esta estan estas este esto estos fue fueron ha han hay la las le les lo los mas
    me mi muy nos o otra otras otro otros para pero por porque puede pueden que se
    segun ser si sin sobre son su sus tambien tiene tienen todo todos tu un una unas
    uno unos y ya debe deben cuyo cuya""".split()  # noqa: SIM905 -- lista legible
    # Ingles, para las guias y revisiones externas. "not" se conserva, igual
    # que "no".
    + """the of and to in is for with on as by be are was were this that these
    those or an at from it its which than also may can has have had been into
    their there such""".split()  # noqa: SIM905 -- lista legible
)


# Palabras que hablan del corpus y no del tema clinico ("¿que recomendaciones
# presentan las guias disponibles...?", "¿que fuentes del corpus
# contienen...?"). En BM25 empujaban secciones administrativas
# ("RECOPILACION DE LA EVIDENCIA", "CRITERIO DE CLASIFICACION DE LAS
# RECOMENDACIONES") por encima del contenido clinico.
_META_WORDS = frozenset(
    """recomendacion recomendaciones recomendada recomendado recomendados presentan
    presenta guia guias disponibles disponible diferencia diferencias existen existe
    encontradas encontrados encontrada fuente fuentes corpus contienen contiene
    informacion relacionada relacionadas relacionado relacionados documento
    documentos protocolo protocolos condicion""".split()  # noqa: SIM905 -- lista legible
)
# Ademas de las anteriores, palabras tan generales que una subpregunta que
# solo tiene estas ("¿que diferencias existen ... para el manejo de esta
# condicion?") no aporta un tema propio: se refiere al de la anterior.
_GENERIC_WORDS = frozenset({"manejo", "tratamiento", "enfoque"})


def retrieval_query(question: str) -> str:
    """Texto con el que se busca: la pregunta sin las subpreguntas que solo
    hablan del corpus o remiten al tema de otra.

    En "¿Que recomendaciones ... sobre el manejo de la bacteriuria
    asintomatica? ¿Que diferencias existen ...? ¿Que fuentes del corpus
    contienen ...?" solo la primera trae el tema. Con las tres, la busqueda
    densa y el reranker subian "ALCANCE", "OBJETIVO" o "INTRODUCCION" y la
    fila de tratamiento de la Tabla 1 quedaba fuera de los 18 candidatos. El
    modelo sigue recibiendo la pregunta completa.
    """
    parts = [p.strip() for p in re.split(r"(?<=\?)", question or "") if p.strip()]
    if len(parts) < 2:
        return question
    kept = [
        p
        for p in parts
        if any(w not in _META_WORDS and w not in _GENERIC_WORDS for w in _tokenize(p))
    ]
    return " ".join(kept) if kept else question


def _bm25_query_tokens(query: str) -> list[str]:
    tokens = [w for w in _tokenize(query) if w not in _META_WORDS]
    return tokens or _tokenize(query)


# Secciones administrativas o de contexto de los protocolos. Ante preguntas
# generales ("¿que factores considerar...?", "¿que recomiendan las
# guias...?") el reranker las ponia por encima del contenido clinico: en
# "factores para decidir el manejo de la neumonia" ocupaban 5 de los 8
# fragmentos. Se les resta retrieval.admin_section_penalty, salvo que la
# pregunta las nombre ("¿cual es el alcance del protocolo?").
_ADMIN_SECTIONS = {
    "objetivo": "objetivo",
    "alcance": "alcance",
    "introduccion": "introduccion",
    "epidemiologia": "epidemiologia",
    "epidemilogia": "epidemiologia",
    "recopilacion de la evidencia": "evidencia",
    "criterio de clasificacion": "clasificacion",
    "control de cambios": "cambios",
    "plan de delegacion": "delegacion",
    # Secciones de las guias y articulos externos que no dan recomendaciones.
    "necesidades de investigacion": "investigacion",
    "esta guia no aplica": "aplica",
    "metodologia": "metodologia",
    "poblacion de interes": "poblacion",
    "usuarios de la guia": "usuarios",
    "usuarios a quienes se dirige": "usuarios",
    "methods": "metodo",
    "research needs": "investigacion",
    "outstanding research": "investigacion",
    "conclusions and research needs": "investigacion",
    "controversies and uncertainties": "controversia",
    # Resumen, introduccion y conclusiones de los articulos: describen el
    # documento ("el panel abordo 16 areas...") en lugar de recomendar.
    "resumen": "resumen",
    "abstract": "resumen",
    "summary": "resumen",
    "introduction": "introduccion",
    "conclusion": "conclusion",
    "overview": "resumen",
}
# Encabezado del documento ("Proceso: Cuidado y tratamiento Código: CT-PL-193
# Versión: 9 Fecha aprobación: ..."): solo sirve si preguntan por la version.
_DOCUMENT_HEADER_RE = re.compile(r"proceso:.*codigo:\s*ct-pl", re.DOTALL)
_DEFAULT_ADMIN_PENALTY = 0.0


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _admin_penalty(
    section: str,
    query: str,
    penalty: float,
    text: str = "",
    source_type: str = INSTITUTIONAL,
    page: int | None = None,
) -> float:
    """Penalizacion (negativa) si la seccion es administrativa y la pregunta
    no la nombra. Se mira el ultimo nivel del titulo ("4. INTRODUCCIÓN —
    Manifestaciones clínicas" no es administrativa; "4. INTRODUCCIÓN" si)."""
    if not penalty:
        return 0.0
    levels = [
        re.sub(r"^\d{1,3}\.\s*", "", level).strip(" :.")
        for level in _plain(section).split(" — ")
    ]
    # Las filas de tabla llevan la condicion al final ("CONTROL DE CAMBIOS —
    # 06"): ahi la tabla es el primer nivel.
    candidates = [levels[-1], levels[0]] if levels[0].startswith("control de cambios") else [levels[-1]]
    query_plain = _plain(query)
    if _DOCUMENT_HEADER_RE.search(_plain(text)):
        return 0.0 if re.search(r"version|codigo|fecha", query_plain) else -penalty
    # Portada de un articulo externo (titulo, autores, resumen): no recomienda.
    if source_type != INSTITUTIONAL and page == 1:
        return -penalty
    for level in candidates:
        for prefix, keyword in _ADMIN_SECTIONS.items():
            if level.startswith(prefix):
                # Por la raiz: "¿como se clasifican...?" nombra el "Criterio
                # de clasificacion"; con la palabra exacta se excluia la tabla
                # de fuerza y calidad de la evidencia del protocolo.
                stem = keyword[:7]
                return 0.0 if stem in query_plain else -penalty
    return 0.0


_RECOMMENDATION_SECTION_RE = re.compile(r"^\d{1,3}\.\s")
# Desactivado por defecto (retrieval.section_boost en conf/config.yaml): se
# diseno para la GPC de NAC 2026, que numeraba sus preguntas clinicas
# ("13. En pacientes con..."). Esa guia ya no esta en el corpus, y en los
# protocolos institucionales lo numerado son secciones genericas ("1.
# OBJETIVO"), que con el impulso desplazaban a fragmentos relevantes.
_DEFAULT_SECTION_BOOST = 0.0
# El boost solo se aplica si el reranker YA le dio a este chunk una senal
# minima de relevancia real. Sin este piso, el boost termina inventando
# relevancia de la nada: para una pregunta totalmente fuera de tema (ej.
# tuberculosis, cuando el corpus solo tiene NAC), TODOS los chunks tienen
# score real cercano a 0 -- y si igual boosteamos las secciones numeradas
# "N. Pregunta clinica...", terminan ganando el top-k solo por su formato,
# no por ser relevantes, lo que le da al LLM contexto fuera de tema y lo
# empuja a alucinar (responder con conocimiento propio) en vez de decir que
# no hay informacion suficiente. Este piso evita ese efecto.
_MIN_SCORE_FOR_BOOST = 0.02


def _recommendation_boost(section: str, raw_score: float, boost: float = _DEFAULT_SECTION_BOOST) -> float:
    """Impulso heuristico para chunks de secciones tipo "N. Pregunta clinica...".

    En una GPC, esas secciones contienen la Recomendacion formal (grado de
    recomendacion, calidad de evidencia, farmaco/dosis exacta) -- el contenido
    normativo que un clinico necesita citar. Un reranker generico (no
    entrenado en dominio clinico) a veces prefiere parrafos narrativos de
    evidencia/discusion, que mencionan mas terminos de la pregunta pero no dan
    la recomendacion en si. Este boost corrige ese sesgo sin descartar al
    reranker: solo suma un margen fijo, no reemplaza su score, y solo cuando
    el reranker ya considero el chunk minimamente relevante (ver
    _MIN_SCORE_FOR_BOOST) -- nunca sobre un chunk que el reranker considero
    irrelevante.
    """
    if raw_score < _MIN_SCORE_FOR_BOOST:
        return 0.0
    return boost if _RECOMMENDATION_SECTION_RE.match(section) else 0.0


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Combina varias listas ordenadas en una sola, sin necesitar scores comparables."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best_chunk:
                best_chunk[key] = item
    fused = [
        RetrievedChunk(chunk=best_chunk[key].chunk, score=score, retrieval_method="hybrid")
        for key, score in scores.items()
    ]
    fused.sort(key=lambda rc: rc.score, reverse=True)
    return fused


def with_institutional_quota(
    ranked: list[RetrievedChunk],
    top_k: int,
    quota: int,
    min_score: float,
    min_final_score: float | None = None,
    max_per_external_document: int | None = None,
) -> list[RetrievedChunk]:
    """Los `top_k` mejores, reservando hasta `quota` lugares para los
    protocolos institucionales que el reranker considero minimamente
    relevantes (`min_score`). Si la pregunta no tiene que ver con los
    protocolos (p. ej. malaria), sus fragmentos no superan el minimo y no se
    fuerzan. El resultado conserva el orden por puntaje.

    `min_final_score`: puntaje minimo (ya penalizado) para entrar al contexto.
    Ante preguntas generales todos los puntajes son bajos y la penalizacion de
    secciones sin recomendaciones solo las reordenaba: "Necesidades de
    investigacion futura" o "Esta guia no aplica a" seguian entre los 8. Con
    el minimo quedan fuera, aunque el contexto tenga menos de 8 fragmentos.

    `max_per_external_document`: maximo de fragmentos de una misma fuente
    externa. En "enfoque para el manejo inicial de la NAC" la revision del
    Lancet ocupaba 5 de los 8 lugares (seguimiento a largo plazo incluido) y
    no entraba ningun fragmento de tratamiento. El protocolo institucional no
    tiene tope: sus respuestas suelen requerir varias secciones."""
    if min_final_score is not None:
        ranked = [rc for rc in ranked if rc.score >= min_final_score]
    if max_per_external_document:
        per_doc: dict[str, int] = {}
        capped = []
        for rc in ranked:
            if rc.chunk.source_type != INSTITUTIONAL:
                per_doc[rc.chunk.source_file] = per_doc.get(rc.chunk.source_file, 0) + 1
                if per_doc[rc.chunk.source_file] > max_per_external_document:
                    continue
            capped.append(rc)
        ranked = capped
    reserved = [
        rc
        for rc in ranked
        if rc.chunk.source_type == INSTITUTIONAL and rc.score >= min_score
    ][:quota]
    reserved_ids = {rc.chunk.chunk_id for rc in reserved}
    rest = [rc for rc in ranked if rc.chunk.chunk_id not in reserved_ids]
    chosen = reserved + rest[: max(top_k - len(reserved), 0)]
    return sorted(chosen, key=lambda rc: rc.score, reverse=True)


# Verbos de la pregunta ("cubrir", "realizan", "egresar", "maneja") que no
# estan en las guias no indican un tema fuera del corpus: solo cuentan los
# sustantivos (la enfermedad o el germen).
_VERB_SUFFIX_RE = re.compile(r"(ar|er|ir|an|en|ando|iendo|ado|ido)$")
_COMMON_VERBS = frozenset(
    """maneja manejo trata usa usar da dar hace hacer indica sospecha debe deben
    puede pueden sirve requiere""".split()  # noqa: SIM905 -- lista legible
)


def _looks_like_verb(word: str) -> bool:
    return word in _COMMON_VERBS or bool(_VERB_SUFFIX_RE.search(word))


def _content_words(query: str) -> list[str]:
    """Sustantivos de la pregunta que pueden nombrar un tema (no verbos ni
    palabras generales)."""
    return [
        w
        for w in _bm25_query_tokens(retrieval_query(query))
        if w not in _GENERIC_WORDS and len(w) >= 4 and not w.isdigit() and not _looks_like_verb(w)
    ]


class HybridRetriever:
    def __init__(self, cfg: DictConfig, embedder: OllamaEmbedder, store: QdrantStore):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []
        self._vocabulary: frozenset[str] = frozenset()
        self._doc_freq: Counter[str] = Counter()
        self._external_topics: dict[str, str] | None = None
        self._reranker = None  # se carga perezosamente, solo si se usa

    def build_bm25_index(self) -> None:
        """Reconstruye el indice BM25 en memoria a partir de lo que hay en Qdrant.

        Llamar despues de cada reindexacion (ver pipelines/feature_pipeline).
        Para corpus muy grandes esto se deberia persistir en disco en vez de
        reconstruirse en memoria en cada arranque; para el volumen tipico de
        GPC de una clinica (decenas/cientos de guias) es suficiente.
        """
        chunks = self.store.scroll_all_chunks()
        self._bm25_chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        per_chunk = [
            {*toks, *_tokenize(c.section)} for c, toks in zip(chunks, tokenized, strict=True)
        ]
        self._doc_freq = Counter(w for words in per_chunk for w in words)
        self._vocabulary = frozenset(self._doc_freq)
        logger.info("Indice BM25 construido con %d chunks.", len(chunks))

    def _bm25_search(
        self, query: str, top_k: int, source_type: str | None = None
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._bm25_chunks:
            return []
        scores = self._bm25.get_scores(_bm25_query_tokens(query))
        pairs = zip(self._bm25_chunks, scores, strict=True)
        if source_type:
            pairs = (pair for pair in pairs if pair[0].source_type == source_type)
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=chunk, score=float(score), retrieval_method="bm25")
            for chunk, score in ranked
            if score > 0
        ]

    @property
    def reranker(self):
        # Import perezoso: evita cargar torch/sentence-transformers si no se usa reranking.
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            # max_length acota el costo en CPU: el modelo admite hasta 8192
            # tokens por par, y los fragmentos son de ~400 (ver chunking).
            self._reranker = CrossEncoder(
                self.cfg.retrieval.reranker_model,
                device="cpu",
                max_length=self.cfg.retrieval.get("reranker_max_length", 1024),
            )
        return self._reranker

    def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, rc.chunk.text) for rc in candidates]
        raw_scores = [float(s) for s in self.reranker.predict(pairs)]
        retrieval = self.cfg.retrieval
        exclude_admin = bool(retrieval.get("exclude_admin_sections", False))
        reranked = []
        for rc, raw_score in zip(candidates, raw_scores, strict=True):
            penalty = _admin_penalty(
                rc.chunk.section,
                query,
                retrieval.get("admin_section_penalty", _DEFAULT_ADMIN_PENALTY),
                rc.chunk.text,
                rc.chunk.source_type,
                rc.chunk.page,
            )
            if exclude_admin and penalty < 0:
                # Seccion sin recomendaciones que la pregunta no nombra: no
                # entra al contexto (con solo restar puntaje, "Necesidades de
                # investigacion futura" o el resumen de la GPC seguian entre
                # los 8 porque el reranker les daba mas de 0.35).
                continue
            boost = _recommendation_boost(
                rc.chunk.section,
                raw_score,
                retrieval.get("section_boost", _DEFAULT_SECTION_BOOST),
            )
            reranked.append(
                RetrievedChunk(
                    chunk=rc.chunk, score=raw_score + boost + penalty, retrieval_method="rerank"
                )
            )
        reranked.sort(key=lambda rc: rc.score, reverse=True)
        return reranked[:top_k]

    def _institutional_candidates(
        self, query: str, query_vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        """Mejores candidatos solo entre los protocolos institucionales.

        Con las guias y revisiones externas el indice paso de ~70 a ~850
        fragmentos, y los de la CUB podian quedar fuera de los 12 primeros de
        la busqueda densa y de BM25: el protocolo institucional ni siquiera
        llegaba al reranker."""
        retrieval = self.cfg.retrieval
        dense = self.store.search(
            query_vector, top_k=retrieval.top_k_dense, source_type=INSTITUTIONAL
        )
        bm25 = self._bm25_search(query, top_k=retrieval.top_k_bm25, source_type=INSTITUTIONAL)
        return _reciprocal_rank_fusion([dense, bm25])[:limit]

    def out_of_corpus_terms(self, query: str) -> list[str]:
        """Palabras de la pregunta que no aparecen en ningun documento indexado
        y que la pregunta no ancla a un tema del corpus (retrieval.topic_terms).

        El reranker no distingue la enfermedad: para "tratamiento recomendado
        para la malaria en adultos" daba 0.49 a "antibiotico recomendado para
        el tratamiento ... en pacientes adultos con NAC", y el RAG respondia
        sobre NAC. "malaria" no aparece en ninguna guia: si la pregunta no
        nombra ademas un tema del corpus (NAC, ITU...), no hay nada que
        responder. Con un tema presente ("pielonefritis en paciente con
        lupus") se responde lo que haya del tema.
        """
        if not self._vocabulary:
            return []
        if self.question_topics(query):
            return []
        return [w for w in _content_words(query) if w not in self._vocabulary]

    def _topic_words(self) -> dict[str, set[str]]:
        """{tema: palabras} de retrieval.topics (o retrieval.topic_terms, sin
        separar por tema)."""
        retrieval = self.cfg.retrieval
        topics = retrieval.get("topics")
        if topics:
            return {
                str(name): {t for term in terms for t in _tokenize(str(term))}
                for name, terms in topics.items()
            }
        return {"corpus": {t for term in retrieval.get("topic_terms", []) for t in _tokenize(str(term))}}

    def question_topics(self, query: str) -> set[str]:
        tokens = set(_tokenize(retrieval_query(query)))
        return {name for name, words in self._topic_words().items() if words & tokens}

    def chunk_topic(self, chunk: Chunk) -> str | None:
        """Tema de un fragmento: el de su protocolo (retrieval.institutional_topics,
        por prefijo del archivo) o el de data/01_raw/externas/fuentes.yaml."""
        retrieval = self.cfg.retrieval
        if chunk.source_type == INSTITUTIONAL:
            for prefix, topic in (retrieval.get("institutional_topics") or {}).items():
                if chunk.source_file.startswith(str(prefix)):
                    return str(topic)
            return None
        if self._external_topics is None:
            directory = Path(retrieval.get("external_catalog_dir", "data/01_raw/externas"))
            try:
                catalog = load_catalog(directory)
            except (OSError, ValueError):
                logger.warning("Sin catalogo de fuentes externas en %s: no se filtra por tema.", directory)
                catalog = {}
            self._external_topics = {key: info.topic for key, info in catalog.items()}
        return self._external_topics.get(_file_key(chunk.source_file))

    def same_topic(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Si la pregunta nombra un solo tema, deja fuera los fragmentos de otro
        (los de tema desconocido se conservan). En "¿en que casos de infeccion
        urinaria se deben solicitar hemocultivos?" entraban las
        recomendaciones de hemocultivos de la GPC de neumonia."""
        topics = self.question_topics(query)
        if len(topics) != 1:
            return candidates
        return [rc for rc in candidates if self.chunk_topic(rc.chunk) in (None, *topics)]

    def uncovered_terms(self, question: str, answer: str) -> list[str]:
        """Palabras poco frecuentes de la pregunta que la respuesta no menciona,
        cuando la pregunta no nombra un tema del corpus.

        "meningitis" aparece en algun documento, asi que el filtro de terminos
        fuera del corpus no actuaba; el selector eligio recomendaciones de
        neumonia y el evaluador las aprobo. Si la respuesta no menciona la
        enfermedad por la que se pregunto, no la responde."""
        if not self._doc_freq or self.question_topics(question):
            return []
        limit = max(3, int(len(self._bm25_chunks) * float(self.cfg.retrieval.get("rare_term_ratio", 0.01))))
        rare = [w for w in _content_words(question) if 0 < self._doc_freq.get(w, 0) <= limit]
        answered = set(_tokenize(answer))
        return [w for w in rare if w not in answered]


    def retrieve(self, query: str) -> list[RetrievedChunk]:
        retrieval = self.cfg.retrieval
        query = retrieval_query(query)
        query_vector = self.embedder.embed_query(query)
        dense_results = self.store.search(query_vector, top_k=retrieval.top_k_dense)
        bm25_results = self._bm25_search(query, top_k=retrieval.top_k_bm25)

        fused = _reciprocal_rank_fusion([dense_results, bm25_results])
        quota = int(retrieval.get("institutional_quota", 0))

        if retrieval.use_reranker and fused:
            candidate_pool = fused[: max(retrieval.top_k_final * 3, 10)]
            if quota:
                seen = {rc.chunk.chunk_id for rc in candidate_pool}
                candidate_pool += [
                    rc
                    for rc in self._institutional_candidates(query, query_vector, quota * 2)
                    if rc.chunk.chunk_id not in seen
                ]
            candidate_pool = self.same_topic(query, candidate_pool)
            ranked = self._rerank(query, candidate_pool, len(candidate_pool))
            return with_institutional_quota(
                ranked,
                retrieval.top_k_final,
                quota,
                float(retrieval.get("institutional_min_score", 0.05)),
                retrieval.get("min_final_score"),
                retrieval.get("max_chunks_per_external_document"),
            )

        return fused[: retrieval.top_k_final]
