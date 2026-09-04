"""Evaluacion del pipeline RAG con RAGAS: Faithfulness, Answer Relevancy,
Context Precision.

Uso:
    uv run python -m gpc_rag.evaluation.ragas_eval --dataset data/05_model_input/preguntas_evaluacion.jsonl

El dataset es un JSONL con una pregunta por linea, idealmente validado por un
clinico de CUB:
    {"question": "...", "ground_truth": "respuesta esperada"}

Corre cada pregunta por el pipeline real (retrieval + generacion) y calcula
las metricas comparando contra el contexto recuperado y la respuesta
generada. Guarda un reporte CSV para comparar entre versiones del pipeline
(ej. antes/despues de agregar reranking).
"""

# ruff: noqa: PLC0415
# Los imports de ragas/langchain/datasets se hacen dentro de la funcion (no al
# nivel del modulo) a proposito: son pesados y este modulo solo se importa de
# verdad cuando corres una evaluacion, no cada vez que arranca la API/chatbot.

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    # Imports pesados (ragas/datasets/langchain) solo cuando se evalua, no al importar el modulo.
    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    from gpc_rag.common.settings import load_config
    from gpc_rag.pipelines.query_pipeline import get_pipeline

    items = load_eval_dataset(dataset_path)
    if not items:
        logger.warning("El dataset de evaluacion %s esta vacio.", dataset_path)
        return

    pipeline = get_pipeline()
    cfg = load_config()

    # IMPORTANTE: RAGAS usa un LLM como "juez" para calcular las metricas. Por
    # defecto intenta usar OpenAI (de pago) -- aqui lo apuntamos al mismo
    # Ollama local para que la evaluacion tambien sea 100% gratis/on-premise.
    judge_llm = LangchainLLMWrapper(ChatOllama(model=cfg.llm.model, base_url=cfg.env.ollama_url))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=cfg.embeddings_cfg.model, base_url=cfg.env.ollama_url)
    )

    questions, answers, contexts, ground_truths = [], [], [], []
    for item in items:
        result = pipeline.answer(item["question"])
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([rc.chunk.text for rc in result.sources])
        ground_truths.append(item.get("ground_truth", ""))

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    scores = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = scores.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Reporte guardado en %s", output_path)
    logger.info("Promedios:\n%s", df[["faithfulness", "answer_relevancy", "context_precision"]].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el pipeline RAG con RAGAS.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL con preguntas de evaluacion.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/08_reporting/ragas_report.csv"),
        help="Donde guardar el CSV con los resultados.",
    )
    args = parser.parse_args()
    run_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()
