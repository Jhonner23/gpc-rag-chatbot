# -*- coding: utf-8 -*-
"""Evalua el pipeline RAG completo con RAGAS, usando Ollama local como juez.

Corre cada pregunta del golden dataset (eval/golden_dataset.json) contra el
pipeline REAL (retrieval + generacion, tal como lo usan la API y el
chatbot), arma un dataset compatible con ragas.evaluate y calcula:
faithfulness, answer_relevancy, context_precision, context_recall.

Importante: esto NO modifica retrieval/generacion ni toca los procesos de
la API/chatbot que ya estan corriendo -- solo instancia su propio
QueryPipeline (un proceso Python aparte) y le hace preguntas, igual que
haria un usuario real. Apunta a la misma Qdrant y al mismo Ollama, asi que
usa el indice ya construido sin reindexar nada.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from gpc_rag.pipelines.query_pipeline import QueryPipeline

# Se puede sobreescribir con una variable de entorno si tienes un modelo mas
# grande descargado (ej. RAGAS_JUDGE_MODEL=qwen2.5:7b-instruct), sin tocar
# codigo. Un juez mas grande suele dar evaluaciones mas confiables.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "qwen2.5:3b-instruct")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL", "bge-m3")


def load_golden_dataset(path: str = "eval/golden_dataset.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pipeline(golden: list[dict]) -> dict:
    """Le hace cada pregunta al pipeline real y recolecta respuesta + contexto usado."""
    pipeline = QueryPipeline()

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []
    refusal_checks: list[dict] = []

    for i, item in enumerate(golden, start=1):
        print(f"  [{i}/{len(golden)}] {item['question'][:70]}...")
        result = pipeline.answer(item["question"])

        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([rc.chunk.text for rc in result.sources])
        ground_truths.append(item["ground_truth"])

        if item.get("should_refuse"):
            answer_lower = result.answer.lower()
            refused = any(
                phrase in answer_lower
                for phrase in [
                    "no encontr",
                    "no cubr",
                    "no teng",
                    "no dispon",
                    "informacion suficiente",
                    "información suficiente",
                ]
            )
            refusal_checks.append(
                {
                    "question": item["question"],
                    "answer": result.answer,
                    "refused_correctly": refused,
                }
            )

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "refusal_checks": refusal_checks,
    }


def main() -> None:
    golden = load_golden_dataset()
    print(f"Corriendo el pipeline real sobre {len(golden)} preguntas (puede tardar varios minutos)...")
    collected = run_pipeline(golden)

    dataset = Dataset.from_dict(
        {
            "question": collected["questions"],
            "answer": collected["answers"],
            "contexts": collected["contexts"],
            "ground_truth": collected["ground_truths"],
        }
    )

    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL, temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    )

    print(f"\nEvaluando con RAGAS (juez local: {JUDGE_MODEL})...")
    # max_workers bajo: Ollama local (CPU) no puede atender muchas peticiones del
    # juez en paralelo -- con concurrencia alta se acumulan y expiran (TimeoutError),
    # lo que arruina las metricas (se ven como 0.0 o nan sin serlo realmente).
    run_config = RunConfig(timeout=240, max_workers=2, max_retries=2)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()
    out_csv = Path("eval/ragas_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Resumen RAGAS (promedio sobre todas las preguntas) ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.3f}")

    print("\n=== Casos de rechazo (preguntas fuera de alcance) ===")
    for check in collected["refusal_checks"]:
        status = "OK" if check["refused_correctly"] else "FALLO -- no rechazo correctamente"
        print(f"  [{status}] {check['question']}")
        if not check["refused_correctly"]:
            print(f"      Respuesta obtenida: {check['answer'][:200]}...")

    print(f"\nResultados detallados (por pregunta) guardados en {out_csv}")


if __name__ == "__main__":
    main()
