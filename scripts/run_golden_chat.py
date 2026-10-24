"""Corre el banco de preguntas contra la API (/chat) y guarda cada respuesta.

Uso (con la API levantada en otra terminal):

    PYTHONPATH=src uv run python scripts/run_golden_chat.py
    PYTHONPATH=src uv run python scripts/run_golden_chat.py --banco eval/preguntas_validacion.json --sin-rondas

eval/preguntas_validacion.json son preguntas que no se usaron para ajustar el
sistema: miden si las correcciones sirven fuera de las preguntas de prueba.

Escribe eval/resultados/respuestas_<fecha>.jsonl, una linea por pregunta, a
medida que avanza: si la corrida se interrumpe, lo ya respondido queda
guardado y `--continuar <archivo>` retoma desde la siguiente pregunta. Los
resultados traen texto de los protocolos (uso exclusivo de la CUB) y no se
versionan (ver .gitignore).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

BANK = Path("eval/golden_protocolos.json")
OUT_DIR = Path("eval/resultados")
API_URL = "http://localhost:8000/chat"
TIMEOUT_SECONDS = 900.0

# Preguntas de las rondas de prueba con la asesora (ademas del banco).
EXTRA_QUESTIONS = [
    (
        (
            "¿Qué recomendaciones presentan las guías disponibles sobre el manejo de la "
            "bacteriuria asintomática? ¿Qué diferencias existen entre las recomendaciones "
            "encontradas para el manejo de esta condición? ¿Qué fuentes del corpus contienen "
            "información relacionada con esta recomendación?"
        ),
        False,
    ),
    ("¿En qué situaciones una bacteriuria asintomática debe ser tratada y en cuáles no?", False),
    ("¿Qué factores deben considerarse al decidir el manejo de un paciente con neumonía?", False),
    ("¿Cuál es el tratamiento recomendado para la malaria en adultos?", True),
    (
        (
            "¿Cuál es el enfoque recomendado para el manejo inicial de la neumonía adquirida "
            "en la comunidad?"
        ),
        False,
    ),
]


def _questions(bank_path: Path = BANK, with_rounds: bool = True) -> list[dict]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    items = [
        {"origen": bank_path.stem, "question": q["question"], "should_refuse": q["should_refuse"]}
        for q in bank
    ]
    if with_rounds:
        items += [
            {"origen": "ronda", "question": q, "should_refuse": refuse}
            for q, refuse in EXTRA_QUESTIONS
        ]
    return [{"n": i, **item} for i, item in enumerate(items, 1)]


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["n"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--continuar", type=Path, help="archivo .jsonl de una corrida a retomar")
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--banco", type=Path, default=BANK, help="archivo .json de preguntas")
    parser.add_argument(
        "--sin-rondas", action="store_true", help="no agregar las 5 preguntas de las rondas de prueba"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.continuar or OUT_DIR / f"respuestas_{datetime.now():%Y%m%d_%H%M}.jsonl"
    done = _done(out)
    questions = [q for q in _questions(args.banco, not args.sin_rondas) if q["n"] not in done]
    print(f"{len(questions)} preguntas por correr -> {out}")

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client, out.open("a", encoding="utf-8") as fh:
        for q in questions:
            start = time.perf_counter()
            try:
                resp = client.post(args.url, json={"question": q["question"]})
                resp.raise_for_status()
                result, error = resp.json(), None
            except httpx.HTTPError as exc:
                result, error = {}, f"{type(exc).__name__}: {exc}"
            seconds = round(time.perf_counter() - start, 1)
            record = {**q, "segundos": seconds, "error": error, "respuesta": result}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            states = {p["agent"]: p["status"] for p in result.get("panels", [])}
            print(f"[{q['n']:>2}] {seconds:>6.1f} s  {states or error}  {q['question'][:70]}")

    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
