# -*- coding: utf-8 -*-
"""Construye el dataset dorado (golden dataset) para evaluar el RAG.

Combina:
1. Las preguntas clinicas numeradas de la GPC, con la respuesta esperada
   redactada a partir de la Recomendacion formal (ver eval/qa_candidates.json,
   generado por scripts/extract_qa_candidates.py).
2. Casos ya probados manualmente en el chatbot real (definicion, diagnostico,
   y el caso fuera de alcance de tuberculosis, que debe generar un rechazo).

Cada entrada trae "question", "ground_truth" (respuesta esperada) y
"expected_section" (para verificar que la cita apunte a la seccion correcta).
Un campo "should_refuse": true marca preguntas que el chatbot NO debe
responder con contenido medico (fuera del alcance de las guias indexadas).
"""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN_DATASET = [
    {
        "question": (
            "En pacientes adultos con sospecha de NAC, ¿debe considerarse la "
            "tomografía computarizada de tórax como alternativa diagnóstica a "
            "la radiografía de tórax?"
        ),
        "ground_truth": (
            "No se recomienda usar la tomografía computarizada de tórax de "
            "rutina para confirmar el diagnóstico de NAC (Fuerte, Moderada "
            "calidad). Se sugiere reservar el uso de la TC para situaciones "
            "específicas: cuando la radiografía es negativa o equívoca pero "
            "persiste alta sospecha clínica de NAC, en pacientes con "
            "evolución atípica, comorbilidades significativas, "
            "inmunosupresión, o cuando el diagnóstico preciso es crítico "
            "para el manejo inmediato."
        ),
        "expected_section": "3.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC, ¿debe obtenerse "
            "una tinción de Gram y cultivo de esputo en el momento del "
            "diagnóstico?"
        ),
        "ground_truth": (
            "No se recomienda realizar tinción de Gram y cultivo de esputo "
            "de forma rutinaria en pacientes con NAC que requieren manejo "
            "ambulatorio (Fuerte, Muy baja calidad). En pacientes con NAC "
            "que requieren manejo intrahospitalario, se recomienda realizar "
            "tinción de Gram y cultivo de esputo previo al inicio de "
            "antibióticos: en NAC moderada (Débil, Muy baja calidad) y en "
            "NAC grave (Fuerte, Muy baja calidad)."
        ),
        "expected_section": "4.",
        "should_refuse": False,
    },
    {
        "question": (
            "En adultos con diagnóstico NAC, ¿debe utilizarse la "
            "procalcitonina para decidir el inicio y la duración del "
            "tratamiento antibiótico?"
        ),
        "ground_truth": (
            "No se recomienda usar procalcitonina sérica como criterio "
            "aislado para decidir el inicio de antibióticos en pacientes "
            "adultos con NAC (Fuerte, Moderada calidad). Se sugiere usar "
            "procalcitonina junto con la evolución clínica para reducir la "
            "duración del tratamiento antibiótico (Débil, Baja calidad)."
        ),
        "expected_section": "9.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC, ¿debe utilizarse "
            "una regla clínica de predicción pronóstica junto con el "
            "juicio clínico para determinar el sitio de atención?"
        ),
        "ground_truth": (
            "Se recomienda usar el juicio clínico acompañado de una regla "
            "de predicción pronóstica validada (PSI, CURB-65 o CRB-65) para "
            "decidir entre manejo ambulatorio u hospitalario (Fuerte, "
            "Moderada calidad). Se recomienda admisión directa a UCI en "
            "pacientes con hipotensión que requiere vasopresores o "
            "insuficiencia respiratoria que requiere ventilación mecánica "
            "(Fuerte, Moderada calidad). Se sugiere emplear SMART-COP como "
            "complemento."
        ),
        "expected_section": "10.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC que requieren "
            "manejo ambulatorio, ¿cuál es el antibiótico recomendado para "
            "el tratamiento empírico?"
        ),
        "ground_truth": (
            "Se recomienda el uso de antibióticos orales como amoxicilina "
            "1 g cada 8 horas, claritromicina 500 mg cada 12 horas (o 1 g "
            "en dosis única diaria en presentaciones que lo permitan), o "
            "doxiciclina 100 mg cada 12 horas, en pacientes adultos con NAC "
            "que requieren manejo ambulatorio sin factores de riesgo para "
            "SAMR o P. aeruginosa (Fuerte, Moderada calidad)."
        ),
        "expected_section": "11.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC moderada que "
            "requieren manejo hospitalario sin factores de riesgo para "
            "SAMR o P. aeruginosa, ¿cuál es el esquema antibiótico "
            "empírico recomendado?"
        ),
        "ground_truth": (
            "Se recomienda terapia empírica con un betalactámico como "
            "ampicilina/sulbactam o ceftriaxona en monoterapia en "
            "pacientes adultos con NAC que requieren manejo "
            "intrahospitalario sin criterios de gravedad y sin factores de "
            "riesgo para SAMR o P. aeruginosa (Fuerte, Moderada calidad). "
            "Se recomienda ajustar el tratamiento una vez se cuente con "
            "aislamientos microbiológicos."
        ),
        "expected_section": "12.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC grave que "
            "requieren manejo hospitalario sin factores de riesgo para "
            "SAMR o P. aeruginosa, ¿cuál es el esquema antibiótico "
            "empírico recomendado?"
        ),
        "ground_truth": (
            "Se recomienda terapia con un betalactámico como "
            "ampicilina/sulbactam o ceftriaxona en combinación con "
            "claritromicina en adultos con diagnóstico de NAC grave "
            "(Fuerte, Moderada calidad)."
        ),
        "expected_section": "13.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC que requieren "
            "manejo hospitalario con factores de riesgo para SAMR o P. "
            "aeruginosa, ¿debe iniciarse tratamiento antibiótico de "
            "espectro ampliado en lugar del régimen estándar?"
        ),
        "ground_truth": (
            "Se recomienda iniciar cobertura empírica para SAMR en "
            "pacientes con NAC moderada o grave que presenten aislamiento "
            "respiratorio previo de SAMR o factores de riesgo clínicos "
            "(Fuerte, Baja calidad). Se recomienda iniciar tratamiento con "
            "vancomicina o linezolid según disponibilidad y perfil "
            "clínico, y realizar cultivo o PCR para confirmar o descartar "
            "la necesidad de continuar la terapia (Fuerte, Baja calidad)."
        ),
        "expected_section": "14.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC que requieren "
            "manejo hospitalario y sospecha de neumonía por aspiración, "
            "¿debe añadirse cobertura anaerobia al tratamiento empírico "
            "estándar?"
        ),
        "ground_truth": (
            "Se sugiere no adicionar cobertura anaerobia de forma "
            "rutinaria en neumonía por aspiración, excepto en pacientes "
            "con sospecha de absceso pulmonar o empiema (Débil, Muy baja "
            "calidad)."
        ),
        "expected_section": "15.",
        "should_refuse": False,
    },
    {
        "question": (
            "En pacientes adultos con diagnóstico de NAC, ¿cuál es la vía "
            "de administración adecuada para el tratamiento antimicrobiano?"
        ),
        "ground_truth": (
            "Se recomienda iniciar antimicrobianos orales en pacientes con "
            "NAC no grave que toleran la vía oral (Fuerte, Moderada "
            "calidad). En pacientes que inician tratamiento endovenoso, se "
            "recomienda reevaluar a las 48 horas para considerar el cambio "
            "de antibiótico intravenoso a oral (Fuerte, Moderada calidad)."
        ),
        "expected_section": "16.",
        "should_refuse": False,
    },
    {
        "question": "¿Cuál es el tratamiento de la tuberculosis pulmonar?",
        "ground_truth": (
            "Las guías indexadas cubren únicamente neumonía adquirida en "
            "la comunidad (NAC), no tuberculosis. El chatbot debe indicar "
            "explícitamente que no cuenta con información suficiente sobre "
            "ese tema en las guías disponibles, sin responder con "
            "conocimiento médico general ni citar fuentes externas."
        ),
        "expected_section": None,
        "should_refuse": True,
    },
]


def main() -> None:
    out_path = Path("eval/golden_dataset.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(GOLDEN_DATASET, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(GOLDEN_DATASET)} preguntas doradas escritas en {out_path}")


if __name__ == "__main__":
    main()
