"""Construccion del prompt para el LLM de generacion.

Reglas clave para un contexto clinico:
1. Responder SOLO con base en el contexto recuperado (nunca "inventar" con
   conocimiento general del modelo) -- reduce alucinaciones peligrosas.
2. Citar siempre la guia/seccion/pagina de donde sale cada afirmacion.
3. Decir explicitamente que no hay informacion suficiente si el contexto no
   responde la pregunta, en vez de improvisar.
4. No solo farmacos/dosis: cualquier criterio clinico enumerado (variables de
   una escala, umbrales numericos, nombres de instrumentos) debe salir
   TEXTUALMENTE del CONTEXTO, no de lo que el modelo "recuerda" sobre el
   instrumento -- ver el caso documentado abajo.

Caso real que motivo la regla 4 (probado en produccion con qwen2.5:3b-instruct
sobre la pregunta "cuales son los criterios de CURB-65"): el CONTEXTO
recuperado SI traia la tabla exacta de la GPC (Confusion=1, Urea >19 mg/dL=1,
Frecuencia respiratoria >=30 rpm=1, Presion baja=1, Edad >=65=1), pero el
modelo igual respondio con datos inventados que coinciden con la
"forma" del acronimo CURB-65 en vez de su contenido real citado: convirtio la
"U" (Urea) en "incontinencia Urinaria", omitio la "R" (frecuencia
Respiratoria) por completo, y cambio el umbral de edad de >=65 a >75 anos.
Es decir, tener el fragmento correcto en el CONTEXTO no fue suficiente -- el
modelo parafraseo con su propio conocimiento general del acronimo en vez de
citar lo que decia el fragmento. La regla 4 existe para cerrar exactamente
ese hueco, que la regla original (solo farmacos/dosis) no cubria.
"""

from __future__ import annotations

from gpc_rag.common.models import RetrievedChunk

SYSTEM_PROMPT = """\
Eres un asistente clinico que responde preguntas sobre Guias de Practica \
Clinica (GPC) usando UNICAMENTE la informacion del CONTEXTO proporcionado.

Reglas estrictas:
- No uses conocimiento medico propio ni supongas informacion que no este en \
el contexto.
- CRITICO: nunca menciones un nombre de farmaco, dosis, esquema posologico o \
cifra especifica a menos que aparezca TEXTUALMENTE en el CONTEXTO. Si el \
contexto solo habla de una clase de farmaco en general (ej. "un \
betalactamico") pero no da el nombre exacto, no lo completes de memoria: di \
que el contexto no especifica el farmaco exacto.
- CRITICO: esta misma regla aplica a CUALQUIER dato especifico, no solo \
farmacos -- variables/criterios de una escala o instrumento clinico (ej. \
CURB-65, PSI, IDSA/ATS), sus umbrales numericos (ej. "edad >= 65 anos", \
"frecuencia respiratoria >= 30"), y el significado de cada sigla/letra de un \
acronimo. NUNCA completes estos datos desde tu conocimiento general del \
instrumento, aunque te "suenen" correctos o el acronimo te parezca \
familiar -- copia (parafraseando solo el estilo, no el contenido) \
UNICAMENTE lo que el fragmento del CONTEXTO dice explicitamente. Si varios \
fragmentos del CONTEXTO mencionan el mismo instrumento, usa el que traiga la \
lista completa (ej. una tabla) en vez de mezclar datos de varios fragmentos \
o de tu memoria. Si ningun fragmento trae la lista completa de \
criterios/variables que pide la pregunta, dilo explicitamente en vez de \
completar los que falten.
- CRITICO: la unica fuente valida es el CONTEXTO proporcionado abajo. NUNCA \
cites, menciones ni inventes fuentes externas -- ni organizaciones (CDC, \
OMS/WHO, sociedades medicas, etc.), ni URLs, ni articulos, ni "fuentes \
academicas" en general. Si necesitas citar algo, solo puede ser con el \
formato [Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>] de un \
fragmento del CONTEXTO. Citar cualquier otra fuente esta prohibido, incluso \
si la informacion parece correcta.
- CRITICO: antes de responder, verifica si el CONTEXTO realmente trata el \
tema de la pregunta (misma enfermedad/condicion). Si la pregunta es sobre un \
tema, enfermedad o condicion distinta a la que cubre el CONTEXTO (por \
ejemplo, preguntan por tuberculosis y el CONTEXTO es sobre neumonia \
adquirida en la comunidad), NO respondas con conocimiento general: di \
explicitamente que las guias disponibles no cubren ese tema.
- Si el contexto no tiene informacion suficiente para responder con \
seguridad, dilo explicitamente: "No encuentro informacion suficiente en las \
guias disponibles para responder esto con seguridad." No inventes una \
respuesta.
- Cita la fuente de cada afirmacion relevante usando el formato \
[Guia: <archivo>, Seccion: <seccion>, Pagina: <pagina>].
- Se claro, conciso y usa terminologia clinica precisa.
- Esta respuesta apoya, pero NO reemplaza, el juicio clinico profesional.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        header = (
            f"[Fragmento {i} | Guia: {c.source_file} | Seccion: {c.section} | Pagina: {c.page}]"
        )
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks) if blocks else "(sin contexto relevante encontrado)"


def build_messages(
    question: str, chunks: list[RetrievedChunk], extra_instruction: str | None = None
) -> list[dict]:
    context = format_context(chunks)
    user_prompt = (
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA DEL USUARIO:\n{question}\n\n"
        "Responde siguiendo las reglas del sistema, citando las fuentes usadas."
    )
    if extra_instruction:
        # Usado por el agente evaluador (ver agents/graph.py) para pedir un
        # reintento mas estricto cuando la primera respuesta no paso su
        # auditoria -- no cambia las reglas base, solo refuerza una de ellas.
        user_prompt += f"\n\nINSTRUCCION ADICIONAL:\n{extra_instruction}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
