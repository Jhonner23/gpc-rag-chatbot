"""Verificacion de fidelidad de citas contra el texto real de un PDF.

Extraido de `scripts/verify_tree_citations.py` para poder reusarlo tambien
desde `trees/extraction.py` (el agente de extraccion asistida por LLM, ver
seccion 10.6 de la arquitectura): antes de aceptar una cita -- la haya
escrito un script generador a mano o la haya propuesto un LLM -- se verifica
con la misma logica que los terminos clinicos y numericos distintivos de la
cita realmente aparecen en la pagina citada del documento.

No es una comparacion de substring exacto: los `source_quote` del bosque son
resumenes compactos de tablas (ej. "Confusion | CURB-65: 1 | CRB-65: 1"), no
texto copiado literal -- extraer una tabla de un PDF reordena filas/columnas,
asi que un string identico no existe en el documento. En su lugar, se extrae
el texto real de la pagina y se verifica que los terminos clinicamente
relevantes de la cita (numeros, umbrales, terminos medicos) aparezcan ahi.
Es el patron de "fidelidad de extraccion, no re-validacion clinica" de la
seccion 10.1: no se re-evalua si el criterio es correcto clinicamente (eso
ya lo aprobo la clinica al proveer el documento) -- solo si la cita
representa fielmente lo que el documento dice.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import fitz  # pymupdf

# Umbral de coincidencia de terminos para considerar una cita fiel. No es
# 100% porque el propio texto de redaccion de la cita (conectores, "Nota:",
# etc.) no tiene por que aparecer literal en el PDF -- lo que importa es que
# los terminos clinicos y numericos distintivos si esten.
UMBRAL_COINCIDENCIA = 0.6

# Un token cuenta como "termino clinico distintivo" si tiene al menos esta
# longitud (evita ruido de palabras cortas que no son stopwords pero tampoco
# aportan senal, como abreviaturas truncadas por la tokenizacion).
LONGITUD_MINIMA_TERMINO = 4

# La linea (0-indexed=1, es decir la segunda) donde esta plantilla de
# revista imprime el numero de pagina -- ver docstring de build_page_map.
LINEAS_MINIMAS_PARA_NUMERO_PAGINA = 2

_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "en", "y", "o", "a", "que", "con", "no",
    "por", "si", "un", "una", "al", "para", "nota", "tabla", "pag", "pagina",
}

# El PDF usa simbolos unicode (≤, ≥) y subindices (PaO₂, FiO₂) que no tienen
# equivalente ASCII directo -- sin esto, terminos que SI estan en el
# documento se reportarian como "no encontrados" solo por la codificacion,
# lo que seria un falso positivo del verificador, no un error real del
# arbol. Descubierto corriendo este mismo verificador contra el PDF real.
_REEMPLAZOS_SIMBOLOS = {
    "≤": "<=",
    "≥": ">=",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}


def normalizar(texto: str) -> str:
    for simbolo, reemplazo in _REEMPLAZOS_SIMBOLOS.items():
        texto = texto.replace(simbolo, reemplazo)
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return sin_acentos.lower()


def terminos_relevantes(quote: str) -> set[str]:
    """Extrae los terminos clinicos/numericos distintivos de una cita.

    Quita las anotaciones tipo "(Tabla 3, pag. 199)" y tokeniza -- se quedan
    palabras de 4+ letras (terminos clinicos) y tokens numericos/porcentajes
    (los umbrales son lo mas facil de verificar y lo mas importante de no
    desalinear).
    """
    sin_anotaciones = re.sub(r"\([^()]*\)", "", quote)
    normalizado = normalizar(sin_anotaciones)
    tokens = re.findall(r"[a-z0-9%/.<>=+-]+", normalizado)
    return {
        t
        for t in tokens
        if t not in _STOPWORDS
        and (len(t) >= LONGITUD_MINIMA_TERMINO or any(ch.isdigit() for ch in t))
    }


def build_page_map(doc: fitz.Document) -> dict[int, int]:
    """Mapea numero de pagina IMPRESO (el que aparece en las citas) -> indice PDF (0-based).

    El PDF suele traer paginas de portada/indice sin numerar antes de que
    empiece la numeracion impresa del articulo, asi que el indice de pymupdf
    y el numero de pagina citado en las guias casi nunca coinciden. No basta
    con buscar "un numero solo en su propia linea" en cualquier parte de la
    pagina: las tablas clinicas (CURB-65, PSI, etc.) tienen celdas de una
    sola cifra que tambien matchean ese patron. En esta plantilla de revista
    el numero de pagina esta siempre en la segunda linea del texto extraido
    (justo despues del encabezado de autor/titulo), asi que se busca solo
    ahi -- si el layout de otra guia fuera distinto, esta funcion es el
    unico lugar que hay que ajustar.
    """
    mapa: dict[int, int] = {}
    for idx in range(len(doc)):
        lineas = doc[idx].get_text().strip().splitlines()
        if len(lineas) < LINEAS_MINIMAS_PARA_NUMERO_PAGINA:
            continue
        segunda_linea = lineas[1].strip()
        if re.fullmatch(r"\d{1,4}", segunda_linea):
            mapa[int(segunda_linea)] = idx
    return mapa


@dataclass(frozen=True)
class ResultadoCobertura:
    """Que tan bien una cita esta respaldada por el texto real de la pagina."""

    cobertura: float  # 0.0 - 1.0
    terminos_encontrados: set[str]
    terminos_faltantes: set[str]

    @property
    def es_fiel(self) -> bool:
        return self.cobertura >= UMBRAL_COINCIDENCIA


def cobertura_cita(quote: str, texto_pagina: str) -> ResultadoCobertura:
    """Calcula que fraccion de los terminos distintivos de `quote` aparece en `texto_pagina`.

    `texto_pagina` debe venir ya normalizado (ver `normalizar`) -- se pasa
    normalizado en vez de normalizarlo aqui porque el caller normalmente
    reusa el mismo texto de pagina para varias citas y no tiene sentido
    normalizarlo una y otra vez.
    """
    terminos = terminos_relevantes(quote)
    if not terminos:
        return ResultadoCobertura(cobertura=1.0, terminos_encontrados=set(), terminos_faltantes=set())
    encontrados = {t for t in terminos if t in texto_pagina}
    faltantes = terminos - encontrados
    return ResultadoCobertura(
        cobertura=len(encontrados) / len(terminos),
        terminos_encontrados=encontrados,
        terminos_faltantes=faltantes,
    )
