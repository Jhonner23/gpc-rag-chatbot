# Bosque de árboles de decisión (`trees/`)

Cada árbol de decisión del proyecto es **reproducible desde cero clonando el repo**, sin depender de ninguna conversación previa: existe un script generador versionado en `scripts/` que reconstruye el JSON exacto a partir de las citas textuales del PDF fuente, y un script de verificación que confirma esas citas contra tu propia copia del PDF. El JSON en `data/` no se edita a mano — si hay que corregir o extender un árbol, se edita su script generador y se vuelve a correr.

Las GPC en PDF **no están versionadas en este repo** (ver `.gitignore`), así que "reproducible" aquí significa: clonas el repo, pones tu propia copia de la guía en una carpeta local, y los dos scripts de abajo te dan (a) el árbol exacto y (b) la confirmación de que sus citas corresponden a tu copia del documento — sin necesitar nada más.

## Contrato de reproducibilidad

1. Cada árbol tiene un script `scripts/gen_<nombre>_tree.py` que escribe su JSON en `data/<gpc>/<archivo>.json`.
2. Correr el script debe producir **siempre el mismo archivo**, byte a byte (son datos estáticos derivados de citas fijas del PDF, no hay nada probabilístico ni un LLM de por medio).
3. El JSON generado se versiona en git junto con su script — así, aunque alguien solo tenga el repo (sin correr nada), puede leer el árbol; y quien quiera verificar o modificar la extracción, corre el generador y compara.
4. Ningún generador corre en producción: son herramientas de construcción offline, igual que una migración de base de datos. El chat (`chat/wizard.py`) solo lee los JSON ya generados, vía `trees/registry.py`.
5. `scripts/verify_tree_citations.py` cierra el círculo: con tu propia copia del PDF, confirma automáticamente que cada `source_quote` del bosque sigue correspondiendo a lo que dice la página citada del documento — sin que tengas que confiar en que la extracción se hizo bien, ni depender de esta conversación.

## Árboles actuales

| Árbol | Generador | JSON | Instrumento / Tabla fuente |
|---|---|---|---|
| `nac-2026-severidad-hospitalizacion` | `scripts/gen_curb65_tree.py` | `data/nac-2026/severidad-hospitalizacion.json` | CURB-65/CRB-65, Tabla 3, pág. 199 |
| `nac-2026-criterios-idsa-ats-uci` | `scripts/gen_idsa_ats_tree.py` | `data/nac-2026/criterios-idsa-ats-uci.json` | Criterios IDSA/ATS 2007, Tabla 5, pág. 199 |

Verificar que un JSON sigue coincidiendo con su generador (por ejemplo, tras clonar el repo en otra máquina o antes de la sustentación):

```bash
python scripts/gen_curb65_tree.py
python scripts/gen_idsa_ats_tree.py
git diff --stat src/gpc_rag/trees/data/   # debe salir vacío
```

Verificar que las citas siguen siendo fieles a tu copia del PDF (requiere `pymupdf`: `uv add --group dev pymupdf`):

```bash
python scripts/verify_tree_citations.py --pdf-dir ./ruta/donde/tengas/las/guias
```

Busca dentro de esa carpeta (recursivo) un archivo cuyo nombre coincida con el `gpc_source` de cada árbol, y reporta, nodo por nodo, si los términos clínicos y numéricos distintivos de cada `source_quote` aparecen en la página citada. No es una comparación de texto exacto — los `source_quote` son resúmenes compactos de tablas, no citas literales copiadas del PDF (extraer una tabla reordena filas/columnas) — así que el chequeo es por cobertura de términos relevantes (nombres de criterios, umbrales numéricos), con un umbral configurable en el script. Si falta una guía en la carpeta, ese árbol se omite con un aviso, no se detiene la verificación de los demás.

## Extracción automática desde tablas (sin LLM)

`table_extraction.py` + `builders.py` automatizan la parte mecánica de
"agregar un árbol nuevo" (abajo) cuando el instrumento viene en una tabla
del PDF con uno de estos dos patrones — el resto del flujo (revisión
humana de la pregunta/redacción y verificación de fidelidad) sigue igual:

- **Suma de puntaje** (tipo CURB-65): encabezado + una fila por criterio,
  cada "Sí" suma 1 punto, y una segunda tabla mapea rangos de puntaje a
  recomendación.
- **Mayor o N menores** (tipo IDSA/ATS): M "criterios mayores" (cualquiera
  → indicación directa) + N "criterios menores" con un umbral (corte
  temprano en cuanto se alcanza).

`table_extraction.py` detecta las tablas por su **geometría real en el
PDF** (`pdfplumber`: bordes/líneas/espaciado, no interpretación de
contenido) y parsea filas con reglas fijas de encabezados/columnas —
**deliberadamente sin ningún LLM**: es un pipeline determinista (mismo PDF
+ misma página + misma receta → siempre el mismo JSON), condición
necesaria para que "reproducible" (ver arriba) siga siendo cierto también
para este paso. Cuando una tabla no matchea con confianza ninguno de los
dos patrones (encabezado desconocido, puntaje ponderado en vez de +1 por
criterio, formato de rango no reconocido, etiquetas de mayores/menores
ausentes), levanta `TablaNoReconocidaError` con el motivo exacto en vez de
adivinar o producir un árbol parcial — un "éxito" silencioso e incorrecto
es peor que un fallo explícito que un humano completa a mano.

`builders.py` generaliza el armado de nodos (`construir_arbol_suma_puntaje`,
`construir_arbol_mayor_o_n_menores`) para que ni el generador a mano ni el
pipeline automático repitan la lógica de enumeración de nodos/trellis; los
dos generadores existentes (`gen_curb65_tree.py`, `gen_idsa_ats_tree.py`)
no se migraron a `builders.py` porque ya estaban verificados como
reproducibles y no valía la pena el riesgo de tocarlos sin necesidad — un
generador nuevo sí debería usar `builders.py` en vez de copiar su patrón.

`scripts/extract_tree_from_table.py` junta ambos módulos en un CLI de dos
pasos (ver el README raíz, sección "Agregar un protocolo nuevo", para el
comando exacto): `listar` para ubicar el índice de tabla correcto en una
página, y `construir` para generar el árbol a partir de una receta JSON
(`recetas/*.json`) — corre la verificación de fidelidad de citas antes de
escribir el archivo, así que un árbol con citas infieles nunca llega a
`data/`.

Lo que el pipeline automático **no** decide por sí solo: la redacción
final de la pregunta que ve el usuario en el wizard (genera una versión
literal a partir del texto de la celda, pensada para que un humano la
revise) ni el texto de las 3 hojas fijas de un instrumento "mayor o N
menores" (redacción clínica de la recomendación) — esas partes siguen
siendo una decisión humana en la receta, igual que en un generador escrito
a mano.

## Agregar un árbol nuevo

1. Identificar el instrumento/tabla de decisión en el PDF (con página y cita textual de cada criterio — ver `trees/schema.py`: `source` y `source_quote` son obligatorios en cada nodo).
2. Construir el árbol de una de dos formas:
   - **A mano**: escribir `scripts/gen_<nombre>_tree.py` siguiendo el patrón de los dos existentes (preguntas booleanas aditivas → trellis `step{N}_score{S}` o `step{N}_count{C}`; corte temprano si el umbral de decisión se puede alcanzar antes del último criterio).
   - **Automático** (si el instrumento es tipo "suma de puntaje" o "mayor o N menores" y está en una tabla del PDF): usar `scripts/extract_tree_from_table.py` — ver la sección de arriba.
3. Correrlo, y agregar sus pruebas en `tests/trees/test_<nombre>.py` (ver `test_idsa_ats_tree.py` como plantilla: recorridos hacia cada hoja, validación de que toda hoja tenga cita, casos de umbral).
4. `trees/registry.py` lo descubre automáticamente (recorre `data/**/*.json`) — no hace falta registrarlo a mano en ningún otro lado, tampoco en el menú del wizard (`chat/wizard.py`).
