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

## Agregar un árbol nuevo

1. Identificar el instrumento/tabla de decisión en el PDF (con página y cita textual de cada criterio — ver `trees/schema.py`: `source` y `source_quote` son obligatorios en cada nodo).
2. Escribir `scripts/gen_<nombre>_tree.py` siguiendo el patrón de los dos existentes (preguntas booleanas aditivas → trellis `step{N}_score{S}` o `step{N}_count{C}`; corte temprano si el umbral de decisión se puede alcanzar antes del último criterio).
3. Correrlo, y agregar sus pruebas en `tests/trees/test_<nombre>.py` (ver `test_idsa_ats_tree.py` como plantilla: recorridos hacia cada hoja, validación de que toda hoja tenga cita, casos de umbral).
4. `trees/registry.py` lo descubre automáticamente (recorre `data/**/*.json`) — no hace falta registrarlo a mano en ningún otro lado, tampoco en el menú del wizard (`chat/wizard.py`).
