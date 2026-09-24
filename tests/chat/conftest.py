"""Instala el chainlit de mentira ANTES de que se importe gpc_rag.chat.wizard.

Se hace aqui (a nivel de modulo, en el conftest de este directorio) porque
`import chainlit as cl` en wizard.py necesita resolver contra algo la primera
vez que se importa ese modulo -- y queremos que sea siempre nuestro doble de
prueba, este o no instalado el chainlit real, para que estas pruebas sean
deterministas y no dependan de si el framework real esta disponible en el
entorno donde corre pytest.

Se inserta manualmente esta carpeta en sys.path (en vez de depender de
`tests` como paquete importable, que varia segun si hay o no un
tests/__init__.py en la raiz) para que `import fake_chainlit` funcione sin
sorpresas sin importar la configuracion de import-mode de pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fake_chainlit  # noqa: E402 -- debe importarse despues de tocar sys.path

sys.modules["chainlit"] = fake_chainlit  # type: ignore[assignment]
