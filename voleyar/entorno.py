"""Detección del entorno de ejecución (local o Hugging Face Spaces con ZeroGPU).

En un Space con ZeroGPU, ``spaces`` debe importarse **antes** que ``torch``; por
eso ``app.py`` importa este módulo en su primera línea. Localmente el paquete
``spaces`` no está instalado y ``gpu`` es un decorador que no hace nada.
"""

from __future__ import annotations

import os

EN_SPACE = os.environ.get("SPACE_ID") is not None

try:  # solo existe en Hugging Face Spaces
    import spaces  # noqa: F401

    HAY_ZEROGPU = True
except ImportError:
    spaces = None
    HAY_ZEROGPU = False


def gpu(duracion: int = 30):
    """``@spaces.GPU(duration=...)`` en ZeroGPU; sin efecto en local.

    En ZeroGPU la función decorada se ejecuta en otro proceso con la GPU
    asignada: sus argumentos y su resultado deben ser serializables (arreglos,
    textos, dataclasses), nunca el modelo.
    """
    def decorador(fn):
        if HAY_ZEROGPU:
            return spaces.GPU(duration=duracion)(fn)
        return fn

    return decorador
