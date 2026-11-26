"""Abre solo el laboratorio de Voley AR (una pestaña por tema del curso).

Uso::

    python laboratorio.py          # http://127.0.0.1:7861
"""

from __future__ import annotations

import voleyar.entorno  # noqa: F401  (primero: en ZeroGPU `spaces` debe cargarse antes que torch)

import argparse  # noqa: E402

from voleyar.ui.app import lanzar  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Laboratorio de Voley AR")
    ap.add_argument("--puerto", type=int, default=7861)
    ap.add_argument("--no-abrir", action="store_true")
    a = ap.parse_args()
    lanzar(a.puerto, abrir=not a.no_abrir, solo_laboratorio=True)
