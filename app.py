"""Lanza el laboratorio interactivo de Voley AR (Gradio, local).

Uso::

    python app.py              # abre http://127.0.0.1:7860
    python app.py --puerto 7861 --no-abrir
"""

from __future__ import annotations

import argparse

from voleyar.ui.app import lanzar

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Laboratorio de Voley AR")
    ap.add_argument("--puerto", type=int, default=7860)
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el navegador")
    a = ap.parse_args()
    lanzar(a.puerto, abrir=not a.no_abrir)
