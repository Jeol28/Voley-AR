"""Laboratorio: una pestaña por tema del curso para ver el juego "por dentro".

No es la cara del juego: vive en ``/laboratorio`` (enlazado desde "Cómo se juega")
o se abre directamente con ``python laboratorio.py``.
"""

from __future__ import annotations

import gradio as gr

from voleyar.ui.pestanas import (calibracion, caracterizacion, filtros, histogramas, morfologia, pipeline,
                                 segmentacion, transformers, umbralizacion)

PESTANAS = [
    ("Calibración", calibracion),
    ("Filtros", filtros),
    ("Histogramas", histogramas),
    ("Umbralización", umbralizacion),
    ("Morfología", morfologia),
    ("Segmentación", segmentacion),
    ("Caracterización", caracterizacion),
    ("Transformers", transformers),
    ("Pipeline", pipeline),
]


def construir(demo: gr.Blocks) -> None:
    from voleyar.ui.comunes import nueva_sesion_componente

    nueva_sesion_componente()
    gr.Markdown("# Voley AR · Laboratorio")
    gr.Markdown("Cada tema del curso, aplicado al juego. [Volver al juego](/)")
    with gr.Tabs():
        for nombre, modulo in PESTANAS:
            with gr.Tab(nombre):
                modulo.construir(demo)
