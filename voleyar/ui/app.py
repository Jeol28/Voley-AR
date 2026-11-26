"""Aplicación de Gradio: una pestaña por tema del curso, más Pipeline y Jugar."""

from __future__ import annotations

import gradio as gr

from voleyar.ui.estado import ESTADO
from voleyar.ui.pestanas import (calibracion, caracterizacion, filtros, histogramas, jugar, morfologia, pipeline,
                                 segmentacion, transformers, umbralizacion)
from voleyar.ui.tema import CSS, crear_tema

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
    ("Jugar", jugar),
]


def crear_app() -> gr.Blocks:
    with gr.Blocks(title="Voley AR · Laboratorio de Visión Artificial") as demo:
        gr.Markdown("# Voley AR")
        estado_cam = ("cámara calibrada (config/camara.json)" if ESTADO.modelo_camara is not None
                      else "cámara **sin calibrar**: empieza por la pestaña Calibración")
        gr.Markdown("Voleibol en realidad aumentada sobre la mesa. Proyecto final de Visión Artificial · "
                    f"Jorge Enrique Olaya Liévano · {estado_cam}.")
        with gr.Tabs():
            for nombre, modulo in PESTANAS:
                with gr.Tab(nombre):
                    modulo.construir(demo)
    return demo


def lanzar(puerto: int = 7860, abrir: bool = True, compartir: bool = False) -> None:
    demo = crear_app()
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=puerto, inbrowser=abrir, share=compartir, theme=crear_tema(),
                css=CSS, footer_links=[])
