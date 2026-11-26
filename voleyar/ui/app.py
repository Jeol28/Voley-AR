"""Aplicación web: el juego es la página principal; el laboratorio vive en ``/laboratorio``."""

from __future__ import annotations

import gradio as gr

from voleyar.entorno import EN_SPACE
from voleyar.ui import juego_web, laboratorio
from voleyar.ui.comunes import nueva_sesion_componente
from voleyar.ui.tema import CSS, crear_tema


def crear_app(con_laboratorio: bool = True) -> gr.Blocks:
    with gr.Blocks(title="Voley AR") as demo:
        gr.Navbar(visible=False)
        nueva_sesion_componente()
        juego_web.construir(demo)
    if con_laboratorio:
        with demo.route("Laboratorio", "/laboratorio", show_in_navbar=False):
            laboratorio.construir(demo)
    return demo


def crear_laboratorio() -> gr.Blocks:
    with gr.Blocks(title="Voley AR · Laboratorio") as demo:
        laboratorio.construir(demo)
    return demo


def lanzar(puerto: int = 7860, abrir: bool = True, compartir: bool = False, solo_laboratorio: bool = False) -> None:
    demo = crear_laboratorio() if solo_laboratorio else crear_app()
    if EN_SPACE:
        # ZeroGPU: los modelos se cargan una vez al arrancar (fuera de @spaces.GPU)
        from voleyar.core.transformers_models import precargar_modelos

        precargar_modelos()
        demo.queue(default_concurrency_limit=4)
        demo.launch(theme=crear_tema(), css=CSS, footer_links=[])  # host y puerto los pone el Space
        return
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=puerto, inbrowser=abrir, share=compartir, theme=crear_tema(),
                css=CSS, footer_links=[])
