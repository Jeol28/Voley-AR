"""Bloques de interfaz reutilizados por todas las pestañas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from voleyar.ui.estado import EJEMPLOS, Frame, frame_desde_rgb, frame_ejemplo, rgb

LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
CAMARA_HD = gr.WebcamOptions(mirror=False, constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}})


def teoria(md: str) -> None:
    """Desplegable "¿Qué está pasando?" con la teoría corta y las fórmulas."""
    with gr.Accordion("¿Qué está pasando?", open=False):
        gr.Markdown(md, latex_delimiters=LATEX)


@dataclass
class Entrada:
    subida: gr.Image
    vivo: gr.Image
    ejemplo: gr.Dropdown
    boton: gr.Button
    estado: gr.State


def bloque_entrada(ejemplo: str = "Mano abierta (remate)", alto: int = 300) -> Entrada:
    """Entrada común: imagen subida o escena de ejemplo, o la cámara web en vivo (streaming)."""
    with gr.Tabs():
        with gr.Tab("Imagen"):
            subida = gr.Image(sources=["upload", "clipboard"], type="numpy", label="Imagen de entrada", height=alto)
            with gr.Row():
                ej = gr.Dropdown(choices=list(EJEMPLOS), value=ejemplo, label="Escena de ejemplo (sintética)",
                                 scale=3)
                boton = gr.Button("Cargar ejemplo", scale=1)
        with gr.Tab("Cámara en vivo"):
            vivo = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Cámara web",
                            webcam_options=CAMARA_HD, height=alto)
    return Entrada(subida, vivo, ej, boton, gr.State(None))


def conectar(entrada: Entrada, fn: Callable, params: list, salidas: list, cada: float = 0.25,
             en_vivo: bool = True) -> None:
    """Conecta la entrada y los parámetros con la función de la pestaña.

    ``fn(frame: Frame, *params) -> tuple`` debe devolver un valor por salida.
    El último frame se guarda en ``entrada.estado`` para volver a procesarlo
    cuando se mueve un parámetro.
    """
    n = len(salidas)

    def _ejecutar(fr: Frame | None, *p):
        if fr is None:
            return tuple([gr.skip()] * n)
        r = fn(fr, *p)
        return r if isinstance(r, tuple) else (r,)

    def desde_subida(img, *p):
        fr = frame_desde_rgb(img)
        return (fr, *_ejecutar(fr, *p))

    def desde_ejemplo(nombre, *p):
        fr = frame_ejemplo(nombre)
        return (fr, rgb(fr.img), *_ejecutar(fr, *p))

    def desde_estado(fr, *p):
        return _ejecutar(fr, *p)

    entrada.subida.input(desde_subida, [entrada.subida, *params], [entrada.estado, *salidas])
    entrada.boton.click(desde_ejemplo, [entrada.ejemplo, *params], [entrada.estado, entrada.subida, *salidas])
    if en_vivo:
        entrada.vivo.stream(desde_subida, [entrada.vivo, *params], [entrada.estado, *salidas], stream_every=cada,
                            concurrency_limit=1, trigger_mode="always_last", show_progress="hidden")
    for p in params:
        evento = p.release if isinstance(p, gr.Slider) else p.change
        evento(desde_estado, [entrada.estado, *params], salidas, show_progress="hidden")


def cargar_al_inicio(demo: gr.Blocks, entrada: Entrada, fn: Callable, params: list, salidas: list) -> None:
    """Procesa la escena de ejemplo al abrir la app, para que ninguna pestaña empiece vacía."""

    def inicial(nombre, *p):
        fr = frame_ejemplo(nombre)
        r = fn(fr, *p)
        r = r if isinstance(r, tuple) else (r,)
        return (fr, rgb(fr.img), *r)

    demo.load(inicial, [entrada.ejemplo, *params], [entrada.estado, entrada.subida, *salidas])
