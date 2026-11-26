"""Bloques de interfaz reutilizados por todas las pestañas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from voleyar.entorno import EN_SPACE
from voleyar.ui.estado import EJEMPLOS, Frame, con_sesion, frame_desde_rgb, frame_ejemplo, rgb

LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
CAMARA_HD = gr.WebcamOptions(mirror=False, constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}})


# En un Space varios visitantes transmiten a la vez; en local basta un flujo por evento.
CONCURRENCIA_VIVO = 4 if EN_SPACE else 1
_SESION: gr.State | None = None


def sesion_componente() -> gr.State:
    """``gr.State`` con el estado de la sesión (calibración, piel). Lo crea ``crear_app``."""
    global _SESION
    if _SESION is None:
        _SESION = gr.State(None)
    return _SESION


def nueva_sesion_componente() -> gr.State:
    global _SESION
    _SESION = gr.State(None)
    return _SESION


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
    S = sesion_componente()

    def _ejecutar(fr: Frame | None, *p):
        if fr is None:
            return tuple([gr.skip()] * n)
        r = fn(fr, *p)
        return r if isinstance(r, tuple) else (r,)

    @con_sesion
    def desde_subida(img, *p):
        fr = frame_desde_rgb(img)
        return (fr, *_ejecutar(fr, *p))

    @con_sesion
    def desde_ejemplo(nombre, *p):
        fr = frame_ejemplo(nombre)
        return (fr, rgb(fr.img), *_ejecutar(fr, *p))

    @con_sesion
    def desde_estado(fr, *p):
        return _ejecutar(fr, *p)

    entrada.subida.input(desde_subida, [S, entrada.subida, *params], [S, entrada.estado, *salidas])
    entrada.boton.click(desde_ejemplo, [S, entrada.ejemplo, *params], [S, entrada.estado, entrada.subida, *salidas])
    if en_vivo:
        entrada.vivo.stream(desde_subida, [S, entrada.vivo, *params], [S, entrada.estado, *salidas], stream_every=cada,
                            concurrency_limit=CONCURRENCIA_VIVO, trigger_mode="always_last", show_progress="hidden")
    for p in params:
        evento = p.release if isinstance(p, gr.Slider) else p.change
        evento(desde_estado, [S, entrada.estado, *params], [S, *salidas], show_progress="hidden")


def cargar_al_inicio(demo: gr.Blocks, entrada: Entrada, fn: Callable, params: list, salidas: list) -> None:
    """Procesa la escena de ejemplo al abrir la app, para que ninguna pestaña empiece vacía."""
    S = sesion_componente()

    @con_sesion
    def inicial(nombre, *p):
        fr = frame_ejemplo(nombre)
        r = fn(fr, *p)
        r = r if isinstance(r, tuple) else (r,)
        return (fr, rgb(fr.img), *r)

    demo.load(inicial, [S, entrada.ejemplo, *params], [S, entrada.estado, entrada.subida, *salidas])
