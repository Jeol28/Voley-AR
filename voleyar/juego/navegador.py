"""Juego dentro del navegador: los frames llegan desde la cámara web de Gradio.

En un Space no hay ventana de OpenCV ni cámara local; en su lugar el navegador
transmite la cámara (``gr.Image`` con *streaming*) y cada frame pasa por el
mismo ``Juego.paso`` que la versión de escritorio. Las teclas se reemplazan
por botones.
"""

from __future__ import annotations

import os

import numpy as np

from voleyar.juego.bucle import Juego, OpcionesJuego


class FuenteExterna:
    """Fuente de video alimentada desde fuera (el último frame recibido)."""

    def __init__(self):
        self.frame: np.ndarray | None = None

    def leer(self) -> tuple[bool, np.ndarray | None]:
        return self.frame is not None, self.frame

    def liberar(self) -> None:
        pass


def crear_juego_navegador(modelo_camara=None, modo: str = "ia", dificultad: str = "normal") -> Juego:
    """Juego sin ventana (en un Space, sin hilos de modelos: la GPU se pide por llamada)."""
    from voleyar.entorno import EN_SPACE

    # En tu computador Depth Anything corre en un hilo (≈ 30 ms en la GPU del Mac) y decide qué está
    # cerca de la cámara; en un Space con ZeroGPU cada llamada pide la GPU, así que solo se usa en su vista.
    profundidad = not EN_SPACE and os.environ.get("VOLEYAR_PROFUNDIDAD", "1") != "0"
    op = OpcionesJuego(modo=modo, profundidad=profundidad, dificultad=dificultad, headless=True, interfaz="web")
    juego = Juego(op, fuente=FuenteExterna(), modelo_camara=modelo_camara)
    # En el servidor cada milisegundo cuenta: se procesa a 640 px y las caras se buscan cada 6 frames
    juego.config.ancho_proceso = 640
    juego.config.caras_cada = 6
    return juego


def paso_navegador(juego: Juego, frame_bgr: np.ndarray) -> np.ndarray | None:
    """Procesa un frame del navegador y devuelve la imagen del juego (BGR)."""
    juego.fuente.frame = frame_bgr
    return juego.paso()
