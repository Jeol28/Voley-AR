"""Juego dentro del navegador: los frames llegan desde la cámara web de Gradio.

En un Space no hay ventana de OpenCV ni cámara local; en su lugar el navegador
transmite la cámara (``gr.Image`` con *streaming*) y cada frame pasa por el
mismo ``Juego.paso`` que la versión de escritorio. Las teclas se reemplazan
por botones.
"""

from __future__ import annotations

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


def crear_juego_navegador(modelo_camara=None, dificultad: str = "normal") -> Juego:
    """Juego sin ventana ni hilos de modelos (en un Space la GPU se pide por llamada)."""
    op = OpcionesJuego(profundidad=False, dificultad=dificultad, headless=True)
    return Juego(op, fuente=FuenteExterna(), modelo_camara=modelo_camara)


def paso_navegador(juego: Juego, frame_bgr: np.ndarray) -> np.ndarray | None:
    """Procesa un frame del navegador y devuelve la imagen del juego (BGR)."""
    juego.fuente.frame = frame_bgr
    return juego.paso()
