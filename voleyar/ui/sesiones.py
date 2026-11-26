"""Sesiones de la página web: qué juego y qué calibración tiene cada visitante.

El video llega por WebRTC a un manejador que no conoce los ``gr.State`` de la
página; los botones (Pausa, temas…) sí. Para que ambos toquen el mismo
``Juego`` se guarda todo aquí, bajo una clave aleatoria que la página crea al
abrirse y le pasa tanto al video como a los botones.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from voleyar.ui.estado import EstadoApp, nuevo_estado

MAX_SESIONES = 64


@dataclass
class Sesion:
    estado: EstadoApp = field(default_factory=nuevo_estado)
    juego: Any = None  # voleyar.juego.bucle.Juego
    modo: str = "2j"
    ajustes: dict = field(default_factory=lambda: {"dificultad": "normal", "puntos": 15, "usar_calibracion": True,
                                                   "espejo": True})
    gestor: Any = None  # GestorCapturaCalibracion de la calibración por video
    ultimo_tablero: Any = None  # ¿se vio el tablero en el último frame de calibración?
    resultado: Any = None  # última calibración calculada (aún sin usar)
    cerrojo: threading.Lock = field(default_factory=threading.Lock)


_sesiones: OrderedDict[str, Sesion] = OrderedDict()
_cerrojo = threading.Lock()


def nueva_clave() -> str:
    return uuid.uuid4().hex


def obtener(clave: str | None) -> Sesion:
    """La sesión de esa clave (se crea si no existe; se olvidan las más antiguas)."""
    clave = clave or "local"
    with _cerrojo:
        s = _sesiones.get(clave)
        if s is None:
            s = Sesion()
            _sesiones[clave] = s
            while len(_sesiones) > MAX_SESIONES:
                _sesiones.popitem(last=False)
        else:
            _sesiones.move_to_end(clave)
        return s


def olvidar(clave: str | None) -> None:
    with _cerrojo:
        _sesiones.pop(clave or "local", None)
