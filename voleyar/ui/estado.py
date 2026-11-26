"""Estado compartido de la UI y utilidades de entrada (sin componentes de Gradio).

Las funciones de cada pestaña son puras respecto a Gradio: reciben arreglos
NumPy y devuelven arreglos o datos, así que se pueden probar sin levantar la UI.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import histogramas as his
from voleyar.core import pipeline as pl
from voleyar.core import segmentacion as seg
from voleyar.core import umbralizacion as umb
from voleyar.utils import sinteticos as sin
from voleyar.utils import visualizacion as vis
from voleyar.entorno import EN_SPACE
from voleyar.utils.io import cargar_modelo_camara


@dataclass
class Frame:
    """Imagen de entrada (BGR) y, si es sintética, su modelo de cámara verdadero."""

    img: np.ndarray
    modelo: cal.ModeloCamara | None = None  # solo en escenas sintéticas
    nombre: str = "imagen"
    verdad: dict = field(default_factory=dict)


@dataclass
class EstadoApp:
    modelo_camara: cal.ModeloCamara | None = None
    pieles: dict = field(default_factory=lambda: {-1: None, 1: None})  # una por jugador (-1 izq., 1 der.)
    piel_origen: str = ""
    config: pl.ConfigPipeline = field(default_factory=pl.ConfigPipeline)

    def recargar_camara(self) -> None:
        self.modelo_camara = cargar_modelo_camara()

    def modelo_para(self, fr: Frame) -> cal.ModeloCamara | None:
        """Modelo calibrado a usar: el verdadero si la escena es sintética, si no ``config/camara.json``."""
        return fr.modelo if fr.modelo is not None else self.modelo_camara


def nuevo_estado() -> EstadoApp:
    """Estado de una sesión nueva. En local carga ``config/camara.json``; en un Space
    empieza sin calibración (cada visitante tiene su propia cámara)."""
    e = EstadoApp()
    if not EN_SPACE:
        e.recargar_camara()
    return e


_global = nuevo_estado()  # respaldo cuando no hay sesión (pruebas, scripts)
_sesion_actual: ContextVar[EstadoApp | None] = ContextVar("sesion_voleyar", default=None)


class _EstadoDeLaSesion:
    """Proxy: ``ESTADO.x`` lee y escribe en el estado de la sesión de Gradio en curso.

    Cada manejador de evento fija la sesión con ``usar_sesion`` antes de llamar a
    las funciones de las pestañas, así la calibración y la piel de un visitante
    no se mezclan con las de otro.
    """

    def __getattr__(self, nombre):
        return getattr(_sesion_actual.get() or _global, nombre)

    def __setattr__(self, nombre, valor):
        setattr(_sesion_actual.get() or _global, nombre, valor)


ESTADO = _EstadoDeLaSesion()


def usar_sesion(sesion: EstadoApp | None) -> tuple[EstadoApp, object]:
    """Activa la sesión (creándola si hace falta) y devuelve ``(sesion, token)``."""
    sesion = sesion if isinstance(sesion, EstadoApp) else nuevo_estado()
    return sesion, _sesion_actual.set(sesion)


def soltar_sesion(token) -> None:
    _sesion_actual.reset(token)


def con_sesion(fn):
    """Envuelve ``fn(*args)`` como manejador de Gradio ``(sesion, *args) -> (sesion, *salidas)``."""
    def envoltura(sesion, *args):
        sesion, token = usar_sesion(sesion)
        try:
            r = fn(*args)
        finally:
            soltar_sesion(token)
        return (sesion, *(r if isinstance(r, tuple) else (r,)))

    # Solo el nombre: con functools.wraps Gradio leería la firma original (sin ``sesion``)
    envoltura.__name__ = getattr(fn, "__name__", "manejador")
    return envoltura


# --------------------------------------------------------------------------- #
# Escenas de ejemplo
# --------------------------------------------------------------------------- #

EJEMPLOS = {
    "Dos jugadores (remate y toque)": dict(dedos=(5, 1)),
    "Dos dedos y puño": dict(dedos=(2, 0)),
    "Tres y cuatro dedos": dict(dedos=(3, 4)),
    "Un jugador contra la IA": dict(dedos=(5,), n=1),
    "Mano delante de la cara": dict(dedos=(5, 2), mano1=(-18.0, 1.0)),
    "Lejos de la cámara (85 cm)": dict(dedos=(4, 1), distancia=85.0),
    "Poca luz": dict(dedos=(5, 3), oscura=True),
}
EJEMPLO_CANCHA = "Cancha impresa con ArUco"

_cache_ejemplos: dict[str, Frame] = {}


def frame_ejemplo(nombre: str) -> Frame:
    """Escena sintética del modo pantalla (dos personas frente a una cámara con distorsión conocida)."""
    if nombre in _cache_ejemplos:
        return _cache_ejemplos[nombre]
    if nombre == EJEMPLO_CANCHA:
        return frame_ejemplo_cancha()
    p = EJEMPLOS.get(nombre, next(iter(EJEMPLOS.values())))
    modelo = sin.camara_ejemplo(1280, 720)
    jugadores = sin.jugadores_por_defecto(p.get("n", 2))
    for jug, d in zip(jugadores, p["dedos"]):
        jug.n_dedos = d
        if "distancia" in p:
            jug.distancia = p["distancia"]
    if "mano1" in p:
        jugadores[0].mano = p["mano1"]
    img, verdad = sin.escena_pantalla(modelo, jugadores, ruido=3.0, semilla=len(nombre))
    if p.get("oscura"):
        img = np.clip(img.astype(np.float32) * 0.38 + 6, 0, 255).astype(np.uint8)
        img = sin.ruido_camara(img, 4.0, 5)
    fr = Frame(img, modelo, nombre, verdad)
    _cache_ejemplos[nombre] = fr
    return fr


def frame_ejemplo_cancha() -> Frame:
    """Escena de la v1 (cancha impresa con ArUco), solo para la demo de pose con solvePnP."""
    if EJEMPLO_CANCHA not in _cache_ejemplos:
        modelo = sin.camara_ejemplo(1280, 720)
        img, verdad = sin.escena_cancha(modelo, sin.pose_juego_por_defecto(), None, ruido=3.0)
        _cache_ejemplos[EJEMPLO_CANCHA] = Frame(img, modelo, EJEMPLO_CANCHA, verdad)
    return _cache_ejemplos[EJEMPLO_CANCHA]


def frame_desde_rgb(img_rgb: np.ndarray | None) -> Frame | None:
    if img_rgb is None:
        return None
    return Frame(vis.rgb_a_bgr(np.asarray(img_rgb)), None, "cámara")


def rgb(img_bgr: np.ndarray | None) -> np.ndarray | None:
    return None if img_bgr is None else vis.bgr_a_rgb(img_bgr)


# --------------------------------------------------------------------------- #
# Pipeline y piel para las pestañas
# --------------------------------------------------------------------------- #


def piel_automatica(img_bgr: np.ndarray, zona: np.ndarray | None = None) -> his.ModeloPiel | None:
    """Modelo de piel sin recuadro: se siembra con el rango fijo de YCrCb (para demos)."""
    m = umb.umbral_ycrcb(img_bgr)
    if zona is not None:
        m = cv2.bitwise_and(m, zona)
    m = cv2.erode(m, np.ones((7, 7), np.uint8))
    if np.count_nonzero(m) < 200:
        return None
    try:
        return his.calibrar_piel(img_bgr, mascara=m)
    except ValueError:
        return None


def piel_para(img_trabajo: np.ndarray, lado: int, zona: np.ndarray | None = None) -> tuple[his.ModeloPiel | None, str]:
    """Piel calibrada del jugador ``lado`` o, si no hay, una automática sembrada con YCrCb en su mitad."""
    if ESTADO.pieles.get(lado) is not None:
        return ESTADO.pieles[lado], ESTADO.piel_origen or "calibrada con el recuadro"
    if zona is None:
        zona = seg.mascara_mitad(img_trabajo.shape, lado)
    return piel_automatica(img_trabajo, zona), "automática (rango YCrCb); calíbrala en Histogramas"


def pieles_para(fr: Frame, cfg: pl.ConfigPipeline) -> dict:
    img, _ = pl.preparar_imagen(fr.img, ESTADO.modelo_para(fr), cfg)
    _, suav = pl.preprocesar(img, cfg)
    return {lado: piel_para(suav, lado)[0] for lado in pl.LADOS}


def ejecutar_pipeline(fr: Frame, usar_calibracion: bool = True, **cambios) -> pl.ResultadoFrame:
    """Pipeline completo (dos mitades) con la configuración de la UI y piel automática si hace falta."""
    cfg = pl.ConfigPipeline(**{**ESTADO.config.__dict__, **cambios})
    cfg.usar_calibracion = usar_calibracion and ESTADO.modelo_para(fr) is not None
    return pl.procesar_frame(fr.img, ESTADO.modelo_para(fr), pieles_para(fr, cfg), cfg)


def tabla_md(filas: list[list], encabezado: list[str]) -> str:
    md = "| " + " | ".join(encabezado) + " |\n|" + "---|" * len(encabezado) + "\n"
    for f in filas:
        md += "| " + " | ".join(str(x) for x in f) + " |\n"
    return md
