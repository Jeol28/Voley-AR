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
    piel: his.ModeloPiel | None = None
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
    "Mano abierta (remate)": dict(n_dedos=5, pos=(-6.0, 1.0), angulo=0),
    "Un dedo (toque)": dict(n_dedos=1, pos=(-5.0, -2.0), angulo=10),
    "Dos dedos": dict(n_dedos=2, pos=(-6.0, 2.0), angulo=-15),
    "Puño (bloqueo)": dict(n_dedos=0, pos=(-7.0, 0.0), angulo=0),
    "Mano tocando un objeto color piel": dict(n_dedos=3, pos=(-6.0, 0.0), angulo=0, objeto=True),
    "Cancha sin mano": dict(sin_mano=True),
    "Poca luz": dict(n_dedos=4, pos=(-6.0, 0.0), angulo=20, oscura=True),
}

_cache_ejemplos: dict[str, Frame] = {}


def frame_ejemplo(nombre: str) -> Frame:
    """Escena sintética con cámara de distorsión conocida (resolución 1280×720)."""
    if nombre in _cache_ejemplos:
        return _cache_ejemplos[nombre]
    p = EJEMPLOS.get(nombre, EJEMPLOS["Mano abierta (remate)"])
    modelo = sin.camara_ejemplo(1280, 720)
    pose = sin.pose_juego_por_defecto()
    mano = None if p.get("sin_mano") else sin.EstadoManoSintetica(pos=p["pos"], n_dedos=p["n_dedos"],
                                                                  angulo=p["angulo"], altura=0.8)
    img, verdad = sin.escena_cancha(modelo, pose, mano, ruido=3.0, semilla=len(nombre))
    if p.get("objeto"):
        # Objeto de color parecido a la piel (p. ej. una pelota de madera) pegado a la mano
        t = np.linspace(0, 2 * np.pi, 40)
        cx, cy = p["pos"][0] + 1.0, p["pos"][1] + 7.5
        pts = np.column_stack([cx + 3.6 * np.cos(t), cy + 3.6 * np.sin(t), np.full_like(t, 0.8)])
        px = np.round(cal.proyectar(pts, pose, modelo)).astype(np.int32)
        capa = img.copy()
        cv2.fillPoly(capa, [px], (110, 145, 200), cv2.LINE_AA)
        img = sin.ruido_camara(capa, 2.0, 3)
    if p.get("oscura"):
        img = np.clip(img.astype(np.float32) * 0.38 + 6, 0, 255).astype(np.uint8)
        img = sin.ruido_camara(img, 4.0, 5)
    fr = Frame(img, modelo, nombre, verdad)
    _cache_ejemplos[nombre] = fr
    return fr


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


def piel_para(img_trabajo: np.ndarray, zona: np.ndarray | None = None) -> tuple[his.ModeloPiel | None, str]:
    if ESTADO.piel is not None:
        return ESTADO.piel, ESTADO.piel_origen or "calibrada con el recuadro"
    return piel_automatica(img_trabajo, zona), "automática (rango YCrCb), calíbrala en Histogramas"


def ejecutar_pipeline(fr: Frame, usar_calibracion: bool = True, **cambios) -> pl.ResultadoFrame:
    """Pipeline completo con la configuración de la UI (y piel automática si hace falta)."""
    cfg = pl.ConfigPipeline(**{**ESTADO.config.__dict__, **cambios})
    cfg.usar_calibracion = usar_calibracion and ESTADO.modelo_para(fr) is not None
    modelo = ESTADO.modelo_para(fr)
    piel = ESTADO.piel
    if piel is None:
        img, m = pl.preparar_imagen(fr.img, modelo, cfg)
        pose = cal.estimar_pose_cancha(cal.detectar_marcadores(img), m)
        zona = seg.zona_busqueda(img.shape, pose, m).mascara if pose is not None else None
        _, suav = pl.preprocesar(img, cfg)
        piel = piel_automatica(suav, zona)
    return pl.procesar_frame(fr.img, modelo, piel, cfg)


def tabla_md(filas: list[list], encabezado: list[str]) -> str:
    md = "| " + " | ".join(encabezado) + " |\n|" + "---|" * len(encabezado) + "\n"
    for f in filas:
        md += "| " + " | ".join(str(x) for x in f) + " |\n"
    return md
