r"""Transformers de visión (solo inferencia, pesos públicos de Hugging Face).

No se entrena ni se ajusta ningún modelo: se descargan bajo demanda la primera
vez que se usan (caché de Hugging Face en ``~/.cache/huggingface``) y se cargan
**una sola vez** por proceso. Se usa MPS (GPU de Apple Silicon) si está
disponible; si no, CPU.

Depth Anything V2 (profundidad monocular)
-----------------------------------------
`depth-anything/Depth-Anything-V2-Small-hf <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>`_
es un ViT (codificador DINOv2) con una cabeza DPT que estima la **profundidad
relativa** (disparidad, :math:`\propto 1/Z`, más alto = más cerca) desde una sola
imagen. Es relativa: está definida salvo una escala y un desplazamiento
desconocidos, así que no da centímetros. Para decidir si la mano está **cerca
de la mesa** se compara la mano con la mesa que la rodea y se normaliza por la
variación de la propia cancha:

.. math::  s = \frac{\operatorname{med}(D_{\text{mano}}) - \operatorname{med}(D_{\text{anillo}})}{P_{95}(D_{\text{cancha}}) - P_{5}(D_{\text{cancha}})}

La resta elimina el desplazamiento y la división elimina la escala. Si
:math:`s` supera un umbral, la mano está levantada (golpe inválido).

*Por qué cada N frames*: el modelo Small tarda decenas o cientos de ms por
imagen en MPS/CPU, frente a ≈ 30 ms que dura un frame a 30 FPS. La altura de
la mano cambia más despacio que su posición, así que el juego lo ejecuta en un
**hilo aparte** cada N frames y usa el último resultado; el bucle nunca espera
al modelo.

OWLv2 (detección de vocabulario abierto)
----------------------------------------
`google/owlv2-base-patch16-ensemble <https://huggingface.co/google/owlv2-base-patch16-ensemble>`_
es un detector basado en CLIP: un ViT produce un embedding por parche y cada
parche predice una caja; la clase se decide comparando el embedding de la
caja con el embedding del texto de la consulta (aquí ``"a hand"``). No
necesita colores calibrados, pero es mucho más lento que el pipeline clásico.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # operaciones no soportadas en MPS → CPU

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from voleyar.entorno import HAY_ZEROGPU, gpu  # noqa: E402

MODELO_PROFUNDIDAD = "depth-anything/Depth-Anything-V2-Small-hf"
MODELO_OWL = "google/owlv2-base-patch16-ensemble"
URL_HF = "https://huggingface.co/"

_cerrojo = threading.Lock()
_cache_modelos: dict[str, object] = {}


def dispositivo() -> str:
    """``"mps"`` en Apple Silicon si está disponible; si no, ``"cpu"``."""
    import torch

    if HAY_ZEROGPU:  # ZeroGPU: .to("cuda") al cargar; la GPU se asigna en cada @spaces.GPU
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transformers_disponible() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


# =========================================================================== #
# Depth Anything V2
# =========================================================================== #


@dataclass
class ResultadoProfundidad:
    mapa: np.ndarray  # float32 HxW, disparidad relativa (más alto = más cerca)
    ms: float
    dispositivo: str


class EstimadorProfundidad:
    """Envoltura perezosa de Depth Anything V2 Small."""

    def __init__(self, nombre: str = MODELO_PROFUNDIDAD, disp: str | None = None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.disp = disp or dispositivo()
        self.procesador = AutoImageProcessor.from_pretrained(nombre)
        self.modelo = AutoModelForDepthEstimation.from_pretrained(nombre).to(self.disp).eval()
        self._cerrojo = threading.Lock()

    def estimar(self, img_bgr: np.ndarray, lado_max: int = 518) -> ResultadoProfundidad:
        import torch

        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if max(h, w) > lado_max * 1.5:  # reducir antes del procesador ahorra tiempo de CPU
            e = lado_max * 1.5 / max(h, w)
            rgb = cv2.resize(rgb, None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
        with self._cerrojo, torch.inference_mode():
            entradas = self.procesador(images=rgb, return_tensors="pt").to(self.disp)
            salida = self.modelo(**entradas).predicted_depth  # (1, h', w')
            mapa = torch.nn.functional.interpolate(salida[:, None], size=(h, w), mode="bilinear",
                                                   align_corners=False)[0, 0]
            mapa = mapa.float().cpu().numpy()
        return ResultadoProfundidad(mapa.astype(np.float32), (time.perf_counter() - t0) * 1000, self.disp)


def obtener_profundidad() -> EstimadorProfundidad:
    """Carga Depth Anything V2 una sola vez por proceso (descarga la primera vez)."""
    with _cerrojo:
        if "profundidad" not in _cache_modelos:
            _cache_modelos["profundidad"] = EstimadorProfundidad()
        return _cache_modelos["profundidad"]  # type: ignore[return-value]


@dataclass
class AlturaMano:
    cerca: bool
    puntaje: float
    med_mano: float
    med_mesa: float
    rango_cancha: float


def mano_cerca_de_mesa(mapa: np.ndarray, mascara_mano: np.ndarray, mascara_cancha: np.ndarray,
                       umbral: float = 0.30, anillo_px: int = 25) -> AlturaMano | None:
    """Decide si la mano está apoyada/cerca de la mesa con la profundidad relativa."""
    if mascara_mano is None or not mascara_mano.any() or not mascara_cancha.any():
        return None
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * anillo_px + 1, 2 * anillo_px + 1))
    anillo = cv2.subtract(cv2.dilate(mascara_mano, ee), cv2.dilate(mascara_mano, np.ones((5, 5), np.uint8)))
    anillo = cv2.bitwise_and(anillo, mascara_cancha)
    mesa_sin_mano = cv2.bitwise_and(mascara_cancha, cv2.bitwise_not(cv2.dilate(mascara_mano, ee)))
    d_mesa_total = mapa[mesa_sin_mano > 0] if mesa_sin_mano.any() else mapa[mascara_cancha > 0]
    d_anillo = mapa[anillo > 0] if np.count_nonzero(anillo) > 50 else d_mesa_total
    # Núcleo de la mano (erosionado) para no mezclar con el borde
    nucleo = cv2.erode(mascara_mano, np.ones((7, 7), np.uint8))
    d_mano = mapa[nucleo > 0] if nucleo.any() else mapa[mascara_mano > 0]
    rango = float(np.percentile(d_mesa_total, 95) - np.percentile(d_mesa_total, 5))
    rango = max(rango, 1e-6)
    med_mano, med_mesa = float(np.median(d_mano)), float(np.median(d_anillo))
    s = (med_mano - med_mesa) / rango
    return AlturaMano(s < umbral, float(s), med_mano, med_mesa, rango)


# =========================================================================== #
# OWLv2
# =========================================================================== #


@dataclass
class Deteccion:
    caja: tuple[float, float, float, float]  # x0, y0, x1, y1
    puntaje: float
    etiqueta: str

    @property
    def centro(self) -> tuple[float, float]:
        return ((self.caja[0] + self.caja[2]) / 2, (self.caja[1] + self.caja[3]) / 2)


@dataclass
class ResultadoOWL:
    detecciones: list[Deteccion]
    ms: float
    dispositivo: str

    @property
    def mejor(self) -> Deteccion | None:
        return max(self.detecciones, key=lambda d: d.puntaje) if self.detecciones else None


class DetectorManoOWL:
    """Envoltura perezosa de OWLv2 (detector de vocabulario abierto)."""

    def __init__(self, nombre: str = MODELO_OWL, disp: str | None = None):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.disp = disp or dispositivo()
        self.procesador = Owlv2Processor.from_pretrained(nombre)
        self.modelo = Owlv2ForObjectDetection.from_pretrained(nombre).to(self.disp).eval()
        self._cerrojo = threading.Lock()

    def detectar(self, img_bgr: np.ndarray, textos: tuple[str, ...] = ("a hand",), umbral: float = 0.2,
                 zona: np.ndarray | None = None) -> ResultadoOWL:
        """Detecta los objetos descritos en ``textos``.

        OWLv2 rellena la imagen hasta hacerla cuadrada (abajo y a la derecha),
        por eso las cajas se reescalan con ``(lado_max, lado_max)``. Si se da
        ``zona``, se descartan las detecciones cuyo centro cae fuera de ella
        (la cara también es "piel" para un detector de manos poco preciso).
        """
        import torch

        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with self._cerrojo, torch.inference_mode():
            entradas = self.procesador(text=[list(textos)], images=rgb, return_tensors="pt").to(self.disp)
            salidas = self.modelo(**entradas)
            lado = max(h, w)
            res = self.procesador.post_process_grounded_object_detection(
                salidas, threshold=umbral, target_sizes=[(lado, lado)], text_labels=[list(textos)])[0]
        dets = []
        etiquetas = res.get("text_labels") or res.get("labels")
        for caja, p, et in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(), etiquetas):
            x0, y0, x1, y1 = np.clip(caja, 0, [w, h, w, h])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            nombre = et if isinstance(et, str) else textos[int(et)]
            d = Deteccion((float(x0), float(y0), float(x1), float(y1)), float(p), nombre)
            if zona is not None:
                cx, cy = (int(v) for v in d.centro)
                if not (0 <= cy < zona.shape[0] and 0 <= cx < zona.shape[1] and zona[cy, cx] > 0):
                    continue
            dets.append(d)
        return ResultadoOWL(dets, (time.perf_counter() - t0) * 1000, self.disp)


def obtener_owl() -> DetectorManoOWL:
    """Carga OWLv2 una sola vez por proceso (descarga la primera vez)."""
    with _cerrojo:
        if "owl" not in _cache_modelos:
            _cache_modelos["owl"] = DetectorManoOWL()
        return _cache_modelos["owl"]  # type: ignore[return-value]


# =========================================================================== #
# Comparación clásico vs. transformer
# =========================================================================== #


def iou_cajas(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def caja_de_mascara(m: np.ndarray) -> tuple[float, float, float, float] | None:
    if m is None or not m.any():
        return None
    x, y, w, h = cv2.boundingRect(m)
    return (float(x), float(y), float(x + w), float(y + h))


@dataclass
class Comparacion:
    iou: float | None
    distancia_px: float | None
    distancia_cm: float | None
    ms_clasico: float
    ms_owl: float
    caja_clasico: tuple | None
    caja_owl: tuple | None


def comparar_clasico_vs_owl(mascara_clasica: np.ndarray | None, ms_clasico: float, owl: ResultadoOWL,
                            a_cm=None) -> Comparacion:
    """Compara la caja de la mano clásica (piel + morfología + contorno) con la de OWLv2.

    ``a_cm`` es una función opcional píxel → (X, Y) en cm (rayo ∩ mesa) para
    expresar la diferencia de ubicación en centímetros.
    """
    cc = caja_de_mascara(mascara_clasica) if mascara_clasica is not None else None
    mejor = owl.mejor
    co = mejor.caja if mejor else None
    iou = iou_cajas(cc, co) if cc and co else None
    dpx = dcm = None
    if cc and co:
        pc = np.array([(cc[0] + cc[2]) / 2, (cc[1] + cc[3]) / 2])
        po = np.array(mejor.centro)
        dpx = float(np.linalg.norm(pc - po))
        if a_cm is not None:
            qc, qo = a_cm(tuple(pc)), a_cm(tuple(po))
            if qc is not None and qo is not None:
                dcm = float(np.linalg.norm(np.asarray(qc)[:2] - np.asarray(qo)[:2]))
    return Comparacion(iou, dpx, dcm, ms_clasico, owl.ms, cc, co)


# =========================================================================== #
# Inferencia con GPU bajo demanda (ZeroGPU) — usar estas funciones desde la UI y el juego
# =========================================================================== #


@gpu(duracion=20)
def estimar_profundidad(img_bgr: np.ndarray) -> ResultadoProfundidad:
    """Depth Anything sobre una imagen. En ZeroGPU pide la GPU solo durante la llamada."""
    return obtener_profundidad().estimar(img_bgr)


@gpu(duracion=30)
def detectar_manos(img_bgr: np.ndarray, textos: tuple[str, ...] = ("a hand",), umbral: float = 0.2,
                   zona: np.ndarray | None = None) -> ResultadoOWL:
    """OWLv2 sobre una imagen. En ZeroGPU pide la GPU solo durante la llamada."""
    return obtener_owl().detectar(img_bgr, textos, umbral, zona)


def precargar_modelos() -> None:
    """Descarga y carga ambos modelos al arrancar (recomendado en ZeroGPU: carga global, no dentro de la GPU)."""
    obtener_profundidad()
    obtener_owl()
