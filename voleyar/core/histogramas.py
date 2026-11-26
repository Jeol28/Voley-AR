r"""Histogramas: modelo de color de la piel, retroproyección, ecualización y CLAHE.

Calibración del color de la piel
--------------------------------
Al iniciar el juego pones la mano dentro de un recuadro. Con esos píxeles se
construye un histograma 2D de **crominancia** (se descarta la luminancia para
que el modelo tolere sombras y cambios de luz):

- **H-S** en HSV: tono (0-179 en OpenCV) y saturación.
- **Cr-Cb** en YCrCb: diferencias de rojo y azul respecto de la luminancia.
  La piel humana de casi cualquier tono ocupa una zona compacta de Cr-Cb.

El histograma normalizado :math:`\hat h(c)` es una estimación de
:math:`P(\text{color} = c \mid \text{piel})`.

Retroproyección
---------------
``cv2.calcBackProject`` reemplaza cada píxel por el valor del histograma en su
color: :math:`B(x, y) = \hat h(c(x, y))`. El resultado es un mapa de
"probabilidad de piel" que luego se umbraliza (Otsu) para obtener la mano.
Aquí se combinan los dos espacios con la media geométrica
:math:`B = \sqrt{B_{HS} \cdot B_{CrCb}}`, que exige que ambos estén de acuerdo.

Ecualización
------------
Con la función de distribución acumulada :math:`\text{cdf}(k) = \sum_{j \le k} h(j)`,
la ecualización aplica

.. math::  s_k = \operatorname{round}\!\left(\frac{\text{cdf}(k) - \text{cdf}_{\min}}{N - \text{cdf}_{\min}}\,(L - 1)\right)

que estira el histograma para ocupar todo el rango. **CLAHE** hace lo mismo en
celdas pequeñas, recortando cada histograma local a un *clip limit* (para no
amplificar el ruido) e interpolando entre celdas. En el juego se aplica CLAHE
a la luminancia (canal L de Lab) para compensar los cambios de luz del cuarto
sin alterar demasiado el color.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BINS_HS = (30, 32)
RANGOS_HS = [0, 180, 0, 256]
BINS_CRCB = (32, 32)
RANGOS_CRCB = [0, 256, 0, 256]


# --------------------------------------------------------------------------- #
# Histogramas básicos
# --------------------------------------------------------------------------- #


def histograma_gris(img: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    """Histograma de 256 niveles de la luminancia (o de la imagen en gris)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([g], [0], mascara, [256], [0, 256]).ravel()


def histogramas_bgr(img: np.ndarray, mascara: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return {n: cv2.calcHist([img], [i], mascara, [256], [0, 256]).ravel() for i, n in enumerate("BGR")}


def histograma_hs(img_bgr: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1], mascara, list(BINS_HS), RANGOS_HS)


def histograma_crcb(img_bgr: np.ndarray, mascara: np.ndarray | None = None) -> np.ndarray:
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.calcHist([ycc], [1, 2], mascara, list(BINS_CRCB), RANGOS_CRCB)


# --------------------------------------------------------------------------- #
# Modelo de piel
# --------------------------------------------------------------------------- #


@dataclass
class ModeloPiel:
    """Histogramas de crominancia de la piel del jugador (normalizados a 0-255)."""

    hist_hs: np.ndarray
    hist_crcb: np.ndarray
    n_pixeles: int = 0

    def a_dict(self) -> dict:
        return {"hist_hs": self.hist_hs.tolist(), "hist_crcb": self.hist_crcb.tolist(), "n_pixeles": self.n_pixeles}

    @classmethod
    def desde_dict(cls, d: dict) -> "ModeloPiel":
        return cls(np.array(d["hist_hs"], np.float32), np.array(d["hist_crcb"], np.float32), int(d.get("n_pixeles", 0)))


def mascara_valida_color(img_bgr: np.ndarray, s_min: int = 25, v_min: int = 40, v_max: int = 250) -> np.ndarray:
    """Excluye píxeles muy oscuros, saturados a blanco o grises: ahí el tono no es fiable."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, s_min, v_min), (180, 255, v_max))


def calibrar_piel(img_bgr: np.ndarray, rect: tuple[int, int, int, int] | None = None,
                  mascara: np.ndarray | None = None, suavizar_hist: bool = True) -> ModeloPiel:
    """Construye el modelo de piel con los píxeles del recuadro ``rect = (x, y, w, h)``.

    Se suaviza el histograma 2D con un gaussiano pequeño para generalizar a
    colores cercanos que no aparecieron en la muestra.
    """
    m = np.zeros(img_bgr.shape[:2], np.uint8)
    if rect is not None:
        x, y, w, h = rect
        m[max(0, y):y + h, max(0, x):x + w] = 255
    if mascara is not None:
        m = mascara.copy() if rect is None else cv2.bitwise_and(m, mascara)
    m = cv2.bitwise_and(m, mascara_valida_color(img_bgr))
    n = int(np.count_nonzero(m))
    if n < 50:
        raise ValueError("Muy pocos píxeles válidos en el recuadro: acerca la mano o mejora la luz.")
    hs = histograma_hs(img_bgr, m)
    cc = histograma_crcb(img_bgr, m)
    if suavizar_hist:
        hs = cv2.GaussianBlur(hs, (3, 3), 0.8)
        cc = cv2.GaussianBlur(cc, (3, 3), 0.8)
    cv2.normalize(hs, hs, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(cc, cc, 0, 255, cv2.NORM_MINMAX)
    return ModeloPiel(hs.astype(np.float32), cc.astype(np.float32), n)


ESPACIOS_RETRO = ("combinada", "hs", "crcb")


def retroproyeccion(img_bgr: np.ndarray, modelo: ModeloPiel, espacio: str = "combinada",
                    disco: int = 0, mascara: np.ndarray | None = None) -> np.ndarray:
    """Mapa de probabilidad de piel (uint8 0-255) por retroproyección.

    ``disco > 0`` convoluciona el mapa con un disco de ese diámetro, que
    rellena los huecos pequeños antes del umbral (como en el tutorial de
    CamShift). ``mascara`` limita el resultado a la zona de búsqueda.
    """
    if espacio not in ESPACIOS_RETRO:
        raise ValueError(f"Espacio desconocido: {espacio}")
    bp_hs = bp_cc = None
    if espacio in ("hs", "combinada"):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bp_hs = cv2.calcBackProject([hsv], [0, 1], modelo.hist_hs, RANGOS_HS, 1)
    if espacio in ("crcb", "combinada"):
        ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        bp_cc = cv2.calcBackProject([ycc], [1, 2], modelo.hist_crcb, RANGOS_CRCB, 1)
    if espacio == "hs":
        bp = bp_hs
    elif espacio == "crcb":
        bp = bp_cc
    else:
        bp = np.sqrt(bp_hs.astype(np.float32) * bp_cc.astype(np.float32)).astype(np.uint8)
    # Los píxeles sin color fiable no pueden ser piel
    bp = cv2.bitwise_and(bp, mascara_valida_color(img_bgr, s_min=15, v_min=30, v_max=255))
    if disco > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disco, disco)).astype(np.float32)
        bp = cv2.filter2D(bp, -1, k / k.sum())
    if mascara is not None:
        bp = cv2.bitwise_and(bp, mascara)
    return bp


# --------------------------------------------------------------------------- #
# Ecualización y CLAHE
# --------------------------------------------------------------------------- #


def ecualizacion_manual(gris: np.ndarray) -> np.ndarray:
    """Ecualización por la CDF (misma fórmula que ``cv2.equalizeHist``)."""
    hist = np.bincount(gris.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    no_cero = np.flatnonzero(hist)
    if len(no_cero) <= 1:
        return gris.copy()
    cdf_min = cdf[no_cero[0]]
    escala = np.float32(255.0 / (gris.size - cdf_min))
    lut = np.clip(np.rint((cdf - cdf_min).astype(np.float32) * escala), 0, 255).astype(np.uint8)
    return lut[gris]


def ecualizar(img: np.ndarray) -> np.ndarray:
    """Ecualización global; en color solo se ecualiza la luminancia (Y de YCrCb)."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def aplicar_clahe(img: np.ndarray, clip: float = 2.0, celdas: int = 8) -> np.ndarray:
    """CLAHE; en color se aplica al canal L de Lab (la crominancia no cambia)."""
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(int(celdas), int(celdas)))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def contraste(img: np.ndarray) -> float:
    """Desviación estándar de la luminancia (medida simple de contraste)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(g.std())


# --------------------------------------------------------------------------- #
# Color del balón: el que menos aparece en la escena
# --------------------------------------------------------------------------- #

# nombre: (tono en OpenCV 0-179, color BGR de los paneles)
COLORES_BALON = {
    "naranja": (9, (53, 107, 255)),
    "amarillo": (28, (20, 215, 255)),
    "cian": (90, (230, 205, 20)),
    "azul": (112, (235, 120, 35)),
    "morado": (140, (210, 70, 150)),
    "rosa": (165, (175, 60, 255)),
}


def histograma_tono(img_bgr: np.ndarray, s_min: int = 60, v_min: int = 50) -> np.ndarray:
    """Histograma de tono (180 bins) de los píxeles **con color**, como fracción de TODA la imagen.

    Los píxeles grises, blancos o muy oscuros no tienen un tono fiable y no compiten con
    el balón, así que no cuentan: en un cuarto gris todas las barras quedan bajas.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, s_min, v_min), (180, 255, 255))
    h = cv2.calcHist([hsv], [0], m, [180], [0, 180]).ravel()
    return (h / max(img_bgr.shape[0] * img_bgr.shape[1], 1)).astype(np.float32)


def masa_de_tono(hist_tono: np.ndarray, tono: int, ancho: int = 12) -> float:
    """Cuánta escena tiene un tono parecido (±``ancho``, el tono es circular: 179 está junto a 0)."""
    idx = np.arange(tono - ancho, tono + ancho + 1) % 180
    return float(hist_tono[idx].sum())


def color_de_contraste(hist_tono: np.ndarray) -> tuple[str, dict[str, float]]:
    """El color del balón que menos se parece a la escena, y cuánto se parece cada uno."""
    masas = {n: masa_de_tono(hist_tono, t) for n, (t, _) in COLORES_BALON.items()}
    return min(masas, key=masas.get), masas
