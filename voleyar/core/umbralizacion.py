r"""Umbralización y binarización.

Convierte los mapas de color o de probabilidad de piel en una máscara binaria
de la mano. Se comparan tres familias:

1. **Rangos fijos de color** (``cv2.inRange``) en HSV y en YCrCb. No necesitan
   calibración, pero dependen de la luz y del tono de piel. En YCrCb la
   luminancia queda separada en Y, así que los rangos de Cr-Cb suelen ser más
   robustos que los de H-S (el tono H de la piel se mezcla con el de la madera
   y con el rojo, que además "da la vuelta" en 0/180).
2. **Otsu** sobre la retroproyección: busca automáticamente el umbral que
   mejor separa "piel" de "no piel" en ese frame.
3. **Adaptativo**: un umbral distinto por vecindad,
   :math:`T(x, y) = \mu_{W}(x, y) - C` (media o media gaussiana en una ventana
   :math:`W`). Útil con iluminación no uniforme, pero sobre la retroproyección
   tiende a marcar bordes internos en lugar de regiones.

Método de Otsu
--------------
Con el histograma normalizado :math:`p_i` y un umbral :math:`t`:

.. math::

   \omega_0(t) = \sum_{i \le t} p_i,\quad
   \mu(t) = \sum_{i \le t} i\,p_i,\quad
   \mu_T = \sum_i i\,p_i

La **varianza entre clases** es

.. math::  \sigma_B^2(t) = \frac{\left[\mu_T\,\omega_0(t) - \mu(t)\right]^2}{\omega_0(t)\,\left[1 - \omega_0(t)\right]}

y el umbral óptimo es :math:`T^* = \arg\max_t \sigma_B^2(t)`. Maximizarla es
equivalente a minimizar la varianza dentro de las clases, y se calcula para
los 256 umbrales con sumas acumuladas en :math:`O(L)`.
"""

from __future__ import annotations

import cv2
import numpy as np

# Rangos de piel por defecto (valores de OpenCV: H ∈ [0, 179], resto ∈ [0, 255])
RANGO_HSV_PIEL = ((0, 40, 60), (25, 200, 255))
RANGO_YCRCB_PIEL = ((0, 135, 80), (255, 175, 128))


def umbral_hsv(img_bgr: np.ndarray, bajo=RANGO_HSV_PIEL[0], alto=RANGO_HSV_PIEL[1]) -> np.ndarray:
    """Máscara de piel por rango en HSV.

    Si ``bajo[0] > alto[0]`` el rango de tono se interpreta como envolvente
    (p. ej. H ∈ [170, 179] ∪ [0, 15]), porque el rojo está en ambos extremos.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if bajo[0] <= alto[0]:
        return cv2.inRange(hsv, tuple(bajo), tuple(alto))
    m1 = cv2.inRange(hsv, (bajo[0], bajo[1], bajo[2]), (179, alto[1], alto[2]))
    m2 = cv2.inRange(hsv, (0, bajo[1], bajo[2]), (alto[0], alto[1], alto[2]))
    return cv2.bitwise_or(m1, m2)


def umbral_ycrcb(img_bgr: np.ndarray, bajo=RANGO_YCRCB_PIEL[0], alto=RANGO_YCRCB_PIEL[1]) -> np.ndarray:
    """Máscara de piel por rango en YCrCb (canales Y, Cr, Cb)."""
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycc, tuple(bajo), tuple(alto))


def otsu_manual(gris: np.ndarray, mascara: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """Umbral de Otsu implementado con NumPy.

    Returns
    -------
    T: umbral óptimo (los píxeles ``> T`` son primer plano, como en OpenCV).
    sigma_b: curva :math:`\\sigma_B^2(t)` para t = 0…255 (para graficarla).

    Si se da ``mascara``, el histograma se calcula solo con esos píxeles (en el
    juego: la zona de la cancha, para que el fondo lejano no influya).
    """
    datos = gris[mascara > 0] if mascara is not None else gris.ravel()
    hist = np.bincount(datos.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0, np.zeros(256)
    p = hist / total
    i = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(i * p)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b)), sigma_b


def otsu_opencv(gris: np.ndarray) -> int:
    """Umbral de Otsu de OpenCV (para comparar)."""
    t, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def binarizar(gris: np.ndarray, T: float, invertir: bool = False) -> np.ndarray:
    """``255`` donde ``gris > T`` (o al revés si ``invertir``)."""
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    return cv2.threshold(gris, T, 255, tipo)[1]


def umbral_adaptativo(gris: np.ndarray, metodo: str = "gaussiano", bloque: int = 31, C: float = -5) -> np.ndarray:
    """Umbral local: media (o media gaussiana) de una ventana ``bloque × bloque`` menos ``C``."""
    bloque = max(3, int(bloque) | 1)
    m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if metodo == "gaussiano" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(gris, 255, m, cv2.THRESH_BINARY, bloque, C)


def umbral_piel_otsu(retro: np.ndarray, mascara: np.ndarray | None = None, t_min: int = 30) -> tuple[np.ndarray, int]:
    """Binariza la retroproyección con Otsu (manual) dentro de la zona de búsqueda.

    ``t_min`` evita que, sin mano en la escena, Otsu "invente" un umbral bajo
    que convierta el ruido de la retroproyección en manchas de piel.
    """
    T, _ = otsu_manual(retro, mascara)
    T = max(T, t_min)
    m = binarizar(retro, T)
    if mascara is not None:
        m = cv2.bitwise_and(m, mascara)
    return m, T


def metricas_mascara(pred: np.ndarray, verdad: np.ndarray) -> dict[str, float]:
    """IoU, precisión y exhaustividad de una máscara frente a la verdad."""
    p, v = pred > 0, verdad > 0
    tp = float(np.logical_and(p, v).sum())
    fp = float(np.logical_and(p, ~v).sum())
    fn = float(np.logical_and(~p, v).sum())
    return {
        "iou": tp / max(1.0, tp + fp + fn),
        "precision": tp / max(1.0, tp + fp),
        "exhaustividad": tp / max(1.0, tp + fn),
    }


def evaluar_en_recuadro(mascara: np.ndarray, rect: tuple[int, int, int, int],
                        zona: np.ndarray | None = None) -> dict[str, float]:
    """Comparación sin verdad completa: % de piel dentro del recuadro de la mano
    (debería ser alto) y % de "falsa piel" fuera de él (debería ser bajo)."""
    x, y, w, h = rect
    dentro = np.zeros(mascara.shape, bool)
    dentro[y:y + h, x:x + w] = True
    fuera = ~dentro if zona is None else np.logical_and(~dentro, zona > 0)
    m = mascara > 0
    return {
        "piel_en_recuadro_%": 100.0 * m[dentro].mean() if dentro.any() else 0.0,
        "falsa_piel_fuera_%": 100.0 * m[fuera].mean() if fuera.any() else 0.0,
    }
