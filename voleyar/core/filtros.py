r"""Filtros espaciales.

En el juego los filtros cumplen dos funciones:

1. **Suavizado antes de segmentar la piel.** La cámara web tiene ruido de
   sensor (más con poca luz) que produce píxeles sueltos con color de piel.
   Un suavizado reduce ese ruido antes de la retroproyección y el umbral.
2. **Bordes de la mano.** Sobel y Canny muestran el contorno de la mano en el
   panel de depuración y sirven para comparar con el contorno de la máscara.

Convolución
-----------
Para una imagen :math:`f` y un núcleo :math:`h` de tamaño :math:`(2a+1)\times(2b+1)`:

.. math::  g(x, y) = (f * h)(x, y) = \sum_{s=-a}^{a}\sum_{t=-b}^{b} h(s, t)\, f(x - s,\, y - t)

``cv2.filter2D`` calcula en realidad la **correlación** (sin voltear el
núcleo): :math:`\sum h(s,t)\,f(x+s, y+t)`. Por eso, para comparar con la
convolución verdadera hay que pasarle el núcleo volteado 180°. Para núcleos
simétricos (gaussiano, caja) ambas operaciones coinciden.

Filtros de suavizado
--------------------
- **Gaussiano**: :math:`h(s,t) \propto e^{-(s^2+t^2)/2\sigma^2}`. Lineal y rápido;
  difumina también los bordes.
- **Mediana**: no lineal, reemplaza cada píxel por la mediana de su vecindad.
  Elimina ruido impulsivo (sal y pimienta) sin crear valores nuevos.
- **Bilateral**: promedia con pesos que combinan distancia espacial y
  diferencia de intensidad, :math:`w = e^{-\|p-q\|^2/2\sigma_s^2}\,e^{-|I_p-I_q|^2/2\sigma_r^2}`.
  Suaviza la piel conservando el borde mano/cancha (pero es el más lento).

Detección de bordes
-------------------
- **Sobel**: aproxima el gradiente con los núcleos
  :math:`G_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}` y
  :math:`G_y = G_x^T`. Magnitud :math:`\sqrt{G_x^2 + G_y^2}`.
- **Canny**: gaussiano → gradiente de Sobel → supresión de no máximos →
  histéresis con dos umbrales (los bordes débiles solo sobreviven si tocan
  uno fuerte).
"""

from __future__ import annotations

import cv2
import numpy as np

METODOS_SUAVIZADO = ("ninguno", "gaussiano", "mediana", "bilateral")


def _impar(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def suavizar(img: np.ndarray, metodo: str = "gaussiano", k: int = 5, sigma: float = 0.0,
             sigma_color: float = 50.0, sigma_espacio: float = 7.0) -> np.ndarray:
    """Aplica el suavizado elegido. ``k`` es el tamaño del núcleo (se fuerza impar)."""
    k = _impar(k)
    if metodo == "ninguno" or k == 1:
        return img.copy()
    if metodo == "gaussiano":
        return cv2.GaussianBlur(img, (k, k), sigma)
    if metodo == "mediana":
        return cv2.medianBlur(img, k)
    if metodo == "bilateral":
        return cv2.bilateralFilter(img, k, sigma_color, sigma_espacio)
    raise ValueError(f"Método de suavizado desconocido: {metodo}")


def _a_gris(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sobel(img: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradientes de Sobel. Devuelve ``(gx, gy, magnitud)`` en float32."""
    g = _a_gris(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=_impar(k))
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=_impar(k))
    return gx, gy, cv2.magnitude(gx, gy)


def canny(img: np.ndarray, umbral_bajo: float = 50, umbral_alto: float = 150, apertura: int = 3) -> np.ndarray:
    """Bordes de Canny (imagen binaria 0/255)."""
    return cv2.Canny(_a_gris(img), umbral_bajo, umbral_alto, apertureSize=_impar(max(3, apertura)))


def bordes_mano(img: np.ndarray, mascara_mano: np.ndarray | None, umbral_bajo: float = 40,
                umbral_alto: float = 120) -> np.ndarray:
    """Canny restringido a la vecindad de la mano (para el panel de depuración)."""
    bordes = canny(suavizar(img, "gaussiano", 5), umbral_bajo, umbral_alto)
    if mascara_mano is None:
        return bordes
    zona = cv2.dilate(mascara_mano, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.bitwise_and(bordes, zona)


# --------------------------------------------------------------------------- #
# Convolución manual
# --------------------------------------------------------------------------- #


def parsear_nucleo(texto: str) -> np.ndarray:
    """Convierte ``"1 2 1; 0 0 0; -1 -2 -1"`` (o una fila por línea) en una matriz.

    Acepta fracciones (``1/9``). Lanza ``ValueError`` si las filas no tienen
    el mismo largo o el núcleo no es de tamaño impar.
    """
    filas = [f for f in texto.replace("\n", ";").split(";") if f.strip()]
    datos = []
    for f in filas:
        valores = []
        for tok in f.replace(",", " ").split():
            if "/" in tok:
                n, d = tok.split("/")
                valores.append(float(n) / float(d))
            else:
                valores.append(float(tok))
        datos.append(valores)
    if not datos or len({len(f) for f in datos}) != 1:
        raise ValueError("Todas las filas del núcleo deben tener el mismo número de valores.")
    k = np.array(datos, dtype=np.float32)
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("El núcleo debe tener tamaño impar (3×3, 5×5, …) para tener un centro.")
    return k


def convolucion_manual(img: np.ndarray, nucleo: np.ndarray) -> np.ndarray:
    """Convolución 2D verdadera implementada con NumPy (borde reflejado 101).

    En lugar de recorrer píxel a píxel (lentísimo en Python), se recorre el
    **núcleo**: para cada coeficiente :math:`h(s,t)` se suma la imagen
    desplazada :math:`f(x-s, y-t)` multiplicada por ese coeficiente. Son
    :math:`(2a+1)(2b+1)` operaciones vectorizadas sobre toda la imagen.

    Devuelve float32 (sin recortar) para poder ver valores negativos (p. ej.
    con núcleos de bordes). Funciona en gris o por canal en color.
    """
    nucleo = np.asarray(nucleo, dtype=np.float32)
    kh, kw = nucleo.shape
    a, b = kh // 2, kw // 2
    f = img.astype(np.float32)
    if f.ndim == 2:
        f = f[..., None]
    # BORDER_REFLECT_101 de OpenCV equivale a np.pad(mode="reflect")
    pad = np.pad(f, ((a, a), (b, b), (0, 0)), mode="reflect")
    h, w = f.shape[:2]
    salida = np.zeros_like(f)
    for s in range(-a, a + 1):
        for t in range(-b, b + 1):
            coef = nucleo[s + a, t + b]
            if coef == 0:
                continue
            # f(x - s, y - t): desplazamiento opuesto al índice del núcleo
            salida += coef * pad[a - s:a - s + h, b - t:b - t + w]
    return salida[..., 0] if img.ndim == 2 else salida


def convolucion_opencv(img: np.ndarray, nucleo: np.ndarray) -> np.ndarray:
    """La misma convolución con ``cv2.filter2D`` (volteando el núcleo, ver docstring del módulo)."""
    nucleo = np.asarray(nucleo, dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), cv2.CV_32F, cv2.flip(nucleo, -1),
                        borderType=cv2.BORDER_REFLECT_101)


NUCLEOS_EJEMPLO: dict[str, str] = {
    "Caja 3×3 (promedio)": "1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9",
    "Gaussiano 3×3": "1/16 2/16 1/16; 2/16 4/16 2/16; 1/16 2/16 1/16",
    "Sobel X": "-1 0 1; -2 0 2; -1 0 1",
    "Sobel Y": "-1 -2 -1; 0 0 0; 1 2 1",
    "Laplaciano": "0 1 0; 1 -4 1; 0 1 0",
    "Realce (sharpen)": "0 -1 0; -1 5 -1; 0 -1 0",
    "Relieve (emboss)": "-2 -1 0; -1 1 1; 0 1 2",
}
