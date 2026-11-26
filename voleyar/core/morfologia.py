r"""Morfología matemática sobre la máscara de la mano.

La máscara que sale del umbral tiene dos defectos típicos: **puntos sueltos**
(ruido de la cámara con color parecido a la piel) y **huecos** (brillos sobre
la piel, sombras entre los dedos). La morfología los corrige usando un
**elemento estructurante** :math:`B` (forma y tamaño configurables).

Operaciones básicas (imágenes binarias)
---------------------------------------
- **Erosión**: :math:`A \ominus B = \{z \mid B_z \subseteq A\}`. Un píxel queda
  encendido solo si *todo* el elemento cabe dentro de la máscara. Encoge las
  regiones y elimina objetos más pequeños que :math:`B`. En niveles de gris
  es el mínimo de la vecindad.
- **Dilatación**: :math:`A \oplus B = \{z \mid \hat B_z \cap A \neq \emptyset\}`.
  Basta con que el elemento toque la máscara. Agranda las regiones y rellena
  huecos pequeños. En gris es el máximo de la vecindad.

Operaciones compuestas
----------------------
- **Apertura** :math:`A \circ B = (A \ominus B) \oplus B`: quita puntos sueltos y
  salientes finos sin cambiar mucho el tamaño de la mano.
- **Cierre** :math:`A \bullet B = (A \oplus B) \ominus B`: rellena huecos y
  grietas pequeñas (por ejemplo, entre los dedos juntos).
- **Gradiente** :math:`(A \oplus B) - (A \ominus B)`: deja solo el borde.

Orden en el juego: primero **apertura** (para no "pegar" el ruido a la mano al
cerrar) y luego **cierre**; al final se rellenan los huecos interiores.
"""

from __future__ import annotations

import cv2
import numpy as np

FORMAS = {"elipse": cv2.MORPH_ELLIPSE, "rectangulo": cv2.MORPH_RECT, "cruz": cv2.MORPH_CROSS}
OPERACIONES = ("erosion", "dilatacion", "apertura", "cierre", "gradiente")


def elemento_estructurante(forma: str = "elipse", tam: int = 5) -> np.ndarray:
    """Elemento estructurante de ``tam × tam`` píxeles (forma: elipse, rectángulo o cruz)."""
    tam = max(1, int(tam))
    return cv2.getStructuringElement(FORMAS[forma], (tam, tam))


def erosion(m: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.erode(m, ee, iterations=iteraciones)


def dilatacion(m: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.dilate(m, ee, iterations=iteraciones)


def apertura(m: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, ee, iterations=iteraciones)


def cierre(m: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ee, iterations=iteraciones)


def gradiente(m: np.ndarray, ee: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, ee)


def aplicar(m: np.ndarray, operacion: str, forma: str = "elipse", tam: int = 5, iteraciones: int = 1) -> np.ndarray:
    """Aplica una operación por nombre (para la UI)."""
    ee = elemento_estructurante(forma, tam)
    if operacion == "erosion":
        return erosion(m, ee, iteraciones)
    if operacion == "dilatacion":
        return dilatacion(m, ee, iteraciones)
    if operacion == "apertura":
        return apertura(m, ee, iteraciones)
    if operacion == "cierre":
        return cierre(m, ee, iteraciones)
    if operacion == "gradiente":
        return gradiente(m, ee)
    raise ValueError(f"Operación desconocida: {operacion}")


# --------------------------------------------------------------------------- #
# Implementación manual (didáctica) de erosión y dilatación
# --------------------------------------------------------------------------- #


def _vecindades(m: np.ndarray, ee: np.ndarray, relleno: int):
    kh, kw = ee.shape
    a, b = kh // 2, kw // 2
    pad = np.pad(m, ((a, a), (b, b)), mode="constant", constant_values=relleno)
    h, w = m.shape
    for i in range(kh):
        for j in range(kw):
            if ee[i, j]:
                yield pad[i:i + h, j:j + w]


def erosion_manual(m: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Erosión como **mínimo** de la vecindad definida por el elemento estructurante.

    Fuera de la imagen se rellena con el máximo (255) para no erosionar los
    bordes, igual que el valor por defecto de OpenCV.
    """
    out = np.full_like(m, 255)
    for v in _vecindades(m, ee, 255):
        np.minimum(out, v, out=out)
    return out


def dilatacion_manual(m: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Dilatación como **máximo** de la vecindad (elemento reflejado; aquí es simétrico)."""
    ee = ee[::-1, ::-1]
    out = np.zeros_like(m)
    for v in _vecindades(m, ee, 0):
        np.maximum(out, v, out=out)
    return out


# --------------------------------------------------------------------------- #
# Limpieza de la máscara de la mano
# --------------------------------------------------------------------------- #


def rellenar_huecos(m: np.ndarray) -> np.ndarray:
    """Rellena los huecos interiores (regiones de fondo que no tocan el borde).

    Se inunda el fondo desde el borde con ``floodFill``; lo que no se alcanzó
    son agujeros dentro de la mano y se suman a la máscara.
    """
    h, w = m.shape
    relleno = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mascara_ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(relleno, mascara_ff, (0, 0), 255)
    huecos = cv2.bitwise_not(relleno)[1:-1, 1:-1]
    return cv2.bitwise_or(m, huecos)


def limpiar_mascara(m: np.ndarray, tam_apertura: int = 5, tam_cierre: int = 5, forma: str = "elipse",
                    rellenar: bool = True) -> np.ndarray:
    """Apertura (quita puntos) → cierre (une grietas) → relleno de huecos."""
    out = m
    if tam_apertura > 1:
        out = apertura(out, elemento_estructurante(forma, tam_apertura))
    if tam_cierre > 1:
        out = cierre(out, elemento_estructurante(forma, tam_cierre))
    if rellenar:
        out = rellenar_huecos(out)
    return out
