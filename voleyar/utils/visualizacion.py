"""Utilidades de visualización: paleta, texto con tildes, mosaicos y gráficas."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# Paleta del proyecto (hex → BGR para OpenCV)
GRAFITO = "#1A2332"
NARANJA = "#FF6B35"
NEUTRO = "#F5F5F7"
VERDE = "#00C896"  # solo datos: golpe válido / acierto
ROJO = "#FF4C4C"   # solo datos: golpe inválido / error


def hex_a_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


BGR_GRAFITO = hex_a_bgr(GRAFITO)
BGR_NARANJA = hex_a_bgr(NARANJA)
BGR_NEUTRO = hex_a_bgr(NEUTRO)
BGR_VERDE = hex_a_bgr(VERDE)
BGR_ROJO = hex_a_bgr(ROJO)

_FUENTES = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=16)
def _fuente(tam: int, negrita: bool = False):
    from PIL import ImageFont

    for ruta in _FUENTES:
        if Path(ruta).exists():
            try:
                f = ImageFont.truetype(ruta, tam)
                if negrita:
                    try:
                        f.set_variation_by_name("Bold")
                    except Exception:
                        pass
                return f
            except Exception:
                continue
    return ImageFont.load_default(size=tam)


@lru_cache(maxsize=512)
def _render_texto(texto: str, tam: int, color: tuple[int, int, int], negrita: bool) -> np.ndarray:
    """Renderiza el texto una sola vez a un parche BGRA (caché LRU)."""
    from PIL import Image, ImageDraw

    f = _fuente(tam, negrita)
    x0, y0, x1, y1 = f.getbbox(texto)
    w, h = max(1, x1 - min(0, x0) + 2), max(1, int(tam * 1.3))
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), texto, font=f, fill=(color[2], color[1], color[0], 255))
    rgba = np.array(im)
    return np.dstack([rgba[..., 2], rgba[..., 1], rgba[..., 0], rgba[..., 3]])


def poner_texto(img: np.ndarray, texto: str, pos: tuple[int, int], tam: int = 22,
                color: tuple[int, int, int] = BGR_NEUTRO, negrita: bool = False,
                fondo: tuple[int, int, int] | None = None, alfa_fondo: float = 0.65) -> np.ndarray:
    """Escribe texto con tildes y eñes (Hershey de OpenCV no las soporta). Modifica ``img``."""
    parche = _render_texto(texto, tam, tuple(int(c) for c in color), negrita)
    h, w = parche.shape[:2]
    x, y = int(pos[0]), int(pos[1])
    if fondo is not None:
        pad = max(4, tam // 4)
        x0, y0 = max(0, x - pad), max(0, y - pad // 2)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad // 2)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1].astype(np.float32)
            roi = roi * (1 - alfa_fondo) + np.array(fondo, np.float32) * alfa_fondo
            img[y0:y1, x0:x1] = roi.astype(np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return img
    p = parche[y0 - y:y1 - y, x0 - x:x1 - x]
    a = p[..., 3:4].astype(np.float32) / 255.0
    img[y0:y1, x0:x1] = (p[..., :3] * a + img[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    return img


def a_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def normalizar_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx - mn < 1e-9:
        return np.zeros(img.shape, np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def mosaico(imagenes: list[np.ndarray], titulos: list[str] | None = None, columnas: int = 3,
            ancho_celda: int = 420) -> np.ndarray:
    """Une varias imágenes en una cuadrícula con título sobre cada una."""
    celdas = []
    for i, im in enumerate(imagenes):
        im = a_bgr(im)
        h, w = im.shape[:2]
        alto = int(round(h * ancho_celda / w))
        c = cv2.resize(im, (ancho_celda, alto), interpolation=cv2.INTER_AREA)
        if titulos:
            poner_texto(c, titulos[i], (8, 6), 18, BGR_NEUTRO, fondo=BGR_GRAFITO)
        celdas.append(c)
    alto_max = max(c.shape[0] for c in celdas)
    celdas = [cv2.copyMakeBorder(c, 0, alto_max - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BGR_GRAFITO)
              for c in celdas]
    while len(celdas) % columnas:
        celdas.append(np.full_like(celdas[0], BGR_GRAFITO))
    filas = [np.hstack(celdas[i:i + columnas]) for i in range(0, len(celdas), columnas)]
    return np.vstack(filas)


def bgr_a_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a_bgr(img), cv2.COLOR_BGR2RGB)


def rgb_a_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Gráficas con Matplotlib (backend Agg, sin ventanas)
# --------------------------------------------------------------------------- #


def _estilo_figura(ax, fig):
    fig.patch.set_facecolor(GRAFITO)
    ax.set_facecolor("#222C3C")
    for s in ax.spines.values():
        s.set_color("#3A4658")
    ax.tick_params(colors=NEUTRO, labelsize=8)
    ax.xaxis.label.set_color(NEUTRO)
    ax.yaxis.label.set_color(NEUTRO)
    ax.title.set_color(NEUTRO)
    ax.grid(color="#2E3A4C", linewidth=0.6)


def figura_a_imagen(fig) -> np.ndarray:
    """Convierte una figura de Matplotlib en imagen RGB (numpy)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    import matplotlib.pyplot as plt

    plt.close(fig)
    return rgba[..., :3].copy()


def grafica_lineas(series: dict[str, np.ndarray], titulo: str = "", xlabel: str = "", ylabel: str = "",
                   marcas_x: dict[str, float] | None = None, tam=(6.0, 3.0), colores: dict[str, str] | None = None,
                   relleno: bool = False) -> np.ndarray:
    """Gráfica de líneas (histogramas, curva de Otsu, etc.) como imagen RGB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tam, dpi=100)
    _estilo_figura(ax, fig)
    paleta = [NARANJA, NEUTRO, "#7FA7D9", "#B8C2D1"]
    for i, (nombre, y) in enumerate(series.items()):
        c = (colores or {}).get(nombre, paleta[i % len(paleta)])
        ax.plot(np.arange(len(y)), y, color=c, linewidth=1.6, label=nombre)
        if relleno:
            ax.fill_between(np.arange(len(y)), y, color=c, alpha=0.18)
    for nombre, x in (marcas_x or {}).items():
        ax.axvline(x, color=NARANJA, linestyle="--", linewidth=1.2)
        ax.text(x, ax.get_ylim()[1] * 0.92, f" {nombre}", color=NARANJA, fontsize=9)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        leg = ax.legend(fontsize=8, facecolor=GRAFITO, edgecolor="#3A4658")
        for t in leg.get_texts():
            t.set_color(NEUTRO)
    fig.tight_layout()
    return figura_a_imagen(fig)


def grafica_barras(valores: list[float], etiquetas: list[str], titulo: str = "", ylabel: str = "",
                   linea: float | None = None, tam=(6.0, 2.8)) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tam, dpi=100)
    _estilo_figura(ax, fig)
    ax.bar(etiquetas, valores, color=NARANJA)
    if linea is not None:
        ax.axhline(linea, color=NEUTRO, linestyle="--", linewidth=1)
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    return figura_a_imagen(fig)


def mapa_2d(h: np.ndarray, titulo: str, xlabel: str, ylabel: str, extent=None, tam=(4.2, 3.4)) -> np.ndarray:
    """Histograma 2D (H-S o Cr-Cb) como mapa de calor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tam, dpi=100)
    _estilo_figura(ax, fig)
    ax.imshow(np.log1p(h), origin="lower", aspect="auto", cmap="inferno", extent=extent)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    fig.tight_layout()
    return figura_a_imagen(fig)
