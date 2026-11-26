r"""Render de realidad aumentada con ``cv2.projectPoints``.

Todo objeto virtual se define en cm en el sistema de la mesa y se proyecta con
la pose (``rvec, tvec``) y el modelo de cámara. Si ``K`` o la distorsión son
incorrectas (modo "sin calibrar"), la red y el balón quedan desalineados.

- **Red**: malla de líneas (cada 1 cm) entre dos postes, con banda superior.
- **Balón**: esfera sombreada (Lambert + especular de Blinn-Phong) cuyo radio en
  píxeles es :math:`f\,R / z_c`: escala con la profundidad, como uno real.
- **Sombra**: el círculo del balón proyectado sobre ``z = 0`` (luz cenital); es
  la pista visual que más ayuda a percibir la altura.
- **Oclusión** balón/red: se compara la profundidad :math:`z_c` del balón con la
  de la red a la misma altura para decidir el orden de dibujo.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from voleyar.core import config as cfg
from voleyar.core.calibracion import ModeloCamara, Pose, profundidad_camara, proyectar
from voleyar.utils import visualizacion as vis


def _px(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _en_imagen(img: np.ndarray, p: np.ndarray, margen: int = 2000) -> bool:
    h, w = img.shape[:2]
    return -margen < p[0] < w + margen and -margen < p[1] < h + margen


# --------------------------------------------------------------------------- #
# Cancha y red
# --------------------------------------------------------------------------- #


def dibujar_lineas_cancha(img: np.ndarray, pose: Pose, modelo: ModeloCamara,
                          color=vis.BGR_NARANJA, grosor: int = 2) -> np.ndarray:
    """Contorno y línea central virtuales: con buena calibración caen sobre las impresas."""
    L, A = cfg.MEDIO_LARGO, cfg.MEDIO_ANCHO
    t = np.linspace(0, 1, 24)
    esquinas = [(-L, -A), (L, -A), (L, A), (-L, A), (-L, -A)]
    for (x0, y0), (x1, y1) in zip(esquinas[:-1], esquinas[1:]):
        p = np.column_stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.zeros_like(t)])
        cv2.polylines(img, [np.round(proyectar(p, pose, modelo)).astype(np.int32)], False, color, grosor, cv2.LINE_AA)
    p = np.column_stack([np.zeros_like(t), -A + 2 * A * t, np.zeros_like(t)])
    cv2.polylines(img, [np.round(proyectar(p, pose, modelo)).astype(np.int32)], False, color, grosor, cv2.LINE_AA)
    return img


def dibujar_red(img: np.ndarray, pose: Pose, modelo: ModeloCamara, altura: float = cfg.ALTURA_RED,
                alfa: float = 0.85) -> np.ndarray:
    """Red virtual en ``x = 0`` como malla 3D proyectada."""
    A = cfg.MEDIO_ANCHO + 0.4
    z0 = 0.6
    capa = img.copy()
    # Fondo translúcido de la red
    marco = np.array([[0, -A, z0], [0, A, z0], [0, A, altura], [0, -A, altura]], float)
    cv2.fillPoly(capa, [np.round(proyectar(marco, pose, modelo)).astype(np.int32)], (40, 40, 40))
    img[:] = cv2.addWeighted(capa, 0.25, img, 0.75, 0)
    capa = img.copy()
    # Malla cada 1 cm
    for y in np.arange(-A, A + 1e-6, 1.0):
        p = proyectar(np.array([[0, y, z0], [0, y, altura]]), pose, modelo)
        cv2.line(capa, _px(p[0]), _px(p[1]), vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    for z in np.arange(z0, altura + 1e-6, 1.0):
        ys = np.linspace(-A, A, 16)
        p = proyectar(np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, z)]), pose, modelo)
        cv2.polylines(capa, [np.round(p).astype(np.int32)], False, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(capa, 0.55 * alfa, img, 1 - 0.55 * alfa, 0)
    # Banda superior y postes
    ys = np.linspace(-A, A, 16)
    banda = proyectar(np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, altura)]), pose, modelo)
    cv2.polylines(img, [np.round(banda).astype(np.int32)], False, vis.BGR_NEUTRO, 4, cv2.LINE_AA)
    for y in (-A, A):
        p = proyectar(np.array([[0, y, 0], [0, y, altura + 0.8]]), pose, modelo)
        cv2.line(img, _px(p[0]), _px(p[1]), vis.BGR_GRAFITO, 6, cv2.LINE_AA)
        cv2.line(img, _px(p[0]), _px(p[1]), vis.BGR_NARANJA, 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Balón
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=256)
def sprite_balon(radio_px: int, giro: int = 0) -> np.ndarray:
    """Esfera sombreada con paneles de balón de voleibol (BGRA, caché por radio y giro)."""
    R = max(2, int(radio_px))
    n = 2 * R + 3
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x, y = (xx - c) / R, (yy - c) / R
    r2 = x * x + y * y
    dentro = r2 <= 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    normal = np.stack([x, y, z], -1)
    luz = np.array([-0.45, -0.6, 0.66])
    luz /= np.linalg.norm(luz)
    difusa = np.clip(normal @ luz, 0, 1)
    medio = luz + np.array([0, 0, 1.0])
    medio /= np.linalg.norm(medio)
    especular = np.clip(normal @ medio, 0, 1) ** 40
    # Paneles: rotamos la normal y vemos qué eje domina
    a = np.deg2rad(giro * 15)
    Rm = cv2.Rodrigues(np.array([a, a * 0.6, 0.3]))[0].astype(np.float32)
    nr = normal @ Rm.T
    eje = np.argmax(np.abs(nr), axis=-1)
    blanco = np.array(vis.BGR_NEUTRO, np.float32)
    naranja = np.array(vis.BGR_NARANJA, np.float32)
    grafito = np.array((80, 70, 60), np.float32)
    base = np.where((eje == 1)[..., None], naranja, blanco)
    # Costuras: donde dos componentes son casi iguales
    s = np.sort(np.abs(nr), axis=-1)
    costura = (s[..., 2] - s[..., 1]) < 0.06
    base = np.where(costura[..., None], grafito, base)
    color = base * (0.35 + 0.65 * difusa[..., None]) + 255 * 0.55 * especular[..., None]
    alfa = np.clip((1 - np.sqrt(r2)) * R + 0.5, 0, 1) * dentro
    out = np.dstack([np.clip(color, 0, 255), alfa * 255]).astype(np.uint8)
    return out


def _pegar_sprite(img: np.ndarray, sprite: np.ndarray, centro: tuple[float, float]) -> None:
    h, w = sprite.shape[:2]
    x0 = int(round(centro[0] - (w - 1) / 2))
    y0 = int(round(centro[1] - (h - 1) / 2))
    H, W = img.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    s = sprite[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4].astype(np.float32) / 255
    img[ya:yb, xa:xb] = (s[..., :3] * a + img[ya:yb, xa:xb] * (1 - a)).astype(np.uint8)


def radio_en_pixeles(pos: np.ndarray, pose: Pose, modelo: ModeloCamara, radio: float = cfg.RADIO_BALON) -> float:
    """Radio aparente :math:`f R / z_c` (crece al acercarse a la cámara)."""
    zc = float(profundidad_camara(np.asarray(pos)[None], pose)[0])
    f = (modelo.K[0, 0] + modelo.K[1, 1]) / 2
    return f * radio / max(zc, 1e-3)


def dibujar_sombra(img: np.ndarray, pos: np.ndarray, pose: Pose, modelo: ModeloCamara,
                   radio: float = cfg.RADIO_BALON) -> None:
    """Sombra elíptica: el círculo del balón proyectado sobre la mesa (se difumina con la altura)."""
    altura = max(0.0, float(pos[2]) - radio)
    r = radio * (1.0 + altura / 25)
    t = np.linspace(0, 2 * np.pi, 28)
    circ = np.column_stack([pos[0] + r * np.cos(t), pos[1] + r * np.sin(t), np.zeros_like(t)])
    p = proyectar(circ, pose, modelo)
    if not _en_imagen(img, p.mean(axis=0)):
        return
    capa = img.copy()
    cv2.fillPoly(capa, [np.round(p).astype(np.int32)], (15, 15, 20), cv2.LINE_AA)
    a = float(np.clip(0.55 - altura / 40, 0.15, 0.55))
    img[:] = cv2.addWeighted(capa, a, img, 1 - a, 0)


def dibujar_balon(img: np.ndarray, pos: np.ndarray, pose: Pose, modelo: ModeloCamara, giro: int = 0,
                  radio: float = cfg.RADIO_BALON) -> None:
    c = proyectar(np.asarray(pos)[None], pose, modelo)[0]
    if not _en_imagen(img, c, 200):
        return
    r = radio_en_pixeles(pos, pose, modelo, radio)
    _pegar_sprite(img, sprite_balon(int(round(np.clip(r, 2, 120))), giro % 24), tuple(c))


def balon_detras_de_red(pos: np.ndarray, pose: Pose) -> bool:
    """¿El balón está más lejos de la cámara que la red (a su misma altura)?"""
    z = profundidad_camara(np.array([pos, [0.0, pos[1], pos[2]]]), pose)
    return bool(z[0] > z[1])


# --------------------------------------------------------------------------- #
# Escena completa
# --------------------------------------------------------------------------- #


def dibujar_rival(img: np.ndarray, pos_xy: np.ndarray, pose: Pose, modelo: ModeloCamara) -> None:
    t = np.linspace(0, 2 * np.pi, 24)
    r = 1.8
    circ = np.column_stack([pos_xy[0] + r * np.cos(t), pos_xy[1] + r * np.sin(t), np.zeros_like(t)])
    p = np.round(proyectar(circ, pose, modelo)).astype(np.int32)
    capa = img.copy()
    cv2.fillPoly(capa, [p], vis.BGR_GRAFITO, cv2.LINE_AA)
    img[:] = cv2.addWeighted(capa, 0.55, img, 0.45, 0)
    cv2.polylines(img, [p], True, vis.BGR_NARANJA, 2, cv2.LINE_AA)
    c = proyectar(np.array([[pos_xy[0], pos_xy[1], 0.0]]), pose, modelo)[0]
    vis.poner_texto(img, "IA", (int(c[0]) - 9, int(c[1]) - 11), 16, vis.BGR_NEUTRO, negrita=True)


def dibujar_marcas_golpe(img: np.ndarray, marcas, t_actual: float, pose: Pose, modelo: ModeloCamara) -> None:
    """Anillo que se expande donde golpeaste (verde válido, rojo inválido)."""
    for m in marcas:
        edad = t_actual - m.t
        r = 1.5 + 6 * edad
        ang = np.linspace(0, 2 * np.pi, 32)
        circ = np.column_stack([m.pos[0] + r * np.cos(ang), m.pos[1] + r * np.sin(ang), np.full_like(ang, m.pos[2])])
        color = vis.BGR_VERDE if m.valido else vis.BGR_ROJO
        cv2.polylines(img, [np.round(proyectar(circ, pose, modelo)).astype(np.int32)], True, color, 3, cv2.LINE_AA)
        if edad < 0.6:
            c = proyectar(np.array([m.pos]), pose, modelo)[0]
            vis.poner_texto(img, m.texto, (int(c[0]) + 14, int(c[1]) - 30), 20, color, negrita=True,
                            fondo=vis.BGR_GRAFITO)


def dibujar_escena(img: np.ndarray, pose: Pose, modelo: ModeloCamara, balon_pos: np.ndarray,
                   rival_xy: np.ndarray | None = None, marcas=(), t: float = 0.0, lineas: bool = False,
                   giro: int = 0) -> np.ndarray:
    """Compone la escena AR respetando el orden de oclusión balón/red."""
    out = img.copy()
    if lineas:
        dibujar_lineas_cancha(out, pose, modelo)
    if rival_xy is not None:
        dibujar_rival(out, rival_xy, pose, modelo)
    dibujar_sombra(out, balon_pos, pose, modelo)
    if balon_detras_de_red(balon_pos, pose):
        dibujar_balon(out, balon_pos, pose, modelo, giro)
        dibujar_red(out, pose, modelo)
    else:
        dibujar_red(out, pose, modelo)
        dibujar_balon(out, balon_pos, pose, modelo, giro)
    dibujar_marcas_golpe(out, marcas, t, pose, modelo)
    return out


def dibujar_hud(img: np.ndarray, marcador: tuple[int, int], mensaje: str = "", modo: str = "",
                fps: float | None = None, gesto: str = "", cerca: bool | None = None,
                rapidez: float | None = None, detector: str = "clásico") -> np.ndarray:
    """Marcador, mensajes y estado (paleta: grafito, naranja y neutro; verde/rojo solo para datos)."""
    h, w = img.shape[:2]
    tam = max(18, int(h / 24))
    texto = f"Tú  {marcador[0]}  :  {marcador[1]}  IA"
    ancho_aprox = int(len(texto) * tam * 0.62)
    vis.poner_texto(img, texto, (w // 2 - ancho_aprox // 2, 12), tam + 6, vis.BGR_NEUTRO, negrita=True,
                    fondo=vis.BGR_GRAFITO, alfa_fondo=0.8)
    if mensaje:
        ancho_m = int(len(mensaje) * tam * 0.55)
        vis.poner_texto(img, mensaje, (w // 2 - ancho_m // 2, h // 2 + int(h * 0.28)), tam + 2, vis.BGR_NARANJA,
                        negrita=True, fondo=vis.BGR_GRAFITO, alfa_fondo=0.75)
    izq = modo + (f"   ·   {fps:.0f} FPS" if fps is not None else "") + f"   ·   detector {detector}"
    vis.poner_texto(img, izq, (14, h - tam - 14), tam - 4, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
    if gesto:
        der = f"{gesto}" + (f" · {rapidez:.0f} cm/s" if rapidez is not None else "")
        x = w - int(len(der) * (tam - 4) * 0.6) - 60
        vis.poner_texto(img, der, (x, h - tam - 14), tam - 4, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        color = vis.BGR_NEUTRO if cerca is None else (vis.BGR_VERDE if cerca else vis.BGR_ROJO)
        cv2.circle(img, (w - 30, h - tam // 2 - 14), 10, color, -1, cv2.LINE_AA)
    return img


def _profundidad_plano(px: np.ndarray, pose: Pose, modelo: ModeloCamara, normal_idx: int, valor: float) -> np.ndarray:
    r"""Profundidad :math:`z_c` en la que el rayo de cada píxel corta el plano ``P_w[normal_idx] = valor``.

    Con :math:`P_c = \lambda (x, y, 1)` y :math:`P_w = R^T (P_c - t)`, la coordenada
    del plano es :math:`n^T P_c - n^T t` con :math:`n = R_{:,i}`; se despeja :math:`\lambda`.
    """
    K = modelo.K
    d = np.column_stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(px))])
    n = pose.R[:, normal_idx]
    den = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (valor + n @ pose.tvec.reshape(3)) / den
    lam[~np.isfinite(lam) | (lam <= 0)] = np.inf
    return lam


def ocluir_con_mano(render: np.ndarray, original: np.ndarray, mascara_mano: np.ndarray | None, pose: Pose,
                    modelo: ModeloCamara, balon_pos: np.ndarray, altura_mano: float = 1.5) -> np.ndarray:
    """Devuelve los píxeles de la mano real allí donde está **delante** de lo virtual.

    Se aproxima la mano como el plano ``z = altura_mano`` y se compara su
    profundidad por píxel con la del balón (constante) y la de la red (plano
    ``x = 0``). Es una prueba de profundidad (z-buffer) simplificada.
    """
    if mascara_mano is None or not mascara_mano.any():
        return render
    ys, xs = np.nonzero(mascara_mano)
    px = np.column_stack([xs, ys]).astype(np.float64)
    z_mano = _profundidad_plano(px, pose, modelo, 2, altura_mano)
    # Balón: disco proyectado con profundidad constante
    c = proyectar(np.asarray(balon_pos)[None], pose, modelo)[0]
    r = radio_en_pixeles(balon_pos, pose, modelo)
    z_balon = float(profundidad_camara(np.asarray(balon_pos)[None], pose)[0])
    en_balon = (xs - c[0]) ** 2 + (ys - c[1]) ** 2 <= (r + 1) ** 2
    # Red: polígono proyectado, profundidad del plano x = 0
    A = cfg.MEDIO_ANCHO + 0.4
    marco = proyectar(np.array([[0, -A, 0], [0, A, 0], [0, A, cfg.ALTURA_RED + 0.8], [0, -A, cfg.ALTURA_RED + 0.8]]),
                      pose, modelo)
    poli = np.zeros(mascara_mano.shape, np.uint8)
    cv2.fillPoly(poli, [np.round(marco).astype(np.int32)], 255)
    en_red = poli[ys, xs] > 0
    z_red = _profundidad_plano(px, pose, modelo, 0, 0.0)
    delante = (en_balon & (z_mano < z_balon)) | (en_red & ~en_balon & (z_mano < z_red))
    out = render.copy()
    out[ys[delante], xs[delante]] = original[ys[delante], xs[delante]]
    return out
