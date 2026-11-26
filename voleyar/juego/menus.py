"""Menús dibujados dentro de la ventana del juego (inicio, ajustes, pausa, fin de partida).

Cada botón tiene una tecla y se puede pulsar con el mouse. No dependen de la
visión: solo dibujan sobre la imagen y responden a clics y teclas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from voleyar.utils import visualizacion as vis


@dataclass
class Boton:
    texto: str
    accion: str
    tecla: str | None = None
    principal: bool = False
    rect: tuple[int, int, int, int] | None = None  # se calcula al dibujar


def oscurecer(img: np.ndarray, cuanto: float = 0.6) -> np.ndarray:
    capa = np.full_like(img, vis.BGR_GRAFITO)
    return cv2.addWeighted(capa, cuanto, img, 1 - cuanto, 0)


def dibujar_menu(img: np.ndarray, titulo: str, botones: list[Boton], subtitulo: str = "",
                 nota: str = "", oscuro: float = 0.62) -> np.ndarray:
    """Título grande y una columna de botones centrada. Guarda el rectángulo de cada botón."""
    out = oscurecer(img, oscuro)
    h, w = out.shape[:2]
    escala = h / 720
    t_titulo = int(70 * escala)
    ancho_titulo = int(len(titulo) * t_titulo * 0.55)
    y = int(h * 0.12)
    vis.poner_texto(out, titulo, (w // 2 - ancho_titulo // 2, y), t_titulo, vis.BGR_NEUTRO, negrita=True)
    y += int(t_titulo * 1.35)
    if subtitulo:
        t_sub = int(26 * escala)
        vis.poner_texto(out, subtitulo, (w // 2 - int(len(subtitulo) * t_sub * 0.27), y), t_sub, vis.BGR_NARANJA)
        y += int(t_sub * 2.2)
    bw, bh = int(w * 0.38), int(62 * escala)
    sep = int(16 * escala)
    t_boton = int(28 * escala)
    for b in botones:
        x0 = w // 2 - bw // 2
        b.rect = (x0, y, bw, bh)
        fondo = vis.BGR_NARANJA if b.principal else (90, 70, 58)  # BGR: grafito azulado
        texto_color = vis.BGR_GRAFITO if b.principal else vis.BGR_NEUTRO
        cv2.rectangle(out, (x0, y), (x0 + bw, y + bh), fondo, -1, cv2.LINE_AA)
        etiqueta = b.texto
        ancho_t = int(len(etiqueta) * t_boton * 0.5)
        vis.poner_texto(out, etiqueta, (w // 2 - ancho_t // 2, y + (bh - t_boton) // 2 - 2), t_boton, texto_color,
                        negrita=b.principal)
        if b.tecla:
            vis.poner_texto(out, b.tecla.upper(), (x0 + 14, y + (bh - int(20 * escala)) // 2), int(18 * escala),
                            texto_color)
        y += bh + sep
    if nota:
        t_nota = int(18 * escala)
        vis.poner_texto(out, nota, (w // 2 - int(len(nota) * t_nota * 0.26), h - int(48 * escala)), t_nota,
                        (180, 182, 196))
    return out


def boton_en(botones: list[Boton], x: int, y: int) -> str | None:
    for b in botones:
        if b.rect is None:
            continue
        x0, y0, bw, bh = b.rect
        if x0 <= x <= x0 + bw and y0 <= y <= y0 + bh:
            return b.accion
    return None


def boton_por_tecla(botones: list[Boton], k: int) -> str | None:
    for b in botones:
        if b.tecla and k in (ord(b.tecla.lower()), ord(b.tecla.upper())):
            return b.accion
    return None


def dibujar_cartel(img: np.ndarray, titulo: str, subtitulo: str = "") -> np.ndarray:
    """Cartel grande en el centro (pausa, ganador, cuenta regresiva)."""
    h, w = img.shape[:2]
    escala = h / 720
    t = int(64 * escala)
    ancho = max(int(len(titulo) * t * 0.55), int(len(subtitulo) * 26 * escala * 0.55)) + int(80 * escala)
    alto = int((t + (50 if subtitulo else 0) * escala) + 60 * escala)
    x0, y0 = w // 2 - ancho // 2, h // 2 - alto // 2 - int(40 * escala)
    capa = img.copy()
    cv2.rectangle(capa, (x0, y0), (x0 + ancho, y0 + alto), vis.BGR_GRAFITO, -1)
    img[:] = cv2.addWeighted(capa, 0.85, img, 0.15, 0)
    cv2.rectangle(img, (x0, y0), (x0 + ancho, y0 + alto), vis.BGR_NARANJA, 3, cv2.LINE_AA)
    vis.poner_texto(img, titulo, (w // 2 - int(len(titulo) * t * 0.27), y0 + int(22 * escala)), t, vis.BGR_NEUTRO,
                    negrita=True)
    if subtitulo:
        ts = int(26 * escala)
        vis.poner_texto(img, subtitulo, (w // 2 - int(len(subtitulo) * ts * 0.26), y0 + t + int(34 * escala)), ts,
                        vis.BGR_NARANJA)
    return img
