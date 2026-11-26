"""Pestaña de caracterización: descriptores, Hu, casco convexo, dedos y punto de golpe."""

from __future__ import annotations

import time

import cv2
import gradio as gr
import numpy as np

from voleyar.core import caracterizacion as car
from voleyar.core import pipeline as pl
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
**Momentos** $m_{pq}=\sum x^p y^q I(x,y)$: área $m_{00}$, centroide $(m_{10}/m_{00},\,m_{01}/m_{00})$.

- **Circularidad** $4\pi A/P^2$ (1 = círculo). **Solidez** $A/A_{casco}$: baja con los dedos abiertos.
- **Excentricidad** $\sqrt{1-\lambda_2/\lambda_1}$ con los autovalores de la covarianza de los momentos centrales.
- **Hu**: 7 invariantes a traslación, escala y rotación, mostrados como $-\mathrm{sign}(h)\log_{10}|h|$.

**Dedos:** el **casco convexo** tiene las puntas como vértices; los **defectos de convexidad** profundos y agudos
son los valles entre dedos. Un dedo es un tramo del contorno a más de $k\,r$ del centro de la palma.
Con la distancia $Z$ de la mano (tamaño de la palma y la $f$ calibrada) cada píxel mide $Z/f$ cm, así que las
áreas y perímetros se pueden expresar en cm.

**Gesto → golpe:** 4-5 dedos = remate (fuerza alta), 2-3 = golpe, 1 = toque suave, puño = bloqueo.
**Punto de golpe:** la punta del dedo (o la palma si la mano está abierta), llevado al plano de la red con el rayo
de la cámara virtual.
La velocidad (cm/s) sale de frames consecutivos con suavizado exponencial (en la cámara en vivo).
"""



def _recorte_mano(img: np.ndarray, r, margen: int = 50) -> np.ndarray:
    ys, xs = np.nonzero(r.componente if r.componente is not None else r.mascara_mano)
    if len(xs) == 0:
        return img
    y0, y1 = max(0, ys.min() - margen), min(img.shape[0], ys.max() + margen)
    x0, x1 = max(0, xs.min() - margen), min(img.shape[1], xs.max() + margen)
    return img[y0:y1, x0:x1]


def procesar(fr: Frame, lado, k_dedo: float, prof_min: float, angulo_max: float):
    lado = int(lado)
    res = ejecutar_pipeline(fr, k_dedo=k_dedo)
    r = res.manos[lado]
    imagen = pl.dibujar_depuracion(res)
    if r.mascara_mano is None or r.analisis is None:
        return rgb(imagen), None, "No se encontró la mano de este jugador.", ""
    a = car.analizar_mano(r.mascara_mano, r.info_corte.direccion_brazo, k_dedo, prof_min, angulo_max)
    # Detalle: silueta de la mano con casco, valles (todos) y puntas
    det = np.zeros((*r.mascara_mano.shape, 3), np.uint8)
    if r.componente is not None:
        det[r.componente > 0] = (70, 70, 70)
    det[r.mascara_mano > 0] = vis.BGR_NEUTRO
    cv2.polylines(det, [a.casco.reshape(-1, 1, 2).astype(np.int32)], True, vis.BGR_NARANJA, 2, cv2.LINE_AA)
    cv2.circle(det, tuple(np.int32(a.centro_palma)), int(a.radio_palma), vis.BGR_NARANJA, 1, cv2.LINE_AA)
    cv2.circle(det, tuple(np.int32(a.centro_palma)), int(a.umbral_dedo), (120, 120, 120), 1, cv2.LINE_AA)
    for d in a.defectos:
        cv2.circle(det, d.valle, 6 if d.valido else 3, vis.BGR_ROJO if d.valido else (120, 120, 120), -1)
    for p in a.puntas:
        cv2.circle(det, p, 8, vis.BGR_VERDE, 2)
    cv2.drawMarker(det, tuple(np.int32(a.punto_golpe_px)), vis.BGR_NARANJA, cv2.MARKER_CROSS, 22, 3)
    d = r.descriptores
    e = r.cm_por_px or 0.0  # cm por píxel en el plano de la mano (Z / f)
    filas = [["Área", f"{d.area:.0f} px² ≈ {d.area * e ** 2:.0f} cm²"],
             ["Perímetro", f"{d.perimetro:.0f} px ≈ {d.perimetro * e:.0f} cm"],
             ["Centroide", f"({d.centroide[0]:.0f}, {d.centroide[1]:.0f}) px"],
             ["Circularidad", f"{d.circularidad:.3f}"], ["Solidez", f"{d.solidez:.3f}"],
             ["Excentricidad", f"{d.excentricidad:.3f}"], ["Orientación", f"{d.orientacion_grados:.1f}°"]]
    filas += [[f"Hu {i + 1}", f"{v:.3f}"] for i, v in enumerate(d.hu_log)]
    validos = sum(x.valido for x in a.defectos)
    v = r.velocidad if fr.nombre == "cámara" else np.zeros(2)
    p = r.punto_plano
    resumen = (f"### {a.n_dedos} dedo{'s' if a.n_dedos != 1 else ''} → **{a.gesto}** (fuerza {a.fuerza:.2f})\n\n"
               + tabla_md([["Defectos de convexidad", f"{len(a.defectos)} ({validos} valles válidos)"],
                           ["Radio de la palma", f"{a.radio_palma:.0f} px ≈ {a.radio_palma * e:.1f} cm"],
                           ["Distancia a la cámara Z = f·W/w", f"{r.distancia_cm:.0f} cm"],
                           ["Punto de golpe", "palma" if a.n_dedos in (4, 5) else ("nudillos" if a.n_dedos == 0
                                                                                   else "punta del dedo")],
                           ["Posición en el plano de juego", "—" if p is None else f"x = {p[0]:.1f}, altura {p[2]:.1f}"],
                           ["Velocidad (cámara en vivo)", f"({v[0]:.0f}, {v[1]:.0f}) unidades/s"]], ["", "Valor"]))
    return rgb(imagen), rgb(_recorte_mano(det, r)), tabla_md(filas, ["Descriptor", "Valor"]), resumen


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Caracterización: forma de la mano, dedos y punto de golpe")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada("Dos dedos y puño")
            jugador = gr.Radio([(pl.NOMBRES_LADO[-1] + " (izquierda)", -1), (pl.NOMBRES_LADO[1] + " (derecha)", 1)],
                               value=-1, label="Jugador")
            k = gr.Slider(1.2, 2.4, value=1.6, step=0.05, label="Un dedo sale a más de k · radio de la palma")
            prof = gr.Slider(0.1, 1.5, value=0.45, step=0.05, label="Profundidad mínima del valle (× radio)")
            ang = gr.Slider(30, 150, value=100, step=5, label="Ángulo máximo del valle (°)")
            resumen = gr.Markdown(latex_delimiters=LATEX)
        with gr.Column(scale=6):
            imagen = gr.Image(label="Contorno, casco, puntas (verde), valles (rojo) y punto de golpe", height=300)
            cen = gr.Image(label="Silueta: casco (naranja), valles (rojo), puntas (verde)", height=280)
            tabla = gr.Markdown()
    params = [jugador, k, prof, ang]
    salidas = [imagen, cen, tabla, resumen]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
