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
Todo se mide en la **vista cenital** (homografía con la pose), así que las áreas salen en cm².

**Gesto → golpe:** 4-5 dedos = remate (fuerza alta), 2-3 = golpe, 1 = toque suave, puño = bloqueo.
**Punto de golpe:** la punta del dedo (o la palma si la mano está abierta), llevado a cm con el rayo ∩ $z=0$.
La velocidad (cm/s) sale de frames consecutivos con suavizado exponencial (en la cámara en vivo).
"""

_velocidad = car.EstimadorVelocidad()


def procesar(fr: Frame, k_dedo: float, prof_min: float, angulo_max: float):
    res = ejecutar_pipeline(fr, k_dedo=k_dedo)
    if res.mano_cenital is None or res.analisis is None:
        return rgb(pl.dibujar_depuracion(res)), None, "No se encontró la mano.", ""
    a = car.analizar_mano(res.mano_cenital, res.info_corte.direccion_brazo, k_dedo, prof_min, angulo_max)
    res.analisis = a
    cen = pl.dibujar_cenital(res)
    for d in a.defectos:  # todos los defectos (válidos en rojo, el resto tenues)
        if not d.valido:
            cv2.circle(cen, d.valle, 3, (120, 120, 120), -1)
    imagen = pl.dibujar_depuracion(res)
    d = res.descriptores
    ppcm = res.vista.ppcm
    filas = [["Área", f"{d.area / ppcm ** 2:.1f} cm²"], ["Perímetro", f"{d.perimetro / ppcm:.1f} cm"],
             ["Centroide", f"({d.centroide[0] / ppcm + res.vista.x_min:.1f}, {res.vista.y_max - d.centroide[1] / ppcm:.1f}) cm"],
             ["Circularidad", f"{d.circularidad:.3f}"], ["Solidez", f"{d.solidez:.3f}"],
             ["Excentricidad", f"{d.excentricidad:.3f}"], ["Orientación", f"{d.orientacion_grados:.1f}°"]]
    filas += [[f"Hu {i + 1}", f"{v:.3f}"] for i, v in enumerate(d.hu_log)]
    validos = sum(x.valido for x in a.defectos)
    p = res.punto_golpe_cm
    v = _velocidad.actualizar(None if p is None else p[:2], time.perf_counter()) if fr.nombre == "cámara" else np.zeros(2)
    resumen = (f"### {a.n_dedos} dedo{'s' if a.n_dedos != 1 else ''} → **{a.gesto}** (fuerza {a.fuerza:.2f})\n\n"
               + tabla_md([["Defectos de convexidad", f"{len(a.defectos)} ({validos} valles válidos)"],
                           ["Radio de la palma", f"{a.radio_palma / ppcm:.1f} cm"],
                           ["Punto de golpe", "palma" if a.n_dedos in (4, 5) else ("nudillos" if a.n_dedos == 0 else "punta del dedo")],
                           ["Posición en la mesa", "—" if p is None else f"({p[0]:.1f}, {p[1]:.1f}) cm"],
                           ["Velocidad", f"({v[0]:.0f}, {v[1]:.0f}) cm/s"]], ["", "Valor"]))
    return rgb(imagen), rgb(cen), tabla_md(filas, ["Descriptor", "Valor"]), resumen


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Caracterización: forma de la mano, dedos y punto de golpe")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada("Dos dedos")
            k = gr.Slider(1.2, 2.4, value=1.6, step=0.05, label="Un dedo sale a más de k · radio de la palma")
            prof = gr.Slider(0.1, 1.5, value=0.45, step=0.05, label="Profundidad mínima del valle (× radio)")
            ang = gr.Slider(30, 150, value=100, step=5, label="Ángulo máximo del valle (°)")
            resumen = gr.Markdown(latex_delimiters=LATEX)
        with gr.Column(scale=6):
            imagen = gr.Image(label="Contorno, casco, puntas (verde), valles (rojo) y punto de golpe", height=300)
            cen = gr.Image(label="Vista cenital métrica", height=260)
            tabla = gr.Markdown()
    params = [k, prof, ang]
    salidas = [imagen, cen, tabla, resumen]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
