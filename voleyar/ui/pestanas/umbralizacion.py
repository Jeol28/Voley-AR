"""Pestaña de umbralización: rangos HSV/YCrCb, Otsu manual vs. OpenCV y adaptativo."""

from __future__ import annotations

import time

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import histogramas as his
from voleyar.core import pipeline as pl
from voleyar.core import segmentacion as seg
from voleyar.core import umbralizacion as umb
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import ESTADO, Frame, piel_para, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
**Rangos fijos** (`inRange`): no necesitan calibración pero dependen de la luz y del tono de piel. En **YCrCb**
el brillo queda aislado en Y, así que Cr-Cb suele separar mejor la piel que H-S (el tono H de la piel se
confunde con la madera y el rojo da la vuelta en 0/180).

**Otsu** sobre la retroproyección (solo dentro de la mitad del jugador, sin su cara). Con $\omega_0(t)=\sum_{i\le t}p_i$,
$\mu(t)=\sum_{i\le t} i\,p_i$ y $\mu_T=\sum_i i\,p_i$:

$$\sigma_B^2(t)=\frac{[\mu_T\,\omega_0(t)-\mu(t)]^2}{\omega_0(t)\,[1-\omega_0(t)]},\qquad T^*=\arg\max_t\sigma_B^2(t)$$

**Adaptativo:** $T(x,y)=\mu_W(x,y)-C$ en cada ventana. Útil con luz desigual, pero sobre la retroproyección
marca bordes internos en vez de regiones: para la mano funciona mejor Otsu.
"""


def procesar(fr: Frame, lado, h_min, h_max, s_min, v_min, cr_min, cr_max, cb_min, cb_max, bloque, C):
    lado = int(lado)
    img, modelo = pl.preparar_imagen(fr.img, ESTADO.modelo_para(fr), ESTADO.config)
    _, suav = pl.preprocesar(img, ESTADO.config)
    # Zona: la mitad del jugador sin su cara (las caras también son piel, pero no son la mano)
    caras = seg.mascara_caras(img.shape, seg.detectar_caras(img))
    zona = cv2.bitwise_and(seg.mascara_mitad(img.shape, lado), cv2.bitwise_not(caras))
    m_hsv = cv2.bitwise_and(umb.umbral_hsv(suav, (int(h_min), int(s_min), int(v_min)), (int(h_max), 255, 255)), zona)
    m_ycc = cv2.bitwise_and(umb.umbral_ycrcb(suav, (0, int(cr_min), int(cb_min)), (255, int(cr_max), int(cb_max))),
                            zona)
    piel, origen = piel_para(suav, lado, zona)
    if piel is None:
        return (rgb(m_hsv), rgb(m_ycc), None, None, None, None, "No se encontró piel para el modelo.")
    retro = his.retroproyeccion(suav, piel, "combinada", 7, zona)
    t0 = time.perf_counter()
    T, curva = umb.otsu_manual(retro, zona)
    t_man = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    T_cv = umb.otsu_opencv(retro[zona > 0].reshape(1, -1))
    t_cv = (time.perf_counter() - t0) * 1000
    m_otsu = cv2.bitwise_and(umb.binarizar(retro, T), zona)
    m_ad = cv2.bitwise_and(umb.umbral_adaptativo(retro, "gaussiano", int(bloque), C), zona)
    hist = np.bincount(retro[zona > 0].ravel(), minlength=256).astype(float)
    graf = vis.grafica_lineas({"σ²_B(t)": curva / max(curva.max(), 1e-9), "histograma": hist / max(hist.max(), 1)},
                              "Otsu sobre la retroproyección (normalizado)", "umbral t", "",
                              marcas_x={f"T* = {T}": T})
    # Comparación de métodos: con la verdad si la escena es sintética, si no con la máscara de Otsu
    pieles_v = fr.verdad.get("mascaras_piel") if fr.verdad else None
    verdad = pieles_v[0 if lado < 0 else 1] if pieles_v and len(pieles_v) > (lado > 0) else None
    if verdad is not None:
        verdad = pl.preparar_imagen(cv2.cvtColor(verdad, cv2.COLOR_GRAY2BGR), ESTADO.modelo_para(fr), ESTADO.config)[0]
        verdad = cv2.bitwise_and(np.where(verdad[..., 0] > 127, 255, 0).astype(np.uint8), zona)
        ref, nombre_ref = verdad, ("piel verdadera de la mano y el antebrazo en la escena sintética (si Haar no "
                                   "encontró la cara, cuenta como falsa piel)")
    else:
        ref, nombre_ref = m_otsu, "Otsu (no hay verdad en una imagen real)"
    filas = []
    for nombre, m in (("Rango HSV", m_hsv), ("Rango YCrCb", m_ycc), ("Otsu (retroproyección)", m_otsu),
                      ("Adaptativo", m_ad)):
        met = umb.metricas_mascara(m, ref)
        filas.append([nombre, f"{met['iou']:.2f}", f"{met['precision']:.2f}", f"{met['exhaustividad']:.2f}"])
    md = (f"**Otsu manual:** T = {T} ({t_man:.2f} ms) · **OpenCV:** T = {T_cv} ({t_cv:.2f} ms) · "
          + ('<span class="dato-ok">coinciden</span>' if T == T_cv else '<span class="dato-mal">difieren</span>')
          + f"\n\nModelo de piel: {origen}. Referencia: {nombre_ref}.\n\n"
          + tabla_md(filas, ["Método", "IoU", "Precisión", "Exhaustividad"]))
    return rgb(m_hsv), rgb(m_ycc), rgb(cv2.applyColorMap(retro, cv2.COLORMAP_INFERNO)), rgb(m_otsu), rgb(m_ad), graf, md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Umbralización: de probabilidad de piel a máscara binaria")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada()
            jugador = gr.Radio([("Jugador 1 (izquierda)", -1), ("Jugador 2 (derecha)", 1)], value=-1,
                               label="Mitad de la imagen")
            with gr.Accordion("Rango HSV", open=False):
                h_min = gr.Slider(0, 179, value=umb.RANGO_HSV_PIEL[0][0], step=1, label="H mínimo")
                h_max = gr.Slider(0, 179, value=umb.RANGO_HSV_PIEL[1][0], step=1, label="H máximo")
                s_min = gr.Slider(0, 255, value=umb.RANGO_HSV_PIEL[0][1], step=1, label="S mínimo")
                v_min = gr.Slider(0, 255, value=umb.RANGO_HSV_PIEL[0][2], step=1, label="V mínimo")
            with gr.Accordion("Rango YCrCb", open=False):
                cr_min = gr.Slider(0, 255, value=umb.RANGO_YCRCB_PIEL[0][1], step=1, label="Cr mínimo")
                cr_max = gr.Slider(0, 255, value=umb.RANGO_YCRCB_PIEL[1][1], step=1, label="Cr máximo")
                cb_min = gr.Slider(0, 255, value=umb.RANGO_YCRCB_PIEL[0][2], step=1, label="Cb mínimo")
                cb_max = gr.Slider(0, 255, value=umb.RANGO_YCRCB_PIEL[1][2], step=1, label="Cb máximo")
            bloque = gr.Slider(3, 101, value=31, step=2, label="Adaptativo: tamaño de la ventana")
            C = gr.Slider(-40, 40, value=-5, step=1, label="Adaptativo: constante C")
        with gr.Column(scale=6):
            with gr.Row():
                m_hsv = gr.Image(label="Rango HSV", height=210)
                m_ycc = gr.Image(label="Rango YCrCb", height=210)
                retro = gr.Image(label="Retroproyección", height=210)
            with gr.Row():
                m_otsu = gr.Image(label="Otsu (manual)", height=210)
                m_ad = gr.Image(label="Adaptativo", height=210)
            graf = gr.Image(label="Curva de varianza entre clases", height=280)
            md = gr.Markdown(latex_delimiters=LATEX)
    params = [jugador, h_min, h_max, s_min, v_min, cr_min, cr_max, cb_min, cb_max, bloque, C]
    salidas = [m_hsv, m_ycc, retro, m_otsu, m_ad, graf, md]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
