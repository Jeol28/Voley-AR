"""Pestaña de histogramas: calibración de la piel, retroproyección, ecualización y CLAHE."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import histogramas as his
from voleyar.core import pipeline as pl
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, sesion_componente, teoria
from voleyar.ui.estado import ESTADO, Frame, con_sesion, rgb
from voleyar.utils import visualizacion as vis

TEORIA = r"""
**Calibración de la piel.** Con los píxeles del recuadro se construyen histogramas 2D de **crominancia**
(H-S en HSV y Cr-Cb en YCrCb), descartando el brillo para tolerar sombras. Normalizados estiman
$P(\text{color}\mid\text{piel})$.

**Retroproyección.** Cada píxel se reemplaza por el valor del histograma en su color:
$B(x,y)=\hat h(c(x,y))$. Se combinan ambos espacios con la media geométrica $B=\sqrt{B_{HS}\,B_{CrCb}}$.

**Ecualización.** $s_k=\mathrm{round}\!\left(\dfrac{\mathrm{cdf}(k)-\mathrm{cdf}_{min}}{N-\mathrm{cdf}_{min}}(L-1)\right)$
estira el histograma. **CLAHE** lo hace por celdas, recortando cada histograma local (*clip limit*) para no
amplificar el ruido. En el juego se aplica CLAHE al canal L de Lab: compensa la luz del cuarto sin cambiar el
color de la piel.
"""


def _rect(img, cx, cy, lado):
    h, w = img.shape[:2]
    s = int(lado / 100 * min(h, w))
    x = int(cx / 100 * w - s / 2)
    y = int(cy / 100 * h - s / 2)
    return (max(0, x), max(0, y), s, s)


def procesar(fr: Frame, cx: float, cy: float, lado: float, espacio: str, clip: float, celdas: int):
    img, _ = pl.preparar_imagen(fr.img, ESTADO.modelo_para(fr), ESTADO.config)
    clahe, suav = pl.preprocesar(img, pl.ConfigPipeline(clahe_clip=clip, clahe_celdas=int(celdas)))
    rect = _rect(img, cx, cy, lado)
    vista = img.copy()
    x, y, w, h = rect
    cv2.rectangle(vista, (x, y), (x + w, y + h), vis.BGR_NARANJA, 3)
    try:
        modelo = his.calibrar_piel(suav, rect)
        hs = vis.mapa_2d(modelo.hist_hs, "Histograma H-S de la piel", "S (saturación)", "H (tono)",
                         extent=[0, 256, 0, 180])
        cc = vis.mapa_2d(modelo.hist_crcb, "Histograma Cr-Cb de la piel", "Cb", "Cr", extent=[0, 256, 0, 256])
        retro = his.retroproyeccion(suav, modelo, espacio, disco=7)
        retro_c = cv2.applyColorMap(retro, cv2.COLORMAP_INFERNO)
        msg = f"{modelo.n_pixeles} píxeles válidos en el recuadro."
    except ValueError as e:
        hs = cc = retro_c = None
        msg = f"**{e}**"
    ecu = his.ecualizar(img)
    hg = {"Original": his.histograma_gris(img), "Ecualizada": his.histograma_gris(ecu),
          "CLAHE": his.histograma_gris(clahe)}
    graf = vis.grafica_lineas(hg, "Histograma de la luminancia", "nivel de gris", "píxeles", relleno=True)
    contraste = (f"Contraste (σ de la luminancia): original **{his.contraste(img):.1f}**, ecualizada "
                 f"**{his.contraste(ecu):.1f}**, CLAHE **{his.contraste(clahe):.1f}**. {msg}")
    return rgb(vista), hs, cc, rgb(retro_c) if retro_c is not None else None, rgb(ecu), rgb(clahe), graf, contraste


def usar_piel(fr: Frame | None, cx, cy, lado):
    if fr is None:
        return "Carga una imagen primero."
    img, _ = pl.preparar_imagen(fr.img, ESTADO.modelo_para(fr), ESTADO.config)
    _, suav = pl.preprocesar(img, ESTADO.config)
    try:
        ESTADO.piel = his.calibrar_piel(suav, _rect(img, cx, cy, lado))
        ESTADO.piel_origen = f"calibrada con el recuadro ({fr.nombre})"
    except ValueError as e:
        return f"**{e}**"
    return ('<span class="dato-ok">Modelo de piel guardado.</span> Las pestañas de Umbralización, Morfología, '
            "Segmentación, Caracterización y Pipeline ya lo usan.")


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Histogramas: el color de tu piel y la luz del cuarto")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada()
            cx = gr.Slider(0, 100, value=52, step=0.5, label="Recuadro: centro X (%)")
            cy = gr.Slider(0, 100, value=42, step=0.5, label="Recuadro: centro Y (%)")
            lado = gr.Slider(2, 30, value=5, step=0.5, label="Recuadro: lado (% de la imagen)")
            boton = gr.Button("Usar este recuadro como modelo de piel", variant="primary")
            aviso = gr.Markdown()
        with gr.Column(scale=6):
            with gr.Row():
                vista = gr.Image(label="Recuadro de calibración", height=250)
                retro = gr.Image(label="Retroproyección", height=250)
            with gr.Row():
                hs = gr.Image(label="H-S", height=250)
                cc = gr.Image(label="Cr-Cb", height=250)
            espacio = gr.Radio(list(his.ESPACIOS_RETRO), value="combinada", label="Espacio de la retroproyección")
    gr.Markdown("### Compensar la luz: ecualización y CLAHE")
    with gr.Row():
        with gr.Column(scale=3):
            clip = gr.Slider(0.5, 8, value=2.0, step=0.1, label="CLAHE: clip limit")
            celdas = gr.Slider(2, 16, value=8, step=1, label="CLAHE: celdas por lado")
            contraste = gr.Markdown()
        ecu = gr.Image(label="Ecualización global (Y)", height=230, scale=3)
        cla = gr.Image(label="CLAHE (L de Lab)", height=230, scale=3)
    graf = gr.Image(label="Histogramas antes y después", height=300)
    params = [cx, cy, lado, espacio, clip, celdas]
    salidas = [vista, hs, cc, retro, ecu, cla, graf, contraste]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    S = sesion_componente()
    boton.click(con_sesion(usar_piel), [S, entrada.estado, cx, cy, lado], [S, aviso])
    teoria(TEORIA)
