"""Pestaña de filtros: suavizado, Sobel, Canny y convolución manual vs. filter2D."""

from __future__ import annotations

import time

import cv2
import gradio as gr
import numpy as np

from voleyar.core import filtros as fil
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
**Para qué en el juego:** la cámara web tiene ruido que crea píxeles sueltos "color piel". Se suaviza **antes**
de la retroproyección para que la máscara de la mano salga limpia. Sobel/Canny muestran el borde de la mano
en el panel de depuración.

**Convolución:** $g(x,y)=\sum_{s}\sum_{t} h(s,t)\,f(x-s,\,y-t)$. `cv2.filter2D` calcula la **correlación**
($f(x+s,y+t)$, sin voltear); por eso se le pasa el núcleo girado 180° para comparar.

- **Gaussiano:** $h\propto e^{-(s^2+t^2)/2\sigma^2}$, lineal, difumina también los bordes.
- **Mediana:** no lineal, ideal para ruido impulsivo (sal y pimienta).
- **Bilateral:** pesos $e^{-\lVert p-q\rVert^2/2\sigma_s^2}\,e^{-|I_p-I_q|^2/2\sigma_r^2}$: suaviza la piel y conserva el
  borde mano/cancha.
- **Sobel:** $G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$, magnitud $\sqrt{G_x^2+G_y^2}$.
- **Canny:** gaussiano → Sobel → supresión de no máximos → histéresis con dos umbrales.
"""


def procesar(fr: Frame, ruido: float, metodo: str, k: int, sigma_color: float, sobel_k: int, canny_bajo: float,
             canny_alto: float, nucleo_txt: str):
    img = fr.img
    if ruido > 0:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(np.float32) + rng.normal(0, ruido, img.shape), 0, 255).astype(np.uint8)
    t0 = time.perf_counter()
    suave = fil.suavizar(img, metodo, int(k), sigma_color=sigma_color)
    t_suave = (time.perf_counter() - t0) * 1000
    _, _, mag = fil.sobel(suave, int(sobel_k))
    bordes = fil.canny(suave, canny_bajo, canny_alto)
    ruido_antes = float(np.std(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    ruido_despues = float(np.std(cv2.Laplacian(cv2.cvtColor(suave, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))
    # Convolución manual vs OpenCV sobre la imagen en gris (reducida para que la manual sea rápida)
    try:
        nucleo = fil.parsear_nucleo(nucleo_txt)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        e = 480 / gris.shape[1]
        gris = cv2.resize(gris, None, fx=e, fy=e, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        manual = fil.convolucion_manual(gris, nucleo)
        t_man = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        ocv = fil.convolucion_opencv(gris, nucleo)
        t_ocv = (time.perf_counter() - t0) * 1000
        dif = float(np.abs(manual - ocv).max())
        conv = vis.mosaico([vis.normalizar_u8(manual), vis.normalizar_u8(ocv)],
                           ["Convolución manual (NumPy)", "cv2.filter2D (núcleo volteado)"], 2, 480)
        md_conv = tabla_md([["Núcleo", f"{nucleo.shape[0]}×{nucleo.shape[1]}, suma = {nucleo.sum():.3f}"],
                            ["Diferencia máxima manual vs. filter2D", f"{dif:.2e}"],
                            ["Tiempo manual", f"{t_man:.1f} ms"], ["Tiempo filter2D", f"{t_ocv:.2f} ms"]],
                           ["", "Valor"])
    except ValueError as e:
        conv, md_conv = None, f"**Núcleo inválido:** {e}"
    md = tabla_md([["Ruido (σ del laplaciano) antes", f"{ruido_antes:.1f}"],
                   ["Ruido después del suavizado", f"{ruido_despues:.1f}"],
                   [f"Tiempo {metodo}", f"{t_suave:.1f} ms"]], ["Medida", "Valor"])
    return (rgb(img), rgb(suave), rgb(vis.normalizar_u8(mag)), rgb(bordes), md, rgb(conv) if conv is not None else None,
            md_conv)


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Filtros: limpiar el ruido antes de buscar la piel")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada("Poca luz")
            ruido = gr.Slider(0, 40, value=12, step=1, label="Ruido gaussiano añadido (σ)")
            metodo = gr.Radio(list(fil.METODOS_SUAVIZADO), value="bilateral", label="Suavizado")
            k = gr.Slider(1, 21, value=7, step=2, label="Tamaño del núcleo (impar)")
            sc = gr.Slider(5, 150, value=50, step=5, label="σ de color (bilateral)")
            sk = gr.Slider(1, 7, value=3, step=2, label="Tamaño de Sobel")
            c1 = gr.Slider(0, 255, value=40, step=1, label="Canny: umbral bajo")
            c2 = gr.Slider(0, 255, value=120, step=1, label="Canny: umbral alto")
        with gr.Column(scale=6):
            with gr.Row():
                original = gr.Image(label="Original (con ruido)", height=230)
                suave = gr.Image(label="Suavizada", height=230)
            with gr.Row():
                sobel = gr.Image(label="Magnitud de Sobel", height=230)
                canny = gr.Image(label="Canny", height=230)
            medidas = gr.Markdown()
    gr.Markdown("### Convolución con tu propio núcleo")
    with gr.Row():
        with gr.Column(scale=3):
            ejemplo = gr.Dropdown(list(fil.NUCLEOS_EJEMPLO), value="Sobel X", label="Núcleos de ejemplo")
            nucleo = gr.Textbox(value=fil.NUCLEOS_EJEMPLO["Sobel X"], lines=4,
                                label="Núcleo (filas separadas por ';' o por línea; admite 1/9)")
            md_conv = gr.Markdown()
        conv = gr.Image(label="Manual vs. OpenCV", height=260, scale=7)
    ejemplo.change(lambda n: fil.NUCLEOS_EJEMPLO[n], ejemplo, nucleo)
    params = [ruido, metodo, k, sc, sk, c1, c2, nucleo]
    salidas = [original, suave, sobel, canny, medidas, conv, md_conv]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
