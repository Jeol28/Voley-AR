"""Pestaña de morfología: erosión, dilatación, apertura, cierre y gradiente sobre la máscara de la mano."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import morfologia as mor
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
Con un **elemento estructurante** $B$:

- **Erosión** $A\ominus B=\{z\mid B_z\subseteq A\}$: mínimo de la vecindad. Encoge y borra lo que es más pequeño que $B$.
- **Dilatación** $A\oplus B=\{z\mid \hat B_z\cap A\neq\emptyset\}$: máximo de la vecindad. Agranda y rellena.
- **Apertura** $A\circ B=(A\ominus B)\oplus B$: quita **puntos sueltos** sin cambiar mucho el tamaño.
- **Cierre** $A\bullet B=(A\oplus B)\ominus B$: rellena **huecos** y grietas (p. ej. entre dedos).
- **Gradiente** $(A\oplus B)-(A\ominus B)$: deja solo el borde.

En el juego: **apertura → cierre → relleno de huecos**. Ojo con el tamaño del cierre: con 9 px los dedos juntos
se pegaban y el conteo fallaba (45/90 aciertos en las pruebas); con 5 px acierta 90/90.
"""


NOMBRES = {"erosion": "Erosión", "dilatacion": "Dilatación", "apertura": "Apertura", "cierre": "Cierre",
           "gradiente": "Gradiente"}


def _ruidosa(m: np.ndarray, nivel: float) -> np.ndarray:
    if nivel <= 0:
        return m
    rng = np.random.default_rng(1)
    r = m.copy()
    ruido = rng.random(m.shape) < nivel / 100
    r[ruido & (m == 0)] = 255
    r[ruido & (m > 0)] = 0
    return r


def ver_elemento(forma: str, tam: int) -> np.ndarray:
    ee = mor.elemento_estructurante(forma, int(tam)) * 255
    celda = max(8, 160 // max(1, int(tam)))
    grande = cv2.resize(ee.astype(np.uint8), None, fx=celda, fy=celda, interpolation=cv2.INTER_NEAREST)
    out = cv2.cvtColor(grande, cv2.COLOR_GRAY2BGR)
    out[grande > 0] = vis.BGR_NARANJA
    for i in range(0, out.shape[0] + 1, celda):
        cv2.line(out, (0, i), (out.shape[1], i), vis.BGR_GRAFITO, 1)
        cv2.line(out, (i, 0), (i, out.shape[0]), vis.BGR_GRAFITO, 1)
    return out


def procesar(fr: Frame, nivel_ruido: float, operacion: str, forma: str, tam: int, iteraciones: int):
    res = ejecutar_pipeline(fr)
    base = res.mascara_umbral
    if base is None:
        return None, None, None, None, "No se detecta la cancha: sin pose no hay zona de búsqueda."
    m = _ruidosa(base, nivel_ruido)
    out = mor.aplicar(m, operacion, forma, int(tam), int(iteraciones))
    ops = [mor.aplicar(m, op, forma, int(tam), int(iteraciones)) for op in mor.OPERACIONES]
    comp = vis.mosaico([m] + ops, ["Entrada"] + [NOMBRES[o] for o in mor.OPERACIONES], 3, 360)
    n0 = cv2.connectedComponents(m)[0] - 1
    n1 = cv2.connectedComponents(out)[0] - 1
    huecos = lambda x: int(np.count_nonzero(mor.rellenar_huecos(x)) - np.count_nonzero(x))  # noqa: E731
    md = tabla_md([["Componentes conexas", n0, n1], ["Píxeles encendidos", int(np.count_nonzero(m)),
                                                       int(np.count_nonzero(out))],
                   ["Píxeles en huecos interiores", huecos(m), huecos(out)]],
                  ["Medida", "Antes", f"Después ({NOMBRES[operacion].lower()})"])
    return rgb(m), rgb(out), rgb(ver_elemento(forma, tam)), rgb(comp), md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Morfología matemática: limpiar la máscara de la mano")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada()
            ruido = gr.Slider(0, 10, value=2, step=0.5, label="Ruido sal y pimienta añadido (%)")
            op = gr.Radio([(NOMBRES[o], o) for o in mor.OPERACIONES], value="apertura", label="Operación")
            forma = gr.Radio([("elipse", "elipse"), ("rectángulo", "rectangulo"), ("cruz", "cruz")], value="elipse",
                             label="Forma del elemento estructurante")
            tam = gr.Slider(1, 25, value=5, step=1, label="Tamaño del elemento (px)")
            it = gr.Slider(1, 5, value=1, step=1, label="Iteraciones")
            elemento = gr.Image(label="Elemento estructurante", height=170)
        with gr.Column(scale=6):
            with gr.Row():
                antes = gr.Image(label="Máscara de piel (Otsu) + ruido", height=260)
                despues = gr.Image(label="Resultado", height=260)
            md = gr.Markdown()
            comp = gr.Image(label="Las cinco operaciones con el mismo elemento", height=360)
    params = [ruido, op, forma, tam, it]
    salidas = [antes, despues, elemento, comp, md]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
