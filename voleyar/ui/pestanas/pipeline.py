"""Pestaña Pipeline: cada paso del procesamiento sobre el frame actual."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import pipeline as pl
from voleyar.juego import render_ar as ra
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import ESTADO, Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
Cada paso es una función pura de `voleyar/core` y el juego llama exactamente a la misma cadena
(`core/pipeline.py::procesar_frame`), una vez por cada mitad de la imagen (un jugador por mitad):

1. **Distorsión** (`calibracion.corregir_distorsion`): `remap` con los mapas de $K$ y $d$.
2. **Caras** (`segmentacion.detectar_caras`, Viola-Jones): una por mitad; se recuerdan si la mano las tapa.
3. **CLAHE** (`histogramas.aplicar_clahe`) + suavizado (`filtros.suavizar`).
4. **Retroproyección** de la piel de cada jugador (`histogramas.retroproyeccion`) en su mitad.
5. **Otsu + morfología** (`umbralizacion.umbral_piel_otsu`, `morfologia.limpiar_mascara`).
6. **Contorno** que entra por abajo, corte del brazo y watershed contra la cara (`segmentacion`).
7. **Casco convexo, dedos, punta** (`caracterizacion.analizar_mano`).
8. **Distancia** $Z=f\,W/w$ y **punto en el plano de la red** con el rayo de la cámara virtual (`calibracion`).
9. **Render AR** (`juego.render_ar`): piso, red, balón y sombra con `projectPoints` y la cámara virtual.
"""


def procesar(fr: Frame, usar_cal: bool, altura: float):
    res = ejecutar_pipeline(fr, usar_calibracion=usar_cal)
    pasos: list[tuple[np.ndarray, str]] = []
    if usar_cal and ESTADO.modelo_para(fr) is not None:
        pasos.append((vis.mosaico([cv2.resize(fr.img, (res.imagen.shape[1], res.imagen.shape[0])), res.imagen],
                                  ["Original", "Sin distorsión"], 2, 480), "1 · Corrección de distorsión"))
    else:
        pasos.append((res.imagen, "1 · Sin corrección (modo sin calibrar)"))
    caras = res.imagen.copy()
    h, w = caras.shape[:2]
    cv2.line(caras, (w // 2, 0), (w // 2, h), vis.BGR_NARANJA, 3)
    for x, y, cw, ch in res.caras:
        cv2.rectangle(caras, (x, y), (x + cw, y + ch), vis.BGR_ROJO, 3)
    pasos.append((caras, f"2 · Mitades por jugador y {len(res.caras)} cara(s) (Haar)"))
    pasos.append((res.clahe, "3 · CLAHE (luminancia)"))
    retros = [m.retro for m in res.manos.values() if m.retro is not None]
    if retros:
        pasos.append((cv2.applyColorMap(np.maximum.reduce(retros), cv2.COLORMAP_INFERNO),
                      "4 · Retroproyección de la piel de cada jugador"))
    ts = ", ".join(f"T{'₁' if l < 0 else '₂'} = {m.T_otsu}" for l, m in res.manos.items() if m.T_otsu is not None)
    if res.mascara_limpia is not None:
        pasos.append((res.mascara_limpia, f"5 · Otsu ({ts}) + apertura y cierre"))
    pasos.append((pl.dibujar_mascaras(res), "6 · Mano (blanco), brazo (naranja), caras (rojo)"))
    pasos.append((pl.dibujar_depuracion(res), "7 · Casco convexo, valles, puntas y punto de golpe"))
    pos = res.imagen.copy()
    for lado, m in res.manos.items():
        if m.hay_mano:
            p = m.punto_plano
            texto = f"Z = {m.distancia_cm:.0f} cm · x = {p[0]:.1f}, altura {p[2]:.1f}"
            u, v = (int(c) for c in m.punto_golpe_px)
            cv2.drawMarker(pos, (u, v), vis.BGR_NARANJA, cv2.MARKER_CROSS, 26, 3)
            vis.poner_texto(pos, texto, (max(0, u - 120), max(0, v - 50)), 18, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
    pasos.append((pos, "8 · Distancia por la palma y punto en el plano de la red"))
    balon = np.array([-5.0, 0.0, altura])
    ar = ra.dibujar_escena_pantalla(res.imagen, res.pose, res.modelo, balon)
    pasos.append((ar, "9 · Render AR con la cámara virtual"))
    total = sum(res.tiempos_ms.values())
    md = tabla_md([[k, f"{v:.1f}"] for k, v in res.tiempos_ms.items()] + [["**Total**", f"**{total:.1f}**"]],
                  ["Etapa", "ms"])
    for lado, m in res.manos.items():
        if m.analisis is not None:
            md += (f"\n\n**{pl.NOMBRES_LADO[lado]}:** {m.analisis.n_dedos} dedos → {m.analisis.gesto}"
                   + (f" · {m.distancia_cm:.0f} cm" if m.distancia_cm else ""))
    return [(rgb(p), t) for p, t in pasos], md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Pipeline completo, paso a paso")
    with gr.Row():
        with gr.Column(scale=3):
            entrada = bloque_entrada()
            usar = gr.Checkbox(label="Con calibración", value=True)
            altura = gr.Slider(1.2, 12, value=7.0, step=0.2, label="Altura del balón (unidades de cancha)")
            md = gr.Markdown(latex_delimiters=LATEX)
        galeria = gr.Gallery(label="Pasos", columns=3, height=720, object_fit="contain", scale=7)
    conectar(entrada, procesar, [usar, altura], [galeria, md], cada=0.3)
    cargar_al_inicio(demo, entrada, procesar, [usar, altura], [galeria, md])
    teoria(TEORIA)
