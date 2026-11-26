"""Pestaña Pipeline: cada paso del procesamiento sobre el frame actual."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import pipeline as pl
from voleyar.juego import render_ar as ra
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import ESTADO, Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
Cada paso es una función pura de `voleyar/core` y el juego llama exactamente a la misma cadena
(`core/pipeline.py::procesar_frame`):

1. **Distorsión** (`calibracion.corregir_distorsion`): `remap` con los mapas de $K$ y $d$.
2. **Pose** (`calibracion.estimar_pose_cancha`): ArUco + `solvePnP` → $R, t$.
3. **CLAHE** (`histogramas.aplicar_clahe`) + suavizado (`filtros.suavizar`).
4. **Retroproyección** de la piel (`histogramas.retroproyeccion`) dentro de la zona de la cancha.
5. **Otsu + morfología** (`umbralizacion.umbral_piel_otsu`, `morfologia.limpiar_mascara`).
6. **Contorno** y corte del brazo (`segmentacion`), en la vista cenital métrica.
7. **Casco convexo, dedos, punta** (`caracterizacion.analizar_mano`).
8. **Posición en cm** (rayo ∩ $z=0$) y velocidad.
9. **Render AR** (`juego.render_ar`): red, balón y sombra con `projectPoints`.
"""


def procesar(fr: Frame, usar_cal: bool, altura: float):
    res = ejecutar_pipeline(fr, usar_calibracion=usar_cal)
    pasos: list[tuple[np.ndarray, str]] = []
    if usar_cal and ESTADO.modelo_para(fr) is not None:
        pasos.append((vis.mosaico([cv2.resize(fr.img, (res.imagen.shape[1], res.imagen.shape[0])), res.imagen],
                                  ["Original", "Sin distorsión"], 2, 480), "1 · Corrección de distorsión"))
    else:
        pasos.append((res.imagen, "1 · Sin corrección (modo sin calibrar)"))
    if res.pose is None:
        pasos.append((res.imagen, "2 · Pose: no se detectan los marcadores"))
        return [(rgb(p), t) for p, t in pasos], "Sin pose el pipeline se detiene."
    ejes = cal.dibujar_ejes(res.imagen, res.pose, res.modelo, 6.0, 3)
    for i, c in res.marcadores.items():
        cv2.polylines(ejes, [c.astype(np.int32)], True, vis.BGR_NARANJA, 2, cv2.LINE_AA)
    pasos.append((ejes, f"2 · Pose: {len(res.marcadores)} marcadores, error {res.pose.error_rms:.2f} px"))
    pasos.append((res.clahe, "3 · CLAHE (luminancia)"))
    if res.retro is not None:
        pasos.append((cv2.applyColorMap(res.retro, cv2.COLORMAP_INFERNO), "4 · Retroproyección de la piel"))
    pasos.append((res.mascara_limpia, f"5 · Otsu (T = {res.T_otsu}) + apertura y cierre"))
    cont = res.imagen.copy()
    if res.contorno is not None:
        cv2.drawContours(cont, [res.contorno], -1, vis.BGR_NEUTRO, 2, cv2.LINE_AA)
    if res.contorno_mano_px is not None:
        cv2.polylines(cont, [np.round(res.contorno_mano_px).astype(np.int32)], True, vis.BGR_NARANJA, 3, cv2.LINE_AA)
    pasos.append((cont, "6 · Contorno de la mano (naranja) sin el brazo" + (" · watershed" if res.uso_watershed else "")))
    pasos.append((pl.dibujar_depuracion(res, ejes=False),
                  "7 · Casco convexo, valles y punta" + (f" · {res.analisis.n_dedos} dedos" if res.analisis else "")))
    cen = pl.dibujar_cenital(res)
    if cen is not None and res.punto_golpe_cm is not None:
        p = res.punto_golpe_cm
        vis.poner_texto(cen, f"({p[0]:.1f}, {p[1]:.1f}) cm", (10, cen.shape[0] - 30), 18, vis.BGR_NARANJA,
                        fondo=vis.BGR_GRAFITO)
    if cen is not None:
        pasos.append((cen, "8 · Posición en cm (vista cenital)"))
    balon = np.array([-4.0, 2.0, altura])
    ar = ra.dibujar_escena(res.imagen, res.pose, res.modelo, balon, np.array([7.0, 0.0]))
    ar = ra.ocluir_con_mano(ar, res.imagen, res.componente, res.pose, res.modelo, balon)
    pasos.append((ar, "9 · Render AR"))
    total = sum(res.tiempos_ms.values())
    md = tabla_md([[k, f"{v:.1f}"] for k, v in res.tiempos_ms.items()] + [["**Total**", f"**{total:.1f}**"]],
                  ["Etapa", "ms"])
    if res.analisis is not None:
        md += (f"\n\n**{res.analisis.n_dedos} dedos → {res.analisis.gesto}** · punto de golpe "
               + ("—" if res.punto_golpe_cm is None else f"({res.punto_golpe_cm[0]:.1f}, {res.punto_golpe_cm[1]:.1f}) cm"))
    return [(rgb(p), t) for p, t in pasos], md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Pipeline completo, paso a paso")
    with gr.Row():
        with gr.Column(scale=3):
            entrada = bloque_entrada()
            usar = gr.Checkbox(label="Con calibración", value=True)
            altura = gr.Slider(1.2, 15, value=5.0, step=0.2, label="Altura del balón (cm)")
            md = gr.Markdown(latex_delimiters=LATEX)
        galeria = gr.Gallery(label="Pasos", columns=3, height=720, object_fit="contain", scale=7)
    conectar(entrada, procesar, [usar, altura], [galeria, md], cada=0.3)
    cargar_al_inicio(demo, entrada, procesar, [usar, altura], [galeria, md])
    teoria(TEORIA)
