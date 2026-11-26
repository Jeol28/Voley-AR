"""Pestaña de transformers: Depth Anything V2, OWLv2 y comparación clásico vs. transformer."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import transformers_models as tm
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = rf"""
Solo **inferencia** con pesos públicos (no se entrena nada). Se descargan la primera vez y se cargan una sola vez.

**[Depth Anything V2 Small]({tm.URL_HF}{tm.MODELO_PROFUNDIDAD})**: ViT (DINOv2) + cabeza DPT que estima la
**profundidad relativa** (disparidad, más alto = más cerca), definida salvo escala y desplazamiento. Para saber si
la mano está **cerca de la mesa** se compara con la mesa que la rodea y se normaliza por la variación de la cancha:

$$ s=\frac{{\mathrm{{med}}(D_{{mano}})-\mathrm{{med}}(D_{{anillo}})}}{{P_{{95}}(D_{{cancha}})-P_5(D_{{cancha}})}} $$

La resta quita el desplazamiento y la división la escala. Si $s$ supera el umbral, el golpe no vale.
**Por qué cada N frames:** tarda ~100 ms por imagen en MPS frente a ~33 ms de un frame; la altura de la mano cambia
más despacio que su posición, así que el juego lo corre en un hilo aparte cada 8 frames y usa el último resultado.

**[OWLv2]({tm.URL_HF}{tm.MODELO_OWL})**: detector de vocabulario abierto basado en CLIP. Cada parche del ViT
propone una caja y se clasifica comparando su embedding con el del texto (`"a hand"`). No necesita calibrar la
piel, pero es ~40 veces más lento que el pipeline clásico.
"""


def _caja(img, caja, color, texto):
    x0, y0, x1, y1 = (int(v) for v in caja)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    vis.poner_texto(img, texto, (x0 + 4, max(0, y0 - 26)), 18, color, True, fondo=vis.BGR_GRAFITO)


def profundidad(fr: Frame, umbral: float, ejecutar: bool):
    if not ejecutar:
        return None, "Activa **Ejecutar Depth Anything** (la primera vez descarga ≈ 100 MB)."
    res = ejecutar_pipeline(fr)
    r = tm.obtener_profundidad().estimar(res.imagen)
    mapa = cv2.applyColorMap(vis.normalizar_u8(r.mapa), cv2.COLORMAP_INFERNO)
    md = f"Latencia: **{r.ms:.0f} ms** en `{r.dispositivo}`."
    if res.mascara_mano is not None and res.zona is not None:
        alt = tm.mano_cerca_de_mesa(r.mapa, res.mascara_mano, res.zona.cancha, umbral)
        cs, _ = cv2.findContours(res.mascara_mano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mapa, cs, -1, vis.BGR_NEUTRO, 2)
        clase = "dato-ok" if alt.cerca else "dato-mal"
        md += (f"\n\nPuntaje de altura $s$ = **{alt.puntaje:.2f}** (umbral {umbral:.2f}) → "
               f'<span class="{clase}">{"cerca de la mesa: golpe válido" if alt.cerca else "mano levantada: golpe inválido"}</span>'
               + "\n\n" + tabla_md([["Mediana mano", f"{alt.med_mano:.3f}"], ["Mediana mesa alrededor", f"{alt.med_mesa:.3f}"],
                                    ["Rango de la cancha (P95−P5)", f"{alt.rango_cancha:.3f}"]], ["", "Valor"]))
    else:
        md += "\n\nNo se encontró la mano o la cancha para calcular el puntaje."
    return rgb(mapa), md


def comparar(fr: Frame, texto: str, umbral_owl: float, ejecutar: bool):
    """Mismo frame: pipeline clásico (piel + morfología + contorno) vs. OWLv2."""
    res = ejecutar_pipeline(fr)
    ms_clasico = sum(res.tiempos_ms.values())  # solo el pipeline, como en el juego
    img = res.imagen.copy()
    caja_c = tm.caja_de_mascara(res.mascara_mano) if res.mascara_mano is not None else None
    if caja_c:
        _caja(img, caja_c, vis.BGR_NARANJA, f"clásico · {ms_clasico:.0f} ms")
    if not ejecutar:
        return rgb(img), "Activa **Ejecutar OWLv2** (la primera vez descarga ≈ 600 MB)."
    zona = res.zona.mascara if res.zona is not None else None
    owl = tm.obtener_owl().detectar(res.imagen, tuple(t.strip() for t in texto.split(",") if t.strip()) or ("a hand",),
                                    umbral_owl, zona)
    for d in owl.detecciones:
        if d is not owl.mejor:
            _caja(img, d.caja, vis.BGR_NEUTRO, f"{d.etiqueta} {d.puntaje:.2f}")
    if owl.mejor:
        _caja(img, owl.mejor.caja, vis.BGR_VERDE, f"OWLv2 {owl.mejor.puntaje:.2f} · {owl.ms:.0f} ms")
    a_cm = (lambda p: cal.pixel_a_mesa(p, res.pose, res.modelo)) if res.pose is not None else None
    c = tm.comparar_clasico_vs_owl(res.mascara_mano, ms_clasico, owl, a_cm)
    filas = [["Latencia por frame", f"{c.ms_clasico:.1f} ms", f"{c.ms_owl:.0f} ms"],
             ["FPS máximos", f"{1000 / max(c.ms_clasico, 1e-3):.0f}", f"{1000 / max(c.ms_owl, 1e-3):.1f}"],
             ["Detecciones", "1" if caja_c else "0", str(len(owl.detecciones))]]
    md = tabla_md(filas, ["", "Clásico", "OWLv2"]) + "\n\n"
    if c.iou is not None:
        md += (f"**IoU entre cajas:** {c.iou:.2f} · **diferencia de centros:** {c.distancia_px:.0f} px"
               + (f" ≈ **{c.distancia_cm:.1f} cm** en la mesa" if c.distancia_cm is not None else "")
               + ". OWLv2 suele incluir parte del antebrazo; el clásico lo corta en la muñeca.")
    else:
        md += "No hay detección en alguno de los dos métodos."
    return rgb(img), md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Transformers: profundidad monocular y detección de vocabulario abierto")
    gr.Markdown(f"Dispositivo: **`{tm.dispositivo()}`** · modelos: "
                f"[{tm.MODELO_PROFUNDIDAD}]({tm.URL_HF}{tm.MODELO_PROFUNDIDAD}) y "
                f"[{tm.MODELO_OWL}]({tm.URL_HF}{tm.MODELO_OWL})")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada()
            ej_prof = gr.Checkbox(label="Ejecutar Depth Anything", value=True)
            umbral = gr.Slider(0.05, 1.0, value=0.30, step=0.01, label="Umbral de altura s")
            ej_owl = gr.Checkbox(label="Ejecutar OWLv2", value=True)
            texto = gr.Textbox(value="a hand", label="Consulta de texto (separa varias con comas)")
            umbral_owl = gr.Slider(0.02, 0.6, value=0.15, step=0.01, label="Umbral de confianza OWLv2")
        with gr.Column(scale=6):
            with gr.Row():
                mapa = gr.Image(label="Profundidad relativa (claro = cerca)", height=280)
                comp = gr.Image(label="Clásico (naranja) vs. OWLv2 (verde)", height=280)
            with gr.Row():
                md_prof = gr.Markdown(latex_delimiters=LATEX)
                md_comp = gr.Markdown(latex_delimiters=LATEX)
    conectar(entrada, profundidad, [umbral, ej_prof], [mapa, md_prof], cada=0.5)
    conectar(entrada, comparar, [texto, umbral_owl, ej_owl], [comp, md_comp], cada=1.0)
    cargar_al_inicio(demo, entrada, profundidad, [umbral, ej_prof], [mapa, md_prof])
    cargar_al_inicio(demo, entrada, comparar, [texto, umbral_owl, ej_owl], [comp, md_comp])
    teoria(TEORIA)
