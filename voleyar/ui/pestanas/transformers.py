"""Pestaña de transformers: Depth Anything V2, OWLv2 y comparación clásico vs. transformer."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import transformers_models as tm
from voleyar.entorno import EN_SPACE
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = rf"""
Solo **inferencia** con pesos públicos (no se entrena nada). Se descargan la primera vez y se cargan una sola vez.

**[Depth Anything V2 Small]({tm.URL_HF}{tm.MODELO_PROFUNDIDAD})**: ViT (DINOv2) + cabeza DPT que estima la
**profundidad relativa** (disparidad, más alto = más cerca), definida salvo escala y desplazamiento. En el modo
pantalla se usa para saber si la mano está **delante del cuerpo** de su jugador: se compara con su cara (Haar) y se
normaliza por la variación de su mitad de la imagen:

$$ s=\frac{{\mathrm{{med}}(D_{{mano}})-\mathrm{{med}}(D_{{cara}})}}{{P_{{95}}-P_5}} $$

La resta quita el desplazamiento y la división la escala. Si la mano queda detrás de la cara (por ejemplo, la de
alguien que pasa por el fondo), el golpe no vale. La distancia en cm la da la calibración ($Z=f\,W/w$).
**Por qué cada N frames:** tarda ~100 ms por imagen en MPS frente a ~33 ms de un frame; la posición de la mano en
profundidad cambia más despacio que en la imagen, así que el juego lo corre en un hilo aparte cada 8 frames.

**[OWLv2]({tm.URL_HF}{tm.MODELO_OWL})**: detector de vocabulario abierto basado en CLIP. Cada parche del ViT
propone una caja y se clasifica comparando su embedding con el del texto (`"a hand"`). No necesita calibrar la
piel, pero es ~40 veces más lento que el pipeline clásico.
"""


def _caja(img, caja, color, texto):
    x0, y0, x1, y1 = (int(v) for v in caja)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    vis.poner_texto(img, texto, (x0 + 4, max(0, y0 - 26)), 18, color, True, fondo=vis.BGR_GRAFITO)


def profundidad(fr: Frame, umbral: float, ejecutar: bool):
    """Depth Anything sobre el frame y, por jugador, si su mano está delante de su cara."""
    if not ejecutar:
        return None, "Activa **Ejecutar Depth Anything** (la primera vez descarga ≈ 100 MB)."
    from voleyar.core import pipeline as pl

    res = ejecutar_pipeline(fr)
    r = tm.estimar_profundidad(res.imagen)
    mapa = cv2.applyColorMap(vis.normalizar_u8(r.mapa), cv2.COLORMAP_INFERNO)
    md = f"Latencia: **{r.ms:.0f} ms** en `{r.dispositivo}`.\n\n"
    filas = []
    w = res.imagen.shape[1]
    for lado in pl.LADOS:
        m = res.manos[lado]
        caras = [c for c in res.caras if (c[0] + c[2] / 2 < w / 2) == (lado < 0)]
        if m.mascara_mano is not None:
            cs, _ = cv2.findContours(m.mascara_mano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mapa, cs, -1, vis.BGR_NEUTRO, 2)
        for x, y, cw, ch in caras:
            cv2.rectangle(mapa, (x, y), (x + cw, y + ch), vis.BGR_NEUTRO, 1)
        if m.mascara_mano is None or not caras:
            filas.append([pl.NOMBRES_LADO[lado], "—", "—", "falta la mano o la cara"])
            continue
        d = tm.mano_delante_de_la_cara(r.mapa, m.mascara_mano, caras[0], margen=-umbral)
        estado = ('<span class="dato-ok">delante del cuerpo: golpe válido</span>' if d.delante
                  else '<span class="dato-mal">detrás de la cara: golpe inválido</span>')
        filas.append([pl.NOMBRES_LADO[lado], f"{d.puntaje:.2f}", f"{m.distancia_cm:.0f} cm", estado])
    md += tabla_md(filas, ["Jugador", "Puntaje s (mano − cara)", "Z por la palma", "Decisión"])
    return rgb(mapa), md


def comparar(fr: Frame, texto: str, umbral_owl: float, ejecutar: bool):
    """Mismo frame: pipeline clásico (piel + morfología + contorno) vs. OWLv2, en cada mitad."""
    from voleyar.core import pipeline as pl

    res = ejecutar_pipeline(fr)
    ms_clasico = sum(res.tiempos_ms.values())  # solo el pipeline, como en el juego
    img = res.imagen.copy()
    w = img.shape[1]
    cv2.line(img, (w // 2, 0), (w // 2, img.shape[0]), vis.BGR_NEUTRO, 1)
    for m in res.manos.values():
        caja = tm.caja_de_mascara(m.mascara_mano) if m.mascara_mano is not None else None
        if caja:
            _caja(img, caja, vis.BGR_NARANJA, f"clásico · {ms_clasico:.0f} ms")
    if not ejecutar:
        return rgb(img), "Activa **Ejecutar OWLv2** (la primera vez descarga ≈ 600 MB)."
    owl = tm.detectar_manos(res.imagen, tuple(t.strip() for t in texto.split(",") if t.strip()) or ("a hand",),
                            umbral_owl, None)
    filas = []
    for lado in pl.LADOS:
        m = res.manos[lado]
        dets = [d for d in owl.detecciones if (d.centro[0] < w / 2) == (lado < 0)]
        sub = tm.ResultadoOWL(dets, owl.ms, owl.dispositivo)
        for d in dets:
            color = vis.BGR_VERDE if d is sub.mejor else vis.BGR_NEUTRO
            _caja(img, d.caja, color, f"OWLv2 {d.puntaje:.2f}")
        a_cm = (lambda p, m=m: None if m.cm_por_px is None else np.array(p) * m.cm_por_px)
        c = tm.comparar_clasico_vs_owl(m.mascara_mano, ms_clasico, sub, a_cm)
        filas.append([pl.NOMBRES_LADO[lado], "sí" if c.caja_clasico else "no", str(len(dets)),
                      "—" if c.iou is None else f"{c.iou:.2f}",
                      "—" if c.distancia_px is None else f"{c.distancia_px:.0f} px"
                      + ("" if c.distancia_cm is None else f" ≈ {c.distancia_cm:.1f} cm")])
    md = (tabla_md([["Latencia por frame", f"{ms_clasico:.1f} ms", f"{owl.ms:.0f} ms"],
                    ["FPS máximos", f"{1000 / max(ms_clasico, 1e-3):.0f}", f"{1000 / max(owl.ms, 1e-3):.1f}"]],
                   ["", "Clásico (2 manos)", "OWLv2"]) + "\n\n"
          + tabla_md(filas, ["Jugador", "Mano clásica", "Cajas OWLv2", "IoU", "Diferencia de centros"])
          + "\n\nLa distancia en cm usa la escala del plano de la mano (Z/f). OWLv2 suele incluir parte del antebrazo; "
            "el clásico lo corta en la muñeca.")
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
            umbral = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Margen: la mano puede estar hasta s detrás de la cara")
            ej_owl = gr.Checkbox(label="Ejecutar OWLv2", value=True)
            texto = gr.Textbox(value="a hand", label="Consulta de texto (separa varias con comas)")
            umbral_owl = gr.Slider(0.02, 0.6, value=0.15, step=0.01, label="Umbral de confianza OWLv2")
        with gr.Column(scale=6):
            with gr.Row():
                mapa = gr.Image(label="Profundidad relativa (claro = cerca); manos y caras en blanco", height=280)
                comp = gr.Image(label="Clásico (naranja) vs. OWLv2 (verde)", height=280)
            with gr.Row():
                md_prof = gr.Markdown(latex_delimiters=LATEX)
                md_comp = gr.Markdown(latex_delimiters=LATEX)
    # En un Space cada llamada pide la GPU (ZeroGPU) y gasta cuota del visitante: en vivo, un frame cada 2-3 s
    conectar(entrada, profundidad, [umbral, ej_prof], [mapa, md_prof], cada=2.0 if EN_SPACE else 0.5)
    conectar(entrada, comparar, [texto, umbral_owl, ej_owl], [comp, md_comp], cada=3.0 if EN_SPACE else 1.0)
    cargar_al_inicio(demo, entrada, profundidad, [umbral, ej_prof], [mapa, md_prof])
    cargar_al_inicio(demo, entrada, comparar, [texto, umbral_owl, ej_owl], [comp, md_comp])
    teoria(TEORIA)
