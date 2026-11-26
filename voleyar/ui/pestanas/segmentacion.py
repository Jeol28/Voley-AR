"""Pestaña de segmentación: mitades por jugador, caras, contornos, corte mano-brazo y watershed."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import pipeline as pl
from voleyar.core import segmentacion as seg
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
1. **Mitades:** la imagen se parte en dos; la izquierda es del jugador 1 y la derecha del jugador 2 (la red virtual
   está en la columna central). Cada mitad se segmenta con **su** modelo de piel.
2. **Caras (Viola-Jones):** rasgos Haar sobre la imagen integral + AdaBoost + cascada. Se busca una cara por mitad
   (la más grande, de al menos el 20 % de la altura). La cara **no** se recorta de la zona —la mano pasa a menudo por
   delante—, pero se descartan las regiones que son casi solo cara.
3. **Contornos** (Suzuki-Abe): la mano es la región que entra por el **borde inferior** con el antebrazo; una cara
   nunca toca ese borde.
4. **Corte mano-brazo** con la **transformada de distancia** $D(p)=\min_{q\notin A}\lVert p-q\rVert$: su máximo es el
   centro de la palma y su valor el radio $r$; el brazo es la parte gruesa que sale del disco de la palma y se corta a
   $1.2\,r$ (método *distancia*) o donde el ancho es mínimo (método *ancho*).
5. **Watershed con marcadores:** si la mano toca la cara, se siembra un marcador en el núcleo de la cara (Haar) y otros
   en los picos de $D$ de la mano; la inundación reparte la zona de contacto.
"""


def _recorte(img: np.ndarray, m: np.ndarray, margen: int = 40) -> np.ndarray:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return img
    y0, y1 = max(0, ys.min() - margen), min(img.shape[0], ys.max() + margen)
    x0, x1 = max(0, xs.min() - margen), min(img.shape[1], xs.max() + margen)
    return img[y0:y1, x0:x1]


def procesar(fr: Frame, lado, metodo: str, factor: float, frac_pico: float):
    lado = int(lado)
    res = ejecutar_pipeline(fr, metodo_corte=metodo)
    img = res.imagen
    r = res.manos[lado]
    # 1. Mitades y caras
    zona = img.copy()
    capa = zona.copy()
    capa[seg.mascara_mitad(img.shape, lado) > 0] = vis.BGR_NARANJA
    if res.mascara_caras is not None:
        capa[res.mascara_caras > 0] = vis.BGR_ROJO
    zona = cv2.addWeighted(capa, 0.3, zona, 0.7, 0)
    for x, y, w, h in res.caras:
        cv2.rectangle(zona, (x, y), (x + w, y + h), vis.BGR_ROJO, 2)
    cv2.line(zona, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), vis.BGR_NEUTRO, 2)
    # 2. Contornos candidatos
    cont = img.copy()
    if r.mascara_limpia is not None:
        for c in seg.contornos(r.mascara_limpia, 50):
            cv2.drawContours(cont, [c], -1, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    if r.contorno is not None:
        cv2.drawContours(cont, [r.contorno], -1, vis.BGR_NARANJA, 3, cv2.LINE_AA)
    filas = [["Caras detectadas (Haar)", str(len(res.caras))]]
    corte = dt_img = perfil = ws = None
    if r.componente is not None and r.componente.any():
        mano, info = seg.separar_mano_brazo(r.componente, r.zona, metodo, factor)
        corte = cv2.cvtColor(r.componente, cv2.COLOR_GRAY2BGR) // 3
        corte[mano > 0] = vis.BGR_NEUTRO
        c = tuple(int(v) for v in info.centro_palma)
        cv2.circle(corte, c, int(info.radio_palma), vis.BGR_NARANJA, 2)
        if info.corte:
            cv2.line(corte, info.corte[0], info.corte[1], vis.BGR_ROJO, 3)
        if info.direccion_brazo:
            p2 = (int(c[0] + 60 * info.direccion_brazo[0]), int(c[1] + 60 * info.direccion_brazo[1]))
            cv2.arrowedLine(corte, c, p2, vis.BGR_NARANJA, 2, tipLength=0.3)
        corte = _recorte(corte, r.componente)
        dt_img = _recorte(cv2.applyColorMap(vis.normalizar_u8(info.transformada), cv2.COLORMAP_INFERNO), r.componente)
        if info.anchos is not None:
            ds0 = int(0.6 * info.radio_palma)
            perfil = vis.grafica_lineas({"ancho": info.anchos}, "Ancho de la silueta a lo largo del brazo",
                                        "distancia al centro de la palma (px)", "px",
                                        marcas_x={"corte": float((info.distancia_corte or ds0) - ds0)})
        f = float(res.modelo.K[0, 0])
        filas += [["Radio de la palma", f"{info.radio_palma:.0f} px"],
                  ["Distancia del corte", f"{(info.distancia_corte or 0):.0f} px "
                   f"({(info.distancia_corte or 0) / max(info.radio_palma, 1):.2f} r)"],
                  ["Distancia de la mano Z = f·W/w", f"{r.distancia_cm:.0f} cm" if r.distancia_cm else "—"],
                  ["Escala en el plano de la mano", f"{r.cm_por_px * 10:.2f} mm/px" if r.cm_por_px else "—"],
                  ["Focal usada", f"{f:.0f} px"]]
        # 3. Watershed: el que usó el pipeline (mano-cara) o una demostración sobre la región
        if r.etiquetas_ws is not None and r.etiquetas_ws.any():
            etiquetas = np.where(r.etiquetas_ws >= 3, 1, np.where(r.etiquetas_ws == 2, 2,
                                                                   np.where(r.etiquetas_ws == -1, -1, 0)))
            nombre_ws = "separó la mano (naranja) de la cara"
        else:
            _, etiquetas, n = seg.watershed_mano(res.suavizada, r.componente, seg.franja_inferior(img.shape), frac_pico)
            nombre_ws = f"{n} regiones sobre la mano y el brazo (no hizo falta separar nada)"
        ws = _recorte(seg.colorear_etiquetas(etiquetas.astype(np.int32)), r.componente)
        filas.append(["Watershed", nombre_ws])
    else:
        filas.append(["Mano", "no se encontró en esta mitad"])
    return (rgb(zona), rgb(cont), rgb(corte) if corte is not None else None,
            rgb(dt_img) if dt_img is not None else None, perfil, rgb(ws) if ws is not None else None,
            tabla_md(filas, ["Medida", "Valor"]))


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Segmentación: encontrar la mano de cada jugador y separarla del brazo y de la cara")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada("Mano delante de la cara")
            jugador = gr.Radio([(pl.NOMBRES_LADO[-1] + " (izquierda)", -1), (pl.NOMBRES_LADO[1] + " (derecha)", 1)],
                               value=-1, label="Jugador")
            metodo = gr.Radio(["distancia", "ancho"], value="distancia", label="Corte mano-brazo")
            factor = gr.Slider(0.8, 2.0, value=1.2, step=0.05, label="Corte a k · radio de la palma (método distancia)")
            frac = gr.Slider(0.3, 0.9, value=0.55, step=0.05, label="Watershed: umbral de los picos (fracción de D máx.)")
            md = gr.Markdown()
        with gr.Column(scale=6):
            with gr.Row():
                zona = gr.Image(label="Mitad del jugador (naranja) y caras de Haar (rojo)", height=250)
                cont = gr.Image(label="Contornos (naranja: la mano elegida)", height=250)
            with gr.Row():
                corte = gr.Image(label="Corte por la muñeca", height=230)
                dt = gr.Image(label="Transformada de distancia", height=230)
            with gr.Row():
                ws = gr.Image(label="Watershed", height=230)
                perfil = gr.Image(label="Perfil de ancho (método ancho)", height=230)
    params = [jugador, metodo, factor, frac]
    salidas = [zona, cont, corte, dt, perfil, ws, md]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
