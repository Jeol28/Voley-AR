"""Pestaña de segmentación: zona por la pose, contornos, corte mano-brazo, watershed y cancha."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import segmentacion as seg
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import Frame, ejecutar_pipeline, rgb, tabla_md
from voleyar.utils import visualizacion as vis

TEORIA = r"""
1. **Zona por la pose:** se proyecta el volumen de juego (cancha × 12 cm de alto) y una franja para el brazo
   detrás de tu borde. Lo de afuera (cara, cuerpo, muebles) no se analiza.
2. **Contornos** (Suzuki-Abe, `findContours`): la mano es la componente que entra por la franja del brazo.
3. **Corte mano-brazo** con la **transformada de distancia** $D(p)=\min_{q\notin A}\lVert p-q\rVert$: su máximo
   es el centro de la palma y su valor el radio $r$. El brazo es la parte gruesa que sale del disco de la palma;
   se corta perpendicular a él a $1.2\,r$ (método *distancia*) o donde el ancho es mínimo (método *ancho*).
   Se hace en la **vista cenital métrica** (homografía de la mesa), donde no hay escorzo.
4. **Watershed:** marcadores en los picos de $D$ (un núcleo por objeto) e inundación; separa la mano de otra
   región de color parecido que la toque.
5. **Cancha:** su máscara (y la de cada lado) sale de proyectar el rectángulo con la pose.
"""


def procesar(fr: Frame, metodo: str, factor: float, frac_pico: float):
    res = ejecutar_pipeline(fr, metodo_corte=metodo)
    img = res.imagen
    if res.zona is None:
        vista = img.copy()
        vis.poner_texto(vista, "No se detectan los marcadores ArUco", (20, 20), 26, vis.BGR_NARANJA,
                        fondo=vis.BGR_GRAFITO)
        return rgb(vista), None, None, None, None, None, "Sin pose no hay zona de búsqueda."
    # 1. Zona y cancha
    zona = img.copy()
    capa = zona.copy()
    capa[res.zona.franja > 0] = vis.BGR_NARANJA
    tuyo, rival = seg.mascara_lados(img.shape, res.pose, res.modelo)
    capa[tuyo > 0] = (capa[tuyo > 0] * 0.4 + np.array(vis.BGR_NEUTRO) * 0.6).astype(np.uint8)
    capa[rival > 0] = (capa[rival > 0] * 0.4 + np.array((150, 110, 60)) * 0.6).astype(np.uint8)
    zona = cv2.addWeighted(capa, 0.45, zona, 0.55, 0)
    cs, _ = cv2.findContours(res.zona.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(zona, cs, -1, vis.BGR_NARANJA, 2, cv2.LINE_AA)
    # 2. Contornos candidatos
    cont = img.copy()
    for i, c in enumerate(seg.contornos(res.mascara_limpia, 50)):
        cv2.drawContours(cont, [c], -1, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    if res.contorno is not None:
        cv2.drawContours(cont, [res.contorno], -1, vis.BGR_NARANJA, 3, cv2.LINE_AA)
    # 3. Corte del brazo (vista cenital), con el factor elegido
    filas = [["Píxeles de la zona", f"{np.count_nonzero(res.zona.mascara)}"]]
    corte = dt_img = perfil = ws = None
    if res.componente_cenital is not None:
        zona_c = res.vista.rectificar(res.zona.mascara)
        mano_c, info = seg.separar_mano_brazo(res.componente_cenital, zona_c, metodo, factor)
        corte = cv2.cvtColor(res.componente_cenital, cv2.COLOR_GRAY2BGR) // 3
        corte[mano_c > 0] = vis.BGR_NEUTRO
        c = tuple(int(v) for v in info.centro_palma)
        cv2.circle(corte, c, int(info.radio_palma), vis.BGR_NARANJA, 2)
        if info.corte:
            cv2.line(corte, info.corte[0], info.corte[1], vis.BGR_ROJO, 3)
        if info.direccion_brazo:
            p2 = (int(c[0] + 40 * info.direccion_brazo[0]), int(c[1] + 40 * info.direccion_brazo[1]))
            cv2.arrowedLine(corte, c, p2, vis.BGR_NARANJA, 2, tipLength=0.3)
        dt_img = cv2.applyColorMap(vis.normalizar_u8(info.transformada), cv2.COLORMAP_INFERNO)
        if info.anchos is not None:
            ds = np.arange(int(0.6 * info.radio_palma), int(0.6 * info.radio_palma) + len(info.anchos))
            perfil = vis.grafica_lineas({"ancho": info.anchos}, "Ancho de la silueta a lo largo del brazo",
                                        "distancia al centro de la palma (px cenitales)", "px",
                                        marcas_x={"corte": float(info.distancia_corte - ds[0])})
        filas += [["Radio de la palma", f"{info.radio_palma / res.vista.ppcm:.1f} cm"],
                  ["Distancia del corte", f"{(info.distancia_corte or 0) / res.vista.ppcm:.1f} cm"],
                  ["Área de la mano sin brazo", f"{np.count_nonzero(mano_c) / res.vista.ppcm ** 2:.0f} cm²"]]
    # 4. Watershed sobre la componente de la imagen
    if res.componente is not None and res.componente.any():
        _, etiquetas, n = seg.watershed_mano(res.suavizada, res.componente, res.zona.franja, frac_pico)
        ws = seg.colorear_etiquetas(etiquetas)
        filas.append(["Regiones del watershed", str(n)])
        filas.append(["¿El pipeline usó watershed?", "sí" if res.uso_watershed else "no (área normal)"])
    return (rgb(zona), rgb(cont), rgb(corte) if corte is not None else None, rgb(dt_img) if dt_img is not None
            else None, perfil, rgb(ws) if ws is not None else None, tabla_md(filas, ["Medida", "Valor"]))


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Segmentación: encontrar la mano y separarla del brazo")
    with gr.Row():
        with gr.Column(scale=4):
            entrada = bloque_entrada("Mano tocando un objeto color piel")
            metodo = gr.Radio(["distancia", "ancho"], value="distancia", label="Corte mano-brazo")
            factor = gr.Slider(0.8, 2.0, value=1.2, step=0.05, label="Corte a k · radio de la palma (método distancia)")
            frac = gr.Slider(0.3, 0.9, value=0.55, step=0.05, label="Watershed: umbral de los picos (fracción de D máx.)")
            md = gr.Markdown()
        with gr.Column(scale=6):
            with gr.Row():
                zona = gr.Image(label="Zona de búsqueda (franja del brazo en naranja) y lados de la cancha", height=250)
                cont = gr.Image(label="Contornos (naranja: la mano)", height=250)
            with gr.Row():
                corte = gr.Image(label="Corte por la muñeca (vista cenital)", height=230)
                dt = gr.Image(label="Transformada de distancia", height=230)
            with gr.Row():
                ws = gr.Image(label="Watershed", height=230)
                perfil = gr.Image(label="Perfil de ancho (método ancho)", height=230)
    params = [metodo, factor, frac]
    salidas = [zona, cont, corte, dt, perfil, ws, md]
    conectar(entrada, procesar, params, salidas)
    cargar_al_inicio(demo, entrada, procesar, params, salidas)
    teoria(TEORIA)
