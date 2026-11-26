"""Pestaña de calibración: captura guiada del tablero y comparación con/sin calibración."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.juego import render_ar as ra
from voleyar.ui.comunes import LATEX, bloque_entrada, cargar_al_inicio, conectar, teoria
from voleyar.ui.estado import ESTADO, Frame, rgb, tabla_md
from voleyar.utils import sinteticos as sin
from voleyar.utils import visualizacion as vis
from voleyar.utils.io import guardar_modelo_camara

TEORIA = r"""
**Modelo pinhole.** Un punto de la mesa $P_w$ pasa a la cámara con los **extrínsecos** $P_c = R P_w + t$
y se proyecta con los **intrínsecos**:

$$ s\begin{bmatrix}u\\v\\1\end{bmatrix} = K\begin{bmatrix}x_c\\y_c\\z_c\end{bmatrix},\qquad
K=\begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{bmatrix} $$

**Distorsión** (sobre $x=x_c/z_c$, $y=y_c/z_c$, $r^2=x^2+y^2$):
radial $x(1+k_1r^2+k_2r^4+k_3r^6)$ y tangencial $2p_1xy + p_2(r^2+2x^2)$.

**Calibración de Zhang.** Cada vista del tablero da una homografía plano→imagen; con varias vistas
**inclinadas** se despeja $K$ y luego se refina todo minimizando el error de reproyección
$e_{RMS}=\sqrt{\frac1N\sum\lVert p_i-\hat p_i\rVert^2}$. Un buen resultado está por debajo de 0.5 px.

**Pose de la cancha.** Las 16 esquinas de los ArUco tienen coordenadas conocidas en cm; `solvePnP` encuentra
$R, t$. Sin calibrar ($f=$ ancho, centro en la mitad, sin distorsión) la pose compensa el error con una
inclinación falsa: lo plano encaja más o menos, pero **lo que tiene altura (la red, el balón) flota**.
"""


# --------------------------------------------------------------------------- #
# Captura guiada
# --------------------------------------------------------------------------- #


def nuevo_gestor(tam=(1280, 720), objetivo: int = 15) -> cal.GestorCapturaCalibracion:
    return cal.GestorCapturaCalibracion(tam, objetivo)


def estado_md(g: cal.GestorCapturaCalibracion | None, detectado: bool | None = None, extra: str = "") -> str:
    if g is None:
        return "Muestra el tablero a la cámara."
    linea = ""
    if detectado is not None:
        linea = ('<span class="dato-ok">● Tablero detectado</span>' if detectado
                 else '<span class="dato-mal">● Buscando el tablero…</span>')
    barra = "▰" * min(len(g.vistas), g.objetivo) + "▱" * max(0, g.objetivo - len(g.vistas))
    return (f"{linea}\n\n**Vistas:** {len(g.vistas)} / {g.objetivo} &nbsp; {barra}\n\n"
            f"**Siguiente:** {g.sugerencia()}" + (f"\n\n{extra}" if extra else ""))


def procesar_vista_tablero(img_rgb, gestor, auto: bool):
    """Detecta el tablero en el frame en vivo y, en modo automático, guarda vistas nuevas."""
    if img_rgb is None:
        return gr.skip(), gestor, gr.skip(), None
    img = vis.rgb_a_bgr(img_rgb)
    h, w = img.shape[:2]
    if gestor is None or gestor.tam_imagen != (w, h):
        gestor = nuevo_gestor((w, h))
    ok, pts = cal.detectar_tablero(img, rapido=True)
    vista = img.copy()
    extra = ""
    if ok:
        cv2.drawChessboardCorners(vista, cfg.TABLERO_ESQUINAS, pts, True)
        if auto and gestor.agregar(pts):
            extra = f"Vista {len(gestor.vistas)} capturada automáticamente."
    for v in gestor.vistas:
        for p in v.reshape(-1, 2)[::6]:
            cv2.circle(vista, tuple(np.int32(p)), 3, vis.BGR_NARANJA, -1)
    return rgb(vista), gestor, estado_md(gestor, ok, extra), (img, pts if ok else None)


def capturar(gestor, ultimo):
    if gestor is None or ultimo is None or ultimo[1] is None:
        return gestor, estado_md(gestor, False, "No hay tablero detectado en el último frame.")
    nuevo = gestor.agregar(ultimo[1])
    return gestor, estado_md(gestor, True, "Vista capturada." if nuevo else
                             "Esa vista se parece mucho a otra: inclina o mueve el tablero.")


def simular_capturas(objetivo: int = 15):
    """Llena el gestor con vistas sintéticas de un tablero (cámara con K y distorsión conocidas)."""
    modelo = sin.camara_ejemplo(1280, 720)
    tex, ppcm, origen = sin.textura_tablero(30.0)
    rng = np.random.default_rng(7)
    g = nuevo_gestor((1280, 720), objetivo)
    centro = np.array([10.0, 6.25, 0.0])
    ultima = None
    while len(g.vistas) < objetivo:
        az, elev, d = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(45, 80)), rng.uniform(32, 46)
        C = centro + d * np.array([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)])
        pose = sin.pose_mirando(C, centro + np.array([rng.uniform(-6, 6), rng.uniform(-4, 4), 0]),
                                rodar=rng.uniform(-20, 20))
        img = sin.ruido_camara(sin.renderizar_plano(tex, ppcm, origen, modelo, pose), 2.0, len(g.vistas))
        ok, pts = cal.detectar_tablero(img)
        if ok:
            g.agregar(pts, forzar=True)
            ultima = img
            cv2.drawChessboardCorners(ultima, cfg.TABLERO_ESQUINAS, pts, True)
    texto = ("Se simularon 15 vistas de un tablero con una cámara **conocida**: "
             f"$f_x$ = {modelo.K[0, 0]:.1f}, $c_x$ = {modelo.K[0, 2]:.1f}, $k_1$ = {modelo.dist[0]:.3f}. "
             "Pulsa **Calibrar** y compara.")
    return rgb(ultima), g, estado_md(g, True, texto)


def calibrar(gestor, cuadro_cm: float):
    """Ejecuta ``cv2.calibrateCamera`` y prepara las salidas (no guarda)."""
    if gestor is None or len(gestor.vistas) < 6:
        return None, "Se necesitan al menos 6 vistas (ideal 15).", None, None, None
    res = cal.calibrar_camara(gestor.vistas, gestor.tam_imagen, tam_cuadro=cuadro_cm)
    m = res.modelo
    K = m.K
    md = (f"**Error de reproyección RMS:** {m.error_rms:.3f} px con {m.n_vistas} vistas "
          + ('<span class="dato-ok">(bueno)</span>' if m.error_rms < 0.5 else
             '<span class="dato-mal">(alto: revisa que el tablero esté plano y bien enfocado)</span>') + "\n\n"
          + tabla_md([["$f_x$", f"{K[0, 0]:.2f}"], ["$f_y$", f"{K[1, 1]:.2f}"], ["$c_x$", f"{K[0, 2]:.2f}"],
                      ["$c_y$", f"{K[1, 2]:.2f}"], ["$k_1$", f"{m.dist[0]:.4f}"], ["$k_2$", f"{m.dist[1]:.4f}"],
                      ["$p_1$", f"{m.dist[2]:.5f}"], ["$p_2$", f"{m.dist[3]:.5f}"],
                      ["$k_3$", f"{m.dist[4]:.4f}" if len(m.dist) > 4 else "—"],
                      ["HFOV", f"{np.degrees(2 * np.arctan(m.ancho / (2 * K[0, 0]))):.1f}°"]],
                     ["Parámetro", "Valor"]))
    grafica = vis.grafica_barras(res.errores_por_vista, [str(i + 1) for i in range(len(res.errores_por_vista))],
                                 "Error de reproyección por vista", "px", linea=m.error_rms)
    # Cuadrícula de prueba: original (curvada) vs corregida (recta)
    w, h = gestor.tam_imagen
    rejilla = np.full((h, w, 3), 245, np.uint8)
    for x in range(0, w, 40):
        cv2.line(rejilla, (x, 0), (x, h), (60, 60, 60), 1)
    for y in range(0, h, 40):
        cv2.line(rejilla, (0, y), (w, y), (60, 60, 60), 1)
    tex = rejilla
    distorsionada = sin.renderizar_plano(tex, 1.0, (w / 2, h / 2), m,
                                         cal.Pose(np.zeros(3), np.array([0, 0, m.K[0, 0]])))
    corregida, _ = cal.corregir_distorsion(distorsionada, m, alpha=1.0)
    comp = vis.mosaico([distorsionada, corregida], ["Rejilla vista con esta lente", "Corregida con K y dist"], 2, 520)
    return res, md, grafica, rgb(comp), m.a_dict()


def guardar(res):
    if res is None:
        return "Primero calibra."
    ruta = guardar_modelo_camara(res.modelo)
    ESTADO.recargar_camara()
    return f"Guardado en `{ruta.relative_to(cfg.RAIZ_PROYECTO)}`. El juego y las demás pestañas ya lo usan."


# --------------------------------------------------------------------------- #
# Con / sin calibración sobre la cancha
# --------------------------------------------------------------------------- #


def comparar_cancha(fr: Frame, modo: str, altura_balon: float):
    """Render AR sobre la imagen original con la K calibrada y/o la aproximada."""
    img = fr.img
    h, w = img.shape[:2]
    marcadores = cal.detectar_marcadores(img)
    modelo_cal = ESTADO.modelo_para(fr)
    modos = cal.comparar_modos(marcadores, modelo_cal, w, h)
    out = img.copy()
    balon = np.array([-5.0, 3.0, altura_balon])
    filas = []
    for clave, nombre in (("calibrado", "Con calibración"), ("sin_calibrar", "Sin calibración")):
        if clave not in modos:
            filas.append([nombre, "—", "—", "no hay config/camara.json"])
            continue
        pose, m = modos[clave]
        if pose is None:
            filas.append([nombre, "—", "—", "no se ven marcadores"])
            continue
        filas.append([nombre, f"{pose.error_rms:.2f}", str(len(pose.ids)),
                      f"{np.linalg.norm(pose.tvec):.1f} cm"])
        dibujar = (modo == "Ambos (superpuestos)") or (modo == nombre)
        if not dibujar:
            continue
        if modo == "Ambos (superpuestos)":
            color = vis.BGR_NARANJA if clave == "calibrado" else vis.BGR_NEUTRO
            ra.dibujar_lineas_cancha(out, pose, m, color, 2)
            pts = np.array([[0, -cfg.MEDIO_ANCHO, cfg.ALTURA_RED], [0, cfg.MEDIO_ANCHO, cfg.ALTURA_RED]])
            p = cal.proyectar(pts, pose, m)
            cv2.line(out, tuple(np.int32(p[0])), tuple(np.int32(p[1])), color, 3, cv2.LINE_AA)
            c = cal.proyectar(balon[None], pose, m)[0]
            cv2.circle(out, tuple(np.int32(c)), int(ra.radio_en_pixeles(balon, pose, m)), color, 2, cv2.LINE_AA)
        else:
            out = cal.dibujar_ejes(out, pose, m, 6.0, 3)
            out = ra.dibujar_escena(out, pose, m, balon, lineas=True)
    if not marcadores:
        vis.poner_texto(out, "No se detectan los marcadores ArUco", (20, 20), 26, vis.BGR_NARANJA,
                        fondo=vis.BGR_GRAFITO)
    origen = "verdadero de la escena sintética" if fr.modelo is not None else (
        "config/camara.json" if ESTADO.modelo_camara is not None else "ninguno")
    md = (f"Modelo calibrado: **{origen}**. En modo *Ambos*: naranja = con calibración, blanco = sin calibración "
          "(contorno, borde superior de la red y balón).\n\n"
          + tabla_md(filas, ["Modo", "Error reproyección ArUco (px)", "Marcadores", "Distancia a la cancha"]))
    return rgb(out), md


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Calibración de la cámara y pose de la cancha")
    with gr.Tabs():
        with gr.Tab("Captura guiada del tablero"):
            gestor = gr.State(None)
            ultimo = gr.State(None)
            resultado = gr.State(None)
            with gr.Row():
                with gr.Column(scale=5):
                    vivo = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Cámara web",
                                    webcam_options=gr.WebcamOptions(mirror=False, constraints={
                                        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}}), height=320)
                    salida = gr.Image(label="Detección del tablero", height=320)
                with gr.Column(scale=4):
                    estado = gr.Markdown(estado_md(None), latex_delimiters=LATEX)
                    auto = gr.Checkbox(label="Capturar vistas nuevas automáticamente", value=True)
                    cuadro = gr.Number(value=cfg.TABLERO_CUADRO_CM, label="Lado del cuadro medido con regla (cm)",
                                       precision=3)
                    with gr.Row():
                        b_cap = gr.Button("Capturar vista")
                        b_rei = gr.Button("Reiniciar")
                    b_sim = gr.Button("Simular 15 vistas sintéticas (sin cámara)")
                    with gr.Row():
                        b_cal = gr.Button("Calibrar", variant="primary")
                        b_gua = gr.Button("Guardar en config/camara.json")
                    guardado = gr.Markdown()
            with gr.Row():
                resumen = gr.Markdown(latex_delimiters=LATEX)
                grafica = gr.Image(label="Error por vista", height=280)
            correccion = gr.Image(label="Efecto de la distorsión y su corrección", height=300)
            json_out = gr.JSON(label="Contenido de camara.json", visible=True)
            vivo.stream(procesar_vista_tablero, [vivo, gestor, auto], [salida, gestor, estado, ultimo],
                        stream_every=0.25, concurrency_limit=1, trigger_mode="always_last", show_progress="hidden")
            b_cap.click(capturar, [gestor, ultimo], [gestor, estado])
            b_rei.click(lambda: (None, estado_md(None)), None, [gestor, estado])
            b_sim.click(simular_capturas, None, [salida, gestor, estado])
            b_cal.click(calibrar, [gestor, cuadro], [resultado, resumen, grafica, correccion, json_out])
            b_gua.click(guardar, resultado, guardado)
            gr.Markdown("También puedes calibrar con `python scripts/calibrar_camara.py`: usa la misma captura de "
                        "OpenCV que el juego, así la resolución coincide.")
        with gr.Tab("Cancha: con y sin calibración"):
            with gr.Row():
                with gr.Column(scale=4):
                    entrada = bloque_entrada()
                    modo = gr.Radio(["Con calibración", "Sin calibración", "Ambos (superpuestos)"],
                                    value="Ambos (superpuestos)", label="Modo")
                    altura = gr.Slider(1.2, 15, value=6.0, step=0.2, label="Altura del balón (cm)")
                with gr.Column(scale=6):
                    render = gr.Image(label="Realidad aumentada sobre la cancha", height=430)
                    tabla = gr.Markdown(latex_delimiters=LATEX)
            conectar(entrada, comparar_cancha, [modo, altura], [render, tabla])
            cargar_al_inicio(demo, entrada, comparar_cancha, [modo, altura], [render, tabla])
    teoria(TEORIA)
