"""Calibración de la cámara desde el juego web: captura guiada del tablero, cálculo y uso en la sesión.

Usa ``core.calibracion`` (Zhang con ``cv2.calibrateCamera``); aquí solo está la parte de interfaz.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.entorno import EN_SPACE
from voleyar.ui.estado import ESTADO, rgb, tabla_md
from voleyar.utils import sinteticos as sin
from voleyar.utils import visualizacion as vis
from voleyar.utils.io import guardar_modelo_camara


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
    """Usa la calibración en esta sesión. En local además escribe ``config/camara.json``;
    en un Space la ofrece para descargar (el disco del Space es compartido y temporal)."""
    if res is None:
        return "Primero calibra.", None
    ESTADO.modelo_camara = res.modelo
    if EN_SPACE:
        ruta = Path(tempfile.mkdtemp()) / "camara.json"
        guardar_modelo_camara(res.modelo, ruta)
        return ("Calibración activa en tu sesión: las demás pestañas y el juego en el navegador ya la usan. "
                "Descarga `camara.json` y cópialo en `config/` para usarlo en tu equipo."), str(ruta)
    ruta = guardar_modelo_camara(res.modelo)
    mostrar = ruta.relative_to(cfg.RAIZ_PROYECTO) if ruta.is_relative_to(cfg.RAIZ_PROYECTO) else ruta
    return f"Guardado en `{mostrar}`. El juego ya la usa.", str(ruta)
