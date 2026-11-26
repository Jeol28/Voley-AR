"""Aplicación web: el juego (menú, ajustes, calibración y partida con sus vistas por tema)."""

from __future__ import annotations

import gradio as gr

from voleyar.entorno import EN_SPACE
from voleyar.ui import juego_web
from voleyar.ui.comunes import nueva_sesion_componente
from voleyar.ui.tema import CSS, crear_tema


def crear_app() -> gr.Blocks:
    with gr.Blocks(title="Voley AR") as demo:
        gr.Navbar(visible=False)
        nueva_sesion_componente()
        juego_web.construir(demo)
    return demo


def medir_rendimiento(frames: int = 40) -> None:
    """Mide en el log cuánto tarda este servidor en procesar un frame del juego (escena sintética)."""
    import os
    import time

    from voleyar.core import pipeline as pl
    from voleyar.juego.navegador import crear_juego_navegador, paso_navegador
    from voleyar.utils import sinteticos as s

    import cv2

    for w, h, hilos in ((640, 360, cv2.getNumThreads()), (640, 360, 4), (640, 360, 2), (640, 360, 1)):
        cv2.setNumThreads(hilos)
        m = s.camara_ejemplo(w, h)
        f = s.FuentePantalla(m)
        j = crear_juego_navegador(m, "2j")
        j.config.ancho_proceso = w
        imgs = [f.leer()[1] for _ in range(4)]
        paso_navegador(j, imgs[0])
        for lado in j.lados:
            j.jugadores[lado].piel = pl.calibrar_piel_en_frame(
                imgs[0], pl.recuadro_piel(j.ultimo_res.imagen.shape, lado), m, j.config)
        j.fase = "jugando"
        t = time.perf_counter()
        for i in range(frames):
            paso_navegador(j, imgs[i % 4])
        ms = (time.perf_counter() - t) / frames * 1000
        print(f"[rendimiento] servidor ({os.cpu_count()} CPU, OpenCV con {hilos} hilos): {w}x{h} → {ms:.0f} ms "
              "por frame del juego", flush=True)
    cv2.setNumThreads(int(os.environ.get("VOLEYAR_HILOS_CV", "4")))


def lanzar(puerto: int = 7860, abrir: bool = True, compartir: bool = False) -> None:
    demo = crear_app()
    if EN_SPACE:
        # ZeroGPU: los modelos se cargan una vez al arrancar (fuera de @spaces.GPU)
        from voleyar.core.transformers_models import precargar_modelos

        precargar_modelos()
        medir_rendimiento()
        demo.queue(default_concurrency_limit=4)
        demo.launch(theme=crear_tema(), css=CSS, footer_links=[])  # host y puerto los pone el Space
        return
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=puerto, inbrowser=abrir, share=compartir, theme=crear_tema(),
                css=CSS, footer_links=[])
