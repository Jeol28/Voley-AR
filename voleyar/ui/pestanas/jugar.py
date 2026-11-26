"""Pestaña Jugar: instrucciones y lanzamiento del juego en su propia ventana."""

from __future__ import annotations

import subprocess
import sys

import gradio as gr
import numpy as np

from voleyar.core import config as cfg
from voleyar.entorno import EN_SPACE
from voleyar.ui.comunes import CONCURRENCIA_VIVO, sesion_componente
from voleyar.ui.estado import ESTADO, con_sesion
from voleyar.utils import visualizacion as vis

INSTRUCCIONES = """
### Antes de jugar
1. Calibra la cámara una vez (pestaña *Calibración* o `python scripts/calibrar_camara.py`). No hace falta imprimir
   ninguna cancha: la cancha, la red y el balón se dibujan en la pantalla.
2. Pon la cámara frente a ustedes. **Jugador 1 a la izquierda, jugador 2 a la derecha** (o tú solo a la izquierda
   contra la IA), a unos 50-75 cm, con buena luz y, si se puede, mangas largas de color oscuro.

### En el juego
- Aparece un **recuadro naranja** en la mitad de cada jugador: cúbranlo con la palma, **a la distancia a la que van a
  jugar**, y presionen **ESPACIO**. Se calibra la piel de cada uno por separado.
- Durante los primeros segundos se mide a qué distancia de la cámara está cada mano (por el tamaño de la palma y la
  `K` calibrada): ese es su **plano de juego**. Un golpe solo vale si la mano está en ese plano.
- **Mano abierta = remate**, **un dedo = toque suave**, 2-3 dedos = golpe, puño = bloqueo. **No crucen la red**: cada
  mano solo se busca en su mitad de la imagen.
- Se permite **un bote**. Gana quien llegue a **15**.

| Tecla | Acción |
|---|---|
| ESPACIO | capturar la piel de cada jugador |
| D | panel de depuración (máscaras, caras, contorno, casco, punta, distancias, FPS) |
| C | con calibración / sin calibración |
| O | detector clásico / OWLv2 |
| P | recalibrar la piel |
| R | reiniciar la partida |
| Q / ESC | salir |

En **modo sintético** (sin cámara) el mouse mueve la mano del jugador 1, 0-5 cambia sus dedos y W/S la acerca o la
aleja; con 2 jugadores, I/J/K/L mueve la mano del jugador 2 y N cambia sus dedos.
"""

_proceso: subprocess.Popen | None = None


def lanzar(modo: str, sintetico: bool, camara: int, dificultad: str, profundidad: bool, sin_calibrar: bool) -> str:
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        return f"El juego ya está abierto (PID {_proceso.pid}). Ciérralo con Q antes de lanzar otro."
    args = [sys.executable, str(cfg.RAIZ_PROYECTO / "jugar.py"), "--modo", modo, "--dificultad", dificultad,
            "--camara", str(int(camara))]
    if sintetico:
        args.append("--sintetico")
    if not profundidad:
        args.append("--sin-profundidad")
    if sin_calibrar:
        args.append("--sin-calibrar")
    _proceso = subprocess.Popen(args, cwd=str(cfg.RAIZ_PROYECTO))
    aviso = "" if ESTADO.modelo_camara is not None or sintetico else (
        "\n\n**Aviso:** no hay `config/camara.json`; el juego arranca sin calibración.")
    return f"Juego lanzado en su ventana (PID {_proceso.pid}): `{' '.join(args[1:])}`{aviso}"


# --------------------------------------------------------------------------- #
# Juego en el navegador (funciona igual en local y en un Space)
# --------------------------------------------------------------------------- #

CAMARA_JUEGO = gr.WebcamOptions(mirror=False, constraints={"video": {"width": {"ideal": 640},
                                                                   "height": {"ideal": 360}}})


def _sincronizar(juego, modo: str = "ia"):
    """Crea el juego de la sesión (o lo recrea si cambió el modo) y le pasa la calibración de la sesión."""
    from voleyar.juego.navegador import crear_juego_navegador

    if juego is None or juego.op.modo != modo:
        return crear_juego_navegador(ESTADO.modelo_camara, modo)
    if juego.modelo_camara is None and ESTADO.modelo_camara is not None:
        juego.modelo_camara = ESTADO.modelo_camara
        juego.config.usar_calibracion = True
    return juego


def jugar_frame(img_rgb, juego, modo: str = "ia"):
    from voleyar.juego.navegador import paso_navegador

    if img_rgb is None:
        return gr.skip(), juego
    juego = _sincronizar(juego, modo)
    out = paso_navegador(juego, vis.rgb_a_bgr(np.asarray(img_rgb)))
    return (gr.skip() if out is None else vis.bgr_a_rgb(out)), juego


def tecla(letra: str):
    def accion(juego, modo: str = "ia"):
        juego = _sincronizar(juego, modo)
        juego.tecla(ord(letra))
        return juego

    accion.__name__ = f"tecla_{letra.strip() or 'espacio'}"
    return accion


INSTRUCCIONES_NAVEGADOR = """
1. Calibra tu cámara en la pestaña **Calibración** (en esta sesión). No hace falta imprimir nada.
2. Elige el modo. Activa la cámara y pulsa grabar. Cada jugador pone la palma sobre **su recuadro naranja** (jugador 1
   a la izquierda, jugador 2 a la derecha) y se pulsa **Capturar piel**.
3. Golpeen el balón con la mano sin cruzar la red. El juego corre en el servidor: según la conexión irá a 5-15 FPS.
   La versión de escritorio (`python jugar.py`) es más fluida y usa Depth Anything como verificación extra.
"""


def construir(demo: gr.Blocks) -> None:
    gr.Markdown("## Jugar")
    with gr.Tabs():
        with gr.Tab("En el navegador"):
            S = sesion_componente()
            juego = gr.State(None)
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown(INSTRUCCIONES_NAVEGADOR)
                    modo = gr.Radio([("Tú contra la IA", "ia"), ("2 jugadores", "2j")], value="2j", label="Modo")
                    camara = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Tu cámara",
                                      webcam_options=CAMARA_JUEGO, height=240)
                    with gr.Row():
                        b_piel = gr.Button("Capturar piel", variant="primary")
                        b_rep = gr.Button("Recalibrar piel")
                    with gr.Row():
                        b_dep = gr.Button("Depuración")
                        b_cal = gr.Button("Con / sin calibración")
                        b_rei = gr.Button("Reiniciar")
                pantalla = gr.Image(label="Voley AR", height=560, scale=7)
            camara.stream(con_sesion(jugar_frame), [S, camara, juego, modo], [S, pantalla, juego], stream_every=0.1,
                          concurrency_limit=CONCURRENCIA_VIVO, trigger_mode="always_last", show_progress="hidden")
            for b, letra in ((b_piel, " "), (b_rep, "p"), (b_dep, "d"), (b_cal, "c"), (b_rei, "r")):
                b.click(con_sesion(tecla(letra)), [S, juego, modo], [S, juego])
        if not EN_SPACE:
            with gr.Tab("En una ventana (escritorio)"):
                construir_ventana()


def construir_ventana() -> None:
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown(INSTRUCCIONES)
        with gr.Column(scale=4):
            modo_v = gr.Radio([("Tú contra la IA", "ia"), ("2 jugadores", "2j")], value="2j", label="Modo")
            sintetico = gr.Checkbox(label="Modo sintético (sin cámara, la mano sigue al mouse)", value=False)
            camara = gr.Number(value=0, precision=0, label="Índice de la cámara")
            dificultad = gr.Radio([("fácil", "facil"), ("normal", "normal"), ("difícil", "dificil")], value="normal",
                                  label="Dificultad de la IA")
            profundidad = gr.Checkbox(label="Usar Depth Anything (mano delante del cuerpo)", value=True)
            sin_cal = gr.Checkbox(label="Empezar sin calibración (para comparar)", value=False)
            boton = gr.Button("Lanzar el juego", variant="primary")
            estado = gr.Markdown()
            boton.click(lanzar, [modo_v, sintetico, camara, dificultad, profundidad, sin_cal], estado)
