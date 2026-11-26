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
1. Imprime `docs/imprimibles/cancha.pdf` **al 100 %** y ponla sobre el escritorio con el borde **TU LADO** hacia ti.
2. Calibra la cámara una vez (pestaña *Calibración* o `python scripts/calibrar_camara.py`).
3. Ubica la cámara de modo que vea los **4 marcadores** y una franja de mesa detrás de tu borde (por donde entra el brazo).

### En el juego
- Al empezar aparece un **recuadro naranja**: cúbrelo con la palma y presiona **ESPACIO** para calibrar tu piel.
- Golpea el balón con la mano **cerca de la mesa**. **Mano abierta = remate**, **un dedo = toque suave**,
  2-3 dedos = golpe, puño = bloqueo. La dirección depende de hacia dónde muevas la mano.
- Se permite **un bote** de tu lado. Gana quien llegue a **15**.

| Tecla | Acción |
|---|---|
| ESPACIO | capturar el color de la piel |
| D | panel de depuración (máscara, contorno, casco, punta, pose, FPS) |
| C | con calibración / sin calibración |
| L | líneas virtuales de la cancha |
| O | detector clásico / OWLv2 |
| P | recalibrar la piel |
| R | reiniciar la partida |
| Q / ESC | salir |

En **modo sintético** (sin cámara) el mouse mueve la mano, las teclas 0-5 cambian los dedos y W/S la suben o bajan.
"""

_proceso: subprocess.Popen | None = None


def lanzar(sintetico: bool, camara: int, dificultad: str, profundidad: bool, sin_calibrar: bool) -> str:
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        return f"El juego ya está abierto (PID {_proceso.pid}). Ciérralo con Q antes de lanzar otro."
    args = [sys.executable, str(cfg.RAIZ_PROYECTO / "jugar.py"), "--dificultad", dificultad, "--camara", str(int(camara))]
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


def _sincronizar(juego):
    """Crea el juego de la sesión o le pasa la calibración que se hizo en la pestaña Calibración."""
    from voleyar.juego.navegador import crear_juego_navegador

    if juego is None:
        return crear_juego_navegador(ESTADO.modelo_camara)
    if juego.modelo_camara is None and ESTADO.modelo_camara is not None:
        juego.modelo_camara = ESTADO.modelo_camara
        juego.config.usar_calibracion = True
    return juego


def jugar_frame(img_rgb, juego):
    from voleyar.juego.navegador import paso_navegador

    if img_rgb is None:
        return gr.skip(), juego
    juego = _sincronizar(juego)
    out = paso_navegador(juego, vis.rgb_a_bgr(np.asarray(img_rgb)))
    return (gr.skip() if out is None else vis.bgr_a_rgb(out)), juego


def tecla(letra: str):
    def accion(juego):
        juego = _sincronizar(juego)
        juego.tecla(ord(letra))
        return juego

    accion.__name__ = f"tecla_{letra.strip() or 'espacio'}"
    return accion


INSTRUCCIONES_NAVEGADOR = """
1. Imprime la cancha y calibra tu cámara en la pestaña **Calibración** (en esta sesión).
2. Activa la cámara aquí abajo y pulsa el botón de grabar. Pon la palma sobre el **recuadro naranja** y pulsa
   **Capturar piel**.
3. Golpea el balón con la mano cerca de la mesa. El juego corre en el servidor: según tu conexión irá a 5-15 FPS.
   La versión de escritorio (`python jugar.py`) es más fluida y usa Depth Anything para validar la altura de la mano.
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
            camara.stream(con_sesion(jugar_frame), [S, camara, juego], [S, pantalla, juego], stream_every=0.1,
                          concurrency_limit=CONCURRENCIA_VIVO, trigger_mode="always_last", show_progress="hidden")
            for b, letra in ((b_piel, " "), (b_rep, "p"), (b_dep, "d"), (b_cal, "c"), (b_rei, "r")):
                b.click(con_sesion(tecla(letra)), [S, juego], [S, juego])
        if not EN_SPACE:
            with gr.Tab("En una ventana (escritorio)"):
                construir_ventana()


def construir_ventana() -> None:
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown(INSTRUCCIONES)
        with gr.Column(scale=4):
            sintetico = gr.Checkbox(label="Modo sintético (sin cámara, la mano sigue al mouse)", value=False)
            camara = gr.Number(value=0, precision=0, label="Índice de la cámara")
            dificultad = gr.Radio(["facil", "normal", "dificil"], value="normal", label="Dificultad de la IA")
            profundidad = gr.Checkbox(label="Usar Depth Anything para validar el golpe", value=True)
            sin_cal = gr.Checkbox(label="Empezar sin calibración (para comparar)", value=False)
            boton = gr.Button("Lanzar el juego", variant="primary")
            estado = gr.Markdown()
            boton.click(lanzar, [sintetico, camara, dificultad, profundidad, sin_cal], estado)
