"""Interfaz web del juego (local y Space): pantallas como en un juego, sin pestañas.

Inicio → (Cómo se juega | Ajustes | Calibrar cámara) → Partida. La partida usa
la cámara del navegador y el mismo ``Juego.paso`` que la versión de escritorio.
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from voleyar.core import config as cfg
from voleyar.entorno import EN_SPACE
from voleyar.ui.comunes import CONCURRENCIA_VIVO, sesion_componente
from voleyar.ui.estado import ESTADO, con_sesion
from voleyar.ui.pestanas import calibracion as cal_ui
from voleyar.utils import visualizacion as vis

PANTALLAS = ("inicio", "como", "ajustes", "calibrar", "juego")
AJUSTES_INICIALES = {"dificultad": "normal", "puntos": 15, "ver_deteccion": False, "usar_calibracion": True}
CAMARA_JUEGO = gr.WebcamOptions(mirror=False, constraints={"video": {"width": {"ideal": 960},
                                                                   "height": {"ideal": 540}}})
CAMARA_CALIBRAR = gr.WebcamOptions(mirror=False, constraints={"video": {"width": {"ideal": 1280},
                                                                      "height": {"ideal": 720}}})

HERO = """
<div class="hero">
  <h1>Voley AR</h1>
  <p class="lema">Voleibol con tus manos. Sin cancha, sin balón, sin nada más que la cámara.</p>
</div>
"""

COMO_SE_JUEGA = """
### Prepárense
1. Pongan la cámara frente a ustedes, a unos **50-75 cm**. **Jugador 1 a la izquierda, jugador 2 a la derecha**
   (o tú solo a la izquierda contra la IA). Buena luz y, si se puede, mangas largas oscuras.
2. Al empezar aparece un **recuadro naranja** en la mitad de cada jugador: pongan la palma encima, a la distancia a la
   que van a jugar, y pulsen **¡Listo!**. Así el juego aprende el color de su piel.

### Golpes
| Mano | Golpe |
|---|---|
| 🖐 Abierta | **Remate**: fuerte y bajo |
| ✌ Dos o tres dedos | **Golpe** normal |
| ☝ Un dedo | **Toque**: suave y alto |
| ✊ Puño | **Bloqueo** |

Hacia dónde muevas la mano decide hacia dónde sale el balón.

### Reglas
- Cada lado devuelve el balón con **un toque** y se permite **un bote**.
- **No crucen la red**: cada mano juega en su mitad de la imagen.
- El golpe vale si la mano está a la distancia a la que empezaste (el juego la mide al inicio de la partida).
- Gana quien llegue primero a los puntos elegidos en Ajustes.

### Para que funcione mejor
Calibra la cámara una vez con el tablero (botón **Calibrar cámara** del inicio): así el juego mide en centímetros
reales a qué distancia está cada mano.
"""

CALIBRAR_INTRO = """
Calibrar la cámara le enseña al juego cómo es **tu** lente, para que la cancha virtual y tus manos encajen y las
distancias salgan en centímetros reales. Se hace una sola vez.

1. Descarga e imprime el tablero **al 100 %** (sin ajustar a la página) y pégalo sobre un cartón.
2. Activa la cámara y muéstrale el tablero desde distintos ángulos: cerca, lejos, inclinado, en las esquinas.
3. Cuando la barra se llene, pulsa **Calibrar**.
"""


def ir(destino: str):
    return tuple(gr.Column(visible=(p == destino)) for p in PANTALLAS)


def chip_camara() -> str:
    if ESTADO.modelo_camara is not None:
        return '<p class="chip ok">● Cámara calibrada</p>'
    return '<p class="chip">● Cámara sin calibrar: el juego funciona, pero las distancias son aproximadas</p>'


# --------------------------------------------------------------------------- #
# Partida
# --------------------------------------------------------------------------- #


def _aplicar_ajustes(juego, ajustes: dict) -> None:
    juego.depurar = bool(ajustes.get("ver_deteccion"))
    juego.config.usar_calibracion = bool(ajustes.get("usar_calibracion")) and juego.modelo_camara is not None


def empezar(juego, modo: str, ajustes: dict):
    """Botón de modo en el inicio: crea la partida (o reinicia la existente, conservando la piel)."""
    from voleyar.juego.navegador import crear_juego_navegador

    if juego is None:
        juego = crear_juego_navegador(ESTADO.modelo_camara, modo, ajustes["dificultad"])
        juego.op.puntos = int(ajustes["puntos"])
        juego.nueva_partida()
    else:
        if juego.modelo_camara is None and ESTADO.modelo_camara is not None:
            juego.modelo_camara = ESTADO.modelo_camara
        juego.nueva_partida(modo, ajustes["dificultad"], int(ajustes["puntos"]))
    _aplicar_ajustes(juego, ajustes)
    return (juego, modo, *ir("juego"))


def jugar_frame(img_rgb, juego, modo: str, ajustes: dict):
    """Un frame de la cámara del navegador → imagen del juego."""
    from voleyar.juego.navegador import paso_navegador

    if img_rgb is None:
        return gr.skip(), juego
    if juego is None:
        juego, *_ = empezar(None, modo, ajustes)
    if juego.modelo_camara is None and ESTADO.modelo_camara is not None:
        juego.modelo_camara = ESTADO.modelo_camara
    _aplicar_ajustes(juego, ajustes)
    out = paso_navegador(juego, vis.rgb_a_bgr(np.asarray(img_rgb)))
    return (gr.skip() if out is None else vis.bgr_a_rgb(out)), juego


def accion(nombre: str):
    def manejador(juego):
        if juego is not None:
            juego.accion(nombre)
        return juego

    manejador.__name__ = f"accion_{nombre}"
    return manejador


def guardar_ajustes(dificultad: str, puntos: int, ver: bool, usar_cal: bool):
    return {"dificultad": dificultad, "puntos": int(puntos), "ver_deteccion": bool(ver), "usar_calibracion": bool(usar_cal)}


# --------------------------------------------------------------------------- #
# Calibración amigable
# --------------------------------------------------------------------------- #


def calibrar_amigable(gestor, cuadro_cm: float):
    res, md, grafica, comp, js = cal_ui.calibrar(gestor, cuadro_cm)
    if res is None:
        return None, f"**{md}**", None, None
    m = res.modelo
    calidad = "excelente" if m.error_rms < 0.3 else ("buena" if m.error_rms < 0.6 else "regular (prueba con más vistas)")
    texto = (f"### Calibración {calidad}\n\nError promedio de **{m.error_rms:.2f} px** con {m.n_vistas} vistas. "
             "Pulsa **Usar esta calibración** para jugar con ella.")
    return res, texto, md, grafica


def usar_calibracion(res):
    texto, ruta = cal_ui.guardar(res)
    return texto, ruta, chip_camara()


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #


def construir(demo: gr.Blocks) -> None:
    S = sesion_componente()
    juego = gr.State(None)
    modo = gr.State("2j")
    ajustes = gr.State(dict(AJUSTES_INICIALES))

    # ---------------------------------------------------------------- inicio
    with gr.Column(visible=True, elem_classes="pantalla") as p_inicio:
        gr.HTML(HERO)
        with gr.Row(elem_classes="fila-botones"):
            b_ia = gr.Button("Contra la IA", variant="primary", elem_classes="boton-grande")
            b_2j = gr.Button("2 jugadores", variant="primary", elem_classes="boton-grande")
        with gr.Row(elem_classes="fila-botones"):
            b_como = gr.Button("Cómo se juega", elem_classes="boton-medio")
            b_aj = gr.Button("Ajustes", elem_classes="boton-medio")
            b_cal = gr.Button("Calibrar cámara", elem_classes="boton-medio")
        chip = gr.HTML(chip_camara())

    # ---------------------------------------------------------- cómo se juega
    with gr.Column(visible=False, elem_classes="pantalla") as p_como:
        gr.Markdown("## Cómo se juega")
        gr.Markdown(COMO_SE_JUEGA)
        gr.Markdown("¿Quieres ver cómo funciona por dentro (filtros, histogramas, segmentación…)? "
                    "Abre el [laboratorio](laboratorio).", elem_classes="nota")
        v1 = gr.Button("Volver", variant="primary", elem_classes="boton-medio")

    # ---------------------------------------------------------------- ajustes
    with gr.Column(visible=False, elem_classes="pantalla") as p_aj:
        gr.Markdown("## Ajustes")
        dificultad = gr.Radio([("Fácil", "facil"), ("Normal", "normal"), ("Difícil", "dificil")], value="normal",
                              label="Dificultad de la IA")
        puntos = gr.Radio([("7", 7), ("11", 11), ("15", 15)], value=15, label="Puntos para ganar")
        ver = gr.Checkbox(label="Ver cómo te detecta el juego (contornos, dedos y distancias)", value=False)
        usar_cal = gr.Checkbox(label="Usar la calibración de la cámara (apágala para comparar)", value=True)
        v2 = gr.Button("Listo", variant="primary", elem_classes="boton-medio")

    # ------------------------------------------------------- calibrar cámara
    with gr.Column(visible=False, elem_classes="pantalla") as p_cal:
        gr.Markdown("## Calibrar la cámara")
        gestor = gr.State(None)
        ultimo = gr.State(None)
        resultado = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown(CALIBRAR_INTRO)
                gr.DownloadButton("Descargar el tablero (PDF)", value=str(cfg.RUTA_IMPRIMIBLES / "tablero.pdf"))
                cuadro = gr.Number(value=cfg.TABLERO_CUADRO_CM, label="Lado de un cuadro medido con regla (cm)",
                                   precision=3)
                progreso = gr.Markdown(cal_ui.estado_md(None), latex_delimiters=[])
                with gr.Row():
                    b_calibrar = gr.Button("Calibrar", variant="primary")
                    b_reiniciar_cal = gr.Button("Empezar de nuevo")
                resumen = gr.Markdown()
                with gr.Row():
                    b_usar = gr.Button("Usar esta calibración")
                    v3 = gr.Button("Volver")
                guardado = gr.Markdown()
                archivo = gr.File(label="camara.json", height=70, visible=EN_SPACE)
                with gr.Accordion("Detalles técnicos", open=False):
                    detalles = gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                    grafica = gr.Image(label="Error por vista", height=220)
            with gr.Column(scale=6):
                cam_cal = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Tu cámara",
                                   webcam_options=CAMARA_CALIBRAR, height=300)
                vista_cal = gr.Image(label="Tablero detectado", height=300)

    # ---------------------------------------------------------------- partida
    with gr.Column(visible=False, elem_classes="pantalla-juego") as p_juego:
        with gr.Row():
            with gr.Column(scale=3, min_width=260):
                gr.Markdown("**1.** Activa la cámara y pulsa grabar.  \n**2.** Palma sobre tu recuadro y **¡Listo!**",
                            elem_classes="nota")
                camara = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Tu cámara",
                                  webcam_options=CAMARA_JUEGO, height=200)
                b_listo = gr.Button("¡Listo!", variant="primary", elem_classes="boton-medio")
                with gr.Row():
                    b_pausa = gr.Button("Pausa / Seguir")
                    b_reini = gr.Button("Reiniciar")
                with gr.Row():
                    b_revancha = gr.Button("Revancha")
                    b_piel = gr.Button("Recalibrar piel")
                b_menu = gr.Button("Menú principal")
            pantalla = gr.Image(label="Voley AR", height=600, scale=9, show_label=False)

    pantallas = [p_inicio, p_como, p_aj, p_cal, p_juego]

    # --------------------------------------------------------------- eventos
    b_como.click(lambda: ir("como"), None, pantallas)
    b_aj.click(lambda: ir("ajustes"), None, pantallas)
    b_cal.click(lambda: ir("calibrar"), None, pantallas)
    for b in (v1, v3):
        b.click(con_sesion(lambda: (chip_camara(), *ir("inicio"))), [S], [S, chip, *pantallas])
    v2.click(guardar_ajustes, [dificultad, puntos, ver, usar_cal], ajustes).then(lambda: ir("inicio"), None, pantallas)
    b_ia.click(con_sesion(lambda j, a: empezar(j, "ia", a)), [S, juego, ajustes], [S, juego, modo, *pantallas])
    b_2j.click(con_sesion(lambda j, a: empezar(j, "2j", a)), [S, juego, ajustes], [S, juego, modo, *pantallas])

    camara.stream(con_sesion(jugar_frame), [S, camara, juego, modo, ajustes], [S, pantalla, juego], stream_every=0.1,
                  concurrency_limit=CONCURRENCIA_VIVO, trigger_mode="always_last", show_progress="hidden")
    for b, nombre in ((b_listo, "listo"), (b_pausa, "pausa"), (b_reini, "reiniciar"), (b_revancha, "revancha"),
                      (b_piel, "piel")):
        b.click(accion(nombre), juego, juego)
    b_menu.click(con_sesion(lambda j: (_pausar(j), chip_camara(), *ir("inicio"))), [S, juego],
                 [S, juego, chip, *pantallas])

    cam_cal.stream(cal_ui.procesar_vista_tablero, [cam_cal, gestor, gr.State(True)],
                   [vista_cal, gestor, progreso, ultimo], stream_every=0.25, concurrency_limit=CONCURRENCIA_VIVO,
                   trigger_mode="always_last", show_progress="hidden")
    b_reiniciar_cal.click(lambda: (None, cal_ui.estado_md(None)), None, [gestor, progreso])
    b_calibrar.click(calibrar_amigable, [gestor, cuadro], [resultado, resumen, detalles, grafica])
    b_usar.click(con_sesion(usar_calibracion), [S, resultado], [S, guardado, archivo, chip])
    demo.load(con_sesion(chip_camara), [S], [S, chip])


def _pausar(juego):
    if juego is not None and not juego.pausado and juego.fase == "jugando":
        juego.pausado = True
    return juego
