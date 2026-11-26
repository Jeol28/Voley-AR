"""Interfaz web del juego (local y Space): pantallas como en un juego, sin pestañas.

Inicio → (Cómo se juega | Ajustes | Calibrar cámara) → Partida.

La cámara viaja como **video por WebRTC** (fastrtc), no como fotos: el navegador
transmite un video continuo y comprimido, el servidor procesa cada frame con el
mismo ``Juego.paso`` de la versión de escritorio y devuelve el video del juego
por el mismo canal. Si el servidor se atrasa, se descartan los frames viejos
(``skip_frames``) para que el retraso no se acumule.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import cv2
import gradio as gr
import numpy as np
from fastrtc import VideoStreamHandler, WebRTC

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.entorno import EN_SPACE
from voleyar.ui import calibracion_web as cal_ui
from voleyar.ui import sesiones
from voleyar.ui.estado import soltar_sesion, usar_sesion
from voleyar.utils import visualizacion as vis

PANTALLAS = ("inicio", "como", "ajustes", "calibrar", "juego")
CONCURRENCIA_VIDEO = 8 if EN_SPACE else 2
VIDEO_JUEGO = {"width": {"ideal": 640}, "height": {"ideal": 360}, "frameRate": {"ideal": 30}}
VIDEO_CALIBRAR = {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 15}}
ETIQUETAS = {"start": "Empezar", "stop": "Detener", "waiting": "Conectando…"}


def _opciones_vista():
    from voleyar.juego.vistas import TITULOS, VISTAS

    from voleyar.juego.vistas import APODOS

    return [("Solo el juego" if v == "juego" else f"{TITULOS[v]}: {APODOS[v]}", v) for v in VISTAS]


OPCIONES_VISTA = _opciones_vista()

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
2. Pulsen **Empezar** en el video y a jugar: no hay nada que calibrar. Muevan la mano (o un cuaderno, o lo que
   quieran) sobre el balón para devolverlo. Lo que se queda quieto es parte del cuarto y no golpea. El juego aprende
   solo el color de su piel en cuanto reconoce sus manos.

### Golpes
| Qué toca el balón | Golpe |
|---|---|
| 🖐 La mano avanzando rápido hacia la cámara | **¡Remate!**: bajo, rápido y al fondo |
| ✊ Puño | **Golpe fuerte**: largo |
| ✌ Dos o tres dedos, o un objeto pequeño | **Golpe** normal |
| 🖐 Mano abierta | **Toque suave**: alto y corto |
| 📓 Un objeto grande y plano (un cuaderno) | **Bloqueo**: rebota corto |

La cara no golpea, y lo que pasa lejos, por detrás, tampoco.

### Gestos (mano quieta medio segundo)
| Gesto | Qué hace |
|---|---|
| ✋ Mano abierta | **Sacar** |
| 👆 Pulgar e índice, como una L | **Pausa** y seguir |
| 👍 Un dedo hacia arriba | **Revancha** al terminar |
| ✌ Dos dedos juntos | **Cambiar** lo que ves |

Alrededor de tu mano aparece un círculo que se va llenando: cuando se completa, el gesto funciona.

### Reglas
- Cada lado devuelve el balón con **un toque** y se permite **un bote**.
- **No crucen la red**: cada jugador golpea el balón en su mitad de la imagen.
- Gana quien llegue primero a los puntos elegidos en Ajustes.

### Para que funcione mejor
Calibra la cámara una vez con el tablero (botón **Calibrar cámara** del inicio): así el juego mide en centímetros
reales a qué distancia está cada mano y a qué velocidad la acercas, que es lo que decide el remate.
"""

VER_TEMAS = """
### ¿Cómo te ve el juego?
Mientras juegas, **¿Qué quieres ver?** te muestra cómo te ve el juego. Cada tema hace algo:

- **Calibración:** mide qué tan cerca está tu mano. ¡Acércala rápido para rematar!
- **Filtros:** sigue puntitos de tu mano para saber qué tan rápido va. ¡Golpea rápido y sale más fuerte!
- **Histogramas:** mira los colores de tu cuarto y pinta el balón del que más resalta.
- **Umbralización:** decide qué se movió. Lo blanco se está moviendo.
- **Morfología:** agranda tu zona de golpe. ¡Más grande es más fácil!
- **Segmentación:** pinta de un color cada cosa que se mueve. Tu cara no cuenta.
- **Caracterización:** cuenta tus dedos: con gestos sacas, pausas o pides revancha. Y la forma decide el golpe.
- **Transformers:** una inteligencia artificial ve qué está cerca. Lo de atrás no golpea.

Con **−** y **+** cambias cómo funciona y lo notas al jugar.
"""

CALIBRAR_INTRO = """
Calibrar la cámara le enseña al juego cómo es **tu** lente, para que la cancha virtual y tus manos encajen y las
distancias salgan en centímetros reales. Se hace una sola vez.

1. Descarga e imprime el tablero **al 100 %** (sin ajustar a la página) y pégalo sobre un cartón.
2. Pulsa **Empezar** en el video y muéstrale el tablero desde distintos ángulos: cerca, lejos, inclinado, en las
   esquinas. Las vistas se guardan solas.
3. Cuando la barra se llene, pulsa **Calibrar**.
"""




# --------------------------------------------------------------------------- #
# WebRTC: servidor TURN en Spaces
# --------------------------------------------------------------------------- #


STUN = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}


def _turn_cloudflare(ttl: int) -> dict | None:
    """Relé TURN de Cloudflare con los secretos ``CLOUDFLARE_TURN_KEY_ID`` y ``CLOUDFLARE_TURN_KEY_API_TOKEN``."""
    key_id, token = os.environ.get("CLOUDFLARE_TURN_KEY_ID"), os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
    if not (key_id and token):
        return None
    try:
        import httpx

        r = httpx.post(f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
                       headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                       json={"ttl": ttl}, timeout=10)
        r.raise_for_status()
        datos = r.json()
        return datos if datos.get("iceServers") else None
    except Exception as e:
        print(f"[webrtc] sin TURN de Cloudflare ({type(e).__name__}: {e})", flush=True)
        return None


def _turn_hf(ttl: int) -> dict | None:
    """Relé TURN de Hugging Face vía fastrtc (secreto ``HF_TOKEN``); ``None`` si no se puede obtener."""
    if not os.environ.get("HF_TOKEN"):
        return None
    try:
        from fastrtc import get_hf_turn_credentials

        cred = get_hf_turn_credentials(ttl=ttl)
        return cred if isinstance(cred, dict) and cred.get("iceServers") else None
    except Exception as e:  # p. ej. certificado vencido en el servidor de credenciales
        print(f"[webrtc] sin TURN de Hugging Face ({type(e).__name__}: {e})", flush=True)
        return None


def _turn(ttl: int) -> dict | None:
    if not EN_SPACE:
        return None
    return _turn_cloudflare(ttl) or _turn_hf(ttl)


def configuracion_rtc():
    """Configuración ICE del navegador.

    En un Space hace falta un relé TURN: primero Cloudflare (secretos ``CLOUDFLARE_TURN_KEY_ID`` y
    ``CLOUDFLARE_TURN_KEY_API_TOKEN``), luego Hugging Face (``HF_TOKEN``). Sin ninguno se intenta STUN,
    que solo funciona si la red permite una conexión directa (en los Spaces normalmente no).
    En local la conexión es directa y no hace falta nada.
    """
    if not EN_SPACE:
        return None

    async def credenciales():
        return _turn(ttl=600) or STUN

    return credenciales


def configuracion_rtc_servidor():
    if not EN_SPACE:
        return None
    return _turn(ttl=360_000) or STUN


# --------------------------------------------------------------------------- #
# Navegación
# --------------------------------------------------------------------------- #


def ir(destino: str):
    return tuple(gr.Column(visible=(p == destino)) for p in PANTALLAS)


@contextmanager
def en_sesion(clave: str | None):
    """Activa el estado de la sesión (calibración) para el código que usa ``ESTADO``."""
    s = sesiones.obtener(clave)
    _, token = usar_sesion(s.estado)
    try:
        yield s
    finally:
        soltar_sesion(token)


def chip_camara(clave: str | None) -> str:
    if sesiones.obtener(clave).estado.modelo_camara is not None:
        return '<p class="chip ok">● Cámara calibrada</p>'
    return '<p class="chip">● Cámara sin calibrar: el juego funciona, pero las distancias son aproximadas</p>'


def abrir_pagina(clave: str | None):
    clave = clave or sesiones.nueva_clave()
    return clave, chip_camara(clave)


# --------------------------------------------------------------------------- #
# Partida
# --------------------------------------------------------------------------- #


def _preparar(s: sesiones.Sesion) -> None:
    """Crea el juego de la sesión si no existe y le pasa la calibración y los ajustes."""
    from voleyar.juego.navegador import crear_juego_navegador

    a = s.ajustes
    if s.juego is None:
        s.juego = crear_juego_navegador(s.estado.modelo_camara, s.modo, a["dificultad"])
        s.juego.op.puntos = int(a["puntos"])
        s.juego.nueva_partida()
    j = s.juego
    if j.modelo_camara is None and s.estado.modelo_camara is not None:
        j.modelo_camara = s.estado.modelo_camara
    j.config.usar_calibracion = bool(a["usar_calibracion"]) and j.modelo_camara is not None
    if j.config.espejo != bool(a.get("espejo", True)):
        j.config.espejo = bool(a.get("espejo", True))
        j.estado_pipe = type(j.estado_pipe)()


def empezar(clave: str, modo: str):
    """Botón de modo en el inicio: partida nueva (la piel ya aprendida se conserva)."""
    s = sesiones.obtener(clave)
    with s.cerrojo:
        s.modo = modo
        if s.juego is None:
            _preparar(s)
        else:
            a = s.ajustes
            s.juego.nueva_partida(modo, a["dificultad"], int(a["puntos"]))
            _preparar(s)
    return ir("juego")


def frame_juego(frame_bgr: np.ndarray, clave: str | None):
    """Un frame del video de la cámara → un frame del video del juego (ambos BGR)."""
    from voleyar.juego.navegador import paso_navegador

    s = sesiones.obtener(clave)
    with s.cerrojo:
        _preparar(s)
        t0 = time.perf_counter()
        out = paso_navegador(s.juego, frame_bgr)
        _registrar_rendimiento(s.juego, (time.perf_counter() - t0) * 1000, frame_bgr.shape)
    return frame_bgr if out is None else out


def _registrar_rendimiento(juego, ms: float, forma) -> None:
    """Cada 300 frames deja en el log el tiempo de proceso y los FPS que llegan (para medir el Space)."""
    juego._ms_frames = getattr(juego, "_ms_frames", []) + [ms]
    if len(juego._ms_frames) >= 300:
        print(f"[rendimiento] video {forma[1]}x{forma[0]} · proceso {np.mean(juego._ms_frames):.0f} ms/frame · "
              f"{juego.fps:.1f} FPS · vista {juego.vista}", flush=True)
        juego._ms_frames = []


def accion(nombre: str):
    def manejador(clave: str):
        s = sesiones.obtener(clave)
        with s.cerrojo:
            if s.juego is not None:
                s.juego.accion(nombre)

    manejador.__name__ = f"accion_{nombre}"
    return manejador


def _texto_param(juego, vista: str) -> str:
    from voleyar.juego.vistas import QUE_HACE, parametro_disponible, texto_parametro

    if vista == "juego" or juego is None:
        return "Elige un tema en **¿Qué quieres ver?** para ver cómo funciona el juego por dentro."
    if not parametro_disponible(juego, vista):
        return f"{QUE_HACE[vista]}  \n**− / +:** {texto_parametro(juego, vista)}."
    return f"{QUE_HACE[vista]}  \n**Con − / + cambias:** {texto_parametro(juego, vista)}"


def _controles_param(juego, vista: str):
    """Texto del ajuste y los botones − / + (apagados si en esta vista no hay nada que cambiar)."""
    from voleyar.juego.vistas import parametro_disponible

    activo = parametro_disponible(juego, vista)
    return _texto_param(juego, vista), gr.update(interactive=activo), gr.update(interactive=activo)


def elegir_vista(clave: str, vista: str):
    s = sesiones.obtener(clave)
    with s.cerrojo:
        if s.juego is not None:
            s.juego.vista = vista
        return _controles_param(s.juego, vista)


def cambiar(paso: int):
    def manejador(clave: str, vista: str):
        from voleyar.juego.vistas import cambiar_parametro

        s = sesiones.obtener(clave)
        with s.cerrojo:
            if s.juego is not None:
                cambiar_parametro(s.juego, vista, paso)
            return _controles_param(s.juego, vista)

    manejador.__name__ = "parametro_mas" if paso > 0 else "parametro_menos"
    return manejador


def datos_tema(clave: str):
    """Datos en vivo y gráficas del tema que se está viendo, en la columna de la izquierda (fuera de la cancha)."""
    s = sesiones.obtener(clave)
    with s.cerrojo:
        j = s.juego
        menu = gr.update()
        if j is not None and j.vista_por_gesto:  # cambió de tema con ✌: el menú lo sigue
            j.vista_por_gesto = False
            menu = gr.update(value=j.vista)
        if j is None or j.vista == "juego":
            return gr.update(value="", visible=False), gr.update(value=None, visible=False), menu
        texto = "  \n".join(j.lineas_tema[:2])
        graficas = list(getattr(j, "graficas_tema", []))
    img = None
    if graficas:  # una al lado de la otra, misma altura
        alto = 160
        partes = [cv2.resize(g, (max(1, int(g.shape[1] * alto / g.shape[0])), alto)) for g in graficas]
        img = vis.bgr_a_rgb(np.hstack(partes))
    return (gr.update(value=texto, visible=bool(texto)),
            gr.update(value=img, visible=img is not None), menu)


CARPETA_FOTOS = cfg.RAIZ_PROYECTO / "capturas_mano"


def guardar_foto_mano(clave: str, carpeta=None) -> str:
    """Guarda en tu computador el frame actual, la máscara de cada mano y lo que el juego leyó.

    Sirve para afinar el conteo de dedos con fotos reales. No se sube a ningún lado
    (la carpeta está en .gitignore y el botón no aparece en el Space).
    """
    import json
    from datetime import datetime
    from pathlib import Path

    from voleyar.juego.gestos import clasificar

    if EN_SPACE:
        return "En la versión en línea no se guardan fotos."
    s = sesiones.obtener(clave)
    with s.cerrojo:
        j = s.juego
        res = None if j is None else j.ultimo_res
        if res is None:
            return "Primero pulsa **Empezar** en el video."
        carpeta = Path(carpeta or CARPETA_FOTOS)
        carpeta.mkdir(parents=True, exist_ok=True)
        nombre = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(str(carpeta / f"{nombre}_imagen.png"), res.imagen)
        datos = {"espejo": j.config.espejo, "manos": {}}
        for lado, r in res.manos.items():
            if r.mascara_mano is not None:
                cv2.imwrite(str(carpeta / f"{nombre}_mano{lado:+d}.png"), r.mascara_mano)
            if r.componente is not None:
                cv2.imwrite(str(carpeta / f"{nombre}_piel{lado:+d}.png"), r.componente)
            a = r.analisis
            datos["manos"][str(lado)] = {
                "hay_mano": bool(r.hay_mano), "presente": r.presente,
                "motivos": list(r.validacion.motivos) if r.validacion is not None else None,
                "dedos_casco": a.n_dedos if a else None, "dedos_estables": r.dedos_estables,
                "dedos_anillo": a.n_anillo if a else None,
                "angulos_anillo": a.angulos_anillo if a else None,
                "gesto": clasificar(a, r.dedos_estables) if a else None,
                "centro_palma": list(map(float, a.centro_palma)) if a else None,
                "radio_palma": float(a.radio_palma) if a else None,
            }
        (carpeta / f"{nombre}_datos.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False))
    return f"Foto guardada en `{carpeta.name}/` (solo en tu computador)."


def guardar_ajustes(clave: str, dificultad: str, puntos: int, usar_cal: bool, espejo: bool = True) -> None:
    s = sesiones.obtener(clave)
    s.ajustes = {"dificultad": dificultad, "puntos": int(puntos), "usar_calibracion": bool(usar_cal),
                 "espejo": bool(espejo)}


def volver_al_menu(clave: str):
    s = sesiones.obtener(clave)
    with s.cerrojo:
        j = s.juego
        if j is not None and not j.pausado and j.fase == "jugando":
            j.pausado = True
    return (chip_camara(clave), *ir("inicio"))


# --------------------------------------------------------------------------- #
# Calibración de la cámara por video
# --------------------------------------------------------------------------- #


def frame_calibracion(frame_bgr: np.ndarray, clave: str | None):
    """Busca el tablero en cada frame del video y guarda solas las vistas nuevas."""
    s = sesiones.obtener(clave)
    h, w = frame_bgr.shape[:2]
    with s.cerrojo:
        if s.gestor is None or s.gestor.tam_imagen != (w, h):
            s.gestor = cal_ui.nuevo_gestor((w, h))
        ok, pts = cal.detectar_tablero(frame_bgr, rapido=True)
        vista = frame_bgr.copy()
        if ok:
            cv2.drawChessboardCorners(vista, cfg.TABLERO_ESQUINAS, pts, True)
            s.gestor.agregar(pts)
        s.ultimo_tablero = ok
        for v in s.gestor.vistas:
            for p in v.reshape(-1, 2)[::6]:
                cv2.circle(vista, tuple(np.int32(p)), 3, vis.BGR_NARANJA, -1)
        n = len(s.gestor.vistas)
    barra = int(w * min(1.0, n / s.gestor.objetivo))
    cv2.rectangle(vista, (0, h - 10), (barra, h), vis.BGR_NARANJA, -1)
    return vista


def progreso_calibracion(clave: str) -> str:
    s = sesiones.obtener(clave)
    return cal_ui.estado_md(s.gestor, s.ultimo_tablero)


def reiniciar_calibracion(clave: str) -> str:
    s = sesiones.obtener(clave)
    with s.cerrojo:
        s.gestor = None
    return cal_ui.estado_md(None)


def calibrar_amigable(clave: str, cuadro_cm: float):
    s = sesiones.obtener(clave)
    with s.cerrojo:
        res, md, grafica, comp, js = cal_ui.calibrar(s.gestor, cuadro_cm)
    s.resultado = res
    if res is None:
        return f"**{md}**", None, None
    m = res.modelo
    calidad = "excelente" if m.error_rms < 0.3 else ("buena" if m.error_rms < 0.6 else "regular (prueba con más vistas)")
    texto = (f"### Calibración {calidad}\n\nError promedio de **{m.error_rms:.2f} px** con {m.n_vistas} vistas. "
             "Pulsa **Usar esta calibración** para jugar con ella.")
    return texto, md, grafica


def usar_calibracion(clave: str):
    with en_sesion(clave) as s:
        texto, ruta = cal_ui.guardar(getattr(s, "resultado", None))
        if s.juego is not None:
            with s.cerrojo:
                s.juego.modelo_camara = s.estado.modelo_camara
    return texto, ruta, chip_camara(clave)


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #


def construir(demo: gr.Blocks) -> None:
    clave = gr.State(None)

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
        chip = gr.HTML()

    # ---------------------------------------------------------- cómo se juega
    with gr.Column(visible=False, elem_classes="pantalla") as p_como:
        gr.Markdown("## Cómo se juega")
        gr.Markdown(COMO_SE_JUEGA)
        gr.Markdown(VER_TEMAS)
        v1 = gr.Button("Volver", variant="primary", elem_classes="boton-medio")

    # ---------------------------------------------------------------- ajustes
    with gr.Column(visible=False, elem_classes="pantalla") as p_aj:
        gr.Markdown("## Ajustes")
        dificultad = gr.Radio([("Fácil", "facil"), ("Normal", "normal"), ("Difícil", "dificil")], value="normal",
                              label="Dificultad de la IA")
        puntos = gr.Radio([("7", 7), ("11", 11), ("15", 15)], value=15, label="Puntos para ganar")
        usar_cal = gr.Checkbox(label="Usar la calibración de la cámara (apágala para comparar)", value=True)
        espejo = gr.Checkbox(label="Ver la cámara como un espejo (quien está a tu izquierda sale a la izquierda)",
                             value=True)
        v2 = gr.Button("Listo", variant="primary", elem_classes="boton-medio")

    # ------------------------------------------------------- calibrar cámara
    with gr.Column(visible=False, elem_classes="pantalla") as p_cal:
        gr.Markdown("## Calibrar la cámara")
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown(CALIBRAR_INTRO)
                gr.DownloadButton("Descargar el tablero (PDF)", value=str(cfg.RUTA_IMPRIMIBLES / "tablero.pdf"))
                cuadro = gr.Number(value=cfg.TABLERO_CUADRO_CM, label="Lado de un cuadro medido con regla (cm)",
                                   precision=3)
                progreso = gr.Markdown(cal_ui.estado_md(None))
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
                video_cal = WebRTC(label="Tu cámara", mode="send-receive", modality="video", mirror_webcam=False,
                                   track_constraints=VIDEO_CALIBRAR, rtc_configuration=configuracion_rtc(),
                                   server_rtc_configuration=configuracion_rtc_servidor(), height=420,
                                   full_screen=False, button_labels=ETIQUETAS)
        reloj = gr.Timer(1.0)

    # ---------------------------------------------------------------- partida
    with gr.Column(visible=False, elem_classes="pantalla-juego") as p_juego:
        with gr.Row():
            with gr.Column(scale=3, min_width=260):
                gr.Markdown("Pulsa **Empezar** en el video y golpea el balón con la mano o con lo que quieras.",
                            elem_classes="nota")
                with gr.Row():
                    b_pausa = gr.Button("Pausa / Seguir")
                    b_reini = gr.Button("Reiniciar")
                with gr.Row():
                    b_revancha = gr.Button("Revancha")
                    b_menu = gr.Button("Menú principal")
                vista = gr.Dropdown(OPCIONES_VISTA, value="juego", label="¿Qué quieres ver?")
                with gr.Row():
                    b_menos = gr.Button("−", min_width=60, interactive=False)
                    b_mas = gr.Button("+", min_width=60, interactive=False)
                param = gr.Markdown(_texto_param(None, "juego"), elem_classes="nota")
                with gr.Row(visible=not EN_SPACE):  # solo en tu computador: las fotos no salen de él
                    b_foto = gr.Button("Guardar foto de mi mano", size="sm")
                foto_msg = gr.Markdown(elem_classes="nota")
                datos = gr.Markdown(visible=False, elem_classes="nota")
                graficas = gr.Image(show_label=False, interactive=False, visible=False, container=False,
                                    show_download_button=False, show_fullscreen_button=False)
                reloj_tema = gr.Timer(1.0)
            with gr.Column(scale=9):
                video = WebRTC(label="Voley AR", mode="send-receive", modality="video", mirror_webcam=False,
                               track_constraints=VIDEO_JUEGO, rtc_configuration=configuracion_rtc(),
                               server_rtc_configuration=configuracion_rtc_servidor(), height=620,
                               full_screen=False, button_labels=ETIQUETAS)

    pantallas = [p_inicio, p_como, p_aj, p_cal, p_juego]

    # --------------------------------------------------------------- eventos
    demo.load(abrir_pagina, clave, [clave, chip])
    b_como.click(lambda: ir("como"), None, pantallas)
    b_aj.click(lambda: ir("ajustes"), None, pantallas)
    b_cal.click(lambda: ir("calibrar"), None, pantallas)
    for b in (v1, v3):
        b.click(lambda c: (chip_camara(c), *ir("inicio")), clave, [chip, *pantallas])
    v2.click(guardar_ajustes, [clave, dificultad, puntos, usar_cal, espejo], None).then(lambda: ir("inicio"), None, pantallas)
    b_ia.click(lambda c: empezar(c, "ia"), clave, pantallas)
    b_2j.click(lambda c: empezar(c, "2j"), clave, pantallas)

    video.stream(VideoStreamHandler(frame_juego, fps=30, skip_frames=True), inputs=[video, clave], outputs=[video],
                 concurrency_limit=CONCURRENCIA_VIDEO, send_input_on="submit")
    for b, nombre in ((b_pausa, "pausa"), (b_reini, "reiniciar"), (b_revancha, "revancha")):
        b.click(accion(nombre), clave, None)
    b_menu.click(volver_al_menu, clave, [chip, *pantallas])
    vista.change(elegir_vista, [clave, vista], [param, b_menos, b_mas])
    b_menos.click(cambiar(-1), [clave, vista], [param, b_menos, b_mas])
    b_mas.click(cambiar(1), [clave, vista], [param, b_menos, b_mas])
    b_foto.click(guardar_foto_mano, clave, foto_msg)
    reloj_tema.tick(datos_tema, clave, [datos, graficas, vista], show_progress="hidden")

    video_cal.stream(VideoStreamHandler(frame_calibracion, fps=15, skip_frames=True), inputs=[video_cal, clave],
                     outputs=[video_cal], concurrency_limit=CONCURRENCIA_VIDEO, send_input_on="submit")
    reloj.tick(progreso_calibracion, clave, progreso, show_progress="hidden")
    b_reiniciar_cal.click(reiniciar_calibracion, clave, progreso)
    b_calibrar.click(calibrar_amigable, [clave, cuadro], [resumen, detalles, grafica])
    b_usar.click(usar_calibracion, clave, [guardado, archivo, chip])
