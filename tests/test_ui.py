"""Pruebas de la interfaz web sin navegador: la página del juego, sus pantallas, las vistas por tema y la calibración."""

import time

import numpy as np
import pytest

from voleyar.core import pipeline as pl
from voleyar.juego.vistas import PARAMETROS, VISTAS
from voleyar.ui import calibracion_web as cal_ui
from voleyar.ui import estado
from voleyar.ui import juego_web as jw
from voleyar.ui.estado import con_sesion
from voleyar.utils import sinteticos as s
from voleyar.utils import visualizacion as vis


def test_la_app_se_construye_sin_laboratorio():
    from voleyar.ui.app import crear_app

    assert crear_app() is not None
    assert [v for _, v in jw.OPCIONES_VISTA] == list(VISTAS)


CASOS = [(n, lado, d) for n, p in estado.EJEMPLOS.items() for lado, d in zip((-1, 1), p["dedos"])
         if n != "Mano delante de la cara"]  # con la mano sobre la cara los dedos se confunden con la cara


@pytest.mark.parametrize("nombre,lado,dedos", CASOS)
def test_pipeline_cuenta_bien_los_dedos_en_las_escenas_de_ejemplo(nombre, lado, dedos):
    r = estado.ejecutar_pipeline(estado.frame_ejemplo(nombre))
    m = r.manos[lado]
    assert m.analisis is not None and m.analisis.n_dedos == dedos


def test_calibracion_mejora_la_distancia_de_la_mano():
    fr = estado.frame_ejemplo("Dos jugadores (remate y toque)")
    con = estado.ejecutar_pipeline(fr, usar_calibracion=True)
    sin = estado.ejecutar_pipeline(fr, usar_calibracion=False)
    for lado, jug in zip((-1, 1), fr.verdad["jugadores"]):
        e_con = abs(con.manos[lado].distancia_cm - jug.distancia)
        e_sin = abs(sin.manos[lado].distancia_cm - jug.distancia)
        assert e_con < 0.12 * jug.distancia and e_sin > 2 * e_con


def test_calibracion_guiada_desde_el_juego():
    _, gestor, _ = cal_ui.simular_capturas(12)
    res, texto, detalles, grafica = jw.calibrar_amigable(gestor, 2.5)
    assert res.modelo.error_rms < 0.5 and "Calibración" in texto


def _partida_web(modo="2j"):
    modelo = s.camara_ejemplo(640, 360)
    fuente = s.FuentePantalla(modelo)
    ajustes = jw.guardar_ajustes("facil", 7, True)
    sesion, juego, modo, *pantallas = con_sesion(lambda j, a: jw.empezar(j, modo, a))(None, None, ajustes)
    sesion.modelo_camara = modelo  # como si hubiera calibrado en esta sesión
    frame = con_sesion(jw.jugar_frame)
    sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, modo, ajustes)
    K = juego.ultimo_res.modelo.K
    for lado, jug in zip((-1, 1), fuente.jugadores):
        x, y, w, h = pl.recuadro_piel(juego.ultimo_res.imagen.shape, lado)
        jug.mano = ((x + w / 2 - K[0, 2]) / K[0, 0] * jug.distancia, (y + h / 2 - K[1, 2]) / K[1, 1] * jug.distancia)
    juego = jw.accion("listo")(juego)
    juego.t_cuenta = time.perf_counter()

    def jugar(n):
        nonlocal sesion, img, juego
        for _ in range(n):
            sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, modo, ajustes)
        return img

    return sesion, juego, pantallas, jugar, ajustes


def test_juego_web_con_los_manejadores_de_gradio():
    sesion, juego, pantallas, jugar, ajustes = _partida_web()
    assert juego.op.interfaz == "web" and juego.op.puntos == 7
    assert [p.visible for p in pantallas] == [p == "juego" for p in jw.PANTALLAS]
    img = jugar(5)
    assert juego.fase == "jugando" and juego.config.usar_calibracion and img.shape == (360, 640, 3)
    juego = jw.accion("pausa")(juego)
    assert juego.pausado
    # Volver al menú y elegir otro modo conserva la piel del jugador 1
    sesion, juego2, *_ = con_sesion(lambda j, a: jw.empezar(j, "ia", a))(sesion, juego, ajustes)
    assert juego2 is juego and juego.op.modo == "ia" and juego.fase == "jugando" and not juego.pausado


@pytest.mark.parametrize("vista", [v for v in VISTAS if v not in ("juego", "transformers")])
def test_cada_tema_se_ve_en_la_partida_y_su_ajuste_cambia(vista):
    sesion, juego, _, jugar, _ = _partida_web()
    jugar(3)
    juego, texto = jw.elegir_vista(juego, vista)
    img = jugar(1)
    assert img.shape[0] == 360 and img.shape[1] > 640  # el panel del tema aparece al lado del juego
    p = PARAMETROS[vista]
    antes = p.leer(juego)
    juego, texto2 = jw.cambiar(1)(juego, vista)
    assert p.leer(juego) != antes and "cambia" in texto2
    jugar(1)  # el pipeline sigue funcionando con el ajuste nuevo
