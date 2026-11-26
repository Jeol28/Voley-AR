"""Pruebas de la UI sin navegador: la app se construye y cada pestaña procesa los ejemplos."""

import pytest

from voleyar.core import filtros as fil
from voleyar.ui import estado
from voleyar.ui.pestanas import (calibracion, caracterizacion, filtros, histogramas, morfologia, pipeline,
                                 segmentacion, umbralizacion)


def test_la_app_se_construye():
    from voleyar.ui.app import crear_app

    assert crear_app() is not None


@pytest.mark.parametrize("nombre", list(estado.EJEMPLOS))
def test_pestanas_procesan_cada_ejemplo(nombre):
    fr = estado.frame_ejemplo(nombre)
    assert calibracion.comparar_distancia(fr, 7.0)[0] is not None
    assert filtros.procesar(fr, 10, "gaussiano", 5, 50, 3, 40, 120, fil.NUCLEOS_EJEMPLO["Laplaciano"])[1] is not None
    assert histogramas.procesar(fr, 36, 66, 5, "combinada", 2.0, 8)[0] is not None
    for lado in (-1, 1):
        umbralizacion.procesar(fr, lado, 0, 25, 40, 60, 135, 175, 80, 128, 31, -5)
        segmentacion.procesar(fr, lado, "ancho", 1.2, 0.55)
        caracterizacion.procesar(fr, lado, 1.6, 0.45, 100)
    morfologia.procesar(fr, 2, "cierre", "cruz", 5, 1)
    galeria, md = pipeline.procesar(fr, True, 7.0)
    assert len(galeria) == 9


def test_demo_de_pose_con_aruco_sigue_funcionando():
    fr = estado.frame_ejemplo(estado.EJEMPLO_CANCHA)
    img, md = calibracion.comparar_cancha(fr, "Ambos (superpuestos)", 6.0)
    assert img is not None and "Con calibración" in md


CASOS = [(n, lado, d) for n, p in estado.EJEMPLOS.items() for lado, d in zip((-1, 1), p["dedos"])
         if n != "Mano delante de la cara"]  # con la mano sobre la cara los dedos se confunden con la cara


@pytest.mark.parametrize("nombre,lado,dedos", CASOS)
def test_pipeline_cuenta_bien_los_dedos_en_los_ejemplos(nombre, lado, dedos):
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


def test_calibracion_simulada_desde_la_ui():
    _, gestor, _ = calibracion.simular_capturas(12)
    res, md, graf, comp, js = calibracion.calibrar(gestor, 2.5)
    assert res.modelo.error_rms < 0.5 and "fx" not in js


def test_juego_web_con_los_manejadores_de_gradio():
    """Simula la página del juego: botón de modo, stream de la cámara y botones de la partida."""
    import time

    from voleyar.core import pipeline as pl
    from voleyar.ui import juego_web as jw
    from voleyar.ui.estado import con_sesion
    from voleyar.utils import sinteticos as s
    from voleyar.utils import visualizacion as vis

    modelo = s.camara_ejemplo(640, 360)
    fuente = s.FuentePantalla(modelo)
    ajustes = jw.guardar_ajustes("facil", 7, True, True)
    empezar = con_sesion(lambda j, a: jw.empezar(j, "2j", a))
    frame = con_sesion(jw.jugar_frame)
    sesion, juego, modo, *pantallas = empezar(None, None, ajustes)
    assert modo == "2j" and juego.op.interfaz == "web" and juego.op.puntos == 7
    assert [p.visible for p in pantallas] == [p == "juego" for p in jw.PANTALLAS]
    sesion.modelo_camara = modelo  # como si hubiera calibrado en esta sesión
    sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, modo, ajustes)
    K = juego.ultimo_res.modelo.K
    for lado, jug in zip((-1, 1), fuente.jugadores):
        x, y, w, h = pl.recuadro_piel(juego.ultimo_res.imagen.shape, lado)
        jug.mano = ((x + w / 2 - K[0, 2]) / K[0, 0] * jug.distancia, (y + h / 2 - K[1, 2]) / K[1, 1] * jug.distancia)
    juego = jw.accion("listo")(juego)
    juego.t_cuenta = time.perf_counter()
    for _ in range(5):
        sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, modo, ajustes)
    assert juego.fase == "jugando" and juego.config.usar_calibracion and juego.depurar
    assert img.shape[0] == 360 and img.shape[1] > 640  # "ver cómo te detecta" agrega el panel
    juego = jw.accion("pausa")(juego)
    assert juego.pausado
    # Volver al menú y elegir otro modo conserva la piel del jugador 1
    sesion, juego2, modo2, *_ = con_sesion(lambda j, a: jw.empezar(j, "ia", a))(sesion, juego, ajustes)
    assert juego2 is juego and juego.op.modo == "ia" and juego.fase == "jugando" and not juego.pausado


def test_la_app_tiene_el_juego_y_el_laboratorio_aparte():
    from voleyar.ui.app import crear_app, crear_laboratorio
    from voleyar.ui.laboratorio import PESTANAS

    assert crear_app() is not None and crear_laboratorio() is not None
    assert "Jugar" not in [n for n, _ in PESTANAS] and len(PESTANAS) == 9
