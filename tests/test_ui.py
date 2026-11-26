"""Pruebas de la UI sin navegador: la app se construye y cada pestaña procesa los ejemplos."""

import pytest

from voleyar.core import filtros as fil
from voleyar.ui import estado
from voleyar.ui.pestanas import (calibracion, caracterizacion, filtros, histogramas, morfologia, pipeline,
                                 segmentacion, umbralizacion)


def test_la_app_se_construye():
    from voleyar.ui.app import PESTANAS, crear_app

    demo = crear_app()
    assert demo is not None and len(PESTANAS) == 10


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


def test_juego_del_navegador_con_los_manejadores_de_gradio():
    """Simula el stream de la cámara y los botones: el estado (sesión y juego) pasa de llamada en llamada."""
    import time

    from voleyar.core import pipeline as pl
    from voleyar.ui.estado import con_sesion
    from voleyar.ui.pestanas import jugar
    from voleyar.utils import sinteticos as s
    from voleyar.utils import visualizacion as vis

    modelo = s.camara_ejemplo(640, 360)
    fuente = s.FuentePantalla(modelo)
    frame = con_sesion(jugar.jugar_frame)
    espacio = con_sesion(jugar.tecla(" "))
    sesion, juego = None, None
    sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, "2j")
    sesion.modelo_camara = modelo  # como si hubiera calibrado en la pestaña Calibración
    sesion, juego = espacio(sesion, juego, "2j")
    juego.t_cuenta = time.perf_counter()
    K = juego.ultimo_res.modelo.K
    for lado, jug in zip((-1, 1), fuente.jugadores):
        x, y, w, h = pl.recuadro_piel(juego.ultimo_res.imagen.shape, lado)
        jug.mano = ((x + w / 2 - K[0, 2]) / K[0, 0] * jug.distancia, (y + h / 2 - K[1, 2]) / K[1, 1] * jug.distancia)
    for _ in range(10):
        sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, "2j")
    assert juego.fase == "jugando" and juego.config.usar_calibracion and juego.op.modo == "2j"
    assert img.shape == (360, 640, 3)
    # Cambiar de modo crea un juego nuevo
    sesion, img, juego2 = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego, "ia")
    assert juego2 is not juego and juego2.op.modo == "ia"
