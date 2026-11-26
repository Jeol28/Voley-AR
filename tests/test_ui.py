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
    assert calibracion.comparar_cancha(fr, "Ambos (superpuestos)", 6.0)[0] is not None
    assert filtros.procesar(fr, 10, "gaussiano", 5, 50, 3, 40, 120, fil.NUCLEOS_EJEMPLO["Laplaciano"])[1] is not None
    assert histogramas.procesar(fr, 52, 42, 5, "combinada", 2.0, 8)[0] is not None
    umbralizacion.procesar(fr, 0, 25, 40, 60, 135, 175, 80, 128, 31, -5)
    morfologia.procesar(fr, 2, "cierre", "cruz", 5, 1)
    segmentacion.procesar(fr, "ancho", 1.2, 0.55)
    caracterizacion.procesar(fr, 1.6, 0.45, 100)
    galeria, md = pipeline.procesar(fr, True, 5.0)
    assert len(galeria) >= 2


@pytest.mark.parametrize("nombre,dedos", [("Mano abierta (remate)", 5), ("Un dedo (toque)", 1), ("Dos dedos", 2),
                                          ("Puño (bloqueo)", 0), ("Mano tocando un objeto color piel", 3),
                                          ("Poca luz", 4)])
def test_pipeline_cuenta_bien_los_dedos_en_los_ejemplos(nombre, dedos):
    r = estado.ejecutar_pipeline(estado.frame_ejemplo(nombre))
    assert r.analisis is not None and r.analisis.n_dedos == dedos
    if "objeto" in nombre:
        assert r.uso_watershed


def test_calibracion_simulada_desde_la_ui():
    _, gestor, _ = calibracion.simular_capturas(12)
    res, md, graf, comp, js = calibracion.calibrar(gestor, 2.5)
    assert res.modelo.error_rms < 0.5 and "fx" not in js


def test_juego_del_navegador_con_los_manejadores_de_gradio():
    """Simula el stream de la cámara y los botones: el estado (sesión y juego) pasa de llamada en llamada."""
    import time

    from voleyar.ui.estado import con_sesion
    from voleyar.ui.pestanas import jugar
    from voleyar.utils import sinteticos as s
    from voleyar.utils import visualizacion as vis

    modelo = s.camara_ejemplo(640, 360)
    fuente = s.FuenteSintetica(modelo)
    frame = con_sesion(jugar.jugar_frame)
    espacio = con_sesion(jugar.tecla(" "))
    sesion, juego = None, None
    sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego)
    sesion.modelo_camara = modelo  # como si hubiera calibrado en la pestaña Calibración
    sesion, juego = espacio(sesion, juego)
    juego.t_cuenta = time.perf_counter()
    fuente.mano.pos = (-6.0, 0.0)
    for _ in range(120):
        b = juego.partida.balon.pos
        if b[0] < 0:
            fuente.mano.pos = (float(b[0]) - 1.0, float(b[1]))
        sesion, img, juego = frame(sesion, vis.bgr_a_rgb(fuente.leer()[1]), juego)
    assert juego.fase == "jugando" and juego.config.usar_calibracion
    assert img.shape == (360, 640, 3) and "golpe" in juego.eventos_totales
