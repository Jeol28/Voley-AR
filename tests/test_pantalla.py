"""Pruebas del modo pantalla (v2): cámara virtual, distancia por la palma, caras y dos jugadores."""

import cv2
import numpy as np
import pytest

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.core import pipeline as pl
from voleyar.core import segmentacion as seg
from voleyar.utils import sinteticos as s


def test_camara_virtual_encuadra_la_cancha():
    for modelo in (s.camara_ejemplo(1280, 720, distorsion=False), cal.modelo_aproximado(960, 540)):
        pose = cal.camara_virtual(modelo)
        L = cfg.MEDIO_LARGO
        p = cal.proyectar(np.array([[0, 0, 0], [-L, 0, 0], [L, 0, 0]], float), pose, modelo)
        assert abs(p[0, 0] - modelo.ancho / 2) < 1e-6  # red en la columna central
        assert abs(p[0, 1] - cfg.FRACCION_PISO * modelo.alto) < 1e-6  # piso en la fila elegida
        assert abs((p[2, 0] - p[1, 0]) / modelo.ancho - cfg.OCUPACION_CANCHA) < 1e-6
        # Ida y vuelta: píxel → plano de la red
        q = cal.pixel_a_plano(cal.proyectar(np.array([[-4.0, 0, 6.0]]), pose, modelo)[0], pose, modelo, 1, 0.0)
        assert np.allclose(q, [-4.0, 0, 6.0], atol=1e-6)


def test_distancia_por_tamano_es_la_proyeccion_inversa():
    f, W, Z = 700.0, 4.0, 60.0
    w = f * W / Z
    assert cal.distancia_por_tamano(w, W, f) == pytest.approx(Z)
    # Con la focal equivocada la distancia sale escalada por el cociente de focales
    assert cal.distancia_por_tamano(w, W, 960.0) == pytest.approx(Z * 960 / 700)


def test_detecta_la_cara_dibujada_y_una_por_mitad():
    img, v = s.escena_pantalla()
    caras = seg.detectar_caras(img)
    assert 1 <= len(caras) <= 2
    lados = [-1 if c[0] + c[2] / 2 < img.shape[1] / 2 else 1 for c in caras]
    assert len(set(lados)) == len(lados)
    x, y, w, h = v["caras"][0]  # la cara del jugador 1 (la cascada la encuentra)
    assert any(abs(c[0] + c[2] / 2 - (x + w / 2)) < w / 3 for c in caras)


def test_memoria_de_caras_cuando_la_mano_la_tapa():
    est = pl.EstadoPipeline()
    vistas = pl.recordar_caras([(100, 50, 200, 200)], 960, est, memoria=5)
    assert len(vistas) == 1
    for _ in range(3):
        est.n_frame += 1
        assert pl.recordar_caras([], 960, est, memoria=5) == vistas
    est.n_frame += 10
    assert pl.recordar_caras([], 960, est, memoria=5) == []


def test_watershed_separa_la_mano_de_la_cara():
    m = np.zeros((400, 400), np.uint8)
    cv2.ellipse(m, (200, 120), (70, 95), 0, 0, 360, 255, -1)  # cara
    mano = np.zeros_like(m)
    cv2.circle(mano, (265, 230), 45, 255, -1)  # palma pegada a la mejilla
    cv2.rectangle(mano, (240, 230), (290, 399), 255, -1)  # antebrazo hasta abajo
    unida = cv2.bitwise_or(m, mano)
    cara = seg.mascara_caras(m.shape, [(130, 25, 140, 190)])
    img = cv2.cvtColor(unida, cv2.COLOR_GRAY2BGR)
    sola, etiquetas = seg.separar_de_la_cara(img, unida, cara)
    inter = np.count_nonzero(cv2.bitwise_and(sola, mano)) / np.count_nonzero(mano)
    fuera = np.count_nonzero(cv2.bitwise_and(sola, cv2.bitwise_not(mano))) / np.count_nonzero(sola)
    assert inter > 0.8 and fuera < 0.15


def test_la_mano_junto_a_la_cara_mantiene_su_distancia():
    modelo = s.camara_ejemplo(1280, 720)
    est = pl.EstadoPipeline()
    img0, _ = s.escena_pantalla(modelo)  # primero ve la cara sin la mano encima
    pl.procesar_frame(img0, modelo, None, pl.ConfigPipeline(), est)
    js = s.jugadores_por_defecto()
    js[0].mano = (-19.0, -2.0)
    img, _ = s.escena_pantalla(modelo, js)
    for _ in range(3):
        r = pl.procesar_frame(img, modelo, None, pl.ConfigPipeline(), est)
    m = r.manos[-1]
    assert m.uso_watershed and abs(m.distancia_cm - 60) < 8


def test_la_mano_se_busca_solo_en_su_mitad():
    js = s.jugadores_por_defecto(1)  # solo el jugador 1
    img, _ = s.escena_pantalla(jugadores=js)
    r = pl.procesar_frame(img, s.camara_ejemplo(), None, pl.ConfigPipeline())
    assert r.manos[-1].hay_mano and not r.manos[1].hay_mano


def test_espejo_conserva_la_geometria():
    """Voltear la imagen y el modelo de cámara a la vez da la misma proyección, volteada."""
    m = s.camara_ejemplo(960, 540)
    pts = np.array([[5.0, -3.0, 60.0], [-20.0, 8.0, 70.0], [12.0, 10.0, 55.0]])
    p, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), m.K, m.dist)
    espejo = m.espejo()
    pts_e = pts * np.array([-1.0, 1.0, 1.0])  # el mundo visto en un espejo
    pe, _ = cv2.projectPoints(pts_e, np.zeros(3), np.zeros(3), espejo.K, espejo.dist)
    assert np.allclose(pe.reshape(-1, 2)[:, 0], m.ancho - 1 - p.reshape(-1, 2)[:, 0], atol=1e-6)
    assert np.allclose(pe.reshape(-1, 2)[:, 1], p.reshape(-1, 2)[:, 1], atol=1e-6)


def test_en_espejo_el_jugador_1_sigue_a_la_izquierda_de_la_pantalla():
    js = s.jugadores_por_defecto(1)
    js[0].mano, js[0].cara = (12.0, 7.0), (28.0, -20.0)  # en la imagen de la cámara la persona está a la derecha…
    img, _ = s.escena_pantalla(jugadores=js)
    r = pl.procesar_frame(img, s.camara_ejemplo(), None, pl.ConfigPipeline(espejo=True))
    assert r.manos[-1].hay_mano and not r.manos[1].hay_mano  # …en espejo queda a la izquierda (jugador 1)


# --------------------------------------------------------------------------- #
# ¿Hay mano o no? Validación por forma, movimiento y seguimiento
# --------------------------------------------------------------------------- #


def _sin_manos():
    js = s.jugadores_por_defecto()
    for j in js:
        j.mano_visible = False
    return js


def test_sin_manos_no_se_inventa_ninguna_aunque_falle_el_detector_de_caras():
    img, _ = s.escena_pantalla(jugadores=_sin_manos())
    for excluir in (True, False):  # también si Haar no encontrara ninguna cara
        r = pl.procesar_frame(img, s.camara_ejemplo(), None, pl.ConfigPipeline(excluir_caras=excluir))
        for lado, m in r.manos.items():
            assert not m.hay_mano, (excluir, lado, m.validacion)


def test_las_manos_reales_pasan_la_validacion():
    img, v = s.escena_pantalla()
    r = pl.procesar_frame(img, s.camara_ejemplo(), None, pl.ConfigPipeline())
    for m in r.manos.values():
        assert m.validacion.valida and m.validacion.confianza > 0.5 and m.hay_mano


def test_seguidor_aparece_desaparece_y_no_parpadea():
    from voleyar.core.caracterizacion import SeguidorMano

    sg = SeguidorMano(frames_aparecer=3, frames_desaparecer=6)
    assert [sg.actualizar(True, 5, 0.3) for _ in range(3)] == [False, False, True]
    # Un frame malo de vez en cuando no la hace desaparecer
    for k in range(20):
        assert sg.actualizar(k % 3 != 0, 5, 0.0)
    assert sg.dedos_estables == 5
    assert [sg.actualizar(False) for _ in range(6)][-1] is False


def test_el_gesto_se_suaviza():
    from voleyar.core.caracterizacion import SeguidorMano

    sg = SeguidorMano()
    for n in (5, 5, 1, 5, 5):  # un frame mal contado
        sg.actualizar(True, n, 0.3)
    assert sg.dedos_estables == 5


def test_una_mano_quieta_desde_el_inicio_no_se_toma_hasta_que_se_mueve():
    """Como un póster o un objeto de color piel: si nunca se mueve, no es un jugador."""
    modelo = s.camara_ejemplo(960, 540)
    fuente = s.FuentePantalla(modelo, n_jugadores=1)
    cfg = pl.ConfigPipeline(seguimiento=True)
    est = pl.EstadoPipeline()
    vistos = [pl.procesar_frame(fuente.leer()[1], modelo, None, cfg, est, lados=(-1,)).manos[-1].hay_mano
              for _ in range(40)]
    assert not any(vistos)
    for k in range(10):  # ahora la mano se mueve
        x, y = fuente.jugadores[0].mano
        fuente.jugadores[0].mano = (x + (2.0 if k % 2 else -2.0), y)
        m = pl.procesar_frame(fuente.leer()[1], modelo, None, cfg, est, lados=(-1,)).manos[-1]
    assert m.hay_mano and m.presente


def test_la_mano_desaparece_al_bajarla():
    modelo = s.camara_ejemplo(960, 540)
    fuente = s.FuentePantalla(modelo, n_jugadores=1)
    cfg = pl.ConfigPipeline(seguimiento=True)
    est = pl.EstadoPipeline()
    rng = np.random.default_rng(0)
    for _ in range(30):  # la mano se mueve como al jugar
        fuente.jugadores[0].mano = (rng.uniform(-18, -8), rng.uniform(2, 10))
        m = pl.procesar_frame(fuente.leer()[1], modelo, None, cfg, est, lados=(-1,)).manos[-1]
    assert m.hay_mano
    fuente.jugadores[0].mano_visible = False
    estados = [pl.procesar_frame(fuente.leer()[1], modelo, None, cfg, est, lados=(-1,)).manos[-1].hay_mano
               for _ in range(8)]
    assert estados[-1] is False
