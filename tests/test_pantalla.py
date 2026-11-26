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
