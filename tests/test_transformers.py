"""Pruebas de los transformers.

Las que descargan/cargan modelos están marcadas como ``lento``:
``pytest -m "not lento"`` las omite.
"""

import numpy as np
import pytest

from voleyar.core import transformers_models as tm
from voleyar.utils import sinteticos as s


def test_iou_y_caja_de_mascara():
    assert tm.iou_cajas((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert tm.iou_cajas((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(tm.iou_cajas((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:30] = 255
    assert tm.caja_de_mascara(m) == (5.0, 10.0, 30.0, 20.0)


def test_mano_cerca_de_mesa_con_profundidad_sintetica():
    h, w = 200, 300
    cancha = np.zeros((h, w), np.uint8)
    cancha[40:180, 30:270] = 255
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    mapa = 1.0 + yy / h  # la mesa se acerca hacia abajo (disparidad crece)
    mano = np.zeros((h, w), np.uint8)
    mano[90:130, 120:170] = 255
    apoyada = tm.mano_cerca_de_mesa(mapa, mano, cancha)
    levantada_mapa = mapa.copy()
    levantada_mapa[mano > 0] += 0.5  # la mano está más cerca de la cámara
    levantada = tm.mano_cerca_de_mesa(levantada_mapa, mano, cancha)
    assert apoyada.cerca and not levantada.cerca
    assert levantada.puntaje > apoyada.puntaje + 0.5


def test_comparacion_sin_owl():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 10:20] = 255
    owl = tm.ResultadoOWL([tm.Deteccion((12, 10, 22, 20), 0.9, "a hand")], 100.0, "cpu")
    c = tm.comparar_clasico_vs_owl(m, 5.0, owl)
    assert c.distancia_px == pytest.approx(2.0) and 0 < c.iou < 1


@pytest.mark.lento
def test_depth_anything_carga_y_estima():
    img, v = s.escena_cancha(mano=s.EstadoManoSintetica(altura=8.0))
    r = tm.obtener_profundidad().estimar(img)
    assert r.mapa.shape == img.shape[:2] and np.isfinite(r.mapa).all()
    assert tm.obtener_profundidad() is tm.obtener_profundidad()  # se carga una sola vez


@pytest.mark.lento
def test_owlv2_detecta_y_respeta_coordenadas():
    img, v = s.escena_cancha(mano=s.EstadoManoSintetica(n_dedos=5))
    r = tm.obtener_owl().detectar(img, ("a hand",), umbral=0.05)
    assert r.ms > 0
    if r.mejor is not None:  # la mano sintética puede no parecer una mano real
        x0, y0, x1, y1 = r.mejor.caja
        assert 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]
