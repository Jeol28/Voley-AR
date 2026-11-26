"""Pruebas de umbralización: Otsu manual vs. OpenCV y rangos de piel."""

import cv2
import numpy as np
import pytest

from voleyar.core import umbralizacion as um
from voleyar.utils import sinteticos as s


@pytest.mark.parametrize("semilla", range(6))
def test_otsu_manual_igual_a_opencv(semilla):
    rng = np.random.default_rng(semilla)
    m1, m2 = rng.uniform(30, 100), rng.uniform(140, 220)
    n = 20000
    frac = rng.uniform(0.2, 0.8)
    datos = np.concatenate([rng.normal(m1, 15, int(n * frac)), rng.normal(m2, 20, n - int(n * frac))])
    g = np.clip(datos, 0, 255).astype(np.uint8).reshape(100, -1)
    T, curva = um.otsu_manual(g)
    assert T == um.otsu_opencv(g)
    assert curva.shape == (256,) and curva[T] == curva.max()
    assert m1 < T < m2


def test_otsu_con_mascara_ignora_lo_de_afuera():
    g = np.zeros((100, 100), np.uint8)
    g[:, :50] = 200  # fuera de la máscara
    g[:50, 50:] = 60
    g[50:, 50:] = 120
    mascara = np.zeros_like(g)
    mascara[:, 50:] = 255
    T, _ = um.otsu_manual(g, mascara)
    assert 60 <= T < 120


def test_rangos_piel_en_hsv_y_ycrcb():
    img, verdad = s.escena_cancha(mano=s.EstadoManoSintetica(n_dedos=5))
    v = verdad["mascara_mano"]
    for f in (um.umbral_hsv, um.umbral_ycrcb):
        met = um.metricas_mascara(f(img), v)
        assert met["exhaustividad"] > 0.8, f.__name__
    # El campo azul nunca debe ser piel
    sin_mano, _ = s.escena_cancha()
    assert um.umbral_ycrcb(sin_mano).mean() < 2


def test_rango_hsv_envolvente():
    rojo = np.zeros((10, 10, 3), np.uint8)
    rojo[:] = (40, 40, 200)  # H ≈ 0
    assert um.umbral_hsv(rojo, (170, 50, 50), (10, 255, 255)).all()


def test_adaptativo_y_metricas():
    g = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    assert um.umbral_adaptativo(g, "media", 15, 2).dtype == np.uint8
    m = np.zeros((10, 10), np.uint8)
    m[:5] = 255
    assert um.metricas_mascara(m, m)["iou"] == 1.0
    r = um.evaluar_en_recuadro(m, (0, 0, 10, 5))
    assert r["piel_en_recuadro_%"] == 100 and r["falsa_piel_fuera_%"] == 0
