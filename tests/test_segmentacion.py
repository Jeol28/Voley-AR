"""Pruebas de segmentación: zona por pose, selección de la mano, corte del brazo y watershed."""

import cv2
import numpy as np
import pytest

from voleyar.core import histogramas as hg
from voleyar.core import morfologia as mo
from voleyar.core import segmentacion as sg
from voleyar.core import umbralizacion as um
from voleyar.utils import sinteticos as s


def _segmentar(mano_estado):
    img, v = s.escena_cancha(mano=mano_estado)
    z = sg.zona_busqueda(img.shape, v["pose"], v["modelo"])
    m = v["mascara_mano"]
    ys, xs = np.nonzero(cv2.erode(m, np.ones((15, 15), np.uint8)))
    cy, cx = int(ys.mean()), int(xs.mean())
    piel = hg.calibrar_piel(img, (cx - 10, cy - 10, 20, 20))
    bp = hg.retroproyeccion(img, piel, mascara=z.mascara)
    mk, _ = um.umbral_piel_otsu(bp, z.mascara)
    c, comp = sg.seleccionar_mano(mo.limpiar_mascara(mk), z.franja)
    return img, v, z, c, comp


def test_zona_de_busqueda_contiene_cancha_y_excluye_lo_lejano():
    img, v = s.escena_cancha()
    z = sg.zona_busqueda(img.shape, v["pose"], v["modelo"])
    cancha = sg.mascara_cancha(img.shape, v["pose"], v["modelo"])
    assert np.all(z.mascara[cancha > 0] > 0)
    assert 0.1 < z.mascara.mean() / 255 < 0.8  # deja fuera buena parte del cuadro
    tuyo, rival = sg.mascara_lados(img.shape, v["pose"], v["modelo"])
    assert abs(int(np.count_nonzero(tuyo)) - int(np.count_nonzero(rival))) < 0.6 * np.count_nonzero(cancha)


@pytest.mark.parametrize("metodo", ["distancia", "ancho"])
@pytest.mark.parametrize("pos,ang,n", [((-6, 1), 10, 5), ((-3, -3), -20, 1), ((-5, 0), 0, 2)])
def test_separar_mano_brazo(metodo, pos, ang, n):
    img, v, z, c, comp = _segmentar(s.EstadoManoSintetica(pos=pos, n_dedos=n, angulo=ang))
    assert c is not None
    antes = um.metricas_mascara(comp, v["mascara_mano"])
    mano, info = sg.separar_mano_brazo(comp, z.mascara, metodo)
    despues = um.metricas_mascara(mano, v["mascara_mano"])
    assert info.direccion_brazo is not None
    assert despues["precision"] > 0.85 and despues["precision"] > antes["precision"] + 0.1
    assert despues["exhaustividad"] > 0.8


def test_sin_brazo_no_se_corta_nada():
    m = s.silueta_mano_2d(5, con_brazo=False)
    mano, info = sg.separar_mano_brazo(m)
    assert info.direccion_brazo is None and np.array_equal(mano, m)


def test_watershed_separa_mano_de_objeto_pegado():
    m = s.silueta_mano_2d(0, con_brazo=False, escala=22)
    ys, xs = np.nonzero(m)
    cx, top = int(xs.mean()), ys.min()
    otro = np.zeros_like(m)
    cv2.circle(otro, (cx, top - 70), 78, 255, -1)  # objeto "color piel" que toca la mano
    unida = cv2.bitwise_or(m, otro)
    assert len(sg.contornos(unida)) == 1
    preferir = np.zeros_like(m)
    preferir[-120:] = 255  # el brazo entra por abajo
    img = cv2.cvtColor(unida, cv2.COLOR_GRAY2BGR)
    mano, etiquetas, n = sg.watershed_mano(img, unida, preferir)
    assert n >= 2
    met = um.metricas_mascara(mano, m)
    assert met["iou"] > 0.8
    assert sg.colorear_etiquetas(etiquetas).shape == img.shape
