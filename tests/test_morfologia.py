"""Pruebas de morfología matemática."""

import cv2
import numpy as np
import pytest

from voleyar.core import morfologia as mo
from voleyar.utils import sinteticos as s


@pytest.mark.parametrize("forma", list(mo.FORMAS))
@pytest.mark.parametrize("tam", [3, 5, 7])
def test_erosion_y_dilatacion_manuales_igual_a_opencv(forma, tam):
    rng = np.random.default_rng(tam)
    m = ((rng.random((40, 50)) > 0.6) * 255).astype(np.uint8)
    ee = mo.elemento_estructurante(forma, tam)
    assert np.array_equal(mo.erosion_manual(m, ee), cv2.erode(m, ee))
    assert np.array_equal(mo.dilatacion_manual(m, ee), cv2.dilate(m, ee))


def test_apertura_quita_ruido_y_cierre_rellena_huecos():
    base = s.silueta_mano_2d(5)
    rng = np.random.default_rng(0)
    ruidosa = base.copy()
    ruido = rng.random(base.shape) < 0.01
    ruidosa[ruido & (base == 0)] = 255  # puntos sueltos fuera
    ruidosa[ruido & (base > 0)] = 0  # agujeros dentro
    limpia = mo.limpiar_mascara(ruidosa, 5, 7)
    err_antes = np.count_nonzero(ruidosa != base)
    err_despues = np.count_nonzero(limpia != base)
    assert err_despues < 0.25 * err_antes


def test_rellenar_huecos_y_gradiente():
    m = np.zeros((50, 50), np.uint8)
    cv2.circle(m, (25, 25), 15, 255, -1)
    cv2.circle(m, (25, 25), 5, 0, -1)
    assert mo.rellenar_huecos(m)[25, 25] == 255
    g = mo.gradiente(m, mo.elemento_estructurante("cruz", 3))
    assert g[25, 25] == 0 and g.max() == 255


def test_aplicar_operaciones_por_nombre():
    m = s.silueta_mano_2d(2)
    for op in mo.OPERACIONES:
        assert mo.aplicar(m, op, "elipse", 5).shape == m.shape
    area = np.count_nonzero(m)
    assert np.count_nonzero(mo.aplicar(m, "erosion")) < area < np.count_nonzero(mo.aplicar(m, "dilatacion"))
