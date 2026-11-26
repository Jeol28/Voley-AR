"""Pruebas de filtros: convolución manual vs. cv2.filter2D y suavizados."""

import numpy as np
import pytest

from voleyar.core import filtros as f


@pytest.fixture
def imagen():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (60, 80), dtype=np.uint8)


@pytest.mark.parametrize("nombre", list(f.NUCLEOS_EJEMPLO))
def test_convolucion_manual_igual_a_filter2d(imagen, nombre):
    k = f.parsear_nucleo(f.NUCLEOS_EJEMPLO[nombre])
    assert np.allclose(f.convolucion_manual(imagen, k), f.convolucion_opencv(imagen, k), atol=1e-3)


def test_convolucion_asimetrica_5x3_y_color(imagen):
    rng = np.random.default_rng(1)
    k = rng.normal(size=(5, 3)).astype(np.float32)
    assert np.allclose(f.convolucion_manual(imagen, k), f.convolucion_opencv(imagen, k), atol=1e-2)
    color = np.dstack([imagen, imagen[::-1], imagen[:, ::-1]])
    assert np.allclose(f.convolucion_manual(color, k), f.convolucion_opencv(color, k), atol=1e-2)


def test_convolucion_voltea_el_nucleo():
    """Un impulso convolucionado reproduce el núcleo (no su versión volteada)."""
    img = np.zeros((7, 7), np.float32)
    img[3, 3] = 1
    k = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    out = f.convolucion_manual(img, k)
    assert np.allclose(out[2:5, 2:5], k)


def test_parsear_nucleo_errores():
    assert f.parsear_nucleo("1 2 3\n4 5 6\n7 8 9").shape == (3, 3)
    with pytest.raises(ValueError):
        f.parsear_nucleo("1 2; 3")
    with pytest.raises(ValueError):
        f.parsear_nucleo("1 2; 3 4")


def test_mediana_elimina_sal_y_pimienta():
    img = np.full((50, 50), 128, np.uint8)
    rng = np.random.default_rng(2)
    idx = rng.random(img.shape) < 0.05
    img[idx] = 255
    limpia = f.suavizar(img, "mediana", 3)
    assert (limpia == 255).sum() < idx.sum() * 0.05


def test_sobel_y_canny_detectan_borde_vertical():
    img = np.zeros((40, 40), np.uint8)
    img[:, 20:] = 200
    gx, gy, mag = f.sobel(img)
    assert np.abs(gx).max() > 0 and np.abs(gy).max() == 0
    assert f.canny(img)[:, 18:22].any()
