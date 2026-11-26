"""Pruebas de histogramas: modelo de piel, retroproyección y ecualización."""

import cv2
import numpy as np

from voleyar.core import histogramas as hg
from voleyar.utils import sinteticos as s


def test_retroproyeccion_encuentra_la_mano():
    img, verdad = s.escena_cancha(mano=s.EstadoManoSintetica(pos=(-6, 0), n_dedos=5))
    m = verdad["mascara_mano"]
    # Recuadro dentro de la palma
    ys, xs = np.nonzero(cv2.erode(m, np.ones((25, 25), np.uint8)))
    cy, cx = int(ys.mean()), int(xs.mean())
    modelo = hg.calibrar_piel(img, (cx - 12, cy - 12, 24, 24))
    for espacio in hg.ESPACIOS_RETRO:
        bp = hg.retroproyeccion(img, modelo, espacio)
        dentro = bp[m > 0].mean()
        fuera = bp[m == 0].mean()
        assert dentro > 4 * fuera, espacio


def test_modelo_piel_serializable():
    img, _ = s.escena_cancha(mano=s.EstadoManoSintetica())
    mod = hg.calibrar_piel(img, (0, 0, img.shape[1], img.shape[0]))
    copia = hg.ModeloPiel.desde_dict(mod.a_dict())
    assert np.allclose(copia.hist_hs, mod.hist_hs)


def test_ecualizacion_manual_igual_a_opencv():
    rng = np.random.default_rng(3)
    for _ in range(5):
        g = np.clip(rng.normal(90, 20, (120, 160)), 0, 255).astype(np.uint8)
        assert np.abs(hg.ecualizacion_manual(g).astype(int) - cv2.equalizeHist(g).astype(int)).max() <= 1


def test_ecualizacion_y_clahe_aumentan_contraste():
    rng = np.random.default_rng(4)
    oscura = np.clip(rng.normal(50, 10, (120, 160, 3)), 0, 255).astype(np.uint8)
    assert hg.contraste(hg.ecualizar(oscura)) > 2 * hg.contraste(oscura)
    assert hg.contraste(hg.aplicar_clahe(oscura, 4.0)) > hg.contraste(oscura)
    assert hg.histograma_gris(oscura).sum() == 120 * 160
