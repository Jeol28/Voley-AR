"""Pruebas de caracterización: descriptores vs. scikit-image, conteo de dedos y velocidad."""

import cv2
import numpy as np
import pytest
from skimage import measure

from voleyar.core import calibracion as cal
from voleyar.core import caracterizacion as ca
from voleyar.core import segmentacion as sg
from voleyar.utils import sinteticos as s


def _formas():
    elipse = np.zeros((200, 260), np.uint8)
    cv2.ellipse(elipse, (130, 100), (90, 40), 25, 0, 360, 255, -1)
    rect = np.zeros((200, 260), np.uint8)
    cv2.rectangle(rect, (40, 50), (200, 130), 255, -1)
    return [elipse, rect, s.silueta_mano_2d(5, escala=14, tam=(300, 400)), s.silueta_mano_2d(2, escala=14, tam=(300, 400))]


@pytest.mark.parametrize("i", range(4))
def test_descriptores_frente_a_scikit_image(i):
    m = _formas()[i]
    d = ca.descriptores_forma(m)
    rp = measure.regionprops(measure.label(m > 0))[0]
    assert d.area == pytest.approx(rp.area, rel=1e-6)
    assert d.centroide[0] == pytest.approx(rp.centroid[1], abs=1e-6)
    assert d.centroide[1] == pytest.approx(rp.centroid[0], abs=1e-6)
    assert d.excentricidad == pytest.approx(rp.eccentricity, abs=1e-6)
    assert d.solidez == pytest.approx(rp.solidity, abs=0.03)
    assert d.perimetro == pytest.approx(rp.perimeter, rel=0.08)
    # Hu: los 6 primeros son invariantes a la reflexión (skimage usa (fila, col)); el 7.º cambia de signo
    hu_sk = rp.moments_hu
    assert np.allclose(d.hu[:6], hu_sk[:6], rtol=1e-4, atol=1e-12)
    assert abs(d.hu[6]) == pytest.approx(abs(hu_sk[6]), rel=1e-3, abs=1e-14)


def test_circularidad_del_circulo():
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 70, 255, -1)
    assert ca.descriptores_forma(m).circularidad > 0.85


@pytest.mark.parametrize("angulo", [90, 45, 160, -30])
def test_conteo_de_dedos_en_siluetas(angulo):
    for n in range(6):
        m = s.silueta_mano_2d(n, angulo=angulo)
        mano, info = sg.separar_mano_brazo(m)
        a = ca.analizar_mano(mano, info.direccion_brazo)
        assert a.n_dedos == n, (angulo, n)
        assert a.gesto == ca.GESTOS[n][0]


@pytest.mark.parametrize("angulo", [0, 30, -30])
def test_conteo_en_perspectiva_con_vista_cenital(angulo):
    modelo = s.camara_ejemplo(distorsion=False)
    for n in range(6):
        img, v = s.escena_cancha(modelo, mano=s.EstadoManoSintetica(pos=(-6, 0), n_dedos=n, angulo=angulo), ruido=0)
        piel = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (0, 40, 60), (25, 255, 255))
        vc = cal.vista_cenital(v["pose"], modelo)
        mano, info = sg.separar_mano_brazo(vc.rectificar(piel))
        assert ca.analizar_mano(mano, info.direccion_brazo).n_dedos == n


def test_defectos_y_punto_de_golpe():
    m = s.silueta_mano_2d(5, con_brazo=False)
    a = ca.analizar_mano(m)
    assert sum(d.valido for d in a.defectos) >= 3
    assert a.punto_golpe_px == a.centro_palma  # mano abierta → palma
    uno = ca.analizar_mano(s.silueta_mano_2d(1, con_brazo=False))
    assert uno.punto_golpe_px == tuple(float(v) for v in uno.punta_principal)
    assert uno.punta_principal[1] < uno.centro_palma[1]  # el índice apunta hacia arriba


def test_vista_cenital_ida_y_vuelta():
    modelo = s.camara_ejemplo(distorsion=False)
    pose = s.pose_juego_por_defecto()
    vc = cal.vista_cenital(pose, modelo)
    p = np.array([[-5.0, 3.0, 0.0], [8.0, -6.0, 0.0]])
    px = cal.proyectar(p, pose, modelo)
    assert np.allclose(vc.a_mesa(vc.desde_imagen(px)), p, atol=1e-6)
    assert np.allclose(vc.a_imagen(vc.desde_imagen(px)), px, atol=1e-6)


def test_estimador_de_velocidad():
    est = ca.EstimadorVelocidad(alfa=1.0)
    est.actualizar(np.array([0.0, 0.0]), 0.0)
    v = est.actualizar(np.array([3.0, 4.0]), 0.1)
    assert np.allclose(v, [30, 40]) and est.rapidez == pytest.approx(50)
    est.actualizar(None, 1.0)  # se perdió la mano mucho tiempo → reinicio
    assert est.rapidez == 0
