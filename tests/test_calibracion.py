"""Pruebas de calibración con datos sintéticos de verdad conocida."""

import cv2
import numpy as np
import pytest

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.utils import sinteticos as s


def _vistas_tablero(modelo, n=14, semilla=1):
    rng = np.random.default_rng(semilla)
    tex, ppcm, origen = s.textura_tablero(30.0)
    centro = np.array([10.0, 6.25, 0.0])
    vistas = []
    intentos = 0
    while len(vistas) < n and intentos < 80:
        intentos += 1
        az = rng.uniform(0, 2 * np.pi)
        elev = np.deg2rad(rng.uniform(45, 80))
        d = rng.uniform(32, 45)
        C = centro + d * np.array([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)])
        objetivo = centro + np.array([rng.uniform(-6, 6), rng.uniform(-4, 4), 0])
        pose = s.pose_mirando(C, objetivo, rodar=rng.uniform(-20, 20))
        img = s.renderizar_plano(tex, ppcm, origen, modelo, pose, s.fondo_mesa(modelo.ancho, modelo.alto))
        img = s.ruido_camara(img, 2.0, len(vistas))
        ok, pts = cal.detectar_tablero(img)
        if ok:
            vistas.append(pts)
    return vistas


@pytest.fixture(scope="module")
def modelo_verdad():
    return s.camara_ejemplo(960, 540, distorsion=True)


@pytest.fixture(scope="module")
def vistas(modelo_verdad):
    return _vistas_tablero(modelo_verdad)


def test_calibracion_recupera_K_y_distorsion(modelo_verdad, vistas):
    assert len(vistas) >= 10
    res = cal.calibrar_camara(vistas, (960, 540))
    K, Kv = res.modelo.K, modelo_verdad.K
    assert res.modelo.error_rms < 0.5
    assert abs(K[0, 0] - Kv[0, 0]) / Kv[0, 0] < 0.02
    assert abs(K[1, 1] - Kv[1, 1]) / Kv[1, 1] < 0.02
    assert abs(K[0, 2] - Kv[0, 2]) < 8
    assert abs(K[1, 2] - Kv[1, 2]) < 8
    # k1 domina la distorsión de barril
    assert abs(res.modelo.dist[0] - modelo_verdad.dist[0]) < 0.05
    # El modelo recuperado deforma los puntos casi igual que el verdadero
    pts = np.array([[0.4, 0.25, 1.0], [-0.5, -0.2, 1.0], [0.1, -0.3, 1.0]])
    p1, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, res.modelo.dist)
    p2, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), Kv, modelo_verdad.dist)
    assert np.max(np.linalg.norm(p1 - p2, axis=-1)) < 2.0


def test_gestor_captura_rechaza_vistas_repetidas(vistas):
    g = cal.GestorCapturaCalibracion((960, 540), objetivo=5)
    assert g.agregar(vistas[0])
    assert not g.agregar(vistas[0] + 0.5)  # misma vista, casi sin movimiento
    for v in vistas[1:]:
        g.agregar(v)
    assert len(g.vistas) >= 5 and g.listo
    assert isinstance(g.sugerencia(), str)


def test_solvepnp_recupera_pose_conocida():
    modelo = s.camara_ejemplo()
    for C in [(38, 2, 34), (25, -10, 30), (-5, 30, 40)]:
        verdad = s.pose_mirando(C, (0, 0, 0))
        img, _ = s.escena_cancha(modelo, verdad, ruido=2.0)
        marcadores = cal.detectar_marcadores(img)
        assert len(marcadores) == 4
        pose = cal.estimar_pose_cancha(marcadores, modelo)
        assert pose is not None
        assert np.linalg.norm(pose.centro_camara - np.array(C)) < 1.0  # < 1 cm
        Rrel = pose.R @ verdad.R.T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        assert ang < 1.0
        assert pose.error_rms < 1.0


def test_sin_calibrar_tiene_mas_error_de_reproyeccion():
    modelo = s.camara_ejemplo()
    img, _ = s.escena_cancha(modelo, ruido=1.0)
    modos = cal.comparar_modos(cal.detectar_marcadores(img), modelo, modelo.ancho, modelo.alto)
    assert modos["sin_calibrar"][0].error_rms > 2 * modos["calibrado"][0].error_rms


def test_pixel_a_mesa_es_inverso_de_proyectar():
    modelo = s.camara_ejemplo()
    pose = s.pose_juego_por_defecto()
    for p in [(0, 0, 0), (-8, 3, 0), (10, -7, 0), (-5, 2, 4)]:
        px = cal.proyectar(np.array([p]), pose, modelo)[0]
        q = cal.pixel_a_mesa(px, pose, modelo, z=p[2])
        assert np.allclose(q, p, atol=1e-3)
        o, d = cal.rayo_desde_pixel(px, pose, modelo)
        assert cal.distancia_rayo_punto(o, d, np.array(p)) < 1e-3


def test_modelo_aproximado_y_escalado():
    m = cal.modelo_aproximado(1280, 720)
    assert m.K[0, 0] == 1280 and m.K[0, 2] == 640 and not m.dist.any()
    e = s.camara_ejemplo(1920, 1080).escalar_a(960, 540)
    assert np.isclose(e.K[0, 2], s.camara_ejemplo(1920, 1080).K[0, 2] / 2)
    d = cal.ModeloCamara.desde_dict(e.a_dict())
    assert np.allclose(d.K, e.K) and np.allclose(d.dist, e.dist)


def test_corregir_distorsion_endereza_rectas():
    modelo = s.camara_ejemplo()
    img, _ = s.escena_cancha(modelo, ruido=0)
    rect, m2 = cal.corregir_distorsion(img, modelo)
    assert rect.shape == img.shape and not m2.dist.any()
    # En la imagen rectificada, la pose con dist = 0 debe ser casi perfecta
    pose = cal.estimar_pose_cancha(cal.detectar_marcadores(rect), m2)
    assert pose.error_rms < 1.0
    assert cfg.esquinas_marcador(0).shape == (4, 3)
