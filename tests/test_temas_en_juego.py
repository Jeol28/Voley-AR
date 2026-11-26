"""Cada tema del curso cambia algo en la partida (no solo se muestra en su lente).

Escenas sintéticas con verdad conocida: un cuarto fijo, objetos que se mueven,
ruido de cámara, cambios de luz, una cara y un mapa de profundidad hecho a mano.
"""

import numpy as np
import pytest

import cv2

from voleyar.core import caracterizacion as car
from voleyar.core import movimiento as movi
from voleyar.core import pipeline as pl
from voleyar.juego.partida import JUGADOR, EntradaJugador, Partida

H, W = 360, 640


def _cuarto(semilla=0, ruido=0.0, rng=None):
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (120, 90, 60)
    tex = np.random.default_rng(semilla).integers(0, 25, (H, W, 3)).astype(np.float32)
    img = base + tex
    if ruido:
        img += (rng or np.random.default_rng()).normal(0, ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _con_bloque(img, x, y, lado=60, color=(40, 200, 60)):
    out = img.copy()
    out[y:y + lado, x:x + lado] = color
    return out


def _analizar(frames, **ajustes):
    cfg = pl.ConfigPipeline(**ajustes)
    a = movi.AnalizadorMovimiento()
    res = None
    for f, caras in frames:
        res = a.aplicar(f, cfg, caras)
    return res, a


def _aprendido(n=25, **k):
    return [(_cuarto(**k), ()) for _ in range(n)]


# ------------------------------------------------------------------ segmentación (MOG2) y lo básico
def test_un_objeto_que_se_mueve_es_un_objeto_y_lo_quieto_no():
    frames = _aprendido() + [(_con_bloque(_cuarto(), 100 + 6 * k, 120), ()) for k in range(4)]
    res, _ = _analizar(frames)
    assert len(res.objetos) == 1 and not res.cambio_luz
    o = res.objetos[0]
    assert abs(o.centroide[0] - (100 + 18 + 30)) < 12 and o.solidez > 0.8
    quieto = [(_con_bloque(_cuarto(), 100, 120), ()) for _ in range(150)]  # el mismo objeto sin moverse
    res, _ = _analizar(_aprendido() + quieto)
    assert res.objetos == []  # pasó a ser entorno


# ------------------------------------------------------------------------------------- filtros
def test_el_suavizado_quita_el_ruido_que_parece_movimiento():
    rng = np.random.default_rng(3)
    frames = [(_cuarto(ruido=14, rng=rng), ()) for _ in range(40)]
    con, _ = _analizar(frames, suavizado="gaussiano")
    sin, _ = _analizar(frames, suavizado="ninguno")
    assert sin.puntos_sueltos > con.puntos_sueltos
    assert np.count_nonzero(sin.cruda) > np.count_nonzero(con.cruda)


# ---------------------------------------------------------------------------------- histogramas
def test_un_cambio_de_luz_se_detecta_con_el_histograma_y_se_reaprende_el_fondo():
    fondo = _cuarto()
    brillo = cv2.convertScaleAbs(fondo, alpha=1.5, beta=30)
    res, a = _analizar(_aprendido() + [(brillo, ())])
    assert res.cambio_luz and res.distancia_luz > 0.3 and a.cambios_luz == 1 and res.objetos == []
    t = movi.toque_en_balon(res, np.array([300.0, 180.0]), 40)
    assert not t.toca and t.motivo == "cambio de luz"
    # Con un umbral muy alto el mismo cambio no se reconoce como luz y todo «se mueve»
    res, a = _analizar(_aprendido() + [(brillo, ())], umbral_luz=1.0)  # 1 = histogramas sin nada en común
    assert not res.cambio_luz and a.cambios_luz == 0


# -------------------------------------------------------------------------------- umbralización
def test_otsu_adapta_el_umbral_y_el_minimo_evita_partir_el_ruido():
    frames = _aprendido() + [(_con_bloque(_cuarto(), 100 + 6 * k, 120, color=(180, 160, 120)), ()) for k in range(4)]
    res, _ = _analizar(frames)
    assert res.T == max(res.T_otsu, 18) and len(res.objetos) == 1
    alto, _ = _analizar(frames, mov_t_min=120)  # un mínimo demasiado alto: el objeto ya no se ve
    assert alto.T == 120 and alto.objetos == []
    quieto, _ = _analizar(_aprendido(40))  # sin nada que se mueva, Otsu partiría el ruido: manda el mínimo
    assert quieto.T >= 18 and quieto.objetos == []


# ---------------------------------------------------------------------------------- morfología
def test_la_apertura_quita_el_ruido_suelto():
    rng = np.random.default_rng(5)
    frames = _aprendido()
    for _ in range(3):  # puntitos sueltos que cambian (polvo, compresión del video)
        f = _cuarto()
        for y, x in zip(rng.integers(0, H - 4, 300), rng.integers(0, W - 4, 300)):
            f[y:y + 4, x:x + 4] = 255  # 2 × 2 px en la imagen reducida
        frames.append((f, ()))
    # sin suavizado, para ver solo el efecto de la morfología
    con, _ = _analizar(frames, mov_apertura=3, mov_area_min=0.0, suavizado="ninguno")
    sin, _ = _analizar(frames, mov_apertura=1, mov_area_min=0.0, suavizado="ninguno")
    assert len(sin.objetos) > 100 and len(con.objetos) < len(sin.objetos) / 3  # la mayoría desaparece


# ------------------------------------------------------------------------- segmentación: la cara
def test_lo_que_se_mueve_en_la_cara_no_es_un_objeto_para_golpear():
    cara = (80, 60, 120, 120)  # recuadro de Viola-Jones
    frames = _aprendido() + [(_con_bloque(_cuarto(), 110 + 4 * k, 90, lado=50), (cara,)) for k in range(4)]
    res, _ = _analizar(frames)
    assert res.objetos == [] and len(res.descartados_cara) == 1
    res, _ = _analizar(frames, ignorar_caras_golpe=False)
    assert len(res.objetos) == 1


# ------------------------------------------------------------------------------- caracterización
def test_el_tipo_de_golpe_sale_de_la_forma():
    assert car.tipo_de_golpe(0) == "fuerte" and car.tipo_de_golpe(5) == "suave" and car.tipo_de_golpe(2) == "golpe"
    assert car.tipo_de_golpe(5, vz_cm=-90) == "remate"  # calibración: hacia la cámara, rápido
    assert car.tipo_de_golpe(5, vz_cm=-20) == "suave"
    assert car.tipo_de_golpe(area_rel=5.0, solidez=0.95) == "bloqueo"  # un cuaderno
    assert car.tipo_de_golpe(area_rel=5.0, solidez=0.5) == "golpe"  # grande pero con forma irregular
    assert car.tipo_de_golpe(area_rel=1.0, solidez=0.95) == "golpe"


@pytest.mark.parametrize("tipo", ["remate", "fuerte", "golpe", "suave", "bloqueo"])
def test_cada_golpe_pasa_la_red_y_cae_en_el_otro_campo(tipo):
    from voleyar.juego.fisica import paso

    p = Partida(modo="2j")
    p.balon.pos = np.array([-6.0, 0.0, 6.0])
    v = p.velocidad_golpe(EntradaJugador(fuerza=0.8, tipo=tipo), JUGADOR)
    from voleyar.juego.fisica import Balon

    b = Balon(p.balon.pos.copy(), v)
    for _ in range(2000):
        ev = paso(b, 1 / 240, p.p)
        if any(e.tipo == "bote" for e in ev):
            break
    assert b.pos[0] > 0  # pasó la red


def test_el_remate_va_mas_lejos_y_mas_bajo_que_el_toque_suave():
    p = Partida(modo="2j")
    p.balon.pos = np.array([-6.0, 0.0, 6.0])
    rem = p.velocidad_golpe(EntradaJugador(tipo="remate"), JUGADOR)
    suave = p.velocidad_golpe(EntradaJugador(tipo="suave"), JUGADOR)
    bloqueo = p.velocidad_golpe(EntradaJugador(tipo="bloqueo"), JUGADOR)
    assert rem[2] < suave[2]  # sube menos
    assert rem[0] > suave[0] > 0 and bloqueo[0] < rem[0]


# ----------------------------------------------------------------------------------- transformers
def test_la_cercania_dice_si_el_objeto_esta_delante_de_la_escena():
    frames = _aprendido() + [(_con_bloque(_cuarto(), 100 + 6 * k, 120), ()) for k in range(4)]
    res, _ = _analizar(frames)
    o = res.objetos[0]
    mapa = np.full((H, W), 0.2, np.float32)  # disparidad: más alto = más cerca
    mapa[:, W // 2:] = 0.4
    x, y, w, h = o.caja
    mapa[y:y + h, x:x + w] = 0.9  # el objeto está pegado a la cámara
    assert movi.cercania(mapa, res, o) > 0.5
    mapa[y:y + h, x:x + w] = 0.1  # el mismo objeto, al fondo del cuarto
    assert movi.cercania(mapa, res, o) < 0


# ------------------------------------------------------------------------------ en la partida real
def _juego(modo="ia"):
    from voleyar.juego.navegador import crear_juego_navegador
    from voleyar.utils import sinteticos as s

    j = crear_juego_navegador(s.camara_ejemplo(W, H), modo)
    j.config.espejo = False
    return j


def _balon(j):
    from voleyar.core import calibracion as cal
    from voleyar.juego import render_ar as ra

    res, pos = j.ultimo_res, j.partida.balon.pos
    return cal.proyectar(pos[None], res.pose, res.modelo)[0], ra.radio_en_pixeles(pos, res.pose, res.modelo)


def _golpear_con(j, lado_px, mapa=None):
    """Aprende el cuarto y mete un bloque de ``lado_px`` moviéndose sobre el balón. Devuelve los eventos."""
    from voleyar.juego.navegador import paso_navegador

    for _ in range(25):
        paso_navegador(j, _cuarto())
    if mapa is not None:
        j._mapa_reciente = lambda: mapa
    c, r = _balon(j)
    for k in range(12):
        x = int(c[0] - lado_px / 2 + (k % 3 - 1) * 4)
        y = int(c[1] - lado_px / 2)
        paso_navegador(j, _con_bloque(_cuarto(), max(0, x), max(0, y), lado_px))
        if "golpe" in j.eventos_totales:
            break
    return j.eventos_totales


def test_un_cuaderno_que_cubre_el_balon_es_un_bloqueo():
    j = _juego()
    c, r = (None, None)
    from voleyar.juego.navegador import paso_navegador

    paso_navegador(j, _cuarto())
    _, r = _balon(j)
    ev = _golpear_con(j, int(4.2 * r))  # bastante más grande que el balón y compacto
    assert "golpe" in ev and j.partida.marcas[-1].texto == "Bloqueo"
    assert j.stats["tipos"].get("bloqueo") == 1


def test_lo_que_esta_lejos_segun_depth_anything_no_golpea():
    j = _juego()
    mapa_lejos = np.full((H, W), 0.8, np.float32)  # toda la escena está más cerca que el objeto…
    from voleyar.juego.navegador import paso_navegador

    paso_navegador(j, _cuarto())
    c, r = _balon(j)
    x0, y0 = int(c[0] - 2 * r), int(c[1] - 2 * r)
    mapa_lejos[max(0, y0):y0 + int(4 * r), max(0, x0):x0 + int(4 * r)] = 0.1  # …que pasa por detrás
    ev = _golpear_con(j, int(1.6 * r), mapa_lejos)
    assert "golpe" not in ev and j.stats["lejos"] >= 1
    j2 = _juego()
    paso_navegador(j2, _cuarto())
    j2.config.cercania_min = 0.0  # sin el filtro de profundidad, el mismo objeto sí golpea
    assert "golpe" in _golpear_con(j2, int(1.6 * r), mapa_lejos)
