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


def test_la_dilatacion_agranda_la_zona_de_golpe():
    # un objeto que pasa justo al lado del balón, sin tocarlo
    frames = _aprendido() + [(_con_bloque(_cuarto(), 100 + 4 * k, 120, lado=50), ()) for k in range(4)]
    # el objeto termina en x = 162; el disco del balón (radio 12 · 1.35 ≈ 16) empieza ~10 px más allá
    balon, radio = np.array([162.0 + 10 + 16.2, 145.0]), 12.0
    normal, _ = _analizar(frames)
    assert not movi.toque_en_balon(normal, balon, radio).toca
    gigante, _ = _analizar(frames, mov_alcance=14)
    t = movi.toque_en_balon(gigante, balon, radio)
    assert t.toca and t.objeto is not None
    assert np.count_nonzero(gigante.zona_golpe) > np.count_nonzero(gigante.etiquetas)


_TEXTURA = np.random.default_rng(9).integers(0, 256, (12, 12, 3)).astype(np.uint8)


def _con_objeto_texturado(img, x, y, lado=70):
    """Un objeto con dibujo (como una mano o un cuaderno con letras): tiene esquinas que seguir."""
    out = img.copy()
    tex = cv2.resize(_TEXTURA, (lado, lado), interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(out.shape[1], x + lado), min(out.shape[0], y + lado)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = tex[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


@pytest.mark.parametrize("paso_px", [3, 12, 30])
def test_los_puntos_seguidos_miden_cuanto_se_movio_el_objeto(paso_px):
    frames = _aprendido() + [(_con_objeto_texturado(_cuarto(), 100 + paso_px * k, 120), ()) for k in range(5)]
    res, _ = _analizar(frames)
    o = max(res.objetos, key=lambda o: o.area)
    esperado = paso_px / W  # fracción del ancho de la imagen por frame
    assert o.puntos is not None and abs(o.desplazamiento - esperado) < max(0.35 * esperado, 0.003)


def test_un_golpe_rapido_sale_mas_fuerte():
    p = Partida(modo="2j")
    p.balon.pos = np.array([-6.0, 0.0, 6.0])
    normal = p.velocidad_golpe(EntradaJugador(tipo="golpe", potencia=0.0), JUGADOR)
    fuerte = p.velocidad_golpe(EntradaJugador(tipo="golpe", potencia=1.0), JUGADOR)
    assert fuerte[0] > normal[0] and fuerte[2] <= normal[2]  # más rápido hacia el rival y más bajo
    bloqueo = [p.velocidad_golpe(EntradaJugador(tipo="bloqueo", potencia=x), JUGADOR) for x in (0.0, 1.0)]
    assert np.allclose(*bloqueo)  # el bloqueo no cambia con la velocidad


@pytest.mark.parametrize("paso_px,rapido", [(2, False), (35, True)])
def test_en_la_partida_un_golpe_rapido_sale_rapidisimo(paso_px, rapido):
    from voleyar.juego.navegador import paso_navegador

    j = _juego()
    for _ in range(25):
        paso_navegador(j, _cuarto())
    c, r = _balon(j)
    x0 = int(c[0] - 35 - paso_px * 6)
    for k in range(12):  # el objeto llega al balón moviéndose paso_px por frame
        paso_navegador(j, _con_objeto_texturado(_cuarto(), x0 + paso_px * k, int(c[1] - 35)))
        if "golpe" in j.eventos_totales:
            break
    assert "golpe" in j.eventos_totales
    assert ("rapidísimo" in j.partida.marcas[-1].texto) == rapido


def test_el_juego_aguanta_que_el_navegador_cambie_la_resolucion_del_video():
    """El navegador baja la resolución si la conexión va lenta (p. ej. 640×360 → 452×254) mientras el mapa
    de profundidad del hilo aún es del tamaño anterior: no debe romperse nada."""
    from types import SimpleNamespace

    from voleyar.juego.navegador import paso_navegador
    from voleyar.utils import sinteticos as s

    j = _juego("2j")
    fuente = s.FuentePantalla(s.camara_ejemplo(W, H))
    mapa_viejo = np.random.default_rng(0).random((H, W)).astype(np.float32)
    j.prof = SimpleNamespace(resultado=SimpleNamespace(mapa=mapa_viejo, ms=30.0, dispositivo="cpu"), edad=0.1,
                             ocupado=True, enviar=lambda *a: None)
    for vista in ("juego", "transformers", "filtros"):
        j.vista = vista
        for k in range(30):
            f = fuente.leer()[1]
            if k % 10 >= 5:  # a ratos llega más pequeño
                f = cv2.resize(f, (452, 254), interpolation=cv2.INTER_AREA)
            assert paso_navegador(j, f) is not None


def test_mano_delante_de_la_cara_con_un_mapa_de_otro_tamano():
    from voleyar.core import transformers_models as tm

    mapa = np.full((H, W), 0.2, np.float32)
    mapa[200:300, 100:200] = 0.9  # la mano, cerca
    mano = np.zeros((254, 452), np.uint8)
    e = 452 / W
    mano[int(200 * e):int(300 * e), int(100 * e):int(200 * e)] = 255
    r = tm.mano_delante_de_la_cara(mapa, mano, (int(120 * e), 10, 60, 60))
    assert r is not None and r.delante


# ------------------------------------------------------------------------- histogramas: color del balón
def _cuarto_de_color(bgr):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = bgr
    return cv2.add(img, np.random.default_rng(0).integers(0, 20, (H, W, 3), dtype=np.uint8))


@pytest.mark.parametrize("pared,no_puede", [((40, 110, 230), "naranja"),  # pared naranja
                                            ((200, 110, 30), "azul"),  # pared azul
                                            ((30, 200, 230), "amarillo")])  # pared amarilla
def test_el_balon_toma_el_color_que_menos_hay_en_el_cuarto(pared, no_puede):
    from voleyar.core import histogramas as his

    img = _cuarto_de_color((120, 120, 120))
    img[:, : W * 2 // 3] = _cuarto_de_color(pared)[:, : W * 2 // 3]  # dos tercios del cuarto de ese color
    hist = his.histograma_tono(img)
    elegido, masas = his.color_de_contraste(hist)
    assert elegido != no_puede and masas[no_puede] > 0.3 and masas[elegido] < 0.01


def test_un_cuarto_gris_no_compite_con_ningun_color():
    from voleyar.core import histogramas as his

    hist = his.histograma_tono(_cuarto_de_color((128, 128, 128)))
    assert max(his.color_de_contraste(hist)[1].values()) < 0.01


def test_en_la_partida_el_balon_cambia_de_color_sin_parpadear():
    from voleyar.juego import render_ar as ra
    from voleyar.juego.navegador import paso_navegador

    j = _juego()
    naranja = cv2.resize(_cuarto_de_color((40, 110, 230)), (W, H))
    paso_navegador(j, naranja)
    assert j.color_balon != "naranja"  # en un cuarto naranja, el balón deja de ser naranja
    elegido = j.color_balon
    for _ in range(40):  # el cuarto sigue igual: el color no cambia
        paso_navegador(j, naranja)
    assert j.color_balon == elegido
    j.color_modo = "rosa"  # o se elige a mano en la lente (−/+)
    paso_navegador(j, naranja)
    assert j.color_balon == "rosa"
    s = ra.sprite_balon(20, 0, (175, 60, 255))
    assert s.shape == (43, 43, 4) and not np.array_equal(s, ra.sprite_balon(20, 0))


@pytest.mark.parametrize("nombre", ["naranja", "azul", "cian", "rosa"])
def test_el_balon_es_todo_de_su_color(nombre):
    from voleyar.core.histogramas import COLORES_BALON
    from voleyar.juego import render_ar as ra

    tono, bgr = COLORES_BALON[nombre]
    s = ra.sprite_balon(40, 3, bgr)
    dentro = s[..., 3] > 200
    hsv = cv2.cvtColor(np.ascontiguousarray(s[..., :3]), cv2.COLOR_BGR2HSV)
    brillantes = dentro & (hsv[..., 2] > 60)  # (las costuras y la sombra son oscuras: sin tono fiable)
    d = np.abs(hsv[..., 0].astype(int) - tono)
    assert (hsv[..., 1][brillantes] > 90).mean() > 0.97  # nada de blanco: todo tiene color
    assert (np.minimum(d, 180 - d)[brillantes] <= 12).mean() > 0.95  # y es el color elegido
