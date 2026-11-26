"""Pruebas del juego: física del balón, trayectorias, reglas, IA y bucle sin ventana."""

import numpy as np
import pytest

from voleyar.core import calibracion as cal
from voleyar.core import config as cfg
from voleyar.core import pipeline as pl
from voleyar.juego import render_ar as ra
from voleyar.juego.fisica import Balon, ParamsFisica, paso, prediccion, velocidad_para_objetivo
from voleyar.juego.ia_rival import IARival
from voleyar.juego.partida import EntradaJugador, Partida
from voleyar.utils import sinteticos as s


def test_rebote_en_la_mesa_conserva_fraccion_de_energia():
    p = ParamsFisica()
    b = Balon(np.array([-5.0, 0.0, 10.0]), np.zeros(3))
    eventos = []
    for _ in range(800):  # un subpaso por llamada: la velocidad se lee justo tras el rebote
        eventos += paso(b, p.sub_paso, p)
        if eventos:
            break
    assert eventos[0].tipo == "bote" and eventos[0].lado == -1 and eventos[0].dentro
    v_impacto = np.sqrt(2 * p.gravedad * (10.0 - p.radio))
    assert b.vel[2] == pytest.approx(p.restitucion_mesa * v_impacto, rel=0.05)


def test_la_red_detiene_un_balon_bajo_y_deja_pasar_uno_alto():
    p = ParamsFisica()
    bajo = Balon(np.array([-3.0, 0.0, 2.5]), np.array([40.0, 0.0, 0.0]))
    ev = []
    for _ in range(30):
        ev += paso(bajo, 1 / 60, p)
    assert any(e.tipo == "red" for e in ev) and bajo.pos[0] < 0
    alto = Balon(np.array([-3.0, 0.0, 9.0]), np.array([40.0, 0.0, 0.0]))
    ev = []
    for _ in range(10):
        ev += paso(alto, 1 / 60, p)
    assert any(e.tipo == "cruza_red" for e in ev) and alto.pos[0] > 0


def test_fuera_de_la_cancha():
    b = Balon(np.array([10.0, 0.0, 3.0]), np.array([30.0, 0.0, 0.0]))
    ev = []
    for _ in range(60):
        ev += paso(b, 1 / 60)
    botes = [e for e in ev if e.tipo == "bote"]
    assert botes and not botes[0].dentro


@pytest.mark.parametrize("objetivo,apice", [((6.0, 2.0), 8.0), ((10.0, -5.0), 14.0), ((3.0, 7.0), 7.0)])
def test_trayectoria_cae_en_el_objetivo_y_pasa_la_red(objetivo, apice):
    p = ParamsFisica()
    b = Balon(np.array([-6.0, 0.0, 2.0]), np.zeros(3))
    b.vel = velocidad_para_objetivo(b.pos, np.array(objetivo), apice, p)

    def parar(t, bb, ev):
        return any(e.tipo in ("bote", "red") for e in ev)

    tray = prediccion(b, p, parar_en=parar)
    final = tray[-1][1]
    assert np.linalg.norm(final[:2] - np.array(objetivo)) < 0.8
    altura_en_red = min(tray, key=lambda tp: abs(tp[1][0]))[1][2]
    assert altura_en_red > cfg.ALTURA_RED + p.radio


def test_golpe_del_jugador_envia_el_balon_al_rival():
    partida = Partida(IARival.con_dificultad("normal", semilla=0))
    b = partida.balon.pos.copy()
    o = np.array([30.0, 0.0, 40.0])
    d = (b - o) / np.linalg.norm(b - o)  # rayo que pasa justo por el balón
    ev = partida.actualizar(1 / 30, EntradaJugador(rayo=(o, d), velocidad=np.array([20.0, 0]), fuerza=1.0,
                                                   gesto="remate", cerca_mesa=True))
    assert "golpe" in ev and partida.estado == "juego" and partida.balon.vel[0] > 0
    # Simular: debe cruzar la red
    for _ in range(90):
        partida.actualizar(1 / 30, None)
        if partida.balon.pos[0] > 0:
            break
    assert partida.balon.pos[0] > 0


def test_golpe_con_mano_levantada_no_vale():
    partida = Partida()
    b = partida.balon.pos.copy()
    ev = partida.actualizar(1 / 30, EntradaJugador(punto_cm=b.copy(), cerca_mesa=False))
    assert ev == ["golpe_invalido"] and partida.estado == "saque"


def test_dos_botes_es_punto_y_se_llega_a_15():
    partida = Partida(puntos_para_ganar=2)
    # El rival saca y nadie devuelve: el balón bota dos veces de tu lado
    partida.sacador = "rival"
    partida._preparar_saque()
    for _ in range(600):
        partida.actualizar(1 / 30, None)
        if partida.estado in ("punto", "fin"):
            break
    assert partida.marcador["rival"] == 1
    for _ in range(600):
        partida.actualizar(1 / 30, None)
        if partida.estado == "fin":
            break
    assert partida.estado == "fin" and partida.ganador == "rival"


def test_ia_devuelve_un_balon_facil():
    p = ParamsFisica()
    ia = IARival.con_dificultad("dificil", semilla=3)
    ia.prob_fallo = 0.0
    b = Balon(np.array([-5.0, 0.0, 3.0]), np.zeros(3))
    b.vel = velocidad_para_objetivo(b.pos, np.array([7.0, 0.0]), 10.0, p)
    ia.nueva_jugada()
    devuelto = None
    for _ in range(240):
        paso(b, 1 / 60, p)
        v = ia.actualizar(b, 1 / 60, p, b.pos[0] > 0)
        if v is not None:
            devuelto = v
            break
    assert devuelto is not None and devuelto[0] < 0


def test_render_ar_dibuja_sin_errores():
    img, v = s.escena_cancha()
    out = ra.dibujar_escena(img, v["pose"], v["modelo"], np.array([-4.0, 1.0, 5.0]), np.array([7.0, 0.0]), lineas=True)
    assert out.shape == img.shape and np.any(out != img)
    assert ra.radio_en_pixeles(np.array([0, 0, 2.0]), v["pose"], v["modelo"]) > 5
    assert ra.sprite_balon(12).shape == (27, 27, 4)


def test_pipeline_en_escena_sintetica_encuentra_la_mano_y_su_posicion():
    modelo = s.camara_ejemplo(1280, 720)
    pose = s.pose_juego_por_defecto()
    cfgp = pl.ConfigPipeline()
    mano = s.EstadoManoSintetica(pos=(-6.0, 1.0), n_dedos=5, altura=0.3)
    img, _ = s.escena_cancha(modelo, pose, mano)
    px = cal.proyectar(np.array([[-6.0, 1.0, 0.3]]), pose, modelo)[0] * 960 / 1280
    piel = pl.calibrar_piel_en_frame(img, (int(px[0]) - 12, int(px[1]) - 12, 24, 24), modelo, cfgp)
    r = pl.procesar_frame(img, modelo, piel, cfgp)
    assert r.hay_mano and r.analisis.n_dedos == 5 and r.analisis.gesto == "remate"
    # Mano abierta: el punto de golpe es la palma → cerca de (-6, 1) en la mesa
    assert np.linalg.norm(r.punto_golpe_cm[:2] - np.array([-6.0, 1.0])) < 1.5


def test_juego_sin_ventana_con_fuente_sintetica():
    from voleyar.juego.bucle import Juego, OpcionesJuego

    j = Juego(OpcionesJuego(sintetico=True, profundidad=False, headless=True, semilla=1, ancho=960, alto=540))
    for _ in range(260):
        b = j.partida.balon.pos
        if j.fase == "jugando" and b[0] < 0:
            j.fuente.mano.pos = (float(b[0]) - 1.0, float(b[1]))
        assert j.paso() is not None
    assert j.fase == "jugando" and j.piel is not None
    assert j.eventos_totales.count("golpe") >= 2
    j.depurar = True
    assert j.paso().shape[1] > 960  # panel de depuración a la derecha
