"""Pruebas del juego: física del balón, trayectorias, reglas, IA y bucle sin ventana."""

import numpy as np
import pytest

import cv2

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


def _rayo_hacia(punto):
    o = np.array([0.0, -30.0, 6.0])  # cámara virtual delante de la red
    d = (punto - o) / np.linalg.norm(punto - o)
    return o, d


@pytest.mark.parametrize("quien,signo", [("jugador", 1), ("rival", -1)])
def test_golpe_humano_envia_el_balon_al_otro_lado(quien, signo):
    partida = Partida(modo="2j")
    if quien == "rival":
        partida.sacador = "rival"
        partida._preparar_saque()
    b = partida.balon.pos.copy()
    e = EntradaJugador(rayo=_rayo_hacia(b), velocidad=np.array([20.0 * signo, 0]), fuerza=1.0, gesto="remate",
                       valida=True)
    ev = partida.actualizar(1 / 30, e if quien == "jugador" else None, e if quien == "rival" else None)
    assert ev and partida.estado == "juego" and np.sign(partida.balon.vel[0]) == signo
    assert partida.balon.vel[1] == 0  # el balón se queda en el plano de la red
    for _ in range(120):
        partida.actualizar(1 / 30)
        if np.sign(partida.balon.pos[0]) == signo:
            break
    assert np.sign(partida.balon.pos[0]) == signo


def test_golpe_con_la_mano_fuera_del_plano_no_vale():
    partida = Partida()
    b = partida.balon.pos.copy()
    ev = partida.actualizar(1 / 30, EntradaJugador(punto=b.copy(), valida=False))
    assert ev == ["golpe_invalido"] and partida.estado == "saque"


def test_en_dos_jugadores_no_se_golpea_el_balon_del_otro_lado():
    partida = Partida(modo="2j")  # saca el jugador 1: el balón está a su lado
    b = partida.balon.pos.copy()
    assert partida.actualizar(1 / 30, None, EntradaJugador(punto=b.copy(), valida=True)) == []
    assert partida.nombres == {"jugador": "Jugador 1", "rival": "Jugador 2"}


def test_dos_botes_es_punto_y_se_llega_a_15():
    partida = Partida(IARival.con_dificultad("normal", semilla=0), puntos_para_ganar=2)
    partida.ia.error = 0.0  # saque sin error: siempre cae de tu lado
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


def test_render_pantalla_con_camara_virtual():
    modelo = cal.ModeloCamara(s.camara_ejemplo(960, 540).K, np.zeros(5), 960, 540)
    pose = cal.camara_virtual(modelo)
    img = np.full((540, 960, 3), 120, np.uint8)
    out = ra.dibujar_escena_pantalla(img, pose, modelo, np.array([-5.0, 0.0, 7.0]), np.array([7.0, 0.0]))
    assert np.any(out != img)
    red = cal.proyectar(np.array([[0.0, 0.0, 0.0]]), pose, modelo)[0]
    assert abs(red[0] - 480) < 1 and abs(red[1] - 0.86 * 540) < 1  # red al centro, piso al 86 %


def _palma_px(jug, modelo_eff):
    return cv2.projectPoints(np.array([[jug.mano[0], jug.mano[1], jug.distancia]]), np.zeros(3), np.zeros(3),
                             modelo_eff.K, modelo_eff.dist)[0].ravel()


def test_pipeline_pantalla_encuentra_las_dos_manos():
    modelo = s.camara_ejemplo(1280, 720)
    cfgp = pl.ConfigPipeline()
    jugadores = s.jugadores_por_defecto()
    img, verdad = s.escena_pantalla(modelo, jugadores)
    _, meff = pl.preparar_imagen(img, modelo, cfgp)
    pieles = {}
    for lado, jug in zip(pl.LADOS, jugadores):
        u, v = _palma_px(jug, meff)
        pieles[lado] = pl.calibrar_piel_en_frame(img, (int(u) - 10, int(v) - 10, 20, 20), modelo, cfgp)
    r = pl.procesar_frame(img, modelo, pieles, cfgp)
    for lado, jug in zip(pl.LADOS, jugadores):
        m = r.manos[lado]
        assert m.hay_mano and m.analisis.n_dedos == jug.n_dedos
        assert abs(m.distancia_cm - jug.distancia) / jug.distancia < 0.15
        assert np.sign(m.punto_plano[0]) == lado  # cada mano queda en su mitad de la cancha


def _jugar(j, n):
    for _ in range(n):
        res = j.ultimo_res
        if j.fase == "jugando" and res is not None:
            b = j.partida.balon.pos
            u, v = cal.proyectar(b[None], res.pose, res.modelo)[0]
            K = res.modelo.K
            for k, jug in enumerate(j.fuente.jugadores):
                lado = -1 if k == 0 else 1
                if b[0] * lado > -0.5:
                    x = (u - K[0, 2]) / K[0, 0] * jug.distancia
                    x = min(x, -6.0) if lado < 0 else max(x, 6.0)
                    jug.n_dedos = 5
                    jug.mano = (x, (v - K[1, 2]) / K[1, 1] * jug.distancia + 1.0)
        assert j.paso() is not None


@pytest.mark.parametrize("modo", ["ia", "2j"])
def test_juego_sin_ventana_con_fuente_sintetica(modo):
    from voleyar.juego.bucle import Juego, OpcionesJuego

    j = Juego(OpcionesJuego(modo=modo, sintetico=True, profundidad=False, headless=True, semilla=1,
                            ancho=960, alto=540))
    _jugar(j, 200)
    assert j.fase == "jugando" and all(x.piel is not None for x in j.jugadores.values())
    assert j.eventos_totales.count("golpe") >= 2
    if modo == "2j":
        assert j.eventos_totales.count("golpe_rival") >= 2  # el jugador 2 también devuelve
        assert all(x.z_ref is not None for x in j.jugadores.values())
    j.vista = "segmentacion"
    assert j.paso().shape[1] > 960  # el panel del tema aparece a la derecha


def test_juego_en_el_navegador_con_frames_externos():
    import time

    from voleyar.juego.navegador import FuenteExterna, crear_juego_navegador, paso_navegador

    modelo = s.camara_ejemplo(640, 360)
    j = crear_juego_navegador(modelo, "2j")
    assert isinstance(j.fuente, FuenteExterna)
    fuente = s.FuentePantalla(modelo)
    assert paso_navegador(j, fuente.leer()[1]) is not None and j.fase == "calibrar_piel"
    res = j.ultimo_res
    K = res.modelo.K
    for lado, jug in zip((-1, 1), fuente.jugadores):  # cada palma sobre su recuadro
        x, y, w, h = pl.recuadro_piel(res.imagen.shape, lado)
        jug.mano = ((x + w / 2 - K[0, 2]) / K[0, 0] * jug.distancia, (y + h / 2 - K[1, 2]) / K[1, 1] * jug.distancia)
    j.tecla(ord(" "))
    j.t_cuenta = time.perf_counter()
    out = paso_navegador(j, fuente.leer()[1])
    assert j.fase == "jugando" and out.shape[:2] == (360, 640)


def test_menus_del_juego_de_escritorio():
    from voleyar.juego.bucle import Juego, OpcionesJuego

    j = Juego(OpcionesJuego(sintetico=True, profundidad=False, headless=True, menu=True, ancho=960, alto=540))
    assert j.pantalla == "menu" and j.paso() is not None
    assert [b.accion for b in j.botones] == ["ia", "2j", "ajustes", "salir"]
    j.tecla(ord("a"))
    j.tecla(ord("d"))  # dificultad: normal → difícil
    j.tecla(ord("p"))  # puntos: 15 → 7
    j.tecla(ord("m"))
    assert j.op.dificultad == "dificil" and j.op.puntos == 7 and j.pantalla == "menu"
    j.tecla(ord("2"))
    assert j.pantalla == "juego" and j.op.modo == "2j" and j.partida.puntos_para_ganar == 7
    for _ in range(20):
        j.paso()
    j.tecla(27)  # ESC → pausa
    assert j.pausado and j.paso() is not None and [b.accion for b in j.botones][0] == "seguir"
    t = j.partida.t
    j.paso()
    assert j.partida.t == t  # en pausa la partida no avanza
    j.tecla(ord("s"))
    j.partida.marcador["jugador"] = 6
    j.partida._punto("jugador", "prueba")
    j.paso()
    assert j.partida.estado == "fin" and [b.accion for b in j.botones][0] == "revancha"
    # Clic con el mouse sobre "Revancha"
    x, y, w, h = j.botones[0].rect
    j._mouse(cv2.EVENT_LBUTTONUP, x + w // 2, y + h // 2, 0, None)
    assert j.partida.marcador == {"jugador": 0, "rival": 0} and j.fase == "jugando"
    assert j.tecla(27) is True and j.pausado  # ESC durante la partida pausa, no cierra el juego


def test_vistas_por_tema_en_la_ventana():
    from voleyar.juego.bucle import Juego, OpcionesJuego
    from voleyar.juego.vistas import VISTAS

    j = Juego(OpcionesJuego(modo="2j", sintetico=True, profundidad=False, headless=True, ancho=960, alto=540))
    for _ in range(25):
        j.paso()
    vistas = []
    for _ in VISTAS[1:-1]:  # sin transformers (carga modelos; se prueba en las pruebas lentas)
        j.tecla(ord("v"))
        vistas.append(j.vista)
        assert j.paso().shape[1] > 960
    assert vistas == list(VISTAS[1:-1])
    j.vista = "morfologia"
    cierre = j.config.tam_cierre
    j.tecla(ord("]"))
    assert j.config.tam_cierre != cierre
    j.tecla(ord("b"))
    assert j.vista == "umbralizacion"


@pytest.mark.lento
def test_vista_de_transformers_con_los_modelos():
    from voleyar.juego.bucle import Juego, OpcionesJuego

    j = Juego(OpcionesJuego(modo="2j", sintetico=True, profundidad=False, headless=True, ancho=960, alto=540,
                            interfaz="web"))
    for _ in range(25):
        j.paso()
    j.vista = "transformers"
    img = j.paso()
    assert j.ultimo_prof is not None and j.ultimo_owl is not None and img.shape[1] > 960
