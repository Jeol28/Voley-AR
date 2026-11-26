"""Caracterización en juego: gestos de la mano para sacar, pausar, pedir revancha y cambiar de tema."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from voleyar.core import caracterizacion as car
from voleyar.juego.gestos import LectorGestos, clasificar
from voleyar.utils import sinteticos as s


@pytest.mark.parametrize("dedos,esperado", [(None, "abierta"), (("indice", "pulgar"), "pulgar_indice"),
                                            (("indice", "medio"), "dos_dedos"), (("indice",), "dedo_arriba"),
                                            ((), None)])
@pytest.mark.parametrize("angulo", [75, 90, 105])
def test_cada_gesto_se_reconoce_por_su_forma(dedos, esperado, angulo):
    m = s.silueta_mano_2d(5 if dedos is None else 0, angulo=angulo, dedos=dedos)
    assert clasificar(car.analizar_mano(m, (0.0, 1.0))) == esperado


def test_un_dedo_hacia_abajo_no_es_revancha():
    m = s.silueta_mano_2d(0, angulo=-90, dedos=("indice",))
    assert clasificar(car.analizar_mano(m, (0.0, -1.0))) is None


def _mano(dedos=None, centro=(100.0, 100.0)):
    a = car.analizar_mano(s.silueta_mano_2d(5 if dedos is None else 0, dedos=dedos), (0.0, 1.0))
    desplazada = SimpleNamespace(**{**a.__dict__, "centro_palma": centro,
                                    "puntas": [(p[0] - a.centro_palma[0] + centro[0], p[1] - a.centro_palma[1] + centro[1])
                                               for p in a.puntas]})
    return SimpleNamespace(analisis=desplazada, hay_mano=True, dedos_estables=a.n_dedos)


def test_el_gesto_se_activa_tras_un_segundo_quieto_y_una_sola_vez():
    g = LectorGestos(espera=1.0)
    todos = {"abierta", "pulgar_indice", "dedo_arriba", "dos_dedos"}
    mano = _mano()
    assert g.actualizar(-1, mano, 0.0, todos) == (None, None)
    disparado, prog = g.actualizar(-1, mano, 0.5, todos)
    assert disparado is None and prog.gesto == "abierta" and abs(prog.fraccion - 0.5) < 1e-6
    assert g.actualizar(-1, mano, 1.05, todos)[0] == "abierta"
    assert g.actualizar(-1, mano, 3.0, todos)[0] is None  # no se repite mientras siga igual
    for k in range(4):  # baja la mano (más que un frame suelto mal leído)…
        g.actualizar(-1, None, 3.1 + 0.01 * k, todos)
    g.actualizar(-1, mano, 3.2, todos)
    assert g.actualizar(-1, mano, 4.3, todos)[0] == "abierta"  # …y al volver a mostrarla, sí


def test_moverse_o_un_gesto_no_permitido_no_activa_nada():
    g = LectorGestos(espera=1.0)
    g.actualizar(-1, _mano(), 0.0, {"abierta"})
    g.actualizar(-1, _mano(centro=(160.0, 100.0)), 0.6, {"abierta"})  # la mano se movió: vuelve a empezar
    assert g.actualizar(-1, _mano(centro=(160.0, 100.0)), 1.2, {"abierta"})[0] is None
    g2 = LectorGestos(espera=1.0)
    g2.actualizar(-1, _mano(), 0.0, {"pulgar_indice"})  # sacar no está permitido ahora
    assert g2.actualizar(-1, _mano(), 2.0, {"pulgar_indice"}) == (None, None)


def _juego_con_gestos():
    from voleyar.juego.bucle import Juego, OpcionesJuego

    j = Juego(OpcionesJuego(modo="ia", sintetico=True, profundidad=False, headless=True, semilla=4, ancho=960, alto=540))
    jug = j.fuente.jugadores[0]
    jug.mano = (-12.0, 7.0)  # abajo a la izquierda, lejos del balón y de la cara
    return j, jug


def _sostener(j, jug, dedos, hasta, max_s=4.0):
    """Mueve un poco la mano (para que el juego la note) y luego la deja quieta con el gesto."""
    jug.dedos = None
    rng = np.random.default_rng(len(dedos or ()))
    for k in range(30):  # (los primeros frames el juego aprende el cuarto: hay que moverse después)
        jug.mano = (-12.0 + rng.uniform(-3, 3), 11.0 + rng.uniform(-2, 2))
        j.paso()
    jug.mano = (-12.0, 11.0)
    jug.dedos = dedos
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < max_s:
        j.paso()
        if hasta():
            return True
    return False


def test_en_la_partida_se_saca_y_se_pausa_con_gestos():
    j, jug = _juego_con_gestos()
    assert j.partida.estado == "saque" and "abre la mano" in j.partida.mensaje
    saques = []  # para comprobar que el saque fue por el gesto y no por tocar el balón sin querer
    original = j.partida.sacar
    j.partida.sacar = lambda quien: saques.append(quien) or original(quien)
    assert _sostener(j, jug, ("indice", "medio", "anular", "menique", "pulgar"), lambda: j.partida.estado == "juego")
    assert saques == ["jugador"] and "golpe" in j.eventos_totales  # el saque salió con la mano abierta
    assert _sostener(j, jug, ("indice", "pulgar"), lambda: j.pausado)  # la L pausa…
    assert _sostener(j, jug, ("indice", "medio"), lambda: j.vista != "juego")  # ✌ cambia el tema (también en pausa)
    assert j.vista == "calibracion"
    jug.dedos = None
    for _ in range(5):
        j.paso()
    assert _sostener(j, jug, ("indice", "pulgar"), lambda: not j.pausado)  # …y la L de nuevo sigue


def test_al_terminar_un_dedo_arriba_pide_la_revancha():
    j, jug = _juego_con_gestos()
    j.partida.marcador["rival"] = j.partida.puntos_para_ganar - 1
    j.partida._punto("rival", "prueba")
    assert j.partida.estado == "fin"
    assert _sostener(j, jug, ("indice",), lambda: j.partida.estado != "fin")
    assert j.partida.marcador == {"jugador": 0, "rival": 0}


def test_un_frame_mal_leido_no_reinicia_el_anillo():
    g = LectorGestos(espera=0.5)
    todos = {"abierta", "pulgar_indice", "dedo_arriba", "dos_dedos"}
    g.actualizar(-1, _mano(), 0.0, todos)
    g.actualizar(-1, _mano(), 0.2, todos)
    g.actualizar(-1, _mano(("indice", "medio")), 0.3, todos)  # un frame lo leyó como ✌
    assert g.actualizar(-1, _mano(), 0.55, todos)[0] == "abierta"  # y el anillo siguió llenándose


def test_si_se_pierde_un_dedo_el_conteo_por_anillo_lo_recupera():
    import cv2

    m = s.silueta_mano_2d(5, angulo=90, con_brazo=False)
    a = car.analizar_mano(m, (0.0, 1.0))
    assert a.n_anillo == 5 and len(a.angulos_anillo) == 5
    from types import SimpleNamespace

    perdida = SimpleNamespace(**{**a.__dict__, "n_dedos": 3})  # los valles del casco vieron solo 3
    assert clasificar(perdida, 3) == "abierta"
