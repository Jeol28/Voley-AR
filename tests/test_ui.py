"""Pruebas de la interfaz web sin navegador: la página del juego, el video por WebRTC (su manejador), las vistas
por tema y la calibración por video."""

import time

import numpy as np
import pytest

from voleyar.core import pipeline as pl
from voleyar.juego.vistas import PARAMETROS, VISTAS
from voleyar.ui import estado
from voleyar.ui import juego_web as jw
from voleyar.ui import sesiones
from voleyar.utils import sinteticos as s


def test_la_app_se_construye_sin_laboratorio():
    from voleyar.ui.app import crear_app

    assert crear_app() is not None
    assert [v for _, v in jw.OPCIONES_VISTA] == list(VISTAS)


CASOS = [(n, lado, d) for n, p in estado.EJEMPLOS.items() for lado, d in zip((-1, 1), p["dedos"])
         if n != "Mano delante de la cara"]  # con la mano sobre la cara los dedos se confunden con la cara


@pytest.mark.parametrize("nombre,lado,dedos", CASOS)
def test_pipeline_cuenta_bien_los_dedos_en_las_escenas_de_ejemplo(nombre, lado, dedos):
    r = estado.ejecutar_pipeline(estado.frame_ejemplo(nombre))
    m = r.manos[lado]
    assert m.analisis is not None and m.analisis.n_dedos == dedos


def test_calibracion_mejora_la_distancia_de_la_mano():
    fr = estado.frame_ejemplo("Dos jugadores (remate y toque)")
    con = estado.ejecutar_pipeline(fr, usar_calibracion=True)
    sin = estado.ejecutar_pipeline(fr, usar_calibracion=False)
    for lado, jug in zip((-1, 1), fr.verdad["jugadores"]):
        e_con = abs(con.manos[lado].distancia_cm - jug.distancia)
        e_sin = abs(sin.manos[lado].distancia_cm - jug.distancia)
        assert e_con < 0.12 * jug.distancia and e_sin > 2 * e_con


def _sesion_con_juego(modo="2j"):
    """Abre la página (clave de sesión), elige el modo y alimenta el manejador de video con frames sintéticos."""
    modelo = s.camara_ejemplo(640, 360)
    fuente = s.FuentePantalla(modelo)
    clave, _ = jw.abrir_pagina(None)
    sesiones.obtener(clave).estado.modelo_camara = modelo  # como si hubiera calibrado en esta sesión
    jw.guardar_ajustes(clave, "facil", 7, True, espejo=False)  # los frames sintéticos ya están sin espejo
    pantallas = jw.empezar(clave, modo)
    jw.frame_juego(fuente.leer()[1], clave)
    juego = sesiones.obtener(clave).juego  # se juega de una, sin calibrar la piel

    def jugar(n):
        out = None
        for _ in range(n):
            out = jw.frame_juego(fuente.leer()[1], clave)
        return out

    return clave, juego, pantallas, jugar


def test_calibracion_por_video_desde_el_juego(tmp_path, monkeypatch):
    # Nunca escribir en config/camara.json real: la calibración de prueba va a un archivo temporal
    from voleyar.ui import calibracion_web
    from voleyar.utils.io import guardar_modelo_camara

    monkeypatch.setattr(calibracion_web, "guardar_modelo_camara",
                        lambda modelo, ruta=None: guardar_modelo_camara(modelo, tmp_path / "camara.json"))
    clave, _ = jw.abrir_pagina(None)
    modelo = s.camara_ejemplo(960, 540)
    tex, ppcm, origen = s.textura_tablero(30.0)
    rng = np.random.default_rng(1)
    centro = np.array([10.0, 6.25, 0.0])
    for _ in range(40):  # el tablero, visto como en un video desde ángulos distintos
        if sesiones.obtener(clave).gestor is not None and len(sesiones.obtener(clave).gestor.vistas) >= 8:
            break
        az, elev, d = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(45, 80)), rng.uniform(32, 45)
        C = centro + d * np.array([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)])
        pose = s.pose_mirando(C, centro, rodar=rng.uniform(-20, 20))
        jw.frame_calibracion(s.renderizar_plano(tex, ppcm, origen, modelo, pose), clave)
    assert len(sesiones.obtener(clave).gestor.vistas) >= 6
    assert "Vistas" in jw.progreso_calibracion(clave)
    texto, detalles, grafica = jw.calibrar_amigable(clave, 2.5)
    assert "Calibración" in texto
    guardado, ruta, chip = jw.usar_calibracion(clave)
    assert sesiones.obtener(clave).estado.modelo_camara is not None and "calibrada" in chip


def test_juego_web_por_video():
    clave, juego, pantallas, jugar = _sesion_con_juego()
    assert juego.op.interfaz == "web" and juego.op.puntos == 7
    assert [p.visible for p in pantallas] == [p == "juego" for p in jw.PANTALLAS]
    out = jugar(5)
    assert juego.fase == "jugando" and juego.config.usar_calibracion and out.shape == (360, 640, 3)
    jw.accion("pausa")(clave)
    assert juego.pausado
    # Volver al menú y elegir otro modo conserva la sesión y su juego
    jw.volver_al_menu(clave)
    jw.empezar(clave, "ia")
    assert sesiones.obtener(clave).juego is juego and juego.op.modo == "ia" and juego.fase == "jugando"
    assert not juego.pausado


def test_cada_sesion_tiene_su_propio_juego():
    a, ja, _, _ = _sesion_con_juego()
    b, jb, _, _ = _sesion_con_juego()
    assert a != b and ja is not jb


@pytest.mark.parametrize("vista", [v for v in VISTAS if v not in ("juego", "transformers")])
def test_cada_tema_se_ve_en_la_partida_y_su_ajuste_cambia(vista):
    clave, juego, _, jugar = _sesion_con_juego()
    jugar(3)
    texto = jw.elegir_vista(clave, vista)
    out = jugar(1)
    assert out.shape == (360, 640, 3)  # el tema se ve dentro del juego, no en un panel aparte
    from voleyar.juego.vistas import lente

    res = juego.ultimo_res
    escena, lineas = lente(juego, res, res.imagen, vista)
    assert lineas and escena.shape == res.imagen.shape and not np.array_equal(escena, res.imagen)
    p = PARAMETROS[vista]
    antes = p.leer(juego)
    texto2 = jw.cambiar(1)(clave, vista)
    assert p.leer(juego) != antes and "cambia" in texto2
    jugar(1)  # el pipeline sigue funcionando con el ajuste nuevo
