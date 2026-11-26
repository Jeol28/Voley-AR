"""Bucle del juego en tiempo real (ventana de OpenCV), modo pantalla (v2).

Una cámara web y dos jugadores lado a lado frente a ella (o uno contra la IA).
La mitad izquierda de la imagen es del jugador 1, la derecha del jugador 2, y la
cancha, la red y el balón se dibujan encima. Golpea cualquier cosa que se mueva
(una mano, un cuaderno…); lo quieto es entorno.

Se juega como cualquier juego: un menú de inicio (contra la IA, 2 jugadores,
ajustes), la partida, la pausa (ESC) y la revancha. No hay que calibrar la piel:
el juego la aprende solo de las manos que reconoce.
Todo se puede elegir con el mouse o con las teclas que aparecen en los botones.

Atajos durante la partida:
    ESC      pausa (seguir, reiniciar, menú, salir)
    V / B    siguiente / anterior tema: cómo te ve el juego, en vivo (calibración, filtros,
             histogramas, umbralización, morfología, segmentación, caracterización, transformers)
    [ / ]    cambiar el parámetro del tema que se está viendo
    C        con calibración / sin calibración
    O        detector de la mano: clásico / OWLv2

Modo sintético (sin cámara): el mouse mueve la mano del jugador 1, 0-5 cambia
sus dedos y W/S la acerca o la aleja; en 2 jugadores, I/J/K/L mueve la mano del
jugador 2 y N cambia sus dedos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import config as cfgc
from voleyar.core import pipeline as pl
from voleyar.juego import render_ar as ra
from voleyar.juego.asincrono import TrabajadorAsincrono
from voleyar.juego.ia_rival import IARival
from voleyar.juego.partida import JUGADOR, RIVAL, EntradaJugador, Partida
from voleyar.utils import visualizacion as vis
from voleyar.utils.io import cargar_modelo_camara

VENTANA = "Voley AR"
QUIEN = {-1: JUGADOR, 1: RIVAL}


@dataclass
class OpcionesJuego:
    camara: int = 0
    ancho: int = 1280
    alto: int = 720
    modo: str = "ia"  # "ia" (tú contra la IA) o "2j" (dos jugadores)
    sintetico: bool = False
    profundidad: bool = True
    cada_n: int = 8
    tolerancia_distancia: float = 0.25  # fracción de la distancia de referencia de cada jugador
    dificultad: str = "normal"
    calibrado: bool = True
    puntos: int = 15
    headless: bool = False
    max_frames: int | None = None
    semilla: int | None = None
    menu: bool = False  # empezar en el menú de inicio (ventana)
    espejo: bool = True  # ver la cámara como un espejo (se ignora con la fuente sintética)
    interfaz: str = "ventana"  # "ventana" (teclas y mouse) o "web" (botones de la página)


DIFICULTADES = ("facil", "normal", "dificil")
NOMBRE_DIFICULTAD = {"facil": "fácil", "normal": "normal", "dificil": "difícil"}
PUNTOS = (7, 11, 15)


@dataclass
class EstadoJugador:
    piel: object = None  # histogramas de la piel, aprendidos solos de la mano confirmada
    n_aprendizajes: int = 0
    z_ref: float | None = None  # distancia de referencia (cm) al plano de juego
    muestras: list = field(default_factory=list)
    delante: bool | None = None  # Depth Anything: mano delante de la cara
    valida: bool | None = None
    contacto: object = None  # movimiento.Toque del último frame
    tipo: str | None = None  # tipo de golpe que daría ahora (caracterización)
    cercania: float | None = None  # Depth Anything: cuánto más cerca está lo que toca el balón
    lejos: bool = False  # lo que toca el balón está detrás (no cuenta)
    z_hist: list = field(default_factory=list)  # (t, Z cm) de la mano: su velocidad hacia la cámara
    vz: float | None = None  # cm/s (negativo = hacia la cámara)
    potencia: float = 0.0  # filtros: qué tan borroso (rápido) viene lo que toca el balón
    anchos: list = field(default_factory=list)  # anchos de borde recientes en su mitad: lo "normal" para él
    c_prev: np.ndarray | None = None  # centro (px) de lo que tocó el balón en el frame anterior
    t_prev: float = 0.0


class Juego:
    def __init__(self, op: OpcionesJuego, fuente=None, modelo_camara=None):
        self.op = op
        self.lados = (-1,) if op.modo == "ia" else (-1, 1)
        if fuente is None:
            if op.sintetico:
                from voleyar.utils.sinteticos import FuentePantalla, camara_ejemplo

                fuente = FuentePantalla(camara_ejemplo(op.ancho, op.alto), n_jugadores=2)
                fuente.jugadores[1].mano_visible = op.modo == "2j"
            else:
                from voleyar.utils.camara import FuenteCamara

                fuente = FuenteCamara(op.camara, op.ancho, op.alto)
        self.fuente = fuente
        self.sintetico = hasattr(fuente, "jugadores")
        if modelo_camara is not None:  # p. ej. la calibración de la sesión en la UI
            self.modelo_camara = modelo_camara
        else:
            self.modelo_camara = fuente.modelo if self.sintetico else cargar_modelo_camara()
        if self.modelo_camara is None:
            print("⚠  No hay config/camara.json: se juega SIN calibración. "
                  "Calibra con `python scripts/calibrar_camara.py` o desde la UI.")
        self.config = pl.ConfigPipeline(usar_calibracion=op.calibrado and self.modelo_camara is not None,
                                        espejo=op.espejo and not self.sintetico,
                                        seguimiento=True,  # presencia estable de cada mano en el tiempo
                                        golpe_por_movimiento=True)  # y cualquier objeto que se mueva golpea
        self.estado_pipe = pl.EstadoPipeline()
        self.jugadores = {lado: EstadoJugador() for lado in self.lados}
        self.partida = self._crear_partida()
        self.fase = "jugando"  # se juega de una: la piel se aprende sola
        self.pantalla = "menu" if op.menu else "juego"  # menu, ajustes o juego
        self.pausado = False
        self.botones: list = []
        self.t_cuenta: float | None = None
        self.vista = "juego"  # o un tema del curso (ver voleyar/juego/vistas.py)
        self.usar_owl = False
        self.ultimo_prof = None  # último resultado de Depth Anything
        self.ultimo_owl = None  # último resultado de OWLv2
        self._t_modelos = 0.0
        self.n_frame = 0
        self.fps = 0.0
        self.ultimo_res: pl.ResultadoFrame | None = None
        self.prof = TrabajadorAsincrono("profundidad") if op.profundidad else None
        self.owl = None
        self._t_prev = time.perf_counter()
        self.eventos_totales: list[str] = []
        self.lineas_tema: list[str] = []  # datos del tema que se está viendo (para el panel)
        self.graficas_tema: list = []
        # Lo que cada tema cambió en la partida (se muestra en su lente)
        self.stats = {"remates": 0, "lejos": 0, "cara": 0, "rapidos": 0, "tipos": {}}

    # compatibilidad con la UI: piel del jugador 1
    @property
    def piel(self):
        return self.jugadores[-1].piel

    # ------------------------------------------------------------ utilidades
    def _modo_texto(self) -> str:
        return "" if self.config.usar_calibracion and self.modelo_camara is not None else "cámara sin calibrar"

    def _tarea_profundidad(self, img):
        from voleyar.core import transformers_models as tm

        return tm.estimar_profundidad(img)

    def _tarea_owl(self, img):
        from voleyar.core import transformers_models as tm

        return tm.detectar_manos(img, ("a hand",), 0.15, None)

    def _validar(self, lado: int, r: pl.ResultadoMano) -> bool | None:
        """¿La mano está en el plano de juego? Compara su distancia (tamaño de la palma + K)
        con la que se midió para ese jugador en los primeros segundos de la partida."""
        j = self.jugadores[lado]
        if r.distancia_cm is None:
            return None
        if j.z_ref is None:
            j.muestras.append(r.distancia_cm)
            if len(j.muestras) >= 15:
                j.z_ref = float(np.median(j.muestras))
            return None
        tol = max(10.0, self.op.tolerancia_distancia * j.z_ref)
        en_plano = abs(r.distancia_cm - j.z_ref) <= tol
        return en_plano and j.delante is not False

    def _cara_de(self, res: pl.ResultadoFrame, lado: int):
        w = res.imagen.shape[1]
        caras = [c for c in res.caras if (c[0] + c[2] / 2 < w / 2) == (lado < 0)]
        return caras[0] if caras else None

    # ------------------------------------------------------------ partidas
    def nueva_partida(self, modo: str | None = None, dificultad: str | None = None, puntos: int | None = None) -> None:
        """Empieza una partida nueva (mismo modo o uno distinto). Si los jugadores ya
        calibraron su piel, se conserva y se juega directamente."""
        if modo is not None:
            self.op.modo = modo
        if dificultad is not None:
            self.op.dificultad = dificultad
        if puntos is not None:
            self.op.puntos = puntos
        self.lados = (-1,) if self.op.modo == "ia" else (-1, 1)
        viejos = getattr(self, "jugadores", {})
        self.jugadores = {lado: EstadoJugador(piel=viejos[lado].piel if lado in viejos else None) for lado in self.lados}
        self.partida = self._crear_partida()
        if self.sintetico and len(self.fuente.jugadores) > 1:
            self.fuente.jugadores[1].mano_visible = self.op.modo == "2j"
        self.fase = "jugando"
        self.pantalla, self.pausado, self.t_cuenta = "juego", False, None

    def _crear_partida(self) -> Partida:
        partida = Partida(IARival.con_dificultad(self.op.dificultad, self.op.semilla),
                          puntos_para_ganar=self.op.puntos, modo=self.op.modo)
        if self.op.interfaz == "web":
            # En línea cada frame viaja al servidor y vuelve (200-400 ms): balón más lento y más alcance
            partida.p.gravedad = 60.0
            partida.alcance = 3.4
            partida.ia.velocidad *= 0.8
        return partida

    def accion(self, accion: str) -> bool:
        """Ejecuta una opción de menú. Devuelve False para salir del juego."""
        if accion == "salir":
            return False
        if accion in ("ia", "2j"):
            self.nueva_partida(accion)
        elif accion == "ajustes":
            self.pantalla = "ajustes"
        elif accion == "menu":
            self.pantalla, self.pausado = "menu", False
        elif accion == "dificultad":
            i = DIFICULTADES.index(self.op.dificultad)
            self.op.dificultad = DIFICULTADES[(i + 1) % len(DIFICULTADES)]
        elif accion == "puntos":
            self.op.puntos = PUNTOS[(PUNTOS.index(self.op.puntos) + 1) % len(PUNTOS)] if self.op.puntos in PUNTOS else 15
        elif accion in ("vista_sig", "vista_ant"):
            from voleyar.juego.vistas import VISTAS

            i = VISTAS.index(self.vista) + (1 if accion == "vista_sig" else -1)
            self.vista = VISTAS[i % len(VISTAS)]
        elif accion.startswith("vista:"):
            self.vista = accion.split(":", 1)[1]
        elif accion in ("param_mas", "param_menos"):
            from voleyar.juego.vistas import cambiar_parametro

            cambiar_parametro(self, self.vista, 1 if accion == "param_mas" else -1)
        elif accion == "calibracion" and self.modelo_camara is not None:
            self.config.usar_calibracion = not self.config.usar_calibracion
            self.estado_pipe = pl.EstadoPipeline()
            for j in self.jugadores.values():  # la distancia de referencia cambia con la K
                j.z_ref, j.muestras = None, []
        elif accion == "espejo":
            self.config.espejo = not self.config.espejo
            self.estado_pipe = pl.EstadoPipeline()
        elif accion == "pausa":
            self.pausado = not self.pausado
        elif accion == "seguir":
            self.pausado = False
        elif accion == "reiniciar":
            self.partida.reiniciar()
            self.pausado = False
        elif accion == "revancha":
            self.nueva_partida()
        elif accion == "owl":
            self.usar_owl = not self.usar_owl
        return True

    def _botones_actuales(self) -> list:
        """Botones del menú visible ahora (o lista vacía si se está jugando)."""
        from voleyar.juego.menus import Boton

        if self.pantalla == "menu":
            return [Boton("Contra la IA", "ia", "1", principal=True), Boton("2 jugadores", "2j", "2", principal=True),
                    Boton("Ajustes", "ajustes", "a"), Boton("Salir", "salir", "q")]
        if self.pantalla == "ajustes":
            cal = ("sí" if self.config.usar_calibracion else "no") if self.modelo_camara is not None else "sin calibrar"
            return [Boton(f"Dificultad de la IA: {NOMBRE_DIFICULTAD[self.op.dificultad]}", "dificultad", "d"),
                    Boton(f"Partida a {self.op.puntos} puntos", "puntos", "p"),
                    Boton(f"Usar la calibración: {cal}", "calibracion", "c"),
                    Boton(f"Cámara como espejo: {'sí' if self.config.espejo else 'no'}", "espejo", "e"),
                    Boton("Volver", "menu", "m", principal=True)]
        if self.op.interfaz == "web":
            return []
        if self.partida.estado == "fin":
            return [Boton("Revancha", "revancha", "r", principal=True), Boton("Menú principal", "menu", "m"),
                    Boton("Salir", "salir", "q")]
        if self.pausado:
            return [Boton("Seguir", "seguir", "s", principal=True),
                    Boton("Reiniciar la partida", "reiniciar", "r"), Boton("Menú principal", "menu", "m"),
                    Boton("Salir", "salir", "q")]
        return []

    def _dibujar_menu(self, img: np.ndarray) -> np.ndarray:
        from voleyar.juego.menus import dibujar_cartel, dibujar_menu

        self.botones = self._botones_actuales()
        m, n = self.partida.marcador, self.partida.nombres
        marcador = f"{n[JUGADOR]} {m[JUGADOR]} – {m[RIVAL]} {n[RIVAL]}"
        if self.pantalla == "menu":
            cam = ("cámara calibrada" if self.modelo_camara is not None
                   else "cámara sin calibrar (python scripts/calibrar_camara.py)")
            return dibujar_menu(img, "Voley AR", self.botones, "Voleibol con tus manos, sin nada más que la cámara",
                                f"Uno a cada lado de la imagen · {cam}")
        if self.pantalla == "ajustes":
            return dibujar_menu(img, "Ajustes", self.botones)
        if self.partida.estado == "fin":
            if self.op.interfaz == "web":
                return dibujar_cartel(img, self.partida.mensaje, marcador)
            return dibujar_menu(img, self.partida.mensaje, self.botones, marcador, oscuro=0.5)
        if self.pausado:
            if self.op.interfaz == "web":
                return dibujar_cartel(img, "Pausa", "Pulsa Seguir para continuar")
            return dibujar_menu(img, "Pausa", self.botones, oscuro=0.5)
        return img

    # ------------------------------------------------------------ un frame
    def paso(self) -> np.ndarray | None:
        ok, frame = self.fuente.leer()
        if not ok or frame is None:
            return None
        ahora = time.perf_counter()
        dt = ahora - self._t_prev
        self._t_prev = ahora
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3)) if self.n_frame else 0.0
        self.n_frame += 1
        if self.pantalla in ("menu", "ajustes"):
            fondo, _ = pl.preparar_imagen(frame, self.modelo_camara, self.config)
            return self._dibujar_menu(fondo)

        pieles = {lado: self.jugadores[lado].piel for lado in self.lados}
        res = pl.procesar_frame(frame, self.modelo_camara, pieles, self.config, self.estado_pipe, ahora,
                                lados=self.lados)
        self.ultimo_res = res
        img = res.imagen.copy()
        lineas_tema: list[str] = []
        if self.vista != "juego":  # el mundo del juego visto a través del tema elegido
            from voleyar.juego.vistas import lente

            img, lineas_tema = lente(self, res, img, self.vista)

        # --- profundidad (hilo aparte, cada N frames) -----------------------------
        mapa = self._modelos(res, ahora)
        if mapa is not None:
            from voleyar.core import transformers_models as tm

            for lado in self.lados:
                r, cara = res.manos[lado], self._cara_de(res, lado)
                d = tm.mano_delante_de_la_cara(mapa.mapa, r.mascara_mano, cara) if cara is not None else None
                self.jugadores[lado].delante = None if d is None else d.delante

        # --- entradas -------------------------------------------------------------------
        entradas: dict[int, EntradaJugador | None] = {}
        for lado in self.lados:
            r = res.manos[lado]
            j = self.jugadores[lado]
            contacto, vel_objeto = self._contacto(lado, res, ahora)
            tipo = j.tipo or "golpe"
            if r.hay_mano:
                j.valida = self._validar(lado, r)
                gesto, fuerza = r.gesto  # suavizado en el tiempo
                # si lo que toca es algo en movimiento, vale aunque la mano esté fuera del plano
                entradas[lado] = EntradaJugador(r.rayo, r.punto_plano, r.velocidad, fuerza, gesto,
                                                None if contacto.toca else j.valida, toca=contacto.toca, tipo=tipo,
                                                potencia=j.potencia)
            elif contacto.toca:  # no es una mano, pero algo que no es el entorno toca el balón
                j.valida = None
                entradas[lado] = EntradaJugador(None, None, vel_objeto, 0.8, "objeto", None, toca=True, tipo=tipo,
                                                potencia=j.potencia)
            else:
                entradas[lado] = None
        if res.mov is not None and res.mov.descartados_cara:
            self.stats["cara"] += 1  # frames en los que la cara se movió y no se contó
        if self.usar_owl:
            entradas = self._entradas_owl(res, entradas)
        self._aprender_piel(res)

        # --- partida y render ---------------------------------------------------------------
        if not self.pausado:
            n_marcas = len(self.partida.marcas)
            ev = self.partida.actualizar(dt, entradas.get(-1), entradas.get(1))
            self.eventos_totales += ev
            if len(self.partida.marcas) > n_marcas and getattr(self.partida, "ultimo_tipo", None) \
                    and self.partida.marcas[-1].valido:
                t = self.partida.ultimo_tipo["tipo"]
                self.stats["tipos"][t] = self.stats["tipos"].get(t, 0) + 1
                if t == "remate":
                    self.stats["remates"] += 1
                if self.partida.ultimo_tipo.get("potencia", 0.0) >= 0.5:
                    self.stats["rapidos"] += 1
        ia_xy = self.partida.ia.pos if self.op.modo == "ia" else None
        img = ra.dibujar_escena_pantalla(img, res.pose, res.modelo, self.partida.balon.pos, ia_xy,
                                         self.partida.ia.altura_golpe, self.partida.marcas, self.partida.t,
                                         giro=int(self.partida.t * 8))
        mensaje = "" if self.partida.estado == "fin" else self.partida.mensaje
        m = self.partida.marcador
        n = self.partida.nombres
        ra.dibujar_hud(img, (m[JUGADOR], m[RIVAL]), mensaje, self._modo_texto(), self.fps, "", None, None,
                       "OWLv2" if self.usar_owl else "clásico", (n[JUGADOR], n[RIVAL]))
        self._estado_jugadores(img, res)
        self.lineas_tema = lineas_tema
        if self.vista != "juego" and self.op.interfaz != "web":
            # En la ventana, los datos del tema van en una franja DEBAJO del video, no encima de la cancha
            from voleyar.juego.vistas import panel

            franja = panel(self, self.vista, lineas_tema, img.shape[1], int(img.shape[0] * 0.3))
            if franja is not None:
                img = np.vstack([img, franja])
        img = self._dibujar_menu(img)
        return img

    def _modelos(self, res: pl.ResultadoFrame, ahora: float):
        """Depth Anything y OWLv2 sin frenar el juego.

        En escritorio corren en hilos (profundidad cada N frames; OWLv2 si se juega con él
        o se mira la vista de transformers). En la web no hay hilos: solo se llaman cuando
        se mira esa vista, como mucho cada 2 s (en ZeroGPU cada llamada pide la GPU).
        """
        mira = self.vista == "transformers"
        if self.op.interfaz == "web" and self.prof is None:
            if mira and ahora - self._t_modelos > 2.0:
                from voleyar.core import transformers_models as tm

                self._t_modelos = ahora
                self.ultimo_prof = tm.estimar_profundidad(res.imagen)
                self.ultimo_owl = tm.detectar_manos(res.imagen, ("a hand",), 0.15, None)
            return self.ultimo_prof if mira else None
        mapa = None
        if self.prof is not None:
            if (self.n_frame % self.op.cada_n == 0 or mira) and not self.prof.ocupado:
                self.prof.enviar(self._tarea_profundidad, res.imagen.copy())
            if self.prof.resultado is not None and self.prof.edad < 1.0:
                mapa = self.prof.resultado
                self.ultimo_prof = mapa
        elif mira and ahora - self._t_modelos > 2.0:  # sin hilo de profundidad: una llamada cada 2 s
            from voleyar.core import transformers_models as tm

            self._t_modelos = ahora
            self.ultimo_prof = tm.estimar_profundidad(res.imagen)
        if mira and not self.usar_owl:
            if self.owl is None:
                self.owl = TrabajadorAsincrono("owl")
            if not self.owl.ocupado:
                self.owl.enviar(self._tarea_owl, res.imagen.copy())
        if self.owl is not None and self.owl.resultado is not None:
            self.ultimo_owl = self.owl.resultado
        return mapa

    def _aprender_piel(self, res: pl.ResultadoFrame) -> None:
        """La piel se aprende sola: histogramas H-S y Cr-Cb de la mano ya confirmada.

        Sin modelo, la piel sale del rango fijo de YCrCb. En cuanto una mano pasa la
        validación de forma y la histéresis, sus píxeles construyen el histograma
        (y a partir de ahí se usa retroproyección + Otsu). Se refresca cada
        ``CADA`` frames mezclando con el anterior para seguir los cambios de luz
        sin que un frame malo lo eche a perder.
        """
        CADA, PESO_NUEVO = 30, 0.3
        from voleyar.core import histogramas as his

        for lado in self.lados:
            r, j = res.manos[lado], self.jugadores[lado]
            if not r.hay_mano or r.presente is not True or r.mascara_mano is None:
                continue
            if j.piel is not None and self.n_frame % CADA:
                continue
            try:
                nueva = his.calibrar_piel(res.suavizada, mascara=r.mascara_mano)
            except ValueError:
                continue
            if j.piel is not None:
                nueva = his.ModeloPiel(cv2.addWeighted(j.piel.hist_hs, 1 - PESO_NUEVO, nueva.hist_hs, PESO_NUEVO, 0),
                                       cv2.addWeighted(j.piel.hist_crcb, 1 - PESO_NUEVO, nueva.hist_crcb, PESO_NUEVO, 0),
                                       nueva.n_pixeles)
            j.piel = nueva
            j.n_aprendizajes += 1

    def _velocidad_z(self, lado: int, r: pl.ResultadoMano, ahora: float) -> float | None:
        """Calibración en juego: velocidad de la mano hacia la cámara (Z = f·W/w, en cm/s)."""
        j = self.jugadores[lado]
        if r.hay_mano and r.distancia_cm:
            j.z_hist.append((ahora, float(r.distancia_cm)))
        else:
            j.z_hist.clear()
        j.z_hist[:] = [(t, z) for t, z in j.z_hist if ahora - t <= 0.3]
        if len(j.z_hist) >= 3 and j.z_hist[-1][0] - j.z_hist[0][0] >= 0.08:
            t, z = np.array(j.z_hist).T
            j.vz = float(np.polyfit(t - t[0], z, 1)[0])
        else:
            j.vz = None
        return j.vz

    def _contacto(self, lado: int, res: pl.ResultadoFrame, ahora: float):
        """¿Qué objeto en movimiento (no el entorno, no la cara) toca el balón en la mitad de este jugador?

        Devuelve el toque y la velocidad del objeto en unidades de cancha (por el
        desplazamiento de su centroide entre frames); deja el tipo de golpe en el jugador.
        """
        from voleyar.core import caracterizacion as car
        from voleyar.core import movimiento as movi

        j, r = self.jugadores[lado], res.manos[lado]
        pos = self.partida.balon.pos
        c = cal.proyectar(np.asarray(pos)[None], res.pose, res.modelo)[0]
        r_px = ra.radio_en_pixeles(pos, res.pose, res.modelo)
        w = res.imagen.shape[1]
        if (c[0] < w / 2) != (lado < 0):  # el balón se ve en la otra mitad
            toque = movi.Toque(False, motivo="balón en la otra mitad")
        else:
            toque = movi.toque_en_balon(res.mov, c, r_px)
        # Transformers: lo que está detrás de la escena (alguien que pasa al fondo) no golpea
        j.cercania, lejos = None, False
        mapa = self._mapa_reciente()
        if toque.objeto is not None and mapa is not None and self.config.cercania_min > 0:
            j.cercania = movi.cercania(mapa, res.mov, toque.objeto)
            if toque.toca and j.cercania is not None and j.cercania < self.config.cercania_min:
                lejos = True
                toque = movi.Toque(False, toque.fraccion, toque.objeto, "está lejos, detrás de la escena")
        if lejos and not j.lejos:
            self.stats["lejos"] += 1
        j.lejos = lejos
        # Velocidad del objeto en la imagen → unidades de cancha
        vel = np.zeros(2)
        if toque.objeto is not None:
            cen = toque.objeto.centroide
            if j.c_prev is not None and 0 < ahora - j.t_prev < 0.3:
                d = (cen - j.c_prev) / (ahora - j.t_prev) * (cfgc.RADIO_BALON / max(r_px, 1.0))
                vel = np.clip(np.array([d[0], -d[1]]), -150.0, 150.0)  # en la imagen y crece hacia abajo
            j.c_prev, j.t_prev = cen, ahora
        else:
            j.c_prev = None
        # Caracterización: ¿es la mano reconocida o un objeto? ¿qué golpe es?
        vz = self._velocidad_z(lado, r, ahora)
        es_mano = r.hay_mano and (toque.objeto is None or self._mano_en(r, toque.objeto))
        if es_mano:
            n = r.dedos_estables if r.dedos_estables is not None else r.analisis.n_dedos
            j.tipo = car.tipo_de_golpe(int(n), vz, remate_vz=self.config.remate_vz)
        elif toque.objeto is not None:
            o = toque.objeto
            j.tipo = car.tipo_de_golpe(area_rel=o.area / (np.pi * r_px ** 2), solidez=o.solidez)
        else:
            j.tipo = None
        # Filtros: lo borroso viene rápido y el golpe sale más fuerte
        self._aprender_anchos(lado, res)
        j.potencia = 0.0
        if toque.objeto is not None:
            j.potencia = float(np.clip(self.borrosidad(lado, toque.objeto) * self.config.potencia_borroso, 0.0, 1.0))
        j.contacto = toque
        return toque, vel

    def _aprender_anchos(self, lado: int, res: pl.ResultadoFrame) -> None:
        """Guarda los anchos de borde de lo que se mueve en la mitad del jugador (los últimos ~5 s)."""
        if res.mov is None:
            return
        w = res.imagen.shape[1]
        j = self.jugadores[lado]
        for o in res.mov.objetos:
            if o.ancho_borde > 0 and (o.centroide[0] < w / 2) == (lado < 0):
                j.anchos.append(o.ancho_borde)
        del j.anchos[:-150]

    def borrosidad(self, lado: int, obj) -> float:
        """Qué tan borroso está ``obj`` comparado con lo normal para este jugador (0 = como siempre, 1 = 2.5 veces).

        Cada cámara, luz y mano tienen bordes más o menos suaves; por eso no se usa un
        ancho fijo sino la mediana de lo que se ha visto en su mitad en los últimos segundos.
        """
        j = self.jugadores[lado]
        if len(j.anchos) < 20 or obj.ancho_borde <= 0:
            return 0.0
        base = max(float(np.median(j.anchos)), 1.0)
        return float(np.clip((obj.ancho_borde / base - 1.5) / 1.0, 0.0, 1.0))  # 1.5× lo normal empieza, 2.5× es el máximo

    @staticmethod
    def _mano_en(r: pl.ResultadoMano, obj) -> bool:
        """¿La mano reconocida es (parte de) ese objeto en movimiento? Sus cajas se cruzan."""
        if r.analisis is None:
            return False
        hx, hy, hw, hh = cv2.boundingRect(r.analisis.contorno)
        ox, oy, ow, oh = obj.caja
        return hx < ox + ow and ox < hx + hw and hy < oy + oh and oy < hy + hh

    def _mapa_reciente(self):
        """Último mapa de Depth Anything si tiene menos de 1.5 s (si no, no se filtra por profundidad)."""
        if self.config.cercania_min <= 0:
            return None
        if self.prof is not None:
            if self.prof.resultado is not None and self.prof.edad < 1.5:
                return self.prof.resultado.mapa
            return None
        if self.ultimo_prof is not None and time.perf_counter() - self._t_modelos < 1.5:
            return self.ultimo_prof.mapa
        return None

    def _estado_jugadores(self, img: np.ndarray, res: pl.ResultadoFrame) -> None:
        """Gesto, distancia y validez de cada mano, en la parte alta de su mitad."""
        h, w = img.shape[:2]
        tam = max(15, h // 34)
        for lado in self.lados:
            r, j = res.manos[lado], self.jugadores[lado]
            if not r.hay_mano:
                texto = "golpea con la mano o con lo que quieras"
            else:  # el golpe que daría ahora su mano
                from voleyar.core.caracterizacion import TIPOS_GOLPE, tipo_de_golpe

                n = r.dedos_estables if r.dedos_estables is not None else r.analisis.n_dedos
                texto = "tu golpe: " + TIPOS_GOLPE[j.tipo if j.tipo else tipo_de_golpe(int(n))].lower()
            x = 16 if lado < 0 else w // 2 + 16
            vis.poner_texto(img, texto, (x + 22, int(h * 0.12)), tam, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            color = vis.BGR_NEUTRO if j.valida is None else (vis.BGR_VERDE if j.valida else vis.BGR_ROJO)
            cv2.circle(img, (x + 8, int(h * 0.12) + tam // 2 + 2), 8, color, -1, cv2.LINE_AA)

    def _entradas_owl(self, res, entradas):
        """Con OWLv2 el punto de golpe de cada mitad es el centro de la caja detectada en ella."""
        if self.owl is None:
            self.owl = TrabajadorAsincrono("owl")
        if not self.owl.ocupado:
            self.owl.enviar(self._tarea_owl, res.imagen.copy())
        r = self.owl.resultado
        if r is None or self.owl.edad > 1.5:
            return {lado: None for lado in self.lados}
        w = res.imagen.shape[1]
        nuevas = {}
        for lado in self.lados:
            dets = [d for d in r.detecciones if (d.centro[0] < w / 2) == (lado < 0)]
            if not dets:
                nuevas[lado] = None
                continue
            c = max(dets, key=lambda d: d.puntaje).centro
            prev = entradas.get(lado)
            nuevas[lado] = EntradaJugador(cal.rayo_desde_pixel(c, res.pose, res.modelo),
                                          cal.pixel_a_plano(c, res.pose, res.modelo, 1, 0.0),
                                          prev.velocidad if prev else np.zeros(2), prev.fuerza if prev else 0.8,
                                          "OWLv2", prev.valida if prev else None)
        return nuevas

    # ------------------------------------------------------------ teclado/mouse
    def tecla(self, k: int) -> bool:
        """Procesa una tecla. Devuelve False para salir."""
        from voleyar.juego.menus import boton_por_tecla

        if k == 255:  # ninguna tecla
            return True
        botones = self._botones_actuales()
        if botones:  # hay un menú en pantalla: sus teclas mandan
            accion = boton_por_tecla(botones, k)
            if accion is not None:
                return self.accion(accion)
            if k == 27:
                return self.accion("seguir" if self.pausado else ("menu" if self.pantalla == "ajustes" else "salir"))
            return True
        if k == 27:
            return self.accion("pausa")
        acciones = {ord("v"): "vista_sig", ord("d"): "vista_sig", ord("b"): "vista_ant", ord("]"): "param_mas",
                    ord("["): "param_menos", ord("c"): "calibracion", ord("o"): "owl", ord("p"): "pausa",
                    ord("r"): "reiniciar", ord("q"): "pausa", 9: "vista_sig"}
        if k in acciones:
            return self.accion(acciones[k])
        if self.sintetico:
            self._tecla_sintetica(k)
        return True

    def _tecla_sintetica(self, k: int) -> None:
        j1 = self.fuente.jugadores[0]
        if ord("0") <= k <= ord("5"):
            j1.n_dedos = k - ord("0")
        elif k == ord("w"):
            j1.distancia = max(30.0, j1.distancia - 5)
        elif k == ord("s"):
            j1.distancia = min(120.0, j1.distancia + 5)
        if len(self.fuente.jugadores) > 1:
            j2 = self.fuente.jugadores[1]
            mov = {ord("i"): (0, -3), ord("k"): (0, 3), ord("j"): (-3, 0), ord("l"): (3, 0)}
            if k in mov:
                j2.mano = (j2.mano[0] + mov[k][0], j2.mano[1] + mov[k][1])
            elif k == ord("n"):
                j2.n_dedos = (j2.n_dedos + 1) % 6

    def _mouse(self, evento, x, y, flags, param) -> None:
        from voleyar.juego.menus import boton_en

        if evento == cv2.EVENT_LBUTTONUP and self.botones:
            accion = boton_en(self.botones, x, y)
            if accion is not None:
                self._salir = not self.accion(accion)
            return
        if not self.sintetico or self.ultimo_res is None or self.botones:
            return
        res = self.ultimo_res
        if x >= res.imagen.shape[1]:
            return
        K = res.modelo.K
        j1 = self.fuente.jugadores[0]
        j1.mano = ((x - K[0, 2]) / K[0, 0] * j1.distancia, (y - K[1, 2]) / K[1, 1] * j1.distancia)

    # ------------------------------------------------------------ bucle
    def ejecutar(self) -> dict:
        if not self.op.headless:
            cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(VENTANA, self._mouse)
        try:
            while True:
                img = self.paso()
                if img is None:
                    break
                if self.op.headless:
                    if self.op.max_frames and self.n_frame >= self.op.max_frames:
                        break
                    continue
                cv2.imshow(VENTANA, img)
                if not self.tecla(cv2.waitKey(1) & 0xFF) or getattr(self, "_salir", False):
                    break
                if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.fuente.liberar()
            for t in (self.prof, self.owl):
                if t is not None:
                    t.detener()
            if not self.op.headless:
                cv2.destroyAllWindows()
        return {"frames": self.n_frame, "marcador": dict(self.partida.marcador), "eventos": self.eventos_totales}
