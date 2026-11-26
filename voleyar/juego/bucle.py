"""Bucle del juego en tiempo real (ventana de OpenCV), modo pantalla (v2).

Una cámara web y dos jugadores lado a lado frente a ella (o uno contra la IA).
La mitad izquierda de la imagen es del jugador 1, la derecha del jugador 2, y la
cancha, la red y el balón se dibujan encima. Lo único físico son las manos.

Teclas:
    ESPACIO  capturar el color de la piel de cada jugador (al inicio)
    D        panel de depuración (máscaras, caras, contorno, casco, punta, distancias, FPS)
    C        alternar con calibración / sin calibración
    O        detector de la mano: clásico / OWLv2
    P        recalibrar la piel
    R        reiniciar la partida
    Q / ESC  salir

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


@dataclass
class EstadoJugador:
    piel: object = None
    z_ref: float | None = None  # distancia de referencia (cm) al plano de juego
    muestras: list = field(default_factory=list)
    delante: bool | None = None  # Depth Anything: mano delante de la cara
    valida: bool | None = None


class Juego:
    def __init__(self, op: OpcionesJuego, fuente=None, modelo_camara=None):
        self.op = op
        self.lados = (-1,) if op.modo == "ia" else (-1, 1)
        if fuente is None:
            if op.sintetico:
                from voleyar.utils.sinteticos import FuentePantalla, camara_ejemplo

                fuente = FuentePantalla(camara_ejemplo(op.ancho, op.alto), n_jugadores=len(self.lados))
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
        self.config = pl.ConfigPipeline(usar_calibracion=op.calibrado and self.modelo_camara is not None)
        self.estado_pipe = pl.EstadoPipeline()
        self.jugadores = {lado: EstadoJugador() for lado in self.lados}
        self.partida = Partida(IARival.con_dificultad(op.dificultad, op.semilla), puntos_para_ganar=op.puntos,
                               modo=op.modo)
        self.fase = "calibrar_piel"
        self.t_cuenta: float | None = None
        self.depurar = False
        self.usar_owl = False
        self.n_frame = 0
        self.fps = 0.0
        self.ultimo_res: pl.ResultadoFrame | None = None
        self.prof = TrabajadorAsincrono("profundidad") if op.profundidad else None
        self.owl = None
        self._t_prev = time.perf_counter()
        self.eventos_totales: list[str] = []

    # compatibilidad con la UI: piel del jugador 1
    @property
    def piel(self):
        return self.jugadores[-1].piel

    # ------------------------------------------------------------ utilidades
    def _modo_texto(self) -> str:
        return "con calibración" if self.config.usar_calibracion else "SIN calibración"

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

        pieles = {lado: self.jugadores[lado].piel for lado in self.lados}
        res = pl.procesar_frame(frame, self.modelo_camara, pieles, self.config, self.estado_pipe, ahora,
                                lados=self.lados)
        self.ultimo_res = res
        img = res.imagen.copy()
        if self.fase == "calibrar_piel":
            return self._paso_calibrar_piel(frame, res, img, ahora)

        # --- profundidad (hilo aparte, cada N frames) -----------------------------
        mapa = None
        if self.prof is not None:
            if self.n_frame % self.op.cada_n == 0 and not self.prof.ocupado:
                self.prof.enviar(self._tarea_profundidad, res.imagen.copy())
            if self.prof.resultado is not None and self.prof.edad < 1.0:
                mapa = self.prof.resultado
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
            if r.hay_mano:
                j.valida = self._validar(lado, r)
                entradas[lado] = EntradaJugador(r.rayo, r.punto_plano, r.velocidad, r.analisis.fuerza,
                                                r.analisis.gesto, j.valida)
            else:
                entradas[lado] = None
        if self.usar_owl:
            entradas = self._entradas_owl(res, entradas)

        # --- partida y render ---------------------------------------------------------------
        ev = self.partida.actualizar(dt, entradas.get(-1), entradas.get(1))
        self.eventos_totales += ev
        ia_xy = self.partida.ia.pos if self.op.modo == "ia" else None
        img = ra.dibujar_escena_pantalla(img, res.pose, res.modelo, self.partida.balon.pos, ia_xy,
                                         self.partida.ia.altura_golpe, self.partida.marcas, self.partida.t,
                                         giro=int(self.partida.t * 8))
        if self.depurar:
            img = pl.dibujar_depuracion(res, img)
        mensaje = self.partida.mensaje + (" · R para jugar otra vez" if self.partida.estado == "fin" else "")
        m = self.partida.marcador
        n = self.partida.nombres
        ra.dibujar_hud(img, (m[JUGADOR], m[RIVAL]), mensaje, self._modo_texto(), self.fps, "", None, None,
                       "OWLv2" if self.usar_owl else "clásico", (n[JUGADOR], n[RIVAL]))
        self._estado_jugadores(img, res)
        if self.depurar:
            img = self._panel_depuracion(img, res, mapa)
        return img

    def _estado_jugadores(self, img: np.ndarray, res: pl.ResultadoFrame) -> None:
        """Gesto, distancia y validez de cada mano, en la parte alta de su mitad."""
        h, w = img.shape[:2]
        tam = max(15, h // 34)
        for lado in self.lados:
            r, j = res.manos[lado], self.jugadores[lado]
            if not r.hay_mano:
                texto = "sin mano"
            else:
                z = f"{r.distancia_cm:.0f} cm" if r.distancia_cm else "—"
                ref = "midiendo…" if j.z_ref is None else f"plano a {j.z_ref:.0f} cm"
                texto = f"{r.analisis.gesto} · {z} ({ref})"
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

    def _paso_calibrar_piel(self, frame, res, img, ahora) -> np.ndarray:
        H, W = img.shape[:2]
        tam = max(18, H // 26)
        rects = {lado: pl.recuadro_piel(img.shape, lado) for lado in self.lados}
        if self.sintetico:  # las manos virtuales se colocan solas sobre su recuadro
            K = res.modelo.K
            for lado, (x, y, w, h) in rects.items():
                jug = self.fuente.jugadores[0 if lado < 0 else 1]
                u, v = x + w / 2, y + h / 2
                jug.mano = ((u - K[0, 2]) / K[0, 0] * jug.distancia, (v - K[1, 2]) / K[1, 1] * jug.distancia)
            if self.t_cuenta is None:
                self.t_cuenta = ahora + 0.5
        for lado, (x, y, w, h) in rects.items():
            cv2.rectangle(img, (x, y), (x + w, y + h), vis.BGR_NARANJA, 3, cv2.LINE_AA)
            nombre = self.partida.nombres[QUIEN[lado]]
            vis.poner_texto(img, nombre, (x, y + h + 8), tam - 2, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        if len(self.lados) == 2:
            cv2.line(img, (W // 2, 0), (W // 2, H), vis.BGR_NEUTRO, 1, cv2.LINE_AA)
        vis.poner_texto(img, "Calibración de la piel", (20, 18), tam + 6, vis.BGR_NEUTRO, True, fondo=vis.BGR_GRAFITO)
        quien = "Cada jugador cubre su recuadro" if len(self.lados) == 2 else "Cubre el recuadro"
        vis.poner_texto(img, f"{quien} con la palma, a la distancia a la que va a jugar, y presiona ESPACIO",
                        (20, 24 + tam + 14), tam, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        if self.t_cuenta is not None:
            falta = self.t_cuenta - ahora
            if falta <= 0:
                try:
                    for lado, rect in rects.items():
                        self.jugadores[lado].piel = pl.calibrar_piel_en_frame(frame, rect, self.modelo_camara,
                                                                              self.config)
                    self.fase = "jugando"
                    self.partida.reiniciar()
                except ValueError as e:
                    print("Calibración de piel fallida:", e)
                self.t_cuenta = None
            else:
                vis.poner_texto(img, f"{falta:.1f}", (W // 2 - 20, H // 2 - 30), tam + 24, vis.BGR_NARANJA, True,
                                fondo=vis.BGR_GRAFITO)
        return img

    def _panel_depuracion(self, img: np.ndarray, res: pl.ResultadoFrame, mapa) -> np.ndarray:
        H, W = img.shape[:2]
        ancho, alto = W // 3, H // 3

        def celda(im, titulo):
            c = np.full((alto, ancho, 3), vis.BGR_GRAFITO, np.uint8)
            if im is not None:
                im = vis.a_bgr(im)
                e = min(ancho / im.shape[1], alto / im.shape[0])
                im = cv2.resize(im, (max(1, int(im.shape[1] * e)), max(1, int(im.shape[0] * e))))
                y0, x0 = (alto - im.shape[0]) // 2, (ancho - im.shape[1]) // 2
                c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            vis.poner_texto(c, titulo, (8, 6), 15, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            return c

        retro = None
        capas = [r.retro for r in res.manos.values() if r.retro is not None]
        if capas:
            retro = cv2.applyColorMap(np.maximum.reduce(capas), cv2.COLORMAP_INFERNO)
        prof = None if mapa is None else cv2.applyColorMap(vis.normalizar_u8(mapa.mapa), cv2.COLORMAP_INFERNO)
        col = np.vstack([celda(pl.dibujar_mascaras(res), "Piel: mano (blanco), brazo (naranja), caras (rojo)"),
                         celda(retro, "Retroproyección de cada jugador"),
                         celda(prof, "Profundidad (Depth Anything V2)")])
        col = col[:H] if col.shape[0] >= H else np.vstack(
            [col, np.full((H - col.shape[0], ancho, 3), vis.BGR_GRAFITO, np.uint8)])
        lineas = [f"FPS {self.fps:.0f} · caras {len(res.caras)} · f = {res.modelo.K[0, 0]:.0f} px",
                  "ms: " + " ".join(f"{k[:6]} {v:.0f}" for k, v in res.tiempos_ms.items())]
        for lado in self.lados:
            r = res.manos[lado]
            if r.analisis is None:
                continue
            p, v = r.punto_plano, r.velocidad
            lineas.append(f"{pl.NOMBRES_LADO[lado]}: {r.analisis.n_dedos} dedos · Otsu T = {r.T_otsu} · "
                          f"Z = {r.distancia_cm:.0f} cm · ({p[0]:.1f}, {p[2]:.1f}) · v ({v[0]:.0f}, {v[1]:.0f})/s"
                          + (" · watershed" if r.uso_watershed else ""))
        y = H - 22 * len(lineas) - 64
        for linea in lineas:
            vis.poner_texto(img, linea, (14, y), 16, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            y += 22
        return np.hstack([img, col])

    # ------------------------------------------------------------ teclado/mouse
    def tecla(self, k: int) -> bool:
        """Procesa una tecla. Devuelve False para salir."""
        if k in (ord("q"), 27):
            return False
        if k == ord(" ") and self.fase == "calibrar_piel":
            self.t_cuenta = time.perf_counter() + 1.5
        elif k == ord("d"):
            self.depurar = not self.depurar
        elif k == ord("c") and self.modelo_camara is not None:
            self.config.usar_calibracion = not self.config.usar_calibracion
            self.estado_pipe = pl.EstadoPipeline()
            for j in self.jugadores.values():  # la distancia de referencia cambia con la K
                j.z_ref, j.muestras = None, []
        elif k == ord("o"):
            self.usar_owl = not self.usar_owl
        elif k == ord("p"):
            self.fase = "calibrar_piel"
            for j in self.jugadores.values():
                j.piel, j.z_ref, j.muestras = None, None, []
        elif k == ord("r"):
            self.partida.reiniciar()
        elif self.sintetico:
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
        if not self.sintetico or self.ultimo_res is None:
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
                if not self.tecla(cv2.waitKey(1) & 0xFF):
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
