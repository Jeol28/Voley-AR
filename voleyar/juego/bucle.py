"""Bucle del juego en tiempo real (ventana de OpenCV).

Teclas:
    ESPACIO  capturar el color de la piel (al inicio)
    D        panel de depuración (máscara, contorno, casco, punta, pose, FPS)
    C        alternar con calibración / sin calibración
    L        líneas virtuales de la cancha (para ver la alineación)
    O        detector de la mano: clásico / OWLv2
    P        recalibrar la piel
    R        reiniciar la partida
    Q / ESC  salir

Modo sintético (sin cámara): el mouse mueve la mano, 0-5 cambia los dedos y
W/S sube o baja la mano.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import pipeline as pl
from voleyar.juego import render_ar as ra
from voleyar.juego.asincrono import TrabajadorAsincrono
from voleyar.juego.ia_rival import IARival
from voleyar.juego.partida import EntradaJugador, Partida
from voleyar.utils import visualizacion as vis
from voleyar.utils.io import cargar_modelo_camara

VENTANA = "Voley AR"


@dataclass
class OpcionesJuego:
    camara: int = 0
    ancho: int = 1280
    alto: int = 720
    sintetico: bool = False
    profundidad: bool = True
    cada_n: int = 8
    umbral_profundidad: float = 0.30
    dificultad: str = "normal"
    calibrado: bool = True
    puntos: int = 15
    headless: bool = False
    max_frames: int | None = None
    semilla: int | None = None


class Juego:
    def __init__(self, op: OpcionesJuego, fuente=None):
        self.op = op
        if fuente is None:
            if op.sintetico:
                from voleyar.utils.sinteticos import FuenteSintetica, camara_ejemplo

                fuente = FuenteSintetica(camara_ejemplo(op.ancho, op.alto))
            else:
                from voleyar.utils.camara import FuenteCamara

                fuente = FuenteCamara(op.camara, op.ancho, op.alto)
        self.fuente = fuente
        self.sintetico = hasattr(fuente, "mano")
        # Modelo de cámara: config/camara.json, o el verdadero en modo sintético
        self.modelo_camara = fuente.modelo if self.sintetico else cargar_modelo_camara()
        if self.modelo_camara is None:
            print("⚠  No hay config/camara.json: se juega SIN calibración. "
                  "Calibra con `python scripts/calibrar_camara.py` o desde la UI.")
        self.config = pl.ConfigPipeline(usar_calibracion=op.calibrado and self.modelo_camara is not None)
        self.estado_pipe = pl.EstadoPipeline()
        self.piel = None
        self.partida = Partida(IARival.con_dificultad(op.dificultad, op.semilla), puntos_para_ganar=op.puntos)
        self.fase = "calibrar_piel"
        self.t_cuenta: float | None = None
        self.depurar = False
        self.lineas = False
        self.usar_owl = False
        self.n_frame = 0
        self.fps = 0.0
        self.ultimo_res: pl.ResultadoFrame | None = None
        self.prof = TrabajadorAsincrono("profundidad") if op.profundidad else None
        self.owl = None
        self._t_prev = time.perf_counter()
        self.eventos_totales: list[str] = []
        self._rect_piel: tuple[int, int, int, int] | None = None

    # ------------------------------------------------------------ utilidades
    def _modo_texto(self) -> str:
        return "con calibración" if self.config.usar_calibracion else "SIN calibración"

    def _rect_calibracion(self, res: pl.ResultadoFrame) -> tuple[int, int, int, int]:
        """Recuadro para la piel: sobre tu lado de la cancha si se ve, si no al centro."""
        h, w = res.imagen.shape[:2]
        if res.pose is not None:
            c = cal.proyectar(np.array([[-6.0, 0.0, 1.5]]), res.pose, res.modelo)[0]
            lado = int(np.clip(2.0 * cal.pixeles_por_cm(res.pose, res.modelo, (-6, 0, 0)), 24, 90))
        else:
            c, lado = np.array([w / 2, h / 2]), 60
        x, y = int(c[0] - lado / 2), int(c[1] - lado / 2)
        return (int(np.clip(x, 0, w - lado)), int(np.clip(y, 0, h - lado)), lado, lado)

    def _tarea_profundidad(self, img, mascara_mano, mascara_cancha):
        from voleyar.core import transformers_models as tm

        r = tm.obtener_profundidad().estimar(img)
        return r, tm.mano_cerca_de_mesa(r.mapa, mascara_mano, mascara_cancha, self.op.umbral_profundidad)

    def _tarea_owl(self, img, zona):
        from voleyar.core import transformers_models as tm

        return tm.obtener_owl().detectar(img, ("a hand",), 0.15, zona)

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

        res = pl.procesar_frame(frame, self.modelo_camara, self.piel, self.config, self.estado_pipe, ahora)
        self.ultimo_res = res
        img = res.imagen.copy()

        if self.fase == "calibrar_piel":
            return self._paso_calibrar_piel(frame, res, img, ahora)

        # --- profundidad (hilo aparte, cada N frames) -----------------------------
        cerca = None
        info_prof = None
        if self.prof is not None:
            if (self.n_frame % self.op.cada_n == 0 and res.mascara_mano is not None and res.zona is not None
                    and not self.prof.ocupado):
                self.prof.enviar(self._tarea_profundidad, res.imagen.copy(), res.mascara_mano.copy(),
                                 res.zona.cancha.copy())
            if self.prof.resultado is not None and self.prof.edad < 1.0:
                info_prof = self.prof.resultado
                if info_prof[1] is not None:
                    cerca = info_prof[1].cerca

        # --- entrada del jugador -----------------------------------------------------
        entrada = None
        gesto, rapidez = "", None
        if res.hay_mano:
            a = res.analisis
            entrada = EntradaJugador(res.rayo_golpe, res.punto_golpe_cm, res.velocidad_cm_s, a.fuerza, a.gesto, cerca)
            gesto, rapidez = f"{a.gesto} ({a.n_dedos} dedos)", float(np.linalg.norm(res.velocidad_cm_s))
        if self.usar_owl:
            entrada = self._entrada_owl(res, entrada, cerca)

        # --- partida ------------------------------------------------------------------
        if res.pose is not None:
            ev = self.partida.actualizar(dt, entrada)
            self.eventos_totales += ev
            img = ra.dibujar_escena(img, res.pose, res.modelo, self.partida.balon.pos,
                                    self.partida.ia.pos, self.partida.marcas, self.partida.t, self.lineas,
                                    giro=int(self.partida.t * 8))
            img = ra.ocluir_con_mano(img, res.imagen, res.componente, res.pose, res.modelo, self.partida.balon.pos)
            mensaje = self.partida.mensaje
            if self.partida.estado == "fin":
                mensaje += " · R para jugar otra vez"
        else:
            mensaje = "No veo la cancha: muestra los 4 marcadores a la cámara"
        if self.depurar:
            img = pl.dibujar_depuracion(res, img)
        m = self.partida.marcador
        ra.dibujar_hud(img, (m["jugador"], m["rival"]), mensaje, self._modo_texto(), self.fps, gesto, cerca, rapidez,
                       "OWLv2" if self.usar_owl else "clásico")
        if self.depurar:
            img = self._panel_depuracion(img, res, info_prof)
        return img

    def _entrada_owl(self, res, entrada, cerca):
        """Con OWLv2 el punto de golpe es el centro de la caja detectada."""
        if self.owl is None:
            self.owl = TrabajadorAsincrono("owl")
        if not self.owl.ocupado and res.zona is not None:
            self.owl.enviar(self._tarea_owl, res.imagen.copy(), res.zona.mascara.copy())
        r = self.owl.resultado
        if r is None or r.mejor is None or self.owl.edad > 1.5 or res.pose is None:
            return None
        c = r.mejor.centro
        rayo = cal.rayo_desde_pixel(c, res.pose, res.modelo)
        punto = cal.pixel_a_mesa(c, res.pose, res.modelo)
        vel = res.velocidad_cm_s if entrada is None else entrada.velocidad
        fuerza = 0.8 if entrada is None else entrada.fuerza
        return EntradaJugador(rayo, punto, vel, fuerza, "OWLv2", cerca)

    def _paso_calibrar_piel(self, frame, res, img, ahora) -> np.ndarray:
        rect = self._rect_calibracion(res)
        self._rect_piel = rect
        if self.sintetico:  # la mano virtual se coloca sola sobre el recuadro
            self.fuente.mano.pos = (-6.0, 0.0)
            if self.t_cuenta is None:
                self.t_cuenta = ahora + 0.5
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), vis.BGR_NARANJA, 3, cv2.LINE_AA)
        H, W = img.shape[:2]
        tam = max(18, H // 26)
        vis.poner_texto(img, "Calibración de la piel", (20, 18), tam + 6, vis.BGR_NEUTRO, True,
                        fondo=vis.BGR_GRAFITO)
        vis.poner_texto(img, "Cubre el recuadro naranja con la palma de la mano y presiona ESPACIO",
                        (20, 24 + tam + 14), tam, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        if res.pose is None:
            vis.poner_texto(img, "Aún no veo la cancha: muestra los 4 marcadores", (20, 30 + 2 * (tam + 14)),
                            tam, vis.BGR_NARANJA, fondo=vis.BGR_GRAFITO)
        if self.t_cuenta is not None:
            falta = self.t_cuenta - ahora
            if falta <= 0:
                try:
                    self.piel = pl.calibrar_piel_en_frame(frame, rect, self.modelo_camara, self.config)
                    self.fase = "jugando"
                    self.partida.reiniciar()
                except ValueError as e:
                    print("Calibración de piel fallida:", e)
                self.t_cuenta = None
            else:
                vis.poner_texto(img, f"{falta:.1f}", (x + w + 12, y), tam + 14, vis.BGR_NARANJA, True,
                                fondo=vis.BGR_GRAFITO)
        return img

    def _panel_depuracion(self, img: np.ndarray, res: pl.ResultadoFrame, info_prof) -> np.ndarray:
        H, W = img.shape[:2]
        ancho = W // 3
        alto = H // 3

        def celda(im, titulo):
            if im is None:
                im = np.full((alto, ancho, 3), vis.BGR_GRAFITO, np.uint8)
            im = vis.a_bgr(im)
            e = min(ancho / im.shape[1], alto / im.shape[0])
            im = cv2.resize(im, (max(1, int(im.shape[1] * e)), max(1, int(im.shape[0] * e))))
            c = np.full((alto, ancho, 3), vis.BGR_GRAFITO, np.uint8)
            y0, x0 = (alto - im.shape[0]) // 2, (ancho - im.shape[1]) // 2
            c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            vis.poner_texto(c, titulo, (8, 6), 15, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            return c

        mascara = res.mascara_limpia
        if mascara is not None and res.componente is not None:
            mascara = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
            mascara[res.componente > 0] = vis.BGR_NARANJA
            if res.mascara_mano is not None:
                mascara[res.mascara_mano > 0] = vis.BGR_NEUTRO
        prof = None
        if info_prof is not None:
            prof = cv2.applyColorMap(vis.normalizar_u8(info_prof[0].mapa), cv2.COLORMAP_INFERNO)
        col = np.vstack([celda(mascara, "Máscara de piel (naranja: brazo)"),
                         celda(pl.dibujar_cenital(res), "Vista cenital (cm)"),
                         celda(prof, "Profundidad (Depth Anything V2)")])
        col = col[:H] if col.shape[0] >= H else np.vstack([col, np.full((H - col.shape[0], ancho, 3), vis.BGR_GRAFITO, np.uint8)])
        lineas = [f"FPS {self.fps:.0f} · pose {'recordada' if res.pose_recordada else ('ok' if res.pose else '—')}"
                  + (f" · err {res.pose.error_rms:.2f} px" if res.pose is not None else ""),
                  "ms: " + " ".join(f"{k[:5]} {v:.0f}" for k, v in res.tiempos_ms.items())]
        if res.analisis is not None:
            a = res.analisis
            p = res.punto_golpe_cm
            lineas.append(f"dedos {a.n_dedos} · {a.gesto} · Otsu T = {res.T_otsu}"
                          + (f" · área {res.area_mano_cm2:.0f} cm²" if res.area_mano_cm2 else ""))
            if p is not None:
                v = res.velocidad_cm_s
                lineas.append(f"golpe ({p[0]:.1f}, {p[1]:.1f}) cm · v ({v[0]:.0f}, {v[1]:.0f}) cm/s")
        if info_prof is not None and info_prof[1] is not None:
            lineas.append(f"altura: puntaje {info_prof[1].puntaje:.2f} → {'cerca' if info_prof[1].cerca else 'levantada'}"
                          f" · {info_prof[0].ms:.0f} ms")
        if res.uso_watershed:
            lineas.append("watershed: mano separada de otra región")
        y = H - 22 * len(lineas) - 64
        for l in lineas:
            vis.poner_texto(img, l, (14, y), 16, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
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
        elif k == ord("l"):
            self.lineas = not self.lineas
        elif k == ord("o"):
            self.usar_owl = not self.usar_owl
        elif k == ord("p"):
            self.fase, self.piel = "calibrar_piel", None
        elif k == ord("r"):
            self.partida.reiniciar()
        elif self.sintetico and ord("0") <= k <= ord("5"):
            self.fuente.mano.n_dedos = k - ord("0")
        elif self.sintetico and k == ord("w"):
            self.fuente.mano.altura = min(20.0, self.fuente.mano.altura + 1.5)
        elif self.sintetico and k == ord("s"):
            self.fuente.mano.altura = max(0.0, self.fuente.mano.altura - 1.5)
        return True

    def _mouse(self, evento, x, y, flags, param) -> None:
        if not self.sintetico or self.ultimo_res is None or self.ultimo_res.pose is None:
            return
        res = self.ultimo_res
        if x >= res.imagen.shape[1]:
            return
        p = cal.pixel_a_mesa((x, y), res.pose, res.modelo, z=self.fuente.mano.altura)
        if p is not None:
            self.fuente.mano.pos = (float(np.clip(p[0], -30, 14)), float(np.clip(p[1], -14, 14)))

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

