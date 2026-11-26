"""Reglas de la partida, sin cámara ni ventana (se puede probar en aislamiento).

Reglas (voleibol de mesa simplificado):

- Gana el primero en llegar a **15 puntos** (opcionalmente con 2 de diferencia).
- Puntuación por jugada: quien gana el punto, saca.
- Cada lado devuelve el balón con **un toque**.
- Se permite **un bote** en tu lado antes de devolverlo (con 0 botes sería
  voleibol real, pero la latencia de la cámara lo hace muy difícil).
- Punto para el rival de quien golpeó si el balón cae **fuera**, se queda en
  **la red** o no pasa a la otra cancha. Punto para quien golpeó si el
  receptor deja botar el balón más de lo permitido.
- El golpe solo vale si la mano está **cerca de la mesa** (según la
  profundidad estimada por Depth Anything) y el balón está de tu lado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voleyar.core.calibracion import distancia_rayo_punto
from voleyar.juego.fisica import Balon, Evento, ParamsFisica, paso, velocidad_para_objetivo
from voleyar.juego.ia_rival import IARival

JUGADOR, RIVAL = "jugador", "rival"
LADO = {JUGADOR: -1, RIVAL: 1}


def _otro(q: str) -> str:
    return RIVAL if q == JUGADOR else JUGADOR


@dataclass
class EntradaJugador:
    """Lo que el pipeline de visión le entrega al juego en cada frame."""

    rayo: tuple[np.ndarray, np.ndarray] | None = None  # rayo de la cámara por el punto de golpe
    punto_cm: np.ndarray | None = None  # punto de golpe sobre la mesa
    velocidad: np.ndarray = field(default_factory=lambda: np.zeros(2))  # cm/s
    fuerza: float = 0.8
    gesto: str = "golpe"
    cerca_mesa: bool | None = None  # None = sin información de profundidad


@dataclass
class MarcaGolpe:
    pos: np.ndarray
    valido: bool
    t: float
    texto: str


class Partida:
    def __init__(self, ia: IARival | None = None, params: ParamsFisica | None = None, puntos_para_ganar: int = 15,
                 ganar_por_dos: bool = False, botes_permitidos: int = 1, alcance: float = 2.6,
                 altura_max_golpe: float = 8.0, enfriamiento: float = 0.35):
        self.ia = ia or IARival.con_dificultad("normal")
        self.p = params or ParamsFisica()
        self.puntos_para_ganar = puntos_para_ganar
        self.ganar_por_dos = ganar_por_dos
        self.botes_permitidos = botes_permitidos
        self.alcance = alcance
        self.altura_max_golpe = altura_max_golpe
        self.enfriamiento = enfriamiento
        self.reiniciar()

    # ------------------------------------------------------------------ estado
    def reiniciar(self) -> None:
        self.marcador = {JUGADOR: 0, RIVAL: 0}
        self.sacador = JUGADOR
        self.ganador: str | None = None
        self.t = 0.0
        self.marcas: list[MarcaGolpe] = []
        self._preparar_saque()

    def _preparar_saque(self) -> None:
        self.estado = "saque"
        self.t_estado = 0.0
        self.ultimo_toque: str | None = None
        self.botes = 0
        self._t_ultimo_golpe = -10.0
        self.ia.reiniciar()
        x = -7.0 if self.sacador == JUGADOR else 7.0
        self.balon = Balon(np.array([x, 0.0, 6.0]), np.zeros(3))
        self.mensaje = "Tu saque: golpea el balón" if self.sacador == JUGADOR else "Saca la IA…"

    def _cambiar(self, estado: str) -> None:
        self.estado = estado
        self.t_estado = 0.0

    def _punto(self, para: str, motivo: str) -> list[str]:
        self.marcador[para] += 1
        self.sacador = para
        quien = "ti" if para == JUGADOR else "la IA"
        self.mensaje = f"Punto para {quien} · {motivo}"
        a, b = self.marcador[JUGADOR], self.marcador[RIVAL]
        lider = max(a, b)
        if lider >= self.puntos_para_ganar and (not self.ganar_por_dos or abs(a - b) >= 2):
            self.ganador = JUGADOR if a > b else RIVAL
            self.mensaje = "¡Ganaste!" if self.ganador == JUGADOR else "Gana la IA"
            self._cambiar("fin")
            return [f"punto_{para}", "fin"]
        self._cambiar("punto")
        return [f"punto_{para}"]

    # -------------------------------------------------------------- golpe tuyo
    def _toca_balon(self, e: EntradaJugador) -> bool:
        if e.rayo is not None:
            o, d = e.rayo
            v = self.balon.pos - o
            lam = max(0.0, float(v @ d))
            mas_cercano = o + lam * d
            return (distancia_rayo_punto(o, d, self.balon.pos) < self.p.radio + self.alcance
                    and mas_cercano[2] <= self.altura_max_golpe)
        if e.punto_cm is not None:
            return (np.linalg.norm(self.balon.pos[:2] - np.asarray(e.punto_cm)[:2]) < self.p.radio + self.alcance
                    and self.balon.pos[2] <= self.altura_max_golpe)
        return False

    def velocidad_golpe(self, e: EntradaJugador) -> np.ndarray:
        """Dirección según la velocidad de la mano; fuerza según el gesto."""
        f = float(np.clip(e.fuerza, 0.4, 1.1))
        v = np.asarray(e.velocidad, float)
        profundidad = np.interp(f, [0.5, 1.05], [4.5, 10.5]) + np.clip(v[0] * 0.02, -2.0, 3.0)
        x_obj = float(np.clip(profundidad, 2.5, 11.5))
        y_obj = float(np.clip(self.balon.pos[1] + np.clip(v[1] * 0.05, -6.0, 6.0), -8.0, 8.0))
        apice = float(np.interp(f, [0.5, 1.05], [15.0, 7.5]))
        return velocidad_para_objetivo(self.balon.pos, np.array([x_obj, y_obj]), apice, self.p)

    def _intentar_golpe(self, e: EntradaJugador | None) -> list[str]:
        if e is None or self.ultimo_toque == JUGADOR or self.balon.pos[0] > 0.5:
            return []
        if self.t - self._t_ultimo_golpe < self.enfriamiento or not self._toca_balon(e):
            return []
        self._t_ultimo_golpe = self.t
        if e.cerca_mesa is False:
            self.marcas.append(MarcaGolpe(self.balon.pos.copy(), False, self.t, "mano levantada"))
            return ["golpe_invalido"]
        self.balon.pos[2] = max(self.balon.pos[2], self.p.radio + 0.5)
        self.balon.vel = self.velocidad_golpe(e)
        self.ultimo_toque = JUGADOR
        self.botes = 0
        self.ia.nueva_jugada()
        self.marcas.append(MarcaGolpe(self.balon.pos.copy(), True, self.t, e.gesto))
        if self.estado == "saque":
            self._cambiar("juego")
            self.mensaje = ""
        return ["golpe"]

    # ------------------------------------------------------------- actualizar
    def actualizar(self, dt: float, entrada: EntradaJugador | None = None) -> list[str]:
        """Avanza la partida ``dt`` segundos. Devuelve los eventos ocurridos."""
        dt = float(min(dt, 0.1))
        self.t += dt
        self.t_estado += dt
        self.marcas = [m for m in self.marcas if self.t - m.t < 0.8]
        if self.estado == "fin":
            return []
        if self.estado == "punto":
            if self.t_estado > 1.6:
                self._preparar_saque()
            return []
        if self.estado == "saque":
            if self.sacador == JUGADOR:
                self.balon.pos = np.array([-7.0, 0.0, 6.0 + 0.8 * np.sin(3 * self.t_estado)])
                return self._intentar_golpe(entrada)
            self.balon.pos = np.array([self.ia.pos[0], self.ia.pos[1], 5.0])
            if self.t_estado > 1.4:
                self.balon.vel = self.ia.golpear(self.balon, self.p)
                self.ultimo_toque = RIVAL
                self._cambiar("juego")
                self.mensaje = ""
                return ["golpe_rival"]
            return []

        # --- juego -------------------------------------------------------------
        salida: list[str] = []
        for ev in paso(self.balon, dt, self.p):
            r = self._procesar_evento(ev)
            salida += r
            if self.estado != "juego":
                return salida
        salida += self._intentar_golpe(entrada)
        if self.ultimo_toque == JUGADOR and self.balon.pos[0] > 0:
            v = self.ia.actualizar(self.balon, dt, self.p, True)
            if v is not None:
                self.balon.vel = v
                self.ultimo_toque = RIVAL
                self.botes = 0
                salida.append("golpe_rival")
        else:
            self.ia.actualizar(self.balon, dt, self.p, False)
        # Balón detenido o perdido lejos de la mesa
        rapidez = np.linalg.norm(self.balon.vel)
        if rapidez < 3.0 and self.balon.pos[2] <= self.p.radio + 0.05:
            lado = JUGADOR if self.balon.pos[0] < 0 else RIVAL
            return salida + self._punto(_otro(lado), "el balón se detuvo")
        if abs(self.balon.pos[0]) > 40 or abs(self.balon.pos[1]) > 30:
            return salida + self._punto(_otro(self.ultimo_toque or JUGADOR), "fuera")
        return salida

    def _procesar_evento(self, ev: Evento) -> list[str]:
        golpeador = self.ultimo_toque
        if golpeador is None:
            return []
        receptor = _otro(golpeador)
        if ev.tipo == "red":
            return ["red"]
        if ev.tipo == "cruza_red":
            self.botes = 0
            return []
        # bote
        if ev.lado == LADO[golpeador]:
            motivo = "en la red" if abs(ev.pos[0]) < 3 else "no pasó la red"
            return ["bote"] + self._punto(receptor, motivo)
        if not ev.dentro and self.botes == 0:
            return ["bote"] + self._punto(receptor, "fuera")
        self.botes += 1
        if self.botes > self.botes_permitidos:
            return ["bote"] + self._punto(golpeador, "dos botes")
        return ["bote"]
