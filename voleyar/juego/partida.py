"""Reglas de la partida, sin cámara ni ventana (se puede probar en aislamiento).

Modo pantalla (v2): la cancha es virtual y vertical, vista de lado. El jugador 1
está a la izquierda (``x < 0``), la red en el centro y a la derecha (``x > 0``)
el jugador 2 o la IA. El balón se mueve en el plano de la red (``y = 0``).

Reglas (voleibol simplificado):

- Gana el primero en llegar a **15 puntos** (opcionalmente con 2 de diferencia).
- Quien gana el punto, saca.
- Cada lado devuelve el balón con **un toque** y se permite **un bote** en el piso.
- Punto para el rival de quien golpeó si el balón cae **fuera**, se queda en **la
  red** o no pasa al otro lado. Punto para quien golpeó si el receptor deja botar
  el balón más de lo permitido.
- El golpe solo vale si la mano está **en el plano de juego**: a la distancia de
  la cámara que se midió para ese jugador al empezar (tamaño de la palma + ``K``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voleyar.core.calibracion import distancia_rayo_punto
from voleyar.juego.fisica import Balon, Evento, ParamsFisica, paso, velocidad_para_objetivo
from voleyar.juego.ia_rival import IARival

JUGADOR, RIVAL = "jugador", "rival"  # izquierda / derecha
LADO = {JUGADOR: -1, RIVAL: 1}
MODOS = ("ia", "2j")


def _otro(q: str) -> str:
    return RIVAL if q == JUGADOR else JUGADOR


@dataclass
class EntradaJugador:
    """Lo que el pipeline de visión le entrega al juego por cada mano, en cada frame."""

    rayo: tuple[np.ndarray, np.ndarray] | None = None  # rayo de la cámara por el punto de golpe
    punto: np.ndarray | None = None  # punto de golpe en el plano de la red (unidades de cancha)
    velocidad: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (vx, vz) en unidades/s
    fuerza: float = 0.8
    gesto: str = "golpe"
    valida: bool | None = None  # False = mano fuera del plano de juego; None = sin información
    toca: bool = False  # algo que no es el entorno (sustracción de fondo) está sobre el balón en la imagen
    tipo: str = "golpe"  # remate, fuerte, golpe, suave o bloqueo (caracterizacion.tipo_de_golpe)


@dataclass
class MarcaGolpe:
    pos: np.ndarray
    valido: bool
    t: float
    texto: str


# Cómo vuela cada tipo de golpe: (profundidad del bote en el campo rival, altura máxima)
PERFILES_GOLPE = {
    "remate": (10.5, 6.2),  # rápido, bajo y al fondo
    "fuerte": (9.5, 7.5),
    "suave": (4.5, 12.5),  # alto y corto, cae cerca de la red
    "bloqueo": (3.0, 6.8),  # rebota corto, justo detrás de la red
}


def params_pantalla() -> ParamsFisica:
    """Física del modo pantalla: gravedad menor para compensar la latencia de la cámara."""
    return ParamsFisica(gravedad=85.0, restitucion_mesa=0.6)


class Partida:
    def __init__(self, ia: IARival | None = None, params: ParamsFisica | None = None, puntos_para_ganar: int = 15,
                 ganar_por_dos: bool = False, botes_permitidos: int = 1, alcance: float = 2.6,
                 enfriamiento: float = 0.35, modo: str = "ia"):
        if modo not in MODOS:
            raise ValueError(f"Modo desconocido: {modo}")
        self.modo = modo
        self.ia = ia or IARival.con_dificultad("normal")
        self.p = params or params_pantalla()
        self.puntos_para_ganar = puntos_para_ganar
        self.ganar_por_dos = ganar_por_dos
        self.botes_permitidos = botes_permitidos
        self.alcance = alcance
        self.enfriamiento = enfriamiento
        self.nombres = ({JUGADOR: "Tú", RIVAL: "IA"} if modo == "ia" else {JUGADOR: "Jugador 1", RIVAL: "Jugador 2"})
        self.reiniciar()

    # ------------------------------------------------------------------ estado
    def humano(self, quien: str) -> bool:
        return quien == JUGADOR or self.modo == "2j"

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
        self._t_ultimo_golpe = {JUGADOR: -10.0, RIVAL: -10.0}
        self.ia.reiniciar()
        x = -7.0 if self.sacador == JUGADOR else 7.0
        self.balon = Balon(np.array([x, 0.0, 7.0]), np.zeros(3))
        if self.humano(self.sacador):
            self.mensaje = ("Tu saque: golpea el balón" if self.modo == "ia"
                            else f"Saca {self.nombres[self.sacador]}: golpea el balón")
        else:
            self.mensaje = "Saca la IA…"

    def _cambiar(self, estado: str) -> None:
        self.estado = estado
        self.t_estado = 0.0

    def _punto(self, para: str, motivo: str) -> list[str]:
        self.marcador[para] += 1
        self.sacador = para
        quien = ("ti" if para == JUGADOR else "la IA") if self.modo == "ia" else self.nombres[para]
        self.mensaje = f"Punto para {quien} · {motivo}"
        a, b = self.marcador[JUGADOR], self.marcador[RIVAL]
        if max(a, b) >= self.puntos_para_ganar and (not self.ganar_por_dos or abs(a - b) >= 2):
            self.ganador = JUGADOR if a > b else RIVAL
            if self.modo == "ia":
                self.mensaje = "¡Ganaste!" if self.ganador == JUGADOR else "Gana la IA"
            else:
                self.mensaje = f"¡Gana {self.nombres[self.ganador]}!"
            self._cambiar("fin")
            return [f"punto_{para}", "fin"]
        self._cambiar("punto")
        return [f"punto_{para}"]

    # ---------------------------------------------------------------- golpes
    def _toca_balon(self, e: EntradaJugador) -> bool:
        if e.toca:  # cualquier objeto en movimiento sobre el balón (no solo una mano)
            return True
        tope = self.p.radio + self.alcance
        if e.rayo is not None:
            o, d = e.rayo
            return distancia_rayo_punto(o, d, self.balon.pos) < tope
        if e.punto is not None:
            return float(np.linalg.norm(self.balon.pos[[0, 2]] - np.asarray(e.punto)[[0, 2]])) < tope
        return False

    def velocidad_golpe(self, e: EntradaJugador, quien: str = JUGADOR) -> np.ndarray:
        """Hacia el otro lado: la distancia depende del gesto y del movimiento de la mano."""
        signo = -LADO[quien]  # el jugador 1 (izq.) envía hacia +x
        f = float(np.clip(e.fuerza, 0.4, 1.1))
        v = np.asarray(e.velocidad, float)
        hacia = signo * v[0]  # rapidez de la mano hacia la red
        if e.tipo in PERFILES_GOLPE:
            base, apice = PERFILES_GOLPE[e.tipo]
            profundidad = base + np.clip(hacia * 0.02, -1.0, 1.0)
        else:
            profundidad = np.interp(f, [0.5, 1.05], [4.5, 10.0]) + np.clip(hacia * 0.03, -2.0, 2.5)
            apice = float(np.interp(f, [0.5, 1.05], [11.5, 7.5]) + np.clip(v[1] * 0.01, 0.0, 1.5))
        x_obj = signo * float(np.clip(profundidad, 2.5, 11.5))
        apice = max(float(apice), self.balon.pos[2] + 0.5)
        vel = velocidad_para_objetivo(self.balon.pos, np.array([x_obj, 0.0]), apice, self.p)
        vel[1] = 0.0
        return vel

    def _intentar_golpe(self, e: EntradaJugador | None, quien: str) -> list[str]:
        if e is None or self.ultimo_toque == quien:
            return []
        if self.balon.pos[0] * LADO[quien] < -0.5:  # el balón está en la otra mitad
            return []
        if self.estado == "saque" and quien != self.sacador:
            return []
        if self.t - self._t_ultimo_golpe[quien] < self.enfriamiento or not self._toca_balon(e):
            return []
        self._t_ultimo_golpe[quien] = self.t
        if e.valida is False:
            self.marcas.append(MarcaGolpe(self.balon.pos.copy(), False, self.t, "muy cerca o muy lejos"))
            return ["golpe_invalido"]
        self.balon.pos[1] = 0.0
        self.balon.pos[2] = max(self.balon.pos[2], self.p.radio + 0.5)
        self.balon.vel = self.velocidad_golpe(e, quien)
        self.ultimo_toque = quien
        self.botes = 0
        if not self.humano(_otro(quien)):
            self.ia.nueva_jugada()
        from voleyar.core.caracterizacion import TIPOS_GOLPE

        self.marcas.append(MarcaGolpe(self.balon.pos.copy(), True, self.t, TIPOS_GOLPE.get(e.tipo, e.gesto)))
        self.ultimo_tipo = {"quien": quien, "tipo": e.tipo}
        if self.estado == "saque":
            self._cambiar("juego")
            self.mensaje = ""
        return ["golpe" if quien == JUGADOR else "golpe_rival"]

    # ------------------------------------------------------------- actualizar
    def actualizar(self, dt: float, entrada: EntradaJugador | None = None,
                   entrada_rival: EntradaJugador | None = None) -> list[str]:
        """Avanza la partida ``dt`` segundos. ``entrada_rival`` solo se usa en modo 2 jugadores."""
        dt = float(min(dt, 0.1))
        self.t += dt
        self.t_estado += dt
        self.marcas = [m for m in self.marcas if self.t - m.t < 0.8]
        entradas = {JUGADOR: entrada, RIVAL: entrada_rival if self.modo == "2j" else None}
        if self.estado == "fin":
            return []
        if self.estado == "punto":
            if self.t_estado > 1.6:
                self._preparar_saque()
            return []
        if self.estado == "saque":
            if self.humano(self.sacador):
                x = -7.0 if self.sacador == JUGADOR else 7.0
                self.balon.pos = np.array([x, 0.0, 7.0 + 0.8 * np.sin(3 * self.t_estado)])
                return self._intentar_golpe(entradas[self.sacador], self.sacador)
            self.balon.pos = np.array([self.ia.pos[0], 0.0, 6.0])
            if self.t_estado > 1.4:
                self.balon.vel = self.ia.golpear(self.balon, self.p)
                self.balon.vel[1] = 0.0
                self.ultimo_toque = RIVAL
                self._cambiar("juego")
                self.mensaje = ""
                return ["golpe_rival"]
            return []

        # --- juego -------------------------------------------------------------
        salida: list[str] = []
        for ev in paso(self.balon, dt, self.p):
            salida += self._procesar_evento(ev)
            if self.estado != "juego":
                return salida
        for quien in (JUGADOR, RIVAL):
            if self.humano(quien):
                salida += self._intentar_golpe(entradas[quien], quien)
        if not self.humano(RIVAL):
            if self.ultimo_toque == JUGADOR and self.balon.pos[0] > 0:
                v = self.ia.actualizar(self.balon, dt, self.p, True)
                if v is not None:
                    self.balon.vel = v
                    self.balon.vel[1] = 0.0
                    self.ultimo_toque = RIVAL
                    self.botes = 0
                    salida.append("golpe_rival")
            else:
                self.ia.actualizar(self.balon, dt, self.p, False)
        rapidez = np.linalg.norm(self.balon.vel)
        if rapidez < 3.0 and self.balon.pos[2] <= self.p.radio + 0.05:
            lado = JUGADOR if self.balon.pos[0] < 0 else RIVAL
            return salida + self._punto(_otro(lado), "el balón se detuvo")
        if abs(self.balon.pos[0]) > 40:
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
        if ev.lado == LADO[golpeador]:
            motivo = "en la red" if abs(ev.pos[0]) < 3 else "no pasó la red"
            return ["bote"] + self._punto(receptor, motivo)
        if not ev.dentro and self.botes == 0:
            return ["bote"] + self._punto(receptor, "fuera")
        self.botes += 1
        if self.botes > self.botes_permitidos:
            return ["bote"] + self._punto(golpeador, "dos botes")
        return ["bote"]
