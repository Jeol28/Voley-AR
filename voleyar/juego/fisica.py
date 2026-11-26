r"""Física del balón en el sistema de la mesa (cm, s).

Modelo de partícula con radio :math:`r`, integrado con Euler semi-implícito en
subpasos de 1/240 s:

.. math::  v_z \leftarrow v_z - g\,\Delta t, \qquad p \leftarrow p + v\,\Delta t

- **Gravedad de juego** :math:`g = 140\ \text{cm/s}^2` (no 981): la cancha mide
  25 cm, así que con la gravedad real una jugada duraría décimas de segundo.
  Escalarla es como jugar "a cámara lenta".
- **Rebote en la mesa** (:math:`z = r`): :math:`v_z \leftarrow -e_{mesa}\, v_z` y
  fricción horizontal :math:`v_{xy} \leftarrow \mu\, v_{xy}`.
- **Red** (plano :math:`x = 0`, :math:`|y| \le` medio ancho, :math:`z \le h_{red}`):
  si el balón cruza :math:`x = 0` por debajo de :math:`h_{red} + r`, rebota con
  :math:`v_x \leftarrow -e_{red}\, v_x`.

Las trayectorias de golpe se calculan como tiros parabólicos que pasan por un
punto de máxima altura (ápice) y caen en un objetivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voleyar.core import config as cfg


@dataclass
class ParamsFisica:
    gravedad: float = 140.0
    restitucion_mesa: float = 0.62
    restitucion_red: float = 0.25
    friccion_mesa: float = 0.85
    radio: float = cfg.RADIO_BALON
    altura_red: float = cfg.ALTURA_RED
    sub_paso: float = 1 / 240


@dataclass
class Balon:
    pos: np.ndarray = field(default_factory=lambda: np.array([-7.0, 0.0, 6.0]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copia(self) -> "Balon":
        return Balon(self.pos.copy(), self.vel.copy())

    @property
    def lado(self) -> int:
        """-1 = tu lado (x < 0), +1 = lado rival."""
        return -1 if self.pos[0] < 0 else 1


@dataclass
class Evento:
    tipo: str  # "bote", "red", "cruza_red"
    pos: np.ndarray
    lado: int
    dentro: bool = True


def dentro_de_cancha(x: float, y: float) -> bool:
    return abs(x) <= cfg.MEDIO_LARGO and abs(y) <= cfg.MEDIO_ANCHO


def paso(balon: Balon, dt: float, p: ParamsFisica | None = None) -> list[Evento]:
    """Avanza la simulación ``dt`` segundos (en subpasos). Modifica ``balon``."""
    p = p or ParamsFisica()
    eventos: list[Evento] = []
    n = max(1, int(np.ceil(dt / p.sub_paso)))
    h = dt / n
    for _ in range(n):
        anterior = balon.pos.copy()
        balon.vel[2] -= p.gravedad * h
        balon.pos += balon.vel * h
        # Red
        if np.sign(anterior[0]) != np.sign(balon.pos[0]) and anterior[0] != 0:
            f = anterior[0] / (anterior[0] - balon.pos[0])
            cruce = anterior + f * (balon.pos - anterior)
            if abs(cruce[1]) <= cfg.MEDIO_ANCHO + p.radio and cruce[2] - p.radio < p.altura_red:
                balon.pos[0] = np.sign(anterior[0]) * p.radio * 0.5
                balon.vel[0] = -p.restitucion_red * balon.vel[0]
                eventos.append(Evento("red", balon.pos.copy(), int(np.sign(anterior[0]))))
            else:
                eventos.append(Evento("cruza_red", cruce, 1 if balon.pos[0] > 0 else -1))
        # Mesa
        if balon.pos[2] < p.radio:
            balon.pos[2] = p.radio
            if balon.vel[2] < 0:
                vz = -balon.vel[2]
                balon.vel[2] = p.restitucion_mesa * vz if vz > 8.0 else 0.0
                balon.vel[:2] *= p.friccion_mesa
                eventos.append(Evento("bote", balon.pos.copy(), balon.lado,
                                      dentro_de_cancha(balon.pos[0], balon.pos[1])))
    return eventos


def velocidad_para_objetivo(origen: np.ndarray, objetivo_xy: np.ndarray, apice: float,
                            p: ParamsFisica | None = None, margen_red: float = 1.0) -> np.ndarray:
    r"""Velocidad inicial para un tiro parabólico de ``origen`` a ``objetivo_xy`` (en la mesa).

    Con ápice :math:`H`: :math:`v_{z0} = \sqrt{2g(H - z_0)}`, subida
    :math:`t_s = v_{z0}/g`, bajada :math:`t_b = \sqrt{2(H - r)/g}` y
    :math:`v_{xy} = (\text{objetivo} - p_0)/(t_s + t_b)`. Si la parábola no pasa
    la red con ``margen_red`` cm de holgura, se sube el ápice.
    """
    p = p or ParamsFisica()
    o = np.asarray(origen, float)
    obj = np.asarray(objetivo_xy, float)
    g = p.gravedad
    H = max(apice, o[2] + 0.5)
    for _ in range(60):
        vz = np.sqrt(2 * g * (H - o[2]))
        T = vz / g + np.sqrt(2 * max(H - p.radio, 0.01) / g)
        vxy = (obj - o[:2]) / T
        if np.sign(o[0]) == np.sign(obj[0]) or abs(vxy[0]) < 1e-6:
            break
        tc = -o[0] / vxy[0]
        zc = o[2] + vz * tc - 0.5 * g * tc ** 2
        if zc - p.radio >= p.altura_red + margen_red:
            break
        H += 0.5
    return np.array([vxy[0], vxy[1], vz])


def prediccion(balon: Balon, p: ParamsFisica | None = None, t_max: float = 3.0, dt: float = 1 / 120,
               parar_en=None) -> list[tuple[float, np.ndarray]]:
    """Simula hacia adelante una copia del balón: lista de ``(t, posición)``.

    ``parar_en(t, balon, eventos) -> bool`` permite cortar la simulación.
    """
    b = balon.copia()
    tray = []
    t = 0.0
    while t < t_max:
        ev = paso(b, dt, p)
        t += dt
        tray.append((t, b.pos.copy()))
        if parar_en is not None and parar_en(t, b, ev):
            break
    return tray
