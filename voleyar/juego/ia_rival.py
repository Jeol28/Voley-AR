"""IA rival simple: predice dónde podrá golpear el balón, se mueve hacia allá y lo
devuelve a tu lado con un error aleatorio.

La "mano" del rival es un punto que se mueve sobre su media cancha con
velocidad máxima limitada y un tiempo de reacción. Para que sea vencible:

- comete errores de puntería (ruido gaussiano en el objetivo),
- a veces falla a propósito (``prob_fallo``),
- no llega a balones muy rápidos o muy lejanos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from voleyar.core import config as cfg
from voleyar.juego.fisica import Balon, ParamsFisica, prediccion, velocidad_para_objetivo

DIFICULTADES = {
    "facil": dict(velocidad=28.0, reaccion=0.35, error=2.8, prob_fallo=0.22),
    "normal": dict(velocidad=38.0, reaccion=0.25, error=2.0, prob_fallo=0.12),
    "dificil": dict(velocidad=52.0, reaccion=0.15, error=1.2, prob_fallo=0.05),
}


@dataclass
class IARival:
    velocidad: float = 38.0  # cm/s
    reaccion: float = 0.25  # s antes de empezar a moverse
    error: float = 2.0  # σ del error de puntería (cm)
    prob_fallo: float = 0.12
    alcance: float = 2.6  # distancia mano-balón para golpear (cm)
    altura_golpe: float = 4.0  # golpea al balón cuando baja por esta altura
    pos: np.ndarray = field(default_factory=lambda: np.array([7.0, 0.0]))
    objetivo: np.ndarray | None = None
    _t_espera: float = 0.0
    _fallara: bool = False
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @classmethod
    def con_dificultad(cls, nombre: str = "normal", semilla: int | None = None) -> "IARival":
        return cls(**DIFICULTADES[nombre], rng=np.random.default_rng(semilla))

    def reiniciar(self) -> None:
        self.pos = np.array([7.0, 0.0])
        self.objetivo = None
        self._t_espera = 0.0

    def punto_de_intercepcion(self, balon: Balon, p: ParamsFisica) -> np.ndarray | None:
        """Dónde estará el balón (x, y) al bajar a ``altura_golpe`` en el lado rival."""
        def parar(t, b, ev):
            return b.pos[0] > 0.8 and b.vel[2] < 0 and b.pos[2] <= self.altura_golpe

        tray = prediccion(balon, p, t_max=3.0, parar_en=parar)
        if not tray:
            return None
        _, ultimo = tray[-1]
        if ultimo[0] <= 0.8:
            return None
        return np.clip(ultimo[:2], [1.0, -cfg.MEDIO_ANCHO], [cfg.MEDIO_LARGO + 1.5, cfg.MEDIO_ANCHO])

    def nueva_jugada(self) -> None:
        """Se llama cuando el balón viene hacia el rival."""
        self.objetivo = None
        self._t_espera = self.reaccion
        self._fallara = bool(self.rng.random() < self.prob_fallo)

    def actualizar(self, balon: Balon, dt: float, p: ParamsFisica, puede_golpear: bool) -> np.ndarray | None:
        """Mueve la mano del rival; si alcanza el balón devuelve la nueva velocidad."""
        if not puede_golpear:
            # Vuelve despacio al centro de su cancha
            self._mover_hacia(np.array([7.0, 0.0]), dt * 0.5)
            return None
        if self._t_espera > 0:
            self._t_espera -= dt
            return None
        if self.objetivo is None or self.rng.random() < 0.1:
            self.objetivo = self.punto_de_intercepcion(balon, p)
        if self.objetivo is not None:
            destino = self.objetivo + (np.array([0, 3.5]) if self._fallara else 0)
            self._mover_hacia(destino, dt)
        distancia = np.linalg.norm(balon.pos[:2] - self.pos)
        if balon.pos[0] > 0.5 and balon.pos[2] <= self.altura_golpe + 2 and distancia < self.alcance:
            return self.golpear(balon, p)
        return None

    def _mover_hacia(self, destino: np.ndarray, dt: float) -> None:
        d = destino - self.pos
        n = np.linalg.norm(d)
        paso = self.velocidad * dt
        self.pos = destino.copy() if n <= paso else self.pos + d / n * paso

    def golpear(self, balon: Balon, p: ParamsFisica) -> np.ndarray:
        """Devuelve el balón a tu lado con error de puntería."""
        objetivo = np.array([self.rng.uniform(-10.5, -3.0), self.rng.uniform(-7.0, 7.0)])
        objetivo += self.rng.normal(0, self.error, 2)
        apice = self.rng.uniform(9.0, 13.0)
        return velocidad_para_objetivo(balon.pos, objetivo, apice, p)
