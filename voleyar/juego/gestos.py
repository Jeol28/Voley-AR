"""Controlar el juego con gestos de la mano (caracterización de la silueta).

Con la mano **quieta un segundo** (que es cuando contar dedos es fiable):

==============================  ===========================  ==========================
Gesto                           Cómo se reconoce             Qué hace
==============================  ===========================  ==========================
Mano abierta (4-5 dedos)        dedos estables ≥ 4           sacar
Pulgar e índice (una "L")       2 dedos muy separados        pausa / seguir
Un dedo hacia arriba (👍 o ☝)   1 dedo con la punta arriba   revancha (al terminar)
Dos dedos juntos (✌)            2 dedos casi paralelos       cambiar el tema que se ve
==============================  ===========================  ==========================

Los dos gestos de dos dedos se separan por el ángulo entre las dos puntas visto
desde el centro de la palma: en la "L" el pulgar y el índice se abren mucho
(> 35°); en la ✌ el índice y el medio van casi juntos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOMBRES = {"abierta": "Sacar", "pulgar_indice": "Pausa", "dedo_arriba": "Revancha", "dos_dedos": "Cambiar tema"}
ANGULO_L = 35.0  # grados entre las puntas desde los que dos dedos son pulgar e índice


def angulo_entre_puntas(centro, p1, p2) -> float:
    a = np.asarray(p1, float) - np.asarray(centro, float)
    b = np.asarray(p2, float) - np.asarray(centro, float)
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def clasificar(analisis, n_dedos: int | None = None) -> str | None:
    """¿Qué gesto de control es esta mano? (o ``None`` si no es ninguno)."""
    if analisis is None:
        return None
    n = analisis.n_dedos if n_dedos is None else n_dedos
    puntas, centro = analisis.puntas, analisis.centro_palma
    if n >= 4:
        return "abierta"
    if n == 2 and len(puntas) >= 2:
        return "pulgar_indice" if angulo_entre_puntas(centro, puntas[0], puntas[1]) > ANGULO_L else "dos_dedos"
    if n == 1 and puntas:
        d = np.asarray(puntas[0], float) - np.asarray(centro, float)
        # hacia arriba en la imagen (y crece hacia abajo), con hasta ~40° de inclinación
        if d[1] < 0 and abs(d[0]) <= abs(d[1]) * 0.85:
            return "dedo_arriba"
    return None


@dataclass
class _Estado:
    gesto: str | None = None
    desde: float = 0.0
    disparado: bool = False
    centro: np.ndarray | None = None


@dataclass
class Progreso:
    gesto: str
    fraccion: float  # 0-1: cuánto falta para que se active
    centro: tuple[float, float]
    radio: float


class LectorGestos:
    """Un gesto se activa cuando se mantiene ``espera`` segundos con la mano casi quieta."""

    def __init__(self, espera: float = 1.0, quieta: float = 0.35):
        self.espera = espera
        self.quieta = quieta  # cuánto puede moverse la palma por frame (en radios de palma)
        self.estados: dict[int, _Estado] = {}

    def actualizar(self, lado: int, r, ahora: float, permitidos: set[str]) -> tuple[str | None, Progreso | None]:
        """Devuelve (gesto que se acaba de activar o None, progreso para dibujar o None)."""
        e = self.estados.setdefault(lado, _Estado())
        a = r.analisis if r is not None and r.hay_mano else None
        gesto = clasificar(a, r.dedos_estables) if a is not None else None
        if gesto not in permitidos:
            gesto = None
        centro = np.asarray(a.centro_palma, float) if a is not None else None
        movida = (centro is not None and e.centro is not None
                  and np.linalg.norm(centro - e.centro) > self.quieta * max(a.radio_palma, 1.0))
        e.centro = centro
        if gesto is None or gesto != e.gesto or movida:
            if gesto != e.gesto:
                e.disparado = False  # cambiar de gesto (o bajar la mano) permite volver a usarlo
            e.gesto, e.desde = gesto, ahora
            return None, None
        if e.disparado:
            return None, None
        fraccion = (ahora - e.desde) / self.espera
        prog = Progreso(gesto, float(min(fraccion, 1.0)), tuple(a.centro_palma), float(a.radio_palma))
        if fraccion >= 1.0:
            e.disparado = True
            return gesto, prog
        return None, prog

    def reiniciar(self) -> None:
        self.estados.clear()
