"""Ejecución de modelos lentos en un hilo aparte sin bloquear el bucle del juego."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable


class TrabajadorAsincrono:
    """Un hilo con una sola tarea a la vez.

    ``enviar`` solo acepta trabajo si el hilo está libre (si no, el frame se
    descarta): así el juego nunca acumula retraso. ``resultado`` es el último
    valor calculado y ``edad`` cuántos segundos tiene.
    """

    def __init__(self, nombre: str = "trabajador"):
        self.nombre = nombre
        self._evento = threading.Event()
        self._cerrojo = threading.Lock()
        self._tarea: tuple[Callable, tuple, dict] | None = None
        self.resultado: Any = None
        self.t_resultado: float = 0.0
        self.error: str | None = None
        self.ocupado = False
        self._vivo = True
        self._hilo = threading.Thread(target=self._bucle, name=nombre, daemon=True)
        self._hilo.start()

    def enviar(self, fn: Callable, *args, **kwargs) -> bool:
        with self._cerrojo:
            if self.ocupado:
                return False
            self.ocupado = True
            self._tarea = (fn, args, kwargs)
        self._evento.set()
        return True

    @property
    def edad(self) -> float:
        return time.perf_counter() - self.t_resultado if self.t_resultado else float("inf")

    def _bucle(self) -> None:
        while self._vivo:
            self._evento.wait(0.2)
            if not self._evento.is_set():
                continue
            self._evento.clear()
            with self._cerrojo:
                tarea = self._tarea
            if tarea is None:
                continue
            fn, args, kwargs = tarea
            try:
                r = fn(*args, **kwargs)
                self.resultado, self.t_resultado, self.error = r, time.perf_counter(), None
            except Exception as e:  # el juego sigue aunque falle el modelo
                self.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                with self._cerrojo:
                    self.ocupado = False
                    self._tarea = None

    def detener(self) -> None:
        self._vivo = False
        self._evento.set()
