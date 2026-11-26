"""Lanza Voley AR en una ventana de OpenCV.

Ejemplos::

    python jugar.py                      # cámara 0, con config/camara.json
    python jugar.py --camara 1           # otra cámara
    python jugar.py --sintetico          # sin cámara: la mano sigue al mouse
    python jugar.py --sin-profundidad    # sin Depth Anything (más rápido)
    python jugar.py --dificultad dificil
"""

from __future__ import annotations

import argparse

from voleyar.juego.bucle import Juego, OpcionesJuego


def main() -> None:
    ap = argparse.ArgumentParser(description="Voley AR: voleibol en realidad aumentada sobre la mesa")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara web")
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--sintetico", action="store_true", help="escena sintética controlada con el mouse")
    ap.add_argument("--sin-profundidad", action="store_true", help="no usar Depth Anything V2")
    ap.add_argument("--cada", type=int, default=8, help="ejecutar la profundidad cada N frames")
    ap.add_argument("--umbral-profundidad", type=float, default=0.30,
                    help="puntaje de altura a partir del cual la mano está levantada")
    ap.add_argument("--dificultad", choices=["facil", "normal", "dificil"], default="normal")
    ap.add_argument("--sin-calibrar", action="store_true", help="empezar en modo sin calibración")
    ap.add_argument("--puntos", type=int, default=15)
    a = ap.parse_args()
    op = OpcionesJuego(camara=a.camara, ancho=a.ancho, alto=a.alto, sintetico=a.sintetico,
                       profundidad=not a.sin_profundidad and not a.sintetico, cada_n=a.cada,
                       umbral_profundidad=a.umbral_profundidad, dificultad=a.dificultad,
                       calibrado=not a.sin_calibrar, puntos=a.puntos)
    print(__doc__.split("Ejemplos")[0].strip())
    print("Teclas: ESPACIO piel · D depuración · C calibración · L líneas · O OWLv2 · P piel · R reiniciar · Q salir")
    r = Juego(op).ejecutar()
    print(f"Marcador final: tú {r['marcador']['jugador']} – IA {r['marcador']['rival']}")


if __name__ == "__main__":
    main()
