"""Lanza Voley AR en una ventana de OpenCV (modo pantalla: sin cancha impresa, solo tus manos).

Ejemplos::

    python jugar.py                      # abre el menú del juego (cámara 0, con config/camara.json)
    python jugar.py --modo 2j            # directo a una partida de dos jugadores
    python jugar.py --modo ia            # directo a una partida contra la IA
    python jugar.py --camara 1           # otra cámara
    python jugar.py --sintetico          # sin cámara: la mano del jugador 1 sigue al mouse
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
    ap.add_argument("--modo", choices=["ia", "2j"], default=None,
                    help="saltar el menú: ia = tú contra la IA; 2j = dos jugadores")
    ap.add_argument("--sintetico", action="store_true", help="escena sintética controlada con el mouse")
    ap.add_argument("--sin-profundidad", action="store_true", help="no usar Depth Anything V2")
    ap.add_argument("--cada", type=int, default=8, help="ejecutar la profundidad cada N frames")
    ap.add_argument("--tolerancia", type=float, default=0.25,
                    help="cuánto puede alejarse la mano de su distancia de referencia (fracción)")
    ap.add_argument("--dificultad", choices=["facil", "normal", "dificil"], default="normal")
    ap.add_argument("--sin-calibrar", action="store_true", help="empezar en modo sin calibración")
    ap.add_argument("--puntos", type=int, default=15)
    a = ap.parse_args()
    op = OpcionesJuego(camara=a.camara, ancho=a.ancho, alto=a.alto, modo=a.modo or "2j", menu=a.modo is None,
                       sintetico=a.sintetico,
                       profundidad=not a.sin_profundidad and not a.sintetico, cada_n=a.cada,
                       tolerancia_distancia=a.tolerancia, dificultad=a.dificultad,
                       calibrado=not a.sin_calibrar, puntos=a.puntos)
    print(__doc__.split("Ejemplos")[0].strip())
    print("Usa el mouse o las teclas de los botones. Durante la partida: ESC pausa · D ver detección")
    r = Juego(op).ejecutar()
    izq, der = ("Tú", "IA") if op.modo == "ia" else ("Jugador 1", "Jugador 2")
    print(f"Marcador final: {izq} {r['marcador']['jugador']} – {der} {r['marcador']['rival']}")


if __name__ == "__main__":
    main()
