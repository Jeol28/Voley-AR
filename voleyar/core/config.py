"""Constantes geométricas compartidas por todo el proyecto.

Sistema de coordenadas de la mesa (en centímetros), fijado por la hoja impresa:

- Origen en el centro de la cancha (sobre la línea central, donde va la red).
- Eje **X** hacia la derecha de la hoja (de *tu lado* hacia el lado *rival*).
- Eje **Y** hacia el borde superior de la hoja.
- Eje **Z** saliendo del papel, hacia arriba (sistema dextrógiro).

El jugador está del lado ``x < 0`` y el rival del lado ``x > 0``. La red es el
plano ``x = 0`` con ``z ∈ [0, ALTURA_RED]``.

Todas las medidas coinciden con ``scripts/generar_imprimibles.py``: si cambias
algo aquí, vuelve a generar los PDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_PROYECTO / "config"
RUTA_CAMARA_JSON = RUTA_CONFIG / "camara.json"
RUTA_IMPRIMIBLES = RAIZ_PROYECTO / "docs" / "imprimibles"

# --------------------------------------------------------------------------- #
# Cancha (cm)
# --------------------------------------------------------------------------- #
CANCHA_LARGO = 25.0  # a lo largo de X (tu lado + lado rival)
CANCHA_ANCHO = 18.0  # a lo largo de Y
MEDIO_LARGO = CANCHA_LARGO / 2  # 12.5
MEDIO_ANCHO = CANCHA_ANCHO / 2  # 9.0
ALTURA_RED = 5.0
RADIO_BALON = 1.2

# Marcadores ArUco: diccionario 4x4 (bits grandes, fáciles de detectar a distancia)
DICCIONARIO_ARUCO = "DICT_4X4_50"
LADO_MARCADOR = 3.0  # lado del cuadro negro del marcador, en cm
MARGEN_BLANCO_MARCADOR = 0.5  # zona blanca alrededor de cada marcador
# Centros de los marcadores (cm), dentro de las esquinas del campo azul.
# IDs 0 y 3 están de tu lado (x < 0); 1 y 2 del lado rival.
CENTROS_MARCADORES: dict[int, tuple[float, float]] = {
    0: (-10.5, 7.0),   # arriba-izquierda
    1: (10.5, 7.0),    # arriba-derecha
    2: (10.5, -7.0),   # abajo-derecha
    3: (-10.5, -7.0),  # abajo-izquierda
}

# Franja de entrada del brazo (cm, más allá de tu borde de la cancha)
FRANJA_BRAZO_LARGO = 16.0
# Altura del "volumen de juego" que se proyecta para buscar la mano (cm)
ALTURA_VOLUMEN_JUEGO = 12.0

# --------------------------------------------------------------------------- #
# Tablero de ajedrez para la calibración intrínseca
# --------------------------------------------------------------------------- #
TABLERO_ESQUINAS = (9, 6)  # esquinas interiores (columnas, filas) → 10 x 7 cuadros
TABLERO_CUADRO_CM = 2.5  # tamaño nominal del cuadro impreso


def esquinas_marcador(id_marcador: int) -> np.ndarray:
    """Devuelve las 4 esquinas 3D (cm, z = 0) de un marcador ArUco.

    El orden coincide con el que entrega ``cv2.aruco.ArucoDetector``:
    superior-izquierda, superior-derecha, inferior-derecha, inferior-izquierda
    (tal como se ve el marcador impreso, con la hoja en horizontal).
    """
    cx, cy = CENTROS_MARCADORES[id_marcador]
    m = LADO_MARCADOR / 2
    return np.array(
        [
            [cx - m, cy + m, 0.0],
            [cx + m, cy + m, 0.0],
            [cx + m, cy - m, 0.0],
            [cx - m, cy - m, 0.0],
        ],
        dtype=np.float64,
    )


def esquinas_cancha(z: float = 0.0, margen: float = 0.0) -> np.ndarray:
    """Esquinas 3D del rectángulo de juego (sentido antihorario visto desde arriba)."""
    x, y = MEDIO_LARGO + margen, MEDIO_ANCHO + margen
    return np.array([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Modo pantalla (v2): sin cancha impresa, todo virtual; solo las manos son reales
# --------------------------------------------------------------------------- #
# La cancha virtual usa las mismas unidades que la física (la cancha de 25 cm de la
# v1). Una cámara virtual con la K real la encuadra para que ocupe toda la imagen;
# la red queda en la columna central y separa al jugador 1 (izq.) del 2 (der.).
# Radio del círculo inscrito en la palma de un adulto (~4 cm), corregido por lo que pierde la máscara
# en el borde (apertura morfológica): en las escenas sintéticas Z salía ~15 % alta con 4.0.
PALMA_RADIO_CM = 3.5
FRACCION_PISO = 0.86  # fila de la imagen (fracción de la altura) donde queda el piso de la cancha
OCUPACION_CANCHA = 0.92  # fracción del ancho de la imagen que ocupa la cancha
ANCHO_VISUAL_CANCHA = 3.0  # medio ancho (profundidad) del piso dibujado, en unidades de cancha
DISTANCIA_JUEGO_CM = 60.0  # distancia típica de las manos a la cámara (se mide al empezar)
