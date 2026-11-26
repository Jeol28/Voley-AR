"""Escenas sintéticas con verdad conocida.

Sirven para tres cosas:

1. **Pruebas automáticas**: se renderiza un tablero o la cancha con una ``K``,
   una distorsión y una pose conocidas, y se verifica que la calibración y
   ``solvePnP`` las recuperen.
2. **UI sin cámara**: cada pestaña puede cargar una escena de ejemplo.
3. **Modo de juego sintético**: una mano virtual que sigue al mouse permite
   jugar y depurar sin cámara web (``python jugar.py --sintetico``).

El render es geométricamente exacto: para cada píxel de salida se invierte la
distorsión, se lanza el rayo y se intersecta con el plano de la mesa (la
operación inversa de ``cv2.projectPoints``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from voleyar.core import config as cfg
from voleyar.core.calibracion import ModeloCamara, Pose

# Hoja carta horizontal (cm)
HOJA_ANCHO = 27.94
HOJA_ALTO = 21.59

COLOR_CAMPO_BGR = (176, 92, 22)  # azul cancha (no se parece a la piel)
COLOR_PIEL_BGR = (120, 150, 205)
COLOR_MESA_BGR = (92, 96, 98)  # gris escritorio


# =========================================================================== #
# Texturas planas
# =========================================================================== #


def textura_cancha(ppcm: float = 24.0) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Imagen de la hoja de la cancha (vista cenital) con el mismo diseño del PDF.

    Devuelve ``(textura, ppcm, origen)`` donde ``origen`` es el píxel de la
    textura que corresponde a ``(X, Y) = (0, 0)`` de la mesa. Un punto de la
    mesa va a ``u = X·ppcm + ox``, ``v = -Y·ppcm + oy``.
    """
    w, h = int(round(HOJA_ANCHO * ppcm)), int(round(HOJA_ALTO * ppcm))
    tex = np.full((h, w, 3), 255, np.uint8)
    ox, oy = w / 2, h / 2

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round(x * ppcm + ox)), int(round(-y * ppcm + oy))

    cv2.rectangle(tex, px(-cfg.MEDIO_LARGO, cfg.MEDIO_ANCHO), px(cfg.MEDIO_LARGO, -cfg.MEDIO_ANCHO),
                  COLOR_CAMPO_BGR, -1)
    g = max(1, int(0.25 * ppcm))
    cv2.rectangle(tex, px(-cfg.MEDIO_LARGO + 0.3, cfg.MEDIO_ANCHO - 0.3),
                  px(cfg.MEDIO_LARGO - 0.3, -cfg.MEDIO_ANCHO + 0.3), (255, 255, 255), g)
    cv2.line(tex, px(0, cfg.MEDIO_ANCHO), px(0, -cfg.MEDIO_ANCHO), (255, 255, 255), g)
    # Línea de ataque (3 m en voleibol real → proporcional aquí)
    for xa in (-4.2, 4.2):
        cv2.line(tex, px(xa, cfg.MEDIO_ANCHO - 0.3), px(xa, -cfg.MEDIO_ANCHO + 0.3), (235, 235, 235), max(1, g // 2))
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.DICCIONARIO_ARUCO))
    for i, (cx, cy) in cfg.CENTROS_MARCADORES.items():
        m = cfg.LADO_MARCADOR / 2 + cfg.MARGEN_BLANCO_MARCADOR
        cv2.rectangle(tex, px(cx - m, cy + m), px(cx + m, cy - m), (255, 255, 255), -1)
        lado = int(round(cfg.LADO_MARCADOR * ppcm))
        marcador = cv2.aruco.generateImageMarker(dic, i, lado)
        x0, y0 = px(cx - cfg.LADO_MARCADOR / 2, cy + cfg.LADO_MARCADOR / 2)
        tex[y0:y0 + lado, x0:x0 + lado] = cv2.cvtColor(marcador, cv2.COLOR_GRAY2BGR)
    return tex, ppcm, (ox, oy)


def textura_tablero(ppcm: float = 24.0, esquinas: tuple[int, int] = cfg.TABLERO_ESQUINAS,
                    cuadro: float = cfg.TABLERO_CUADRO_CM) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Tablero de ajedrez cuyo origen (0, 0) es la primera esquina interior.

    El eje Y del tablero se dibuja hacia abajo para que coincida con el orden
    de ``puntos_objeto_tablero`` (filas crecientes).
    """
    cols, filas = esquinas[0] + 1, esquinas[1] + 1
    margen = 1.5
    w = int(round((cols * cuadro + 2 * margen) * ppcm))
    h = int(round((filas * cuadro + 2 * margen) * ppcm))
    tex = np.full((h, w, 3), 255, np.uint8)
    for j in range(filas):
        for i in range(cols):
            if (i + j) % 2 == 0:
                x0 = int(round((margen + i * cuadro) * ppcm))
                y0 = int(round((margen + j * cuadro) * ppcm))
                x1 = int(round((margen + (i + 1) * cuadro) * ppcm))
                y1 = int(round((margen + (j + 1) * cuadro) * ppcm))
                tex[y0:y1, x0:x1] = 0
    ox = (margen + cuadro) * ppcm
    oy = (margen + cuadro) * ppcm
    # En el tablero Y crece hacia abajo → usamos v = +Y·ppcm + oy (signo distinto).
    return tex, ppcm, (ox, oy)


# =========================================================================== #
# Render geométrico de un plano
# =========================================================================== #


def _normalizar_inverso(u: np.ndarray, v: np.ndarray, K: np.ndarray, dist: np.ndarray,
                        iteraciones: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Invierte la distorsión por punto fijo (vectorizado para toda la imagen)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    yd = (v - cy) / fy
    xd = (u - cx - K[0, 1] * yd) / fx
    d = np.zeros(5)
    d[: min(5, len(dist))] = dist[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xd.copy(), yd.copy()
    for _ in range(iteraciones):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def renderizar_plano(textura: np.ndarray, ppcm: float, origen: tuple[float, float], modelo: ModeloCamara,
                     pose: Pose, fondo: np.ndarray | None = None, y_hacia_abajo: bool = False) -> np.ndarray:
    """Renderiza una textura plana (z = 0) vista por la cámara ``modelo`` en ``pose``."""
    w, h = modelo.ancho, modelo.alto
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xn, yn = _normalizar_inverso(uu, vv, modelo.K, modelo.dist)
    R = cv2.Rodrigues(np.asarray(pose.rvec, dtype=np.float64))[0]
    Hm = np.column_stack([R[:, 0], R[:, 1], np.asarray(pose.tvec, dtype=np.float64).reshape(3)])
    Hinv = np.linalg.inv(Hm)
    pts = np.stack([xn, yn, np.ones_like(xn)], axis=-1) @ Hinv.T
    valido = pts[..., 2] > 0
    X = pts[..., 0] / pts[..., 2]
    Y = pts[..., 1] / pts[..., 2]
    mapx = (X * ppcm + origen[0]).astype(np.float32)
    signo = 1.0 if y_hacia_abajo else -1.0
    mapy = (signo * Y * ppcm + origen[1]).astype(np.float32)
    mapx[~valido] = -1
    mapy[~valido] = -1
    out = cv2.remap(textura, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 2, 3))
    dentro = cv2.remap(np.full(textura.shape[:2], 255, np.uint8), mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if fondo is None:
        fondo = fondo_mesa(w, h)
    a = (dentro.astype(np.float32) / 255.0)[..., None]
    return (out * a + fondo * (1 - a)).astype(np.uint8)


def fondo_mesa(w: int, h: int, semilla: int = 0) -> np.ndarray:
    """Escritorio gris con textura suave y un degradado de iluminación."""
    rng = np.random.default_rng(semilla)
    ruido = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 3)
    grad = np.linspace(0.9, 1.1, w, dtype=np.float32)[None, :]
    base = np.array(COLOR_MESA_BGR, np.float32)[None, None, :]
    img = base * grad[..., None] + ruido[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def ruido_camara(img: np.ndarray, sigma: float = 4.0, semilla: int | None = None) -> np.ndarray:
    """Añade ruido gaussiano y un leve desenfoque, como una cámara web real."""
    rng = np.random.default_rng(semilla)
    out = cv2.GaussianBlur(img, (3, 3), 0.6).astype(np.float32)
    out += rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================================== #
# Cámaras y poses de ejemplo
# =========================================================================== #


def camara_ejemplo(ancho: int = 960, alto: int = 540, distorsion: bool = True) -> ModeloCamara:
    """Cámara web típica: HFOV ≈ 70°, distorsión de barril moderada."""
    f = ancho / (2 * np.tan(np.deg2rad(70) / 2))
    K = np.array([[f, 0, ancho / 2 + 6], [0, f * 1.002, alto / 2 - 4], [0, 0, 1]])
    dist = np.array([-0.32, 0.12, 0.0008, -0.0006, -0.02]) if distorsion else np.zeros(5)
    return ModeloCamara(K, dist, ancho, alto, True)


def pose_mirando(centro_camara: tuple[float, float, float], objetivo: tuple[float, float, float] = (0, 0, 0),
                 rodar: float = 0.0) -> Pose:
    """Pose de una cámara situada en ``centro_camara`` (cm, mesa) mirando a ``objetivo``."""
    C = np.asarray(centro_camara, dtype=np.float64)
    z = np.asarray(objetivo, dtype=np.float64) - C
    z /= np.linalg.norm(z)
    arriba = np.array([0, 0, 1.0])
    x = np.cross(z, arriba)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    if rodar:
        c, s = np.cos(np.deg2rad(rodar)), np.sin(np.deg2rad(rodar))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    t = -R @ C
    return Pose(cv2.Rodrigues(R)[0].ravel(), t)


def pose_juego_por_defecto() -> Pose:
    """Cámara del portátil detrás del lado rival, a ≈ 50 cm, mirando la cancha."""
    return pose_mirando((38.0, 2.0, 34.0), (-2.0, 0.0, 0.0))


# =========================================================================== #
# Mano sintética
# =========================================================================== #


def poligonos_mano(n_dedos: int, con_brazo: bool = True) -> list[np.ndarray]:
    """Silueta de una mano en cm, en un sistema local con la palma en el origen.

    Los dedos apuntan hacia +x y el antebrazo sale hacia -x. ``n_dedos``
    extendidos (0 = puño): índice, medio, anular, meñique y por último pulgar.
    """
    n_dedos = int(np.clip(n_dedos, 0, 5))
    polis: list[np.ndarray] = []
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    if n_dedos == 0:
        palma = np.column_stack([4.3 * np.cos(t), 4.0 * np.sin(t)])
    else:
        palma = np.column_stack([4.6 * np.cos(t), 4.1 * np.sin(t)])
    polis.append(palma)

    def rect(x0, y0, largo, ancho, ang_deg):
        a = np.deg2rad(ang_deg)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        p0 = np.array([x0, y0])
        pts = [p0 - n * ancho / 2, p0 + n * ancho / 2,
               p0 + n * ancho / 2 + d * largo, p0 - n * ancho / 2 + d * largo]
        # Punta redondeada
        tip = p0 + d * largo
        arco = [tip + (np.cos(b) * n + np.sin(b) * d) * ancho / 2 for b in np.linspace(0, np.pi, 9)]
        return np.array([pts[0], pts[1], *arco])

    # (base_x, base_y, largo, ancho, ángulo) – mano derecha vista desde arriba (palma abajo)
    dedos = [
        (3.4, 2.4, 7.0, 1.6, 18),    # índice (dedos algo separados, como al golpear con la mano abierta)
        (3.9, 0.65, 7.6, 1.65, 6),   # medio
        (3.7, -1.15, 7.0, 1.55, -7),  # anular
        (3.1, -2.85, 5.6, 1.4, -21),  # meñique
        (0.6, 3.4, 5.6, 2.0, 62),    # pulgar
    ]
    for d in dedos[:n_dedos]:
        polis.append(rect(*d))
    if con_brazo:
        # Antebrazo: angosto en la muñeca (5.4 cm) y más ancho hacia el codo (7.6 cm)
        polis.append(np.array([[-2.5, 2.7], [-2.5, -2.7], [-26.0, -3.8], [-26.0, 3.8]]))
    return polis


def mano_a_mesa(polis: list[np.ndarray], pos: tuple[float, float], angulo: float) -> list[np.ndarray]:
    a = np.deg2rad(angulo)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return [p @ R.T + np.asarray(pos)[None, :] for p in polis]


def silueta_mano_2d(n_dedos: int, escala: float = 22.0, angulo: float = 90.0, con_brazo: bool = True,
                    tam: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Máscara binaria de una mano (para probar el conteo de dedos).

    Los dedos apuntan hacia arriba en la imagen con ``angulo = 90`` y el brazo
    sale por el borde inferior.
    """
    h, w = tam
    polis = mano_a_mesa(poligonos_mano(n_dedos, con_brazo), (0, 0), angulo)
    centro = np.array([w / 2, h * 0.58])
    m = np.zeros((h, w), np.uint8)
    for p in polis:
        q = np.column_stack([p[:, 0] * escala + centro[0], -p[:, 1] * escala + centro[1]])
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 255)
    return m


@dataclass
class EstadoManoSintetica:
    pos: tuple[float, float] = (-8.0, 0.0)
    n_dedos: int = 5
    angulo: float = 0.0  # 0 → dedos hacia el rival (+x)
    altura: float = 1.5  # cm sobre la mesa


def dibujar_mano(img: np.ndarray, modelo: ModeloCamara, pose: Pose, mano: EstadoManoSintetica,
                 semilla: int = 0, con_piel: bool = False):
    """Proyecta la mano sintética (a ``mano.altura`` cm) sobre la imagen.

    Devuelve la imagen y la máscara verdadera de la mano (sin el brazo); con
    ``con_piel=True`` también la máscara de toda la piel (mano + antebrazo).
    """
    out = img.copy()
    polis = mano_a_mesa(poligonos_mano(mano.n_dedos, True), mano.pos, mano.angulo)
    verdad = np.zeros(img.shape[:2], np.uint8)
    capa = np.zeros(img.shape[:2], np.uint8)
    for k, p in enumerate(polis):
        p3 = np.column_stack([p, np.full(len(p), mano.altura)])
        px, _ = cv2.projectPoints(p3, pose.rvec, pose.tvec, modelo.K, modelo.dist)
        q = np.round(px.reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(capa, [q], 255)
        if k < len(polis) - 1:
            cv2.fillPoly(verdad, [q], 255)
    rng = np.random.default_rng(semilla)
    h, w = img.shape[:2]
    variacion = cv2.GaussianBlur(rng.normal(0, 10, (h, w)).astype(np.float32), (0, 0), 6)
    # Sombreado: más oscuro hacia los bordes de la silueta
    dist = cv2.distanceTransform(capa, cv2.DIST_L2, 5)
    sombra = np.clip(dist / 12.0, 0, 1) * 0.25 + 0.75
    piel = np.array(COLOR_PIEL_BGR, np.float32)[None, None, :] * sombra[..., None] + variacion[..., None]
    piel = np.clip(piel, 0, 255).astype(np.uint8)
    capa_suave = cv2.GaussianBlur(capa, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    out = (piel * capa_suave + out * (1 - capa_suave)).astype(np.uint8)
    return (out, verdad, capa) if con_piel else (out, verdad)


def escena_cancha(modelo: ModeloCamara | None = None, pose: Pose | None = None,
                  mano: EstadoManoSintetica | None = None, ruido: float = 3.0,
                  semilla: int = 0) -> tuple[np.ndarray, dict]:
    """Escena completa: escritorio + cancha impresa (+ mano opcional).

    Devuelve la imagen BGR y un diccionario con la verdad: modelo, pose y
    máscara de la mano.
    """
    modelo = modelo or camara_ejemplo()
    pose = pose or pose_juego_por_defecto()
    tex, ppcm, origen = _textura_cancha_cache()
    img = renderizar_plano(tex, ppcm, origen, modelo, pose, fondo_mesa(modelo.ancho, modelo.alto, semilla))
    verdad: dict = {"modelo": modelo, "pose": pose, "mascara_mano": None}
    if mano is not None:
        img, m, piel = dibujar_mano(img, modelo, pose, mano, semilla, con_piel=True)
        verdad["mascara_mano"] = m
        verdad["mascara_piel"] = piel
        verdad["mano"] = mano
    if ruido > 0:
        img = ruido_camara(img, ruido, semilla)
    return img, verdad


_tex_cache: tuple | None = None


def _textura_cancha_cache() -> tuple[np.ndarray, float, tuple[float, float]]:
    global _tex_cache
    if _tex_cache is None:
        _tex_cache = textura_cancha(24.0)
    return _tex_cache


class FuenteSintetica:
    """Fuente de video sintética: la mano virtual se controla desde fuera.

    Para que sea rápida, el fondo (mesa + cancha) se renderiza una sola vez y
    en cada frame solo se dibuja la mano y el ruido.
    """

    def __init__(self, modelo: ModeloCamara | None = None, pose: Pose | None = None):
        self.modelo = modelo or camara_ejemplo()
        self.pose = pose or pose_juego_por_defecto()
        self.mano = EstadoManoSintetica()
        self.mostrar_mano = True
        self._fondo, _ = escena_cancha(self.modelo, self.pose, None, ruido=0)
        self._i = 0

    def leer(self) -> tuple[bool, np.ndarray]:
        self._i += 1
        img = self._fondo
        if self.mostrar_mano:
            img, _ = dibujar_mano(img, self.modelo, self.pose, self.mano, semilla=self._i % 7)
        return True, ruido_camara(img, 2.5, self._i % 5)

    def liberar(self) -> None:  # interfaz común con la cámara
        pass


# =========================================================================== #
# Modo pantalla (v2): dos personas frente a la cámara, sin cancha impresa
# =========================================================================== #

PIELES_JUGADORES = [(120, 150, 205), (80, 112, 168)]  # dos tonos de piel distintos (BGR)
COLOR_PARED = (150, 140, 128)
COLOR_CAMISA = [(70, 50, 35), (40, 55, 40)]


@dataclass
class JugadorSintetico:
    """Una persona frente a la cámara (coordenadas de la cámara, cm: x derecha, y abajo, z adelante)."""

    mano: tuple[float, float] = (-12.0, 7.0)  # centro de la palma (x, y) en cm
    distancia: float = 60.0  # z de la mano en cm
    n_dedos: int = 5
    angulo: float = 90.0  # 90 = dedos hacia arriba
    cara: tuple[float, float] = (-28.0, -20.0)  # centro de la cara (x, y) en cm, 15 cm detrás de la mano
    piel: tuple[int, int, int] = (120, 150, 205)
    camisa: tuple[int, int, int] = (70, 50, 35)
    visible: bool = True
    mano_visible: bool = True  # la persona sigue en la imagen aunque no juegue (modo contra la IA)


def jugadores_por_defecto(n: int = 2) -> list[JugadorSintetico]:
    j = [JugadorSintetico((-12.0, 7.0), 60.0, 5, 90.0, (-28.0, -20.0), PIELES_JUGADORES[0], COLOR_CAMISA[0])]
    if n > 1:
        j.append(JugadorSintetico((13.0, 8.0), 64.0, 1, 100.0, (29.0, -19.0), PIELES_JUGADORES[1], COLOR_CAMISA[1]))
    return j


def _proyectar_cam(pts_cm: np.ndarray, modelo: ModeloCamara) -> np.ndarray:
    px, _ = cv2.projectPoints(np.asarray(pts_cm, np.float64).reshape(-1, 3), np.zeros(3), np.zeros(3),
                              modelo.K, modelo.dist)
    return px.reshape(-1, 2)


def _dibujar_cara(img, cx, cy, s, piel) -> tuple[int, int, int, int]:
    """Cara de dibujo (la cascada Haar la reconoce). Devuelve su caja (x, y, w, h)."""
    oscuro = tuple(int(c * 0.55) for c in piel)
    cv2.ellipse(img, (cx, cy), (int(0.42 * s), int(0.55 * s)), 0, 0, 360, piel, -1, cv2.LINE_AA)
    for dx in (-0.17, 0.17):
        ex, ey = int(cx + dx * s), int(cy - 0.1 * s)
        cv2.line(img, (ex - int(0.11 * s), ey - int(0.12 * s)), (ex + int(0.11 * s), ey - int(0.13 * s)), (40, 45, 55),
                 max(2, int(0.035 * s)))
        cv2.ellipse(img, (ex, ey), (int(0.09 * s), int(0.045 * s)), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (ex, ey), int(0.04 * s), (40, 30, 30), -1)
    cv2.line(img, (cx, cy - int(0.05 * s)), (cx - int(0.04 * s), cy + int(0.12 * s)), oscuro, max(2, int(0.025 * s)))
    cv2.ellipse(img, (cx, cy + int(0.27 * s)), (int(0.15 * s), int(0.05 * s)), 0, 0, 180, (70, 70, 150),
                max(2, int(0.03 * s)))
    cv2.ellipse(img, (cx, cy - int(0.5 * s)), (int(0.45 * s), int(0.2 * s)), 0, 180, 360, (30, 35, 45), -1)
    return (int(cx - 0.42 * s), int(cy - 0.55 * s), int(0.84 * s), int(1.1 * s))


def fondo_pantalla(modelo: ModeloCamara, jugadores: list[JugadorSintetico], semilla: int = 0):
    """Pared, torsos con camisa y caras (sin manos). Devuelve la imagen y las cajas de las caras."""
    w, h = modelo.ancho, modelo.alto
    rng = np.random.default_rng(semilla)
    img = np.empty((h, w, 3), np.float32)
    img[:] = COLOR_PARED
    img *= np.linspace(1.08, 0.85, h, dtype=np.float32)[:, None, None]
    img += cv2.GaussianBlur(rng.normal(0, 5, (h, w)).astype(np.float32), (0, 0), 4)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    caras = []
    f = modelo.K[0, 0]
    for j in jugadores:
        if not j.visible:
            continue
        zc = j.distancia + 15.0
        c = _proyectar_cam([[j.cara[0], j.cara[1], zc]], modelo)[0]
        s = f * 20.0 / zc
        # Torso (camisa oscura) desde los hombros hasta abajo
        hombros = _proyectar_cam([[j.cara[0] - 20, j.cara[1] + 15, zc], [j.cara[0] + 20, j.cara[1] + 15, zc],
                                  [j.cara[0] + 24, j.cara[1] + 80, zc], [j.cara[0] - 24, j.cara[1] + 80, zc]], modelo)
        cv2.fillPoly(img, [np.round(hombros).astype(np.int32)], j.camisa, cv2.LINE_AA)
        cuello = _proyectar_cam([[j.cara[0] - 5, j.cara[1] + 6, zc], [j.cara[0] + 5, j.cara[1] + 6, zc],
                                 [j.cara[0] + 5, j.cara[1] + 16, zc], [j.cara[0] - 5, j.cara[1] + 16, zc]], modelo)
        cv2.fillPoly(img, [np.round(cuello).astype(np.int32)], tuple(int(v * 0.9) for v in j.piel), cv2.LINE_AA)
        caras.append(_dibujar_cara(img, int(c[0]), int(c[1]), s, j.piel))
    return cv2.GaussianBlur(img, (3, 3), 0.8), caras


def dibujar_mano_pantalla(img: np.ndarray, modelo: ModeloCamara, j: JugadorSintetico, semilla: int = 0):
    """Mano levantada frente a la cámara (plano z = distancia). Devuelve imagen, mano verdadera y toda la piel."""
    polis = mano_a_mesa(poligonos_mano(j.n_dedos, True), (0.0, 0.0), j.angulo)
    out = img.copy()
    verdad = np.zeros(img.shape[:2], np.uint8)
    capa = np.zeros(img.shape[:2], np.uint8)
    for k, p in enumerate(polis):
        # local (x, y-arriba) → cámara (x, y-abajo)
        p3 = np.column_stack([j.mano[0] + p[:, 0], j.mano[1] - p[:, 1], np.full(len(p), j.distancia)])
        q = np.round(_proyectar_cam(p3, modelo)).astype(np.int32)
        cv2.fillPoly(capa, [q], 255)
        if k < len(polis) - 1:
            cv2.fillPoly(verdad, [q], 255)
    rng = np.random.default_rng(semilla)
    h, w = img.shape[:2]
    variacion = cv2.GaussianBlur(rng.normal(0, 9, (h, w)).astype(np.float32), (0, 0), 6)
    dist = cv2.distanceTransform(capa, cv2.DIST_L2, 5)
    sombra = np.clip(dist / 14.0, 0, 1) * 0.22 + 0.78
    piel = np.clip(np.array(j.piel, np.float32)[None, None, :] * sombra[..., None] + variacion[..., None], 0, 255)
    a = cv2.GaussianBlur(capa, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    out = (piel * a + out * (1 - a)).astype(np.uint8)
    return out, verdad, capa


def escena_pantalla(modelo: ModeloCamara | None = None, jugadores: list[JugadorSintetico] | None = None,
                    ruido: float = 3.0, semilla: int = 0) -> tuple[np.ndarray, dict]:
    """Escena del modo pantalla con su verdad: máscaras de cada mano, caras y distancias."""
    modelo = modelo or camara_ejemplo(1280, 720)
    jugadores = jugadores_por_defecto() if jugadores is None else jugadores
    img, caras = fondo_pantalla(modelo, jugadores, semilla)
    manos, pieles = [], []
    for i, j in enumerate(jugadores):
        if j.visible and j.mano_visible:
            img, m, p = dibujar_mano_pantalla(img, modelo, j, semilla + i)
        else:
            m = p = np.zeros(img.shape[:2], np.uint8)
        manos.append(m)
        pieles.append(p)
    if ruido > 0:
        img = ruido_camara(img, ruido, semilla)
    return img, {"modelo": modelo, "jugadores": jugadores, "mascaras_mano": manos, "mascaras_piel": pieles,
                 "caras": caras}


class FuentePantalla:
    """Fuente sintética del modo pantalla: las manos de los jugadores se mueven desde fuera."""

    def __init__(self, modelo: ModeloCamara | None = None, n_jugadores: int = 2):
        self.modelo = modelo or camara_ejemplo(1280, 720)
        self.jugadores = jugadores_por_defecto(n_jugadores)
        self._fondo, self.caras = fondo_pantalla(self.modelo, self.jugadores)
        self._i = 0

    @property
    def mano(self) -> JugadorSintetico:  # compatibilidad: el jugador 1 se controla con el mouse
        return self.jugadores[0]

    def leer(self) -> tuple[bool, np.ndarray]:
        self._i += 1
        img = self._fondo
        for k, j in enumerate(self.jugadores):
            if j.visible and j.mano_visible:
                img, _, _ = dibujar_mano_pantalla(img, self.modelo, j, semilla=(self._i + k) % 7)
        return True, ruido_camara(img, 2.5, self._i % 5)

    def liberar(self) -> None:
        pass
