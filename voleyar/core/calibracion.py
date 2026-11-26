r"""Modelos de cámara y calibración.

Este módulo es la base geométrica del juego. Sin la matriz intrínseca ``K`` y los
coeficientes de distorsión de **tu** cámara web, la red y el balón virtuales no
quedan "pegados" a la cancha impresa: flotan y se deslizan cuando mueves la hoja.

Modelo pinhole (cámara estenopeica)
-----------------------------------
Un punto 3D :math:`P_w = (X, Y, Z)` del sistema de la mesa pasa al sistema de la
cámara con los **parámetros extrínsecos** (rotación ``R`` y traslación ``t``):

.. math::  P_c = R\,P_w + t

y se proyecta en la imagen con los **parámetros intrínsecos**:

.. math::

    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
    K \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix},\qquad
    K = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}

- :math:`f_x, f_y`: distancia focal en píxeles (focal física / tamaño del píxel).
- :math:`c_x, c_y`: punto principal (donde el eje óptico corta el sensor).
- :math:`\gamma`: sesgo entre ejes (≈ 0 en cámaras modernas).

Distorsión
----------
Las lentes gran angular de las cámaras web curvan las rectas. Sobre las
coordenadas normalizadas :math:`x = x_c/z_c,\; y = y_c/z_c` y con
:math:`r^2 = x^2 + y^2`:

- **Radial** (barril / cojín, por la curvatura de la lente):

  .. math::  x_r = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)

- **Tangencial** (lente no paralela al sensor):

  .. math::

     x_t = 2 p_1 x y + p_2 (r^2 + 2x^2),\qquad
     y_t = p_1 (r^2 + 2y^2) + 2 p_2 x y

OpenCV usa el vector ``dist = [k1, k2, p1, p2, k3]``.

Calibración intrínseca (Zhang)
------------------------------
Con varias vistas de un tablero plano de medidas conocidas, cada vista da una
homografía plano → imagen. Con ≥ 3 vistas en orientaciones distintas se despeja
``K`` en forma cerrada y luego se refina todo (``K``, ``dist`` y la pose de cada
vista) por mínimos cuadrados no lineales (Levenberg-Marquardt) minimizando el
**error de reproyección**:

.. math::  e_{RMS} = \sqrt{\tfrac{1}{N}\sum_i \lVert p_i - \hat p_i(K, d, R_j, t_j)\rVert^2}

Por eso la captura guiada pide **variedad de ángulos**: vistas paralelas entre sí
no aportan información sobre la focal.

Pose de la cancha (extrínsecos)
-------------------------------
Las esquinas de los 4 marcadores ArUco tienen coordenadas 3D conocidas (cm) en
la mesa. ``cv2.solvePnP`` encuentra ``R, t`` que llevan esos puntos a sus
detecciones en la imagen, dados ``K`` y ``dist``. Con la pose se puede:

- proyectar cualquier objeto 3D (red, balón) con ``cv2.projectPoints``;
- hacer lo inverso: llevar un píxel a la mesa intersectando su rayo con ``z = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import config as cfg

# =========================================================================== #
# Modelo de cámara
# =========================================================================== #


@dataclass
class ModeloCamara:
    """Parámetros intrínsecos de una cámara y la resolución en que se midieron."""

    K: np.ndarray
    dist: np.ndarray
    ancho: int
    alto: int
    calibrada: bool = True
    error_rms: float | None = None
    n_vistas: int = 0
    tam_cuadro_cm: float | None = None

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)

    @property
    def tam(self) -> tuple[int, int]:
        return (self.ancho, self.alto)

    def escalar_a(self, ancho: int, alto: int) -> "ModeloCamara":
        """Adapta ``K`` a otra resolución con la misma relación de aspecto.

        Si la cámara entrega 1280×720 y se calibró a 1920×1080, la focal y el
        punto principal escalan en la misma proporción; la distorsión (que
        actúa sobre coordenadas normalizadas) no cambia.
        """
        if (ancho, alto) == self.tam:
            return self
        sx, sy = ancho / self.ancho, alto / self.alto
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return ModeloCamara(K, self.dist.copy(), ancho, alto, self.calibrada,
                            self.error_rms, self.n_vistas, self.tam_cuadro_cm)

    def espejo(self) -> "ModeloCamara":
        r"""Modelo de la misma cámara con la imagen volteada horizontalmente (vista de espejo).

        Voltear :math:`u \to (W - 1) - u` equivale a :math:`x \to -x` en coordenadas
        normalizadas: el punto principal pasa a :math:`c_x' = W - 1 - c_x`; la distorsión
        radial no cambia y en la tangencial :math:`p_2' = -p_2` (:math:`p_1` se conserva).
        """
        K = self.K.copy()
        K[0, 2] = self.ancho - 1 - K[0, 2]
        K[0, 1] = -K[0, 1]
        dist = self.dist.copy()
        if len(dist) > 3:
            dist[3] = -dist[3]
        return ModeloCamara(K, dist, self.ancho, self.alto, self.calibrada, self.error_rms, self.n_vistas,
                            self.tam_cuadro_cm)

    def a_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "ancho": self.ancho,
            "alto": self.alto,
            "calibrada": self.calibrada,
            "error_rms": self.error_rms,
            "n_vistas": self.n_vistas,
            "tam_cuadro_cm": self.tam_cuadro_cm,
        }

    @classmethod
    def desde_dict(cls, d: dict) -> "ModeloCamara":
        return cls(
            K=np.array(d["K"]),
            dist=np.array(d["dist"]),
            ancho=int(d["ancho"]),
            alto=int(d["alto"]),
            calibrada=bool(d.get("calibrada", True)),
            error_rms=d.get("error_rms"),
            n_vistas=int(d.get("n_vistas", 0)),
            tam_cuadro_cm=d.get("tam_cuadro_cm"),
        )


def modelo_aproximado(ancho: int, alto: int) -> ModeloCamara:
    """Modelo "sin calibrar": la suposición ingenua que hace cualquier demo AR.

    - :math:`f_x = f_y = \\text{ancho}` (equivale a un campo de visión
      horizontal de ≈ 53°, típico pero no exacto).
    - Punto principal en el centro de la imagen.
    - Sin distorsión.

    Sirve de comparación: con esta ``K`` la pose que da solvePnP compensa el
    error con una distancia/inclinación falsas, así que los objetos con altura
    (la red, el balón en el aire) quedan desalineados y "flotan".
    """
    K = np.array([[ancho, 0, ancho / 2], [0, ancho, alto / 2], [0, 0, 1]], dtype=np.float64)
    return ModeloCamara(K, np.zeros(5), ancho, alto, calibrada=False)


# =========================================================================== #
# Corrección de distorsión
# =========================================================================== #

_cache_mapas: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def modelo_rectificado(modelo: ModeloCamara, alpha: float = 0.0) -> tuple[ModeloCamara, np.ndarray, np.ndarray]:
    """Calcula la cámara ideal (sin distorsión) y los mapas de ``remap``.

    ``alpha = 0`` recorta para que no queden bordes negros; ``alpha = 1``
    conserva todos los píxeles originales. Los mapas se guardan en caché porque
    calcularlos cuesta más que aplicarlos.
    """
    clave = (modelo.K.tobytes(), modelo.dist.tobytes(), modelo.ancho, modelo.alto, alpha)
    if clave not in _cache_mapas:
        K_nueva, _ = cv2.getOptimalNewCameraMatrix(modelo.K, modelo.dist, modelo.tam, alpha, modelo.tam)
        mapa1, mapa2 = cv2.initUndistortRectifyMap(
            modelo.K, modelo.dist, None, K_nueva, modelo.tam, cv2.CV_16SC2
        )
        _cache_mapas[clave] = (K_nueva, mapa1, mapa2)
    K_nueva, mapa1, mapa2 = _cache_mapas[clave]
    rect = ModeloCamara(K_nueva, np.zeros(5), modelo.ancho, modelo.alto, True,
                        modelo.error_rms, modelo.n_vistas, modelo.tam_cuadro_cm)
    return rect, mapa1, mapa2


def corregir_distorsion(img: np.ndarray, modelo: ModeloCamara, alpha: float = 0.0) -> tuple[np.ndarray, ModeloCamara]:
    """Quita la distorsión de ``img`` y devuelve también el modelo equivalente.

    La imagen resultante cumple el modelo pinhole puro (rectas → rectas) con la
    nueva matriz ``K'`` y distorsión cero, que es lo que usa el resto del pipeline.
    """
    h, w = img.shape[:2]
    modelo = modelo.escalar_a(w, h)
    if not modelo.calibrada or not np.any(modelo.dist):
        return img, modelo
    rect, m1, m2 = modelo_rectificado(modelo, alpha)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR), rect


# =========================================================================== #
# Calibración intrínseca con tablero de ajedrez
# =========================================================================== #


def puntos_objeto_tablero(esquinas: tuple[int, int] = cfg.TABLERO_ESQUINAS,
                          tam_cuadro: float = cfg.TABLERO_CUADRO_CM) -> np.ndarray:
    """Coordenadas 3D (cm, z = 0) de las esquinas interiores del tablero."""
    cols, filas = esquinas
    obj = np.zeros((cols * filas, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:filas].T.reshape(-1, 2) * tam_cuadro
    return obj


def detectar_tablero(img: np.ndarray, esquinas: tuple[int, int] = cfg.TABLERO_ESQUINAS,
                     rapido: bool = False) -> tuple[bool, np.ndarray | None]:
    """Busca las esquinas interiores del tablero y las refina a subpíxel.

    ``cornerSubPix`` ajusta cada esquina buscando el punto donde los gradientes
    de su vecindad son ortogonales al vector que los une con él (precisión
    ≈ 0.1 px), lo que reduce mucho el error de reproyección final.

    Con ``rapido=True`` se trabaja sobre una copia reducida (útil en vivo).
    """
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    escala = 1.0
    busqueda = gris
    if rapido and max(gris.shape) > 800:
        escala = 640 / max(gris.shape)
        busqueda = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    banderas = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if rapido:
        banderas |= cv2.CALIB_CB_FAST_CHECK
    ok, pts = cv2.findChessboardCorners(busqueda, esquinas, banderas)
    if not ok:
        return False, None
    pts = (pts / escala).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    pts = cv2.cornerSubPix(gris, pts, (11, 11), (-1, -1), criterio)
    return True, pts


@dataclass
class ResultadoCalibracion:
    modelo: ModeloCamara
    errores_por_vista: list[float]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]


def calibrar_camara(lista_esquinas: list[np.ndarray], tam_imagen: tuple[int, int],
                    esquinas: tuple[int, int] = cfg.TABLERO_ESQUINAS,
                    tam_cuadro: float = cfg.TABLERO_CUADRO_CM) -> ResultadoCalibracion:
    """Calibración intrínseca con ``cv2.calibrateCamera`` (método de Zhang).

    Parameters
    ----------
    lista_esquinas: detecciones de ``detectar_tablero`` (una por vista).
    tam_imagen: ``(ancho, alto)`` en píxeles.
    tam_cuadro: tamaño **medido con regla** del cuadro impreso (cm). Si la
        impresora escaló la hoja, la focal sigue siendo correcta (no depende de
        la escala del tablero) pero las distancias de las poses del tablero no.
    """
    if len(lista_esquinas) < 3:
        raise ValueError("Se necesitan al menos 3 vistas del tablero (idealmente 12-20).")
    obj = puntos_objeto_tablero(esquinas, tam_cuadro)
    objetos = [obj] * len(lista_esquinas)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objetos, lista_esquinas, tam_imagen, None, None)
    errores = []
    for o, p, r, t in zip(objetos, lista_esquinas, rvecs, tvecs):
        proy, _ = cv2.projectPoints(o, r, t, K, dist)
        errores.append(float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - p.reshape(-1, 2)) ** 2, axis=1)))))
    modelo = ModeloCamara(K, dist.ravel(), tam_imagen[0], tam_imagen[1], True,
                          float(rms), len(lista_esquinas), tam_cuadro)
    return ResultadoCalibracion(modelo, errores, list(rvecs), list(tvecs))


@dataclass
class GestorCapturaCalibracion:
    """Lógica (sin UI) de la captura guiada de vistas del tablero.

    Acepta una vista nueva solo si es **distinta** de las anteriores: el
    tablero debe haberse movido en la imagen o cambiado de inclinación. Además
    lleva el control de qué zonas de la imagen ya se cubrieron, para sugerir
    al usuario hacia dónde mover el tablero.
    """

    tam_imagen: tuple[int, int]
    objetivo: int = 15
    esquinas: tuple[int, int] = cfg.TABLERO_ESQUINAS
    vistas: list[np.ndarray] = field(default_factory=list)
    _firmas: list[np.ndarray] = field(default_factory=list)

    def _firma(self, pts: np.ndarray) -> np.ndarray:
        """Resume una vista: centro, tamaño aparente e inclinación (normalizados)."""
        p = pts.reshape(-1, 2)
        w, h = self.tam_imagen
        centro = p.mean(axis=0) / np.array([w, h])
        area = cv2.contourArea(cv2.convexHull(p.astype(np.float32))) / (w * h)
        # Inclinación: razón (log) entre lados opuestos del cuadrilátero exterior.
        cols, filas = self.esquinas
        q = p.reshape(filas, cols, 2)
        sup, inf = np.linalg.norm(q[0, -1] - q[0, 0]), np.linalg.norm(q[-1, -1] - q[-1, 0])
        izq, der = np.linalg.norm(q[-1, 0] - q[0, 0]), np.linalg.norm(q[-1, -1] - q[0, -1])
        incl_h = np.log((sup + 1e-6) / (inf + 1e-6))
        incl_v = np.log((izq + 1e-6) / (der + 1e-6))
        return np.array([centro[0], centro[1], np.sqrt(area), incl_h * 2, incl_v * 2])

    def es_vista_nueva(self, pts: np.ndarray, umbral: float = 0.12) -> bool:
        if not self._firmas:
            return True
        f = self._firma(pts)
        dist = min(np.linalg.norm(f - g) for g in self._firmas)
        return dist > umbral

    def agregar(self, pts: np.ndarray, forzar: bool = False) -> bool:
        if forzar or self.es_vista_nueva(pts):
            self.vistas.append(pts.copy())
            self._firmas.append(self._firma(pts))
            return True
        return False

    def reiniciar(self) -> None:
        self.vistas.clear()
        self._firmas.clear()

    @property
    def listo(self) -> bool:
        return len(self.vistas) >= self.objetivo

    def sugerencia(self) -> str:
        """Consejo para la siguiente vista según lo que falta cubrir."""
        n = len(self.vistas)
        if n == 0:
            return "Muestra el tablero completo a la cámara."
        f = np.array(self._firmas)
        if n >= self.objetivo:
            return "¡Listo! Ya puedes calibrar."
        if np.abs(f[:, 3]).max() < 0.25:
            return "Inclina el tablero hacia adelante o hacia atrás (≈ 30°)."
        if np.abs(f[:, 4]).max() < 0.25:
            return "Gira el tablero hacia la izquierda o la derecha (≈ 30°)."
        # Zonas: 3x3 de la imagen
        zonas = set((min(2, int(cx * 3)), min(2, int(cy * 3))) for cx, cy in f[:, :2])
        nombres = {(0, 0): "arriba a la izquierda", (1, 0): "arriba", (2, 0): "arriba a la derecha",
                   (0, 1): "a la izquierda", (1, 1): "al centro", (2, 1): "a la derecha",
                   (0, 2): "abajo a la izquierda", (1, 2): "abajo", (2, 2): "abajo a la derecha"}
        for z in [(0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (1, 2), (0, 1), (2, 1)]:
            if z not in zonas:
                return f"Lleva el tablero {nombres[z]} de la imagen."
        if f[:, 2].max() < 0.5:
            return "Acerca el tablero a la cámara para que ocupe más de la imagen."
        return "Sigue variando ángulo y distancia."


# =========================================================================== #
# Marcadores ArUco y pose de la cancha
# =========================================================================== #


def crear_detector_aruco() -> cv2.aruco.ArucoDetector:
    """Detector ArUco con refinamiento subpíxel de esquinas (mejor pose)."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.DICCIONARIO_ARUCO))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dic, params)


_detector_por_defecto: cv2.aruco.ArucoDetector | None = None


def detectar_marcadores(img: np.ndarray, detector: cv2.aruco.ArucoDetector | None = None) -> dict[int, np.ndarray]:
    """Detecta los marcadores de la cancha. Devuelve ``{id: esquinas (4×2)}``."""
    global _detector_por_defecto
    if detector is None:
        if _detector_por_defecto is None:
            _detector_por_defecto = crear_detector_aruco()
        detector = _detector_por_defecto
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    esquinas, ids, _ = detector.detectMarkers(gris)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), esquinas) if int(i) in cfg.CENTROS_MARCADORES}


@dataclass
class Pose:
    """Transformación mesa → cámara (``P_c = R P_w + t``)."""

    rvec: np.ndarray
    tvec: np.ndarray
    error_rms: float = 0.0
    ids: tuple[int, ...] = ()
    puntos_img: np.ndarray | None = None
    puntos_obj: np.ndarray | None = None

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def centro_camara(self) -> np.ndarray:
        """Posición del centro óptico en coordenadas de la mesa: ``C = -Rᵀ t``."""
        return (-self.R.T @ self.tvec.reshape(3)).ravel()


def estimar_pose_cancha(marcadores: dict[int, np.ndarray], modelo: ModeloCamara,
                        pose_previa: Pose | None = None) -> Pose | None:
    """Estima la pose de la cancha con ``solvePnP`` sobre las esquinas ArUco.

    Con un solo marcador ya hay 4 puntos coplanares (suficiente con
    ``SOLVEPNP_IPPE``), pero usar los 4 marcadores (16 puntos repartidos por la
    hoja) hace la pose mucho más estable. Si hay una pose previa, se usa como
    semilla (``useExtrinsicGuess``) para evitar saltos entre frames.
    """
    if not marcadores:
        return None
    ids = tuple(sorted(marcadores))
    obj = np.concatenate([cfg.esquinas_marcador(i) for i in ids]).astype(np.float64)
    img = np.concatenate([marcadores[i] for i in ids]).astype(np.float64)
    if len(ids) == 1:
        ok, rvec, tvec = cv2.solvePnP(obj, img, modelo.K, modelo.dist, flags=cv2.SOLVEPNP_IPPE)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, modelo.K, modelo.dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            ok, rvec, tvec = cv2.solvePnP(obj, img, modelo.K, modelo.dist, rvec, tvec,
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # La cancha está delante de la cámara: z_c > 0 para el origen.
    if float(np.ravel(tvec)[2]) <= 0:
        return None
    pose = Pose(rvec.reshape(3), tvec.reshape(3), 0.0, ids, img, obj)
    pose.error_rms = error_reproyeccion(pose, modelo)
    return pose


def error_reproyeccion(pose: Pose, modelo: ModeloCamara) -> float:
    """RMS (px) entre las esquinas ArUco detectadas y las reproyectadas con la pose."""
    if pose.puntos_obj is None or pose.puntos_img is None:
        return float("nan")
    proy = proyectar(pose.puntos_obj, pose, modelo)
    return float(np.sqrt(np.mean(np.sum((proy - pose.puntos_img) ** 2, axis=1))))


def proyectar(puntos: np.ndarray, pose: Pose, modelo: ModeloCamara) -> np.ndarray:
    """Proyecta puntos 3D de la mesa (cm) a píxeles con el modelo completo."""
    puntos = np.asarray(puntos, dtype=np.float64).reshape(-1, 3)
    px, _ = cv2.projectPoints(puntos, pose.rvec, pose.tvec, modelo.K, modelo.dist)
    return px.reshape(-1, 2)


def profundidad_camara(puntos: np.ndarray, pose: Pose) -> np.ndarray:
    """Coordenada ``z_c`` (cm) de puntos de la mesa en el sistema de la cámara."""
    puntos = np.asarray(puntos, dtype=np.float64).reshape(-1, 3)
    return (puntos @ pose.R.T + pose.tvec.reshape(1, 3))[:, 2]


def rayo_desde_pixel(px: tuple[float, float], pose: Pose, modelo: ModeloCamara) -> tuple[np.ndarray, np.ndarray]:
    """Rayo de visión de un píxel, en coordenadas de la mesa: ``origen + λ·dirección``.

    1. ``undistortPoints`` invierte ``K`` y la distorsión → (x, y) normalizados.
    2. En la cámara el rayo es ``λ (x, y, 1)``.
    3. Se lleva a la mesa con la pose inversa: ``P_w = Rᵀ (P_c - t)``.
    """
    norm = cv2.undistortPoints(np.array([[px]], dtype=np.float64), modelo.K, modelo.dist).reshape(2)
    d_cam = np.array([norm[0], norm[1], 1.0])
    R = pose.R
    origen = pose.centro_camara
    direccion = R.T @ d_cam
    return origen, direccion / np.linalg.norm(direccion)


def pixel_a_mesa(px: tuple[float, float], pose: Pose, modelo: ModeloCamara, z: float = 0.0) -> np.ndarray | None:
    """Intersección del rayo de un píxel con el plano horizontal ``Z = z`` (cm).

    Resolviendo :math:`o_z + \\lambda d_z = z` → :math:`\\lambda = (z - o_z)/d_z`.
    Devuelve ``None`` si el rayo es paralelo al plano o el corte queda detrás.
    """
    o, d = rayo_desde_pixel(px, pose, modelo)
    if abs(d[2]) < 1e-9:
        return None
    lam = (z - o[2]) / d[2]
    if lam <= 0:
        return None
    return o + lam * d


def distancia_rayo_punto(origen: np.ndarray, direccion: np.ndarray, punto: np.ndarray) -> float:
    """Distancia mínima (cm) entre un rayo y un punto 3D (p. ej. el centro del balón)."""
    v = np.asarray(punto, dtype=np.float64) - origen
    lam = max(0.0, float(v @ direccion))
    return float(np.linalg.norm(v - lam * direccion))


def pixeles_por_cm(pose: Pose, modelo: ModeloCamara, punto: tuple[float, float, float] = (0, 0, 0)) -> float:
    """Escala local aproximada (px/cm) en un punto de la mesa: :math:`f / z_c`."""
    z = float(profundidad_camara(np.array([punto]), pose)[0])
    return float((modelo.K[0, 0] + modelo.K[1, 1]) / 2 / max(z, 1e-6))


def dibujar_ejes(img: np.ndarray, pose: Pose, modelo: ModeloCamara, largo: float = 6.0,
                 grosor: int = 3) -> np.ndarray:
    """Dibuja los ejes X (rojo), Y (verde) y Z (azul) del sistema de la mesa."""
    out = img.copy()
    cv2.drawFrameAxes(out, modelo.K, modelo.dist, pose.rvec, pose.tvec, largo, grosor)
    return out


def comparar_modos(marcadores: dict[int, np.ndarray], modelo_calibrado: ModeloCamara | None,
                   ancho: int, alto: int) -> dict[str, tuple[Pose | None, ModeloCamara]]:
    """Calcula la pose de la cancha con la K calibrada y con la K aproximada.

    Ambas se calculan sobre la imagen **original** (con distorsión), así la
    comparación es justa: el modelo calibrado modela la distorsión y el
    aproximado no.
    """
    res: dict[str, tuple[Pose | None, ModeloCamara]] = {}
    aprox = modelo_aproximado(ancho, alto)
    res["sin_calibrar"] = (estimar_pose_cancha(marcadores, aprox), aprox)
    if modelo_calibrado is not None:
        m = modelo_calibrado.escalar_a(ancho, alto)
        res["calibrado"] = (estimar_pose_cancha(marcadores, m), m)
    return res


# =========================================================================== #
# Vista cenital métrica (rectificación del plano de la mesa)
# =========================================================================== #


@dataclass
class VistaCenital:
    """Homografía imagen → vista desde arriba del plano ``Z = z`` (``ppcm`` px por cm).

    Un punto de la mesa ``(X, Y)`` cae en el píxel ``u = (X - x_min)·ppcm``,
    ``v = (y_max - Y)·ppcm`` de la vista cenital.
    """

    H: np.ndarray  # imagen → cenital
    tam: tuple[int, int]
    ppcm: float
    x_min: float
    y_max: float
    z: float = 0.0

    def a_mesa(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.column_stack([uv[:, 0] / self.ppcm + self.x_min, self.y_max - uv[:, 1] / self.ppcm,
                                np.full(len(uv), self.z)])

    def a_imagen(self, uv) -> np.ndarray:
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(uv, np.linalg.inv(self.H)).reshape(-1, 2)

    def desde_imagen(self, px) -> np.ndarray:
        px = np.asarray(px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(px, self.H).reshape(-1, 2)

    def rectificar(self, img: np.ndarray, interpolacion: int = cv2.INTER_NEAREST) -> np.ndarray:
        return cv2.warpPerspective(img, self.H, self.tam, flags=interpolacion)


def vista_cenital(pose: Pose, modelo: ModeloCamara, ppcm: float = 10.0,
                  limites: tuple[float, float, float, float] | None = None, z: float = 0.0) -> VistaCenital:
    r"""Homografía que "endereza" el plano de la mesa a una vista desde arriba.

    Para los puntos del plano :math:`Z = z`, el modelo pinhole se reduce a una
    homografía :math:`\tilde p \sim K\,[r_1\;r_2\;(z\,r_3 + t)]\,(X, Y, 1)^T`. Su
    inversa, compuesta con el cambio de escala a ``ppcm``, quita la perspectiva:
    en la vista cenital los dedos tienen su largo real en cm sin importar la
    inclinación de la cámara. Supone imagen sin distorsión (``modelo.dist = 0``,
    p. ej. tras ``corregir_distorsion``).
    """
    if limites is None:
        limites = (-cfg.MEDIO_LARGO - 2 - cfg.FRANJA_BRAZO_LARGO, cfg.MEDIO_LARGO + 2,
                   -cfg.MEDIO_ANCHO - 5, cfg.MEDIO_ANCHO + 5)
    x0, x1, y0, y1 = limites
    R = pose.R
    P = modelo.K @ np.column_stack([R[:, 0], R[:, 1], z * R[:, 2] + pose.tvec.reshape(3)])
    A = np.array([[ppcm, 0, -x0 * ppcm], [0, -ppcm, y1 * ppcm], [0, 0, 1]])
    H = A @ np.linalg.inv(P)
    return VistaCenital(H / H[2, 2], (int(round((x1 - x0) * ppcm)), int(round((y1 - y0) * ppcm))), ppcm, x0, y1, z)


# =========================================================================== #
# Cámara virtual del modo pantalla (v2)
# =========================================================================== #


def camara_virtual(modelo: ModeloCamara, ocupacion: float = cfg.OCUPACION_CANCHA,
                   fraccion_piso: float = cfg.FRACCION_PISO) -> Pose:
    r"""Pose de una cámara virtual que encuadra la cancha virtual en la imagen.

    Sin cancha impresa no hay pose que medir: se **elige** una. La cámara virtual
    usa la misma ``K`` que la real y mira horizontalmente la red (plano
    :math:`y = 0`) desde :math:`y = -D`. Con :math:`u = c_x + f_x\,X/Z`:

    - :math:`D = f_x \cdot 2L / (\text{ocupación}\cdot W)` hace que la cancha
      (largo :math:`2L`) ocupe esa fracción del ancho :math:`W`;
    - la altura :math:`h = (\text{fracción}\cdot H - c_y)\,D/f_y` lleva el piso
      a la fila deseada;
    - el desplazamiento lateral :math:`-(W/2 - c_x)\,D/f_x` deja la red en la
      columna central aunque el punto principal no esté en el centro.

    Por eso la calibración sigue importando: con la ``K`` y la distorsión reales,
    la red, el balón y las manos comparten la misma geometría que la imagen
    corregida.
    """
    W, H = modelo.ancho, modelo.alto
    fx, fy, cx, cy = modelo.K[0, 0], modelo.K[1, 1], modelo.K[0, 2], modelo.K[1, 2]
    D = fx * 2 * cfg.MEDIO_LARGO / (ocupacion * W)
    h = (fraccion_piso * H - cy) * D / fy
    Cx = -(W / 2 - cx) * D / fx
    C = np.array([Cx, -D, h])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # x→x, z(arriba)→-y, y(profundidad)→z
    t = -R @ C
    return Pose(cv2.Rodrigues(R)[0].ravel(), t)


def pixel_a_plano(px: tuple[float, float], pose: Pose, modelo: ModeloCamara, eje: int = 1,
                  valor: float = 0.0) -> np.ndarray | None:
    """Intersección del rayo de un píxel con el plano ``P_w[eje] = valor`` (eje 1 = plano de la red)."""
    o, d = rayo_desde_pixel(px, pose, modelo)
    if abs(d[eje]) < 1e-9:
        return None
    lam = (valor - o[eje]) / d[eje]
    return None if lam <= 0 else o + lam * d


def distancia_por_tamano(tamano_px: float, tamano_cm: float, f: float) -> float:
    r"""Distancia a la cámara de un objeto de tamaño conocido: :math:`Z = f\,W / w`.

    Es la proyección pinhole al revés (:math:`w = f\,W/Z`). Con la focal
    equivocada (sin calibrar) la distancia sale escalada por
    :math:`f_{aprox}/f_{real}`.
    """
    return float(f * tamano_cm / max(tamano_px, 1e-6))
