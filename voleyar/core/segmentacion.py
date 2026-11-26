r"""Segmentación: zona de búsqueda, región de la mano, separación mano-brazo y watershed.

1. **Zona de búsqueda guiada por la pose.** Con la pose de la cancha se
   proyecta el "volumen de juego" (la cancha extendida hasta 12 cm de altura)
   y una franja de entrada para el brazo detrás de tu borde. Todo lo que quede
   fuera (cara, cuello, cuerpo, muebles de madera) se descarta antes de
   segmentar: es una segmentación por **región de interés geométrica**.
2. **Contornos.** ``cv2.findContours`` (algoritmo de Suzuki-Abe) extrae los
   bordes de cada componente conexa de la máscara. La mano es la componente
   que entra por la franja del brazo (o la de mayor área).
3. **Separación mano-brazo** con la **transformada de distancia**
   :math:`D(p) = \min_{q \notin A} \lVert p - q \rVert`: su máximo está en el centro
   de la palma y su valor es el radio :math:`r` de la palma (el círculo
   inscrito más grande). El brazo es la parte gruesa (:math:`D > 0.5 r`) que
   sale del disco de la palma; se corta por la muñeca, perpendicular a la
   dirección del brazo, a :math:`\approx 1.2 r` del centro (método
   ``distancia``) o donde el ancho del contorno es mínimo (método ``ancho``).
4. **Watershed** para cuando la mano toca otra región de color parecido: se
   siembran marcadores en los "picos" de la transformada de distancia (los
   núcleos de cada objeto), y la inundación del gradiente separa las regiones
   por las "líneas divisorias de aguas".
5. **Segmentación de la cancha**: la máscara del área de juego (y de cada
   lado) se obtiene proyectando su rectángulo con la pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import config as cfg
from voleyar.core.calibracion import ModeloCamara, Pose, profundidad_camara

# =========================================================================== #
# Zonas a partir de la pose
# =========================================================================== #


def _puntos_caja(xmin, xmax, ymin, ymax, zmin, zmax, n: int = 10) -> np.ndarray:
    """Puntos muestreados sobre las 12 aristas de una caja (para proyectar con distorsión)."""
    t = np.linspace(0, 1, n)
    vs = np.array([[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)], float)
    aristas = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(vs[i] != vs[j]) == 1:
                aristas.append(vs[i][None, :] * (1 - t[:, None]) + vs[j][None, :] * t[:, None])
    return np.concatenate(aristas)


def _rellenar_proyeccion(shape, puntos3d: np.ndarray, pose: Pose, modelo: ModeloCamara, convexo: bool = True) -> np.ndarray:
    """Proyecta puntos, descarta los que quedan detrás de la cámara y rellena su casco."""
    m = np.zeros(shape[:2], np.uint8)
    z = profundidad_camara(puntos3d, pose)
    pts = puntos3d[z > 1.0]
    if len(pts) < 3:
        return m
    px, _ = cv2.projectPoints(pts, pose.rvec, pose.tvec, modelo.K, modelo.dist)
    px = px.reshape(-1, 2)
    lim = 4 * max(shape[:2])
    px = np.clip(px, -lim, lim)
    poli = cv2.convexHull(px.astype(np.float32)) if convexo else px.reshape(-1, 1, 2)
    cv2.fillPoly(m, [np.round(poli).astype(np.int32)], 255)
    return m


def poligono_cancha(pose: Pose, modelo: ModeloCamara, margen: float = 0.0, z: float = 0.0, n: int = 12) -> np.ndarray:
    """Contorno de la cancha proyectado (con los bordes curvados por la distorsión)."""
    x, y = cfg.MEDIO_LARGO + margen, cfg.MEDIO_ANCHO + margen
    t = np.linspace(0, 1, n, endpoint=False)
    esquinas = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
    lados = [esquinas[i][None, :] * (1 - t[:, None]) + esquinas[(i + 1) % 4][None, :] * t[:, None] for i in range(4)]
    p = np.concatenate(lados)
    p3 = np.column_stack([p, np.full(len(p), z)])
    px, _ = cv2.projectPoints(p3, pose.rvec, pose.tvec, modelo.K, modelo.dist)
    return np.round(px.reshape(-1, 2)).astype(np.int32)


def mascara_cancha(shape, pose: Pose, modelo: ModeloCamara, margen: float = 0.0) -> np.ndarray:
    """Máscara del área de juego (segmentación de la cancha por la pose)."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [poligono_cancha(pose, modelo, margen)], 255)
    return m


def mascara_lados(shape, pose: Pose, modelo: ModeloCamara) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de tu lado (x < 0) y del lado rival (x > 0)."""
    L, A = cfg.MEDIO_LARGO, cfg.MEDIO_ANCHO
    tuyo = _rellenar_proyeccion(shape, _puntos_caja(-L, 0, -A, A, 0, 0, 12), pose, modelo)
    rival = _rellenar_proyeccion(shape, _puntos_caja(0, L, -A, A, 0, 0, 12), pose, modelo)
    return tuyo, rival


@dataclass
class ZonaBusqueda:
    mascara: np.ndarray  # volumen de juego ∪ franja del brazo
    franja: np.ndarray  # solo la franja de entrada del brazo
    cancha: np.ndarray  # cancha a z = 0


def zona_busqueda(shape, pose: Pose, modelo: ModeloCamara, margen: float = 2.0,
                  altura: float = cfg.ALTURA_VOLUMEN_JUEGO, franja: float = cfg.FRANJA_BRAZO_LARGO) -> ZonaBusqueda:
    """Zona donde se busca la mano: volumen de juego + franja de entrada del brazo."""
    L, A = cfg.MEDIO_LARGO + margen, cfg.MEDIO_ANCHO + margen
    vol = _rellenar_proyeccion(shape, _puntos_caja(-L, L, -A, A, 0, altura), pose, modelo)
    fr = _rellenar_proyeccion(shape, _puntos_caja(-L - franja, -L, -A - 3, A + 3, 0, altura), pose, modelo)
    return ZonaBusqueda(cv2.bitwise_or(vol, fr), fr, mascara_cancha(shape, pose, modelo))


# =========================================================================== #
# Contornos y selección de la mano
# =========================================================================== #


def contornos(mascara: np.ndarray, area_min: float = 0) -> list[np.ndarray]:
    """Contornos externos ordenados de mayor a menor área."""
    cs, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= area_min]
    return sorted(cs, key=cv2.contourArea, reverse=True)


def seleccionar_mano(mascara: np.ndarray, franja: np.ndarray | None = None,
                     area_min: float = 800) -> tuple[np.ndarray | None, np.ndarray]:
    """Elige la componente de la mano: la que entra por la franja del brazo; si
    ninguna la toca, la de mayor área. Devuelve ``(contorno, máscara_componente)``."""
    cs = contornos(mascara, area_min)
    comp = np.zeros_like(mascara)
    if not cs:
        return None, comp
    elegido = cs[0]
    if franja is not None and franja.any():
        mejor = -1.0
        for c in cs[:5]:
            m = np.zeros_like(mascara)
            cv2.drawContours(m, [c], -1, 255, -1)
            toque = np.count_nonzero(cv2.bitwise_and(m, franja))
            puntaje = cv2.contourArea(c) * (2.0 if toque > 0 else 1.0)
            if puntaje > mejor:
                mejor, elegido = puntaje, c
    cv2.drawContours(comp, [elegido], -1, 255, -1)
    # conservar huecos reales de la máscara original dentro de la componente
    comp = cv2.bitwise_and(comp, mascara)
    return elegido, comp


# =========================================================================== #
# Separación mano-brazo
# =========================================================================== #


@dataclass
class InfoCorte:
    centro_palma: tuple[float, float]
    radio_palma: float
    direccion_brazo: tuple[float, float] | None = None
    corte: tuple[tuple[int, int], tuple[int, int]] | None = None  # segmento de la muñeca
    distancia_corte: float | None = None
    anchos: np.ndarray | None = None  # perfil de ancho a lo largo del brazo (método "ancho")
    transformada: np.ndarray | None = field(default=None, repr=False)


def transformada_distancia(mascara: np.ndarray) -> np.ndarray:
    """Transformada de distancia euclídea tratando el borde de la imagen como fondo."""
    pad = cv2.copyMakeBorder(mascara, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1]


def centro_palma(mascara: np.ndarray) -> tuple[tuple[float, float], float, np.ndarray]:
    """Centro de la palma = máximo de la transformada de distancia; su valor es el radio.

    Se agrega un marco de ceros antes de la transformada: OpenCV no considera
    el borde de la imagen como fondo, y un brazo que sale del cuadro parecería
    más grueso que la palma.
    """
    dt = transformada_distancia(mascara)
    _, r, _, (x, y) = cv2.minMaxLoc(dt)
    return (float(x), float(y)), float(r), dt


def _direccion_brazo(mascara: np.ndarray, dt: np.ndarray, c, r, zona: np.ndarray | None) -> np.ndarray | None:
    """Dirección (unitaria) del brazo desde el centro de la palma, o None si no hay brazo."""
    h, w = mascara.shape
    afuera = mascara.copy()
    cv2.circle(afuera, (int(round(c[0])), int(round(c[1]))), int(1.5 * r), 0, -1)
    n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(afuera)
    borde = np.zeros_like(mascara)
    borde[[0, -1], :] = 255
    borde[:, [0, -1]] = 255
    if zona is not None:
        borde = cv2.bitwise_or(borde, cv2.bitwise_xor(zona, cv2.erode(zona, np.ones((5, 5), np.uint8))))
    mejor, mejor_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        comp = etiquetas == i
        grueso = dt[comp].max() > 0.5 * r
        toca_borde = bool(np.any(borde[comp]))
        if (grueso or toca_borde) and area > mejor_area:
            mejor, mejor_area = i, area
    if mejor is None:
        return None
    v = centroides[mejor] - np.asarray(c)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-6 else None


def separar_mano_brazo(mascara: np.ndarray, zona: np.ndarray | None = None, metodo: str = "distancia",
                       factor_corte: float = 1.2) -> tuple[np.ndarray, InfoCorte]:
    """Quita el antebrazo de la máscara cortando por la muñeca.

    - ``distancia``: corte perpendicular al brazo a ``factor_corte · r`` del
      centro de la palma.
    - ``ancho``: se mide el ancho de la silueta a lo largo del eje del brazo y
      se corta en el mínimo (la muñeca es más angosta que la palma y que el
      antebrazo) entre ``0.8 r`` y ``1.8 r`` (rango anatómico de la muñeca).
    """
    if not mascara.any():
        return mascara.copy(), InfoCorte((0.0, 0.0), 0.0)
    c, r, dt = centro_palma(mascara)
    info = InfoCorte(c, r, transformada=dt)
    u = _direccion_brazo(mascara, dt, c, r, zona)
    if u is None:
        return mascara.copy(), info
    info.direccion_brazo = (float(u[0]), float(u[1]))
    h, w = mascara.shape
    yy, xx = np.mgrid[0:h, 0:w]
    proy = (xx - c[0]) * u[0] + (yy - c[1]) * u[1]  # distancia a lo largo del brazo
    lat = -(xx - c[0]) * u[1] + (yy - c[1]) * u[0]  # distancia lateral
    d_corte = factor_corte * r
    if metodo == "ancho":
        ds = np.arange(int(0.6 * r), int(3.0 * r) + 1)
        banda = (np.abs(lat) < 2.5 * r) & (mascara > 0)
        indice = np.round(proy[banda]).astype(int)
        validos = (indice >= ds[0]) & (indice <= ds[-1])
        anchos = np.bincount(indice[validos] - ds[0], minlength=len(ds)).astype(float)
        anchos = np.convolve(anchos, np.ones(5) / 5, mode="same")
        info.anchos = anchos
        rango = (ds >= 0.8 * r) & (ds <= 1.8 * r) & (anchos > 0)
        if rango.any():
            d_corte = float(ds[rango][np.argmin(anchos[rango])])
    info.distancia_corte = float(d_corte)
    mano = mascara.copy()
    mano[proy > d_corte] = 0
    # Nos quedamos con la componente que contiene la palma
    n, et = cv2.connectedComponents(mano)
    if n > 2:
        lbl = et[int(round(c[1])), int(round(c[0]))]
        if lbl > 0:
            mano = np.where(et == lbl, 255, 0).astype(np.uint8)
    p = np.array(c) + d_corte * u
    nrm = np.array([-u[1], u[0]])
    info.corte = (tuple(np.round(p - 1.6 * r * nrm).astype(int)), tuple(np.round(p + 1.6 * r * nrm).astype(int)))
    return mano, info


# =========================================================================== #
# Watershed
# =========================================================================== #


def watershed_regiones(img_bgr: np.ndarray, mascara: np.ndarray, frac_pico: float = 0.55) -> tuple[np.ndarray, int]:
    """Watershed con marcadores en los picos de la transformada de distancia.

    Devuelve ``(etiquetas, n_regiones)``: etiquetas ``1..n`` por objeto, ``-1``
    en las líneas divisorias y ``0`` en el fondo.
    """
    dt = transformada_distancia(mascara)
    if dt.max() <= 0:
        return np.zeros(mascara.shape, np.int32), 0
    seguro_fg = np.uint8(dt > frac_pico * dt.max()) * 255
    # Picos locales adicionales (objetos más pequeños que el principal)
    picos = (dt == cv2.dilate(dt, np.ones((25, 25), np.uint8))) & (dt > 0.35 * dt.max())
    seguro_fg = cv2.bitwise_or(seguro_fg, cv2.dilate(np.uint8(picos) * 255, np.ones((7, 7), np.uint8)))
    seguro_fg = cv2.bitwise_and(seguro_fg, mascara)
    seguro_bg = cv2.dilate(mascara, np.ones((3, 3), np.uint8), iterations=3)
    desconocido = cv2.subtract(seguro_bg, seguro_fg)
    n, marcadores = cv2.connectedComponents(seguro_fg)
    marcadores = marcadores + 1  # fondo = 1
    marcadores[desconocido > 0] = 0
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    marcadores = cv2.watershed(img, marcadores.astype(np.int32))
    etiquetas = np.where(marcadores > 1, marcadores - 1, np.where(marcadores == -1, -1, 0)).astype(np.int32)
    return etiquetas, n - 1


def watershed_mano(img_bgr: np.ndarray, mascara: np.ndarray, preferir: np.ndarray | None = None,
                   frac_pico: float = 0.55) -> tuple[np.ndarray, np.ndarray, int]:
    """Separa la mano de otra región pegada de color parecido.

    Se queda con la región que más se solapa con ``preferir`` (la franja del
    brazo: la mano está unida al brazo) o, si no hay, con la que contiene el
    mayor pico de la transformada de distancia.
    """
    etiquetas, n = watershed_regiones(img_bgr, mascara, frac_pico)
    if n <= 1:
        return mascara.copy(), etiquetas, n
    elegido = None
    if preferir is not None and preferir.any():
        solapes = [np.count_nonzero((etiquetas == i) & (preferir > 0)) for i in range(1, n + 1)]
        if max(solapes) > 0:
            elegido = int(np.argmax(solapes)) + 1
    if elegido is None:
        _, _, dt = centro_palma(mascara)
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        elegido = int(etiquetas[y, x]) if etiquetas[y, x] > 0 else 1
    mano = np.uint8(etiquetas == elegido) * 255
    # Las líneas divisorias (-1) vecinas a la región se le devuelven
    mano = cv2.bitwise_and(cv2.morphologyEx(mano, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), mascara)
    return mano, etiquetas, n


def colorear_etiquetas(etiquetas: np.ndarray) -> np.ndarray:
    """Imagen en color de las regiones del watershed (líneas divisorias en blanco)."""
    rng = np.random.default_rng(7)
    paleta = np.vstack([[26, 35, 50], rng.integers(60, 255, (max(1, etiquetas.max()) + 1, 3))]).astype(np.uint8)
    paleta[1] = (53, 107, 255)  # primera región en naranja (BGR)
    img = paleta[np.clip(etiquetas, 0, None)]
    img[etiquetas == -1] = (245, 245, 247)
    return img


def hay_region_extra(mascara_mano: np.ndarray, info: InfoCorte, frac_grosor: float = 0.5) -> bool:
    """¿Quedó pegada a la mano una región **gruesa** que no es un dedo?

    Fuera del disco de la palma (radio 1.5 r) solo deberían quedar dedos, cuyo
    grosor (máximo de la transformada de distancia) es ≈ 0.2 r. Una región con
    grosor mayor que ``frac_grosor · r`` es otro objeto de color parecido (otra
    mano, un objeto de madera…) y conviene separarla con watershed.
    """
    if info.radio_palma <= 0 or not mascara_mano.any():
        return False
    afuera = mascara_mano.copy()
    c = (int(round(info.centro_palma[0])), int(round(info.centro_palma[1])))
    cv2.circle(afuera, c, int(1.5 * info.radio_palma), 0, -1)
    dt = transformada_distancia(afuera)
    return bool(dt.max() > frac_grosor * info.radio_palma)


# =========================================================================== #
# Modo pantalla (v2): mitades por jugador y exclusión de caras
# =========================================================================== #

_cascada_caras: cv2.CascadeClassifier | None = None


def detectar_caras(img_bgr: np.ndarray, ancho: int = 640, factor: float = 1.05,
                   vecinos: int = 4) -> list[tuple[int, int, int, int]]:
    """Caras frontales con el detector de Viola-Jones (cascada Haar de OpenCV).

    Rasgos Haar (diferencias de sumas de rectángulos, calculadas con la imagen
    integral) + AdaBoost + cascada de clasificadores que descarta rápido las
    ventanas sin cara. Se ejecuta sobre una copia reducida para ir rápido y las
    cajas se devuelven en coordenadas de la imagen original.
    """
    global _cascada_caras
    if _cascada_caras is None:
        _cascada_caras = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    h, w = img_bgr.shape[:2]
    e = min(1.0, ancho / w)
    gris = cv2.cvtColor(cv2.resize(img_bgr, None, fx=e, fy=e, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    # Una persona por mitad y a menos de ~1.2 m: caras grandes. Filtra falsos positivos (p. ej. sobre la mano).
    lado_min = max(24, int(0.20 * gris.shape[0]))
    caras = _cascada_caras.detectMultiScale(gris, factor, vecinos, minSize=(lado_min, lado_min))
    caras = [tuple(int(round(v / e)) for v in c) for c in (caras if len(caras) else [])]
    mejor: dict[int, tuple[int, int, int, int]] = {}
    for c in caras:
        lado = -1 if c[0] + c[2] / 2 < w / 2 else 1
        if lado not in mejor or c[2] * c[3] > mejor[lado][2] * mejor[lado][3]:
            mejor[lado] = c
    return [mejor[k] for k in sorted(mejor)]


def mascara_caras(shape, caras, margen: float = 0.0, cuello: float = 0.8) -> np.ndarray:
    """Máscara (255) del óvalo de cada cara y del cuello debajo: zona prohibida para la mano.

    Sin margen a propósito: si la mano se acerca a la cara, ampliar la zona le
    cortaría los dedos. Lo que quede pegado de la cara lo separa el watershed.
    """
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in caras:
        cx, cy = x + w / 2, y + h / 2
        cv2.ellipse(m, (int(cx), int(cy)), (int(w * (0.42 + margen)), int(h * (0.55 + margen))), 0, 0, 360, 255, -1)
        cv2.rectangle(m, (int(cx - 0.22 * w), int(cy)), (int(cx + 0.22 * w), int(y + h * (1 + cuello))), 255, -1)
    return m


def mascara_mitad(shape, lado: int, margen_red: int = 0) -> np.ndarray:
    """Mitad izquierda (``lado = -1``, jugador 1) o derecha (``+1``, jugador 2) de la imagen."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    if lado < 0:
        m[:, : max(0, w // 2 - margen_red)] = 255
    else:
        m[:, min(w, w // 2 + margen_red):] = 255
    return m


def franja_inferior(shape, fraccion: float = 0.04) -> np.ndarray:
    """Borde inferior de la imagen: por ahí entra el antebrazo de cada jugador."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[int(h * (1 - fraccion)):, :] = 255
    return m


def separar_de_la_cara(img_bgr: np.ndarray, componente: np.ndarray, mascara_cara: np.ndarray,
                       frac_pico: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Separa la mano de la cara cuando se tocan en la imagen (watershed con marcadores).

    Marcadores: fondo (1), el núcleo de la cara que encontró Haar (2) y los picos
    de la transformada de distancia de lo que queda fuera de la cara (3, 4, …).
    La inundación reparte la zona de contacto entre la cara y la mano siguiendo
    los bordes de la imagen. Devuelve ``(mano, etiquetas)``.
    """
    comp_cara = cv2.bitwise_and(mascara_cara, componente)
    if not comp_cara.any():
        return componente.copy(), np.zeros(componente.shape, np.int32)
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    nucleo = cv2.erode(comp_cara, ee)
    resto = cv2.bitwise_and(componente, cv2.bitwise_not(cv2.dilate(mascara_cara, ee)))
    marcadores = np.zeros(componente.shape, np.int32)
    marcadores[cv2.dilate(componente, np.ones((3, 3), np.uint8), iterations=3) == 0] = 1
    marcadores[nucleo > 0] = 2
    if resto.any():
        dt = transformada_distancia(resto)
        semillas = np.uint8(dt > frac_pico * dt.max()) * 255
        n, et = cv2.connectedComponents(semillas)
        marcadores[et > 0] = et[et > 0] + 2
    else:  # la mano está completamente sobre la cara: no hay de dónde sembrarla
        return cv2.bitwise_and(componente, cv2.bitwise_not(nucleo)), marcadores
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    etiquetas = cv2.watershed(img, marcadores.copy())
    mano = cv2.bitwise_and(np.uint8(etiquetas >= 3) * 255, componente)
    return mano, etiquetas


# =========================================================================== #
# Segmentación por movimiento (sustracción de fondo)
# =========================================================================== #


class DetectorMovimiento:
    r"""Primer plano en movimiento con una mezcla de gaussianas por píxel (MOG2).

    Cada píxel se modela como una mezcla de :math:`K` gaussianas que se actualiza
    con una tasa de aprendizaje :math:`\alpha`; un píxel es *primer plano* si su
    valor no encaja en las gaussianas que representan el fondo. En el juego sirve
    para no confundir con una mano a un objeto de color piel que nunca se mueve.
    Trabaja sobre una copia reducida para que cueste ~1 ms.
    """

    def __init__(self, ancho: int = 320, historia: int = 300, umbral: float = 25.0, calentamiento: int = 15):
        self.ancho = ancho
        self.calentamiento = calentamiento
        self.n = 0
        self.mog = cv2.createBackgroundSubtractorMOG2(history=historia, varThreshold=umbral, detectShadows=False)

    def aplicar(self, img_bgr: np.ndarray) -> np.ndarray:
        """Máscara de lo que se movió. Los primeros frames (mientras aprende el fondo) no
        cuentan: al arrancar todo parecería movimiento."""
        h, w = img_bgr.shape[:2]
        e = self.ancho / w
        peq = cv2.resize(img_bgr, (self.ancho, max(1, int(round(h * e)))), interpolation=cv2.INTER_AREA)
        fg = self.mog.apply(peq)
        self.n += 1
        if self.n <= self.calentamiento:
            return np.zeros((h, w), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def fraccion_en_movimiento(mascara: np.ndarray | None, movimiento: np.ndarray | None) -> float | None:
    if mascara is None or movimiento is None or not mascara.any():
        return None
    return float(np.count_nonzero(cv2.bitwise_and(mascara, movimiento)) / np.count_nonzero(mascara))
