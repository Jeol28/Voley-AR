r"""Caracterización de la mano: descriptores de forma, dedos, punto de golpe y velocidad.

Descriptores de forma (a partir de los momentos)
------------------------------------------------
Momentos de la máscara :math:`m_{pq} = \sum_{x,y} x^p y^q I(x,y)`:

- **Área** :math:`A = m_{00}` y **centroide** :math:`(\bar x, \bar y) = (m_{10}/m_{00},\, m_{01}/m_{00})`.
- **Perímetro** :math:`P`: longitud del contorno.
- **Circularidad** :math:`4\pi A / P^2` (1 = círculo; la mano abierta da ≈ 0.2-0.3).
- **Solidez** :math:`A / A_{\text{casco}}`: baja con los dedos abiertos (los
  huecos entre dedos están en el casco pero no en la mano).
- **Excentricidad**: con los momentos centrales normalizados se forma la
  matriz de covarianza :math:`\begin{bmatrix}\mu_{20}&\mu_{11}\\\mu_{11}&\mu_{02}\end{bmatrix}/m_{00}`
  de autovalores :math:`\lambda_1 \ge \lambda_2`; :math:`e = \sqrt{1 - \lambda_2/\lambda_1}`
  (0 = círculo, → 1 = alargada).
- **Momentos de Hu**: 7 combinaciones de los momentos centrales normalizados
  :math:`\eta_{pq} = \mu_{pq} / \mu_{00}^{1 + (p+q)/2}`, invariantes a traslación,
  escala y rotación. Se muestran como :math:`-\operatorname{sign}(h)\log_{10}|h|`.

Dedos con casco convexo y defectos de convexidad
------------------------------------------------
- El **casco convexo** es el menor polígono convexo que contiene la mano. Las
  puntas de los dedos extendidos son vértices del casco.
- Un **defecto de convexidad** es una zona del contorno entre dos vértices
  del casco que se aleja de él; su punto más profundo (*far*) es el valle
  entre dos dedos, y su profundidad indica cuánto se separan.

Algoritmo de conteo: el centro de la palma :math:`c` y su radio :math:`r` salen de
la transformada de distancia. Un dedo extendido es un tramo continuo del
contorno que se aleja más de :math:`k\,r` de :math:`c` (con :math:`k \approx 1.6`); su
punta es el punto más lejano del tramo (siempre un vértice del casco). Los
valles válidos (defectos profundos con ángulo agudo) separan dedos vecinos.

Gesto → golpe: mano abierta (4-5 dedos) = **remate**; 2-3 = **golpe**;
1 dedo = **toque** suave; puño = **bloqueo**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core.calibracion import ModeloCamara, Pose, pixel_a_mesa

# =========================================================================== #
# Descriptores de forma
# =========================================================================== #


@dataclass
class Descriptores:
    area: float
    perimetro: float
    centroide: tuple[float, float]
    circularidad: float
    solidez: float
    excentricidad: float
    orientacion_grados: float
    hu: np.ndarray
    hu_log: np.ndarray
    caja: tuple[int, int, int, int]
    relacion_aspecto: float

    def como_tabla(self) -> list[list]:
        filas = [
            ["Área (px²)", f"{self.area:.0f}"],
            ["Perímetro (px)", f"{self.perimetro:.1f}"],
            ["Centroide (px)", f"({self.centroide[0]:.1f}, {self.centroide[1]:.1f})"],
            ["Circularidad 4πA/P²", f"{self.circularidad:.3f}"],
            ["Solidez A/A_casco", f"{self.solidez:.3f}"],
            ["Excentricidad", f"{self.excentricidad:.3f}"],
            ["Orientación (°)", f"{self.orientacion_grados:.1f}"],
            ["Relación de aspecto", f"{self.relacion_aspecto:.2f}"],
        ]
        filas += [[f"Hu {i + 1} (−sign·log₁₀|h|)", f"{v:.3f}"] for i, v in enumerate(self.hu_log)]
        return filas


def contorno_principal(mascara: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea) if cs else None


def descriptores_forma(mascara: np.ndarray, contorno: np.ndarray | None = None) -> Descriptores | None:
    """Descriptores de la región binaria ``mascara`` (momentos de imagen + contorno)."""
    if contorno is None:
        contorno = contorno_principal(mascara)
    if contorno is None:
        return None
    M = cv2.moments(mascara, binaryImage=True)
    if M["m00"] == 0:
        return None
    area = M["m00"]
    cx, cy = M["m10"] / area, M["m01"] / area
    perimetro = cv2.arcLength(contorno, True)
    area_casco = cv2.contourArea(cv2.convexHull(contorno))
    solidez = cv2.contourArea(contorno) / area_casco if area_casco > 0 else 0.0
    cov = np.array([[M["mu20"], M["mu11"]], [M["mu11"], M["mu02"]]]) / area
    l1, l2 = sorted(np.linalg.eigvalsh(cov), reverse=True)
    exc = float(np.sqrt(max(0.0, 1 - l2 / l1))) if l1 > 0 else 0.0
    orient = 0.5 * np.degrees(np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"]))
    hu = cv2.HuMoments(M).ravel()
    with np.errstate(divide="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    x, y, w, h = cv2.boundingRect(contorno)
    return Descriptores(float(area), float(perimetro), (float(cx), float(cy)),
                        float(4 * np.pi * area / perimetro ** 2) if perimetro > 0 else 0.0,
                        float(solidez), exc, float(orient), hu, hu_log, (x, y, w, h), float(w / max(h, 1)))


# =========================================================================== #
# Casco convexo, defectos y dedos
# =========================================================================== #


@dataclass
class Defecto:
    inicio: tuple[int, int]
    fin: tuple[int, int]
    valle: tuple[int, int]
    profundidad: float
    angulo: float
    valido: bool


def casco_y_defectos(contorno: np.ndarray) -> tuple[np.ndarray, list[Defecto]]:
    """Casco convexo (puntos) y defectos de convexidad del contorno."""
    casco_pts = cv2.convexHull(contorno)
    defectos: list[Defecto] = []
    if len(contorno) < 5:
        return casco_pts, defectos
    idx = cv2.convexHull(contorno, returnPoints=False)
    try:
        d = cv2.convexityDefects(contorno, np.sort(idx, axis=0) if idx is not None else idx)
    except cv2.error:
        d = None
    if d is None:
        return casco_pts, defectos
    for s, e, f, prof in np.asarray(d).reshape(-1, 4):  # OpenCV 4 da (N,1,4); OpenCV 5, (N,4)
        p_s, p_e, p_f = (tuple(int(v) for v in contorno[i][0]) for i in (s, e, f))
        a = np.array(p_s, float) - p_f
        b = np.array(p_e, float) - p_f
        cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        defectos.append(Defecto(p_s, p_e, p_f, prof / 256.0, float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), False))
    return casco_pts, defectos


GESTOS = {  # dedos levantados → nombre de la mano y fuerza
    0: ("puño", 1.00),
    1: ("puño", 0.95),
    2: ("golpe", 0.80),
    3: ("golpe", 0.85),
    4: ("mano abierta", 0.60),
    5: ("mano abierta", 0.55),
}

# Tipos de golpe del juego y su nombre en pantalla
TIPOS_GOLPE = {
    "remate": "¡Remate!",
    "fuerte": "Golpe fuerte",
    "golpe": "Golpe",
    "suave": "Toque suave",
    "bloqueo": "Bloqueo",
}


def tipo_de_golpe(n_dedos: int | None = None, vz_cm: float | None = None, area_rel: float | None = None,
                  solidez: float | None = None, remate_vz: float = 60.0) -> str:
    """Qué golpe es, según lo que tocó el balón.

    - Una mano reconocida que avanza hacia la cámara más rápido que ``remate_vz``
      (cm/s, con :math:`Z = f\\,W/w` de la calibración) → **remate**.
    - Puño (0-1 dedos) → **fuerte**; mano abierta (4-5) → **suave**; si no, golpe normal.
    - Sin mano: un objeto grande (≥ 3 veces el disco del balón) y compacto (solidez
      ≥ 0.8), como un cuaderno, → **bloqueo**; si no, golpe normal.
    """
    if n_dedos is not None:
        if vz_cm is not None and vz_cm <= -remate_vz:
            return "remate"
        if n_dedos <= 1:
            return "fuerte"
        if n_dedos >= 4:
            return "suave"
        return "golpe"
    if area_rel is not None and area_rel >= 3.0 and (solidez or 0.0) >= 0.8:
        return "bloqueo"
    return "golpe"


@dataclass
class AnalisisMano:
    n_dedos: int
    puntas: list[tuple[int, int]]
    punta_principal: tuple[int, int] | None
    centro_palma: tuple[float, float]
    radio_palma: float
    casco: np.ndarray
    defectos: list[Defecto]
    gesto: str
    fuerza: float
    punto_golpe_px: tuple[float, float]
    contorno: np.ndarray
    distancias: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    umbral_dedo: float = 0.0
    n_anillo: int | None = None  # dedos contados recorriendo círculos alrededor de la palma (segunda opinión)
    angulos_anillo: list[float] = field(default_factory=list)  # dirección (grados, imagen) de cada dedo cruzado
    radios_anillo: list[float] = field(default_factory=list)  # radios de los círculos usados (px)


def analizar_mano(mascara_mano: np.ndarray, direccion_brazo: tuple[float, float] | None = None,
                  k_dedo: float = 1.6, prof_min: float = 0.45, angulo_max: float = 100.0) -> AnalisisMano | None:
    """Cuenta los dedos extendidos y elige el punto de golpe.

    Parameters
    ----------
    direccion_brazo: vector unitario (imagen) del brazo desde la palma; las
        "puntas" en esa dirección (restos del antebrazo) se descartan y sirve
        para elegir el nudillo como punto de golpe del puño.
    k_dedo: un tramo del contorno a más de ``k_dedo · r`` del centro es un dedo.
    prof_min, angulo_max: un defecto es un valle entre dedos si su
        profundidad supera ``prof_min · r`` y su ángulo es menor que ``angulo_max``.
    """
    contorno = contorno_principal(mascara_mano)
    if contorno is None or cv2.contourArea(contorno) < 100:
        return None
    from voleyar.core.segmentacion import transformada_distancia

    dt = transformada_distancia(mascara_mano)
    _, r, _, c = cv2.minMaxLoc(dt)
    c = (float(c[0]), float(c[1]))
    r = max(float(r), 1.0)
    pts = contorno[:, 0, :].astype(np.float64)
    d = np.linalg.norm(pts - np.array(c), axis=1)
    umbral = k_dedo * r
    u = np.array(direccion_brazo, float) if direccion_brazo is not None else None

    # Tramos contiguos del contorno lejos de la palma = dedos
    lejos = d > umbral
    puntas: list[tuple[int, int]] = []
    if lejos.any() and not lejos.all():
        inicio = int(np.argmin(lejos))  # empieza en un punto "cerca" para no partir un tramo
        orden = np.roll(np.arange(len(pts)), -inicio)
        tramo: list[int] = []
        tramos: list[list[int]] = []
        for i in orden:
            if lejos[i]:
                tramo.append(i)
            elif tramo:
                tramos.append(tramo)
                tramo = []
        if tramo:
            tramos.append(tramo)
        largo_min = max(3, int(0.15 * r))
        for t in tramos:
            if len(t) < largo_min:
                continue
            j = t[int(np.argmax(d[t]))]
            v = pts[j] - np.array(c)
            if u is not None and v @ u / (np.linalg.norm(v) + 1e-9) > 0.5:
                continue  # resto del brazo, no es un dedo
            puntas.append((int(pts[j][0]), int(pts[j][1])))

    casco, defectos = casco_y_defectos(contorno)
    for df in defectos:
        dist_valle = np.linalg.norm(np.array(df.valle, float) - np.array(c))
        df.valido = df.profundidad > prof_min * r and df.angulo < angulo_max and dist_valle < 2.2 * r

    n = min(5, len(puntas))
    gesto, fuerza = GESTOS[n]
    punta_principal = None
    if puntas:
        if u is not None:
            punta_principal = max(puntas, key=lambda p: -((p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1]))
        else:
            punta_principal = max(puntas, key=lambda p: np.hypot(p[0] - c[0], p[1] - c[1]))
    if n >= 4 or n == 0:
        if n == 0 and u is not None:
            # Puño: los nudillos, el punto del contorno más alejado en sentido contrario al brazo
            j = int(np.argmax(-(pts - np.array(c)) @ u))
            punto = (float(pts[j][0]), float(pts[j][1]))
        else:
            punto = c  # mano abierta: centro de la palma
    else:
        punto = (float(punta_principal[0]), float(punta_principal[1]))
    out = AnalisisMano(n, puntas, punta_principal, c, r, casco, defectos, gesto, fuerza, punto, contorno, d, umbral)
    out.n_anillo, out.angulos_anillo, out.radios_anillo = dedos_por_anillo(mascara_mano, c, r, direccion_brazo)
    return out


def dedos_por_anillo(mascara: np.ndarray, centro, radio: float, direccion_brazo=None,
                     factores=(1.5, 1.8, 2.1), ancho_min: float = 3.0, ancho_max: float = 45.0,
                     excluir_brazo: float = 50.0) -> tuple[int, list[float], list[float]]:
    r"""Cuenta los dedos recorriendo círculos alrededor de la palma.

    Cada círculo de radio :math:`R = f\,r_{palma}` corta cada dedo extendido en un arco
    de piel. Se cuentan los arcos de ancho angular razonable (ni ruido ni la palma
    entera) que no estén hacia el brazo, y se toma la mediana de los tres círculos.
    Es menos sensible que los defectos del casco convexo cuando el borde es irregular.
    Devuelve (dedos, dirección de cada dedo en grados, radios usados).
    """
    H, W = mascara.shape[:2]
    brazo = (np.degrees(np.arctan2(direccion_brazo[1], direccion_brazo[0]))
             if direccion_brazo is not None else None)
    t = np.radians(np.arange(0, 360, 1.0))
    cuentas, por_radio, radios = [], [], []
    for f in factores:
        R = f * radio
        xs = np.round(centro[0] + R * np.cos(t)).astype(int)
        ys = np.round(centro[1] + R * np.sin(t)).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        v = np.zeros(360, bool)
        v[ok] = mascara[ys[ok], xs[ok]] > 0
        dedos: list[float] = []
        if v.any() and not v.all():
            k = int(np.argmin(v))  # empezar fuera de la piel para no partir un arco en dos
            v = np.roll(v, -k)
            d = np.diff(np.r_[0, v.astype(int), 0])
            for i, j in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                c = ((i + j - 1) / 2 + k) % 360
                if not (ancho_min <= j - i <= ancho_max):
                    continue
                if brazo is not None and abs((c - brazo + 180) % 360 - 180) < excluir_brazo:
                    continue
                dedos.append(float(c))
        cuentas.append(len(dedos))
        por_radio.append(dedos)
        radios.append(float(R))
    n = int(np.median(cuentas))
    elegido = next((d for d in por_radio if len(d) == n), [])
    return n, elegido, radios

# =========================================================================== #
# Posición en la mesa y velocidad
# =========================================================================== #


def posicion_en_mesa(punto_px: tuple[float, float], pose: Pose, modelo: ModeloCamara, z: float = 0.0) -> np.ndarray | None:
    """Pasa el punto de golpe a centímetros sobre la cancha (rayo ∩ plano ``Z = z``)."""
    return pixel_a_mesa(punto_px, pose, modelo, z)


class EstimadorVelocidad:
    r"""Velocidad de la mano en cm/s con suavizado exponencial.

    .. math::  v_k = \alpha\,\frac{p_k - p_{k-1}}{t_k - t_{k-1}} + (1 - \alpha)\,v_{k-1}

    Si la mano se pierde más de ``max_hueco`` segundos se reinicia, para no
    calcular una velocidad enorme al reaparecer en otro lugar.
    """

    def __init__(self, alfa: float = 0.5, max_hueco: float = 0.3, v_max: float = 400.0):
        self.alfa = alfa
        self.max_hueco = max_hueco
        self.v_max = v_max
        self.reiniciar()

    def reiniciar(self) -> None:
        self.pos: np.ndarray | None = None
        self.t: float | None = None
        self.vel = np.zeros(2)

    def actualizar(self, pos_cm: np.ndarray | None, t: float) -> np.ndarray:
        if pos_cm is None:
            if self.t is not None and t - self.t > self.max_hueco:
                self.reiniciar()
            return self.vel
        p = np.asarray(pos_cm, float)[:2]
        if self.pos is None or self.t is None or t - self.t > self.max_hueco or t <= self.t:
            self.pos, self.t = p, t
            self.vel = np.zeros(2)
            return self.vel
        v = (p - self.pos) / (t - self.t)
        norma = np.linalg.norm(v)
        if norma > self.v_max:
            v *= self.v_max / norma
        self.vel = self.alfa * v + (1 - self.alfa) * self.vel
        self.pos, self.t = p, t
        return self.vel

    @property
    def rapidez(self) -> float:
        return float(np.linalg.norm(self.vel))


# =========================================================================== #
# ¿Es de verdad una mano? Validación por forma y seguimiento en el tiempo
# =========================================================================== #


@dataclass
class ValidacionMano:
    valida: bool
    confianza: float  # 0-1
    motivos: list[str]  # por qué se rechazó (vacío si es válida)


def validar_mano(analisis: AnalisisMano | None, descriptores: Descriptores | None, f: float,
                 prob_piel: float | None = None, radio_palma_cm: float = 3.5,
                 z_min: float = 35.0, z_max: float = 130.0) -> ValidacionMano:
    r"""Decide si una región de piel tiene la forma de una mano (caracterización como filtro).

    - **Tamaño**: con el radio de la palma :math:`r` (px) la distancia sería
      :math:`Z = f\,W/r`; si queda fuera de lo posible para alguien jugando
      (35-130 cm), no es una mano. Una cara que se escapó del detector de caras se ve
      como una "palma" enorme, a menos de ~32 cm: se descarta por eso.
    - **Área/palma**: el área de la mano dividida por el área del círculo de la palma
      :math:`\pi r^2` está entre ~0.75 (puño) y ~4.5 (mano abierta con dedos largos).
    - **Solidez** :math:`A/A_{casco}`: entre 0.4 (dedos muy abiertos) y 0.99.
    - **Piel**: la probabilidad media de la retroproyección dentro de la región.

    La confianza es el producto de puntajes suaves; es válida si no hay motivos de
    rechazo.
    """
    if analisis is None or descriptores is None:
        return ValidacionMano(False, 0.0, ["sin región"])
    motivos, puntajes = [], []
    r = max(analisis.radio_palma, 1.0)
    z = f * radio_palma_cm / r
    if not z_min <= z <= z_max:
        motivos.append(f"tamaño imposible para una mano (estaría a {z:.0f} cm)")
    puntajes.append(1.0 if z_min <= z <= z_max else 0.2)
    rel = descriptores.area / (np.pi * r * r)
    if not 0.75 <= rel <= 4.5:
        motivos.append(f"área/palma = {rel:.2f}")
    puntajes.append(float(np.clip(1.0 - max(0.0, 0.75 - rel, rel - 4.5), 0.1, 1.0)))
    if not 0.4 <= descriptores.solidez <= 0.99:
        motivos.append(f"solidez = {descriptores.solidez:.2f}")
    puntajes.append(1.0 if 0.4 <= descriptores.solidez <= 0.99 else 0.3)
    if prob_piel is not None:
        if prob_piel < 0.25:
            motivos.append(f"poca piel ({prob_piel:.2f})")
        puntajes.append(float(np.clip(prob_piel / 0.5, 0.1, 1.0)))
    return ValidacionMano(not motivos, float(np.prod(puntajes)), motivos)


class SeguidorMano:
    """Presencia estable de la mano de un jugador (histéresis) y gesto suavizado.

    - **Aparece** tras ``frames_aparecer`` frames válidos seguidos y, si se pide
      movimiento, solo si la región se movió en alguno de ellos (un objeto de color
      piel que nunca se mueve no se confunde con una mano).
    - **Desaparece** tras ``frames_desaparecer`` frames sin mano válida: un frame
      malo no la hace parpadear.
    - El **gesto** es el más frecuente de los últimos ``ventana_gesto`` frames.
    """

    def __init__(self, frames_aparecer: int = 3, frames_desaparecer: int = 6, ventana_gesto: int = 5,
                 movimiento_min: float = 0.04):
        self.frames_aparecer = frames_aparecer
        self.frames_desaparecer = frames_desaparecer
        self.ventana_gesto = ventana_gesto
        self.movimiento_min = movimiento_min
        self.reiniciar()

    def reiniciar(self) -> None:
        self.presente = False
        self._validos = 0
        self._invalidos = 0
        self._se_movio = False
        self._dedos: list[int] = []

    def actualizar(self, valida: bool, n_dedos: int | None = None, movimiento: float | None = None) -> bool:
        """``movimiento``: fracción de la región que cambió respecto al fondo (``None`` = no se exige)."""
        if valida:
            self._validos += 1
            self._invalidos = 0
            if movimiento is None or movimiento >= self.movimiento_min:
                self._se_movio = True
            if n_dedos is not None:
                self._dedos = (self._dedos + [int(n_dedos)])[-self.ventana_gesto:]
            if not self.presente and self._validos >= self.frames_aparecer and self._se_movio:
                self.presente = True
        else:
            self._invalidos += 1
            self._validos = 0
            if self._invalidos >= self.frames_desaparecer:
                self.presente = False
                self._se_movio = False
                self._dedos = []
        return self.presente

    @property
    def dedos_estables(self) -> int | None:
        if not self._dedos:
            return None
        valores, cuentas = np.unique(self._dedos, return_counts=True)
        return int(valores[np.argmax(cuentas)])
