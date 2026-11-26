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


GESTOS = {
    0: ("bloqueo", 0.70),
    1: ("toque", 0.50),
    2: ("golpe", 0.80),
    3: ("golpe", 0.85),
    4: ("remate", 1.00),
    5: ("remate", 1.05),
}


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
    return AnalisisMano(n, puntas, punta_principal, c, r, casco, defectos, gesto, fuerza, punto, contorno, d, umbral)


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
