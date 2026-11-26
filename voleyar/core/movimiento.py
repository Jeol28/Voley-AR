"""Qué se mueve frente a la cámara y qué toca el balón: los temas del curso trabajando juntos.

El balón se devuelve con cualquier cosa que no sea el entorno. Para decidirlo, cada
frame pasa por los temas del curso, y cada uno cambia lo que ocurre en la partida:

1. **Filtros.** El frame (reducido a 320 px) se suaviza antes de todo: el ruido del
   sensor deja de parecer movimiento. Además, con los gradientes de Sobel se mide qué
   tan **borroso** está cada objeto: lo que se mueve muy rápido sale borroso en la
   cámara, y ese golpe sale más fuerte (``ancho_de_borde``).
2. **Histogramas.** Se compara el histograma de gris del frame con el de referencia
   (distancia de Bhattacharyya). Si cambia de golpe es la luz (una lámpara, la
   exposición automática), no un objeto: el juego olvida el fondo y lo reaprende en
   vez de golpear el balón.
3. **Segmentación por sustracción de fondo.** MOG2 modela el fondo píxel a píxel; lo
   que lleva quieto un rato pasa a ser entorno.
4. **Umbralización.** La diferencia entre el frame y el fondo se binariza con Otsu
   (con un mínimo): el umbral se adapta a cada cuarto y a cada luz. Un píxel se mueve
   si MOG2 no lo explica **y** su diferencia supera ese umbral.
5. **Morfología.** Apertura (quita puntos de ruido) y cierre (une el objeto). Al final,
   una dilatación agranda la **zona de golpe** de cada objeto (ajustable: ayuda a
   quien empieza a alcanzar el balón).
6. **Segmentación en objetos.** Componentes conexas; se descartan las que son la cara
   (Viola-Jones), que es lo que más se mueve sin querer.
7. **Caracterización.** Área relativa al balón y solidez de cada objeto (con ellas se
   decide, en el juego, si es un bloqueo).
8. **Transformers.** Con el mapa de Depth Anything, ``cercania`` dice si un objeto está
   más cerca de la cámara que el resto de la escena: lo que pasa por detrás no golpea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import filtros as fil
from voleyar.core import morfologia as mor
from voleyar.core import umbralizacion as umb

BINS_LUZ = 64


@dataclass
class ObjetoMovil:
    etiqueta: int  # etiqueta en ``ResultadoMovimiento.etiquetas`` (imagen reducida)
    area: float  # px de la imagen de trabajo
    caja: tuple[int, int, int, int]  # (x, y, w, h) en la imagen de trabajo
    centroide: np.ndarray  # px de la imagen de trabajo
    solidez: float  # área / área del casco convexo
    en_cara: bool = False  # descartado: es (casi todo) la cara de un jugador
    contorno: np.ndarray | None = None  # en la imagen de trabajo
    ancho_borde: float = 0.0  # px de la imagen reducida que ocupa su borde (crece con el desenfoque)
    borrosidad: float = 0.0  # 0 = nítido (lento), 1 = muy borroso (muy rápido)


@dataclass
class ResultadoMovimiento:
    escala: float  # imagen reducida = imagen de trabajo · escala
    forma: tuple[int, int]  # (alto, ancho) de la imagen de trabajo
    mascara: np.ndarray  # objetos válidos (sin la cara), tamaño de la imagen de trabajo
    diferencia: np.ndarray | None = None  # |frame - fondo| (reducida)
    cruda: np.ndarray | None = None  # diferencia > T, antes de la morfología (reducida)
    limpia: np.ndarray | None = None  # después de apertura + cierre (reducida)
    etiquetas: np.ndarray | None = None  # componentes conexas de ``limpia`` (reducida)
    zona_golpe: np.ndarray | None = None  # ``etiquetas`` dilatadas: hasta dónde alcanza cada objeto (reducida)
    T: int = 0  # umbral usado
    T_otsu: int = 0  # el que eligió Otsu
    curva_otsu: np.ndarray | None = None
    hist_luz: np.ndarray | None = None
    hist_ref: np.ndarray | None = None
    distancia_luz: float = 0.0
    cambio_luz: bool = False
    calentando: bool = True
    objetos: list[ObjetoMovil] = field(default_factory=list)  # válidos
    descartados_cara: list[ObjetoMovil] = field(default_factory=list)
    puntos_sueltos: int = 0  # regiones diminutas en ``cruda`` (ruido que se movió)
    px_quitados: int = 0  # píxeles que quitó la apertura

    def objeto(self, etiqueta: int) -> ObjetoMovil | None:
        return next((o for o in self.objetos if o.etiqueta == etiqueta), None)


def _hist_gris(peq: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([g], [0], None, [BINS_LUZ], [0, 256]).ravel()
    return (h / max(h.sum(), 1.0)).astype(np.float32)


class AnalizadorMovimiento:
    def __init__(self, ancho: int = 320, historia: int = 900, calentamiento: int = 15):
        self.ancho = ancho
        self.historia = historia
        self.calentamiento = calentamiento
        self.cambios_luz = 0
        self._reiniciar()
        self.hist_ref: np.ndarray | None = None

    def _reiniciar(self) -> None:
        self.mog = cv2.createBackgroundSubtractorMOG2(history=self.historia, varThreshold=16, detectShadows=False)
        self.n = 0

    def aplicar(self, img_bgr: np.ndarray, config, caras=()) -> ResultadoMovimiento:
        h, w = img_bgr.shape[:2]
        e = self.ancho / w
        hp = max(1, int(round(h * e)))
        vacia = np.zeros((h, w), np.uint8)
        peq = cv2.resize(img_bgr, (self.ancho, hp), interpolation=cv2.INTER_AREA)
        gris_crudo = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)  # sin suavizar: para medir lo borroso
        # 1. Filtros
        peq = fil.suavizar(peq, config.suavizado, config.k_suavizado)
        res = ResultadoMovimiento(e, (h, w), vacia)
        # 2. Histogramas: ¿cambió la luz?
        hl = _hist_gris(peq)
        res.hist_luz = hl
        if self.hist_ref is None:
            self.hist_ref = hl
        res.hist_ref = self.hist_ref
        d = float(cv2.compareHist(hl, self.hist_ref, cv2.HISTCMP_BHATTACHARYYA))
        res.distancia_luz = d
        if d > config.umbral_luz:
            self.cambios_luz += 1
            self.hist_ref = hl
            self._reiniciar()  # el cuarto se ve distinto: se reaprende el fondo
            res.cambio_luz = True
        else:
            self.hist_ref = (0.97 * self.hist_ref + 0.03 * hl).astype(np.float32)
        # 3. Sustracción de fondo
        # Tasa de aprendizaje fija tras el calentamiento: con α = 1/historia, algo quieto tarda
        # ~0.1·historia frames (≈ 3 s a 30 FPS) en volverse entorno. La tasa automática (1/n)
        # se tragaría en pocos frames una mano casi quieta al empezar la partida.
        alfa = -1 if self.n < self.calentamiento else 1.0 / self.historia
        fg = self.mog.apply(peq, learningRate=alfa)  # lo que MOG2 no explica con su modelo del fondo
        self.n += 1
        if self.n <= self.calentamiento:
            return res
        res.calentando = False
        fondo = self.mog.getBackgroundImage()
        if fondo is None:
            return res
        dif = cv2.absdiff(peq, fondo).max(axis=2)
        res.diferencia = dif
        # 4. Umbralización (Otsu con un mínimo: sin nada que se mueva, Otsu partiría el propio ruido)
        T_otsu, curva = umb.otsu_manual(dif)
        res.T_otsu, res.curva_otsu = T_otsu, curva
        res.T = max(T_otsu, int(config.mov_t_min))
        # MOG2 propone (píxel a píxel, con su mezcla de gaussianas) y Otsu confirma que el cambio es grande
        cruda = np.where((dif > res.T) & (fg > 0), 255, 0).astype(np.uint8)
        res.cruda = cruda
        # 5. Morfología
        abierta = mor.apertura(cruda, mor.elemento_estructurante("elipse", config.mov_apertura)) \
            if config.mov_apertura > 1 else cruda
        res.px_quitados = int(np.count_nonzero(cruda) - np.count_nonzero(abierta))
        limpia = mor.cierre(abierta, mor.elemento_estructurante("elipse", config.mov_cierre)) \
            if config.mov_cierre > 1 else abierta
        res.limpia = limpia
        # 6. Segmentación en objetos
        area_min = config.mov_area_min * self.ancho * hp
        n0 = cv2.connectedComponents(cruda)[0] - 1
        n, etiquetas, stats, cents = cv2.connectedComponentsWithStats(limpia)
        res.puntos_sueltos = max(0, n0 - (n - 1))
        caras_peq = [(x * e, y * e, cw * e, ch * e) for x, y, cw, ch in caras]
        validas = np.zeros(n, bool)
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area < area_min:
                res.puntos_sueltos += 1
                continue
            comp = (etiquetas == k).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            casco = cv2.contourArea(cv2.convexHull(c))
            x, y, bw, bh = stats[k, :4]
            obj = ObjetoMovil(k, area / e ** 2, (int(x / e), int(y / e), int(bw / e), int(bh / e)),
                              np.asarray(cents[k]) / e, float(cv2.contourArea(c) / casco) if casco > 0 else 0.0,
                              contorno=np.round(c / e).astype(np.int32))
            # 7. ¿Es la cara? (más de la mitad del objeto dentro de un recuadro de Viola-Jones agrandado)
            if config.ignorar_caras_golpe and caras_peq:
                ys, xs = np.nonzero(comp)
                for cx, cy, cw, ch in caras_peq:
                    mx, my = cw * 0.2, ch * 0.2
                    dentro = (xs >= cx - mx) & (xs <= cx + cw + mx) & (ys >= cy - my) & (ys <= cy + ch + my)
                    if dentro.mean() > 0.5:
                        obj.en_cara = True
                        break
            obj.ancho_borde = ancho_de_borde(gris_crudo, comp)
            obj.borrosidad = float(np.clip((obj.ancho_borde - config.borde_nitido)
                                           / (config.borde_borroso - config.borde_nitido), 0.0, 1.0))
            if obj.en_cara:
                res.descartados_cara.append(obj)
            else:
                res.objetos.append(obj)
                validas[k] = True
        etiquetas = np.where(validas[etiquetas], etiquetas, 0).astype(np.int32)
        res.etiquetas = etiquetas
        # 6b. Morfología otra vez: la dilatación agranda la zona con la que cada objeto golpea el balón
        if config.mov_alcance > 0 and validas.any():
            ee = mor.elemento_estructurante("elipse", 2 * int(config.mov_alcance) + 1)
            res.zona_golpe = cv2.dilate(etiquetas.astype(np.float32), ee).astype(np.int32)
        else:
            res.zona_golpe = etiquetas
        res.mascara = cv2.resize(np.where(etiquetas > 0, 255, 0).astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
        return res


def ancho_de_borde(gris: np.ndarray, mascara: np.ndarray) -> float:
    r"""Qué tan ancho es el borde de un objeto, en píxeles: mide el desenfoque por movimiento.

    Un borde nítido de contraste :math:`C` sube de golpe (gradiente máximo
    :math:`\approx C`); si el objeto se movió durante la exposición, el borde se
    reparte en :math:`w` píxeles y el gradiente máximo baja a :math:`\approx C / w`.
    Así :math:`w \approx C / \max|\nabla I|`, con :math:`C` = percentil 95 − 5 del gris
    alrededor del objeto y el gradiente de Sobel (normalizado). Se calcula por
    separado en x y en y y se toma el mayor: un objeto que se mueve de lado solo se
    desenfoca de lado.
    """
    zona = cv2.dilate(mascara, np.ones((7, 7), np.uint8)) > 0
    if zona.sum() < 20:
        return 0.0
    g = gris.astype(np.float32)
    p5, p95 = np.percentile(g[zona], [5, 95])
    C = float(p95 - p5)
    if C < 15:  # sin contraste no hay borde que medir
        return 0.0
    anchos = []
    for dx, dy in ((1, 0), (0, 1)):
        d = np.abs(cv2.Sobel(g, cv2.CV_32F, dx, dy, ksize=3)) / 8.0
        anchos.append(C / max(float(np.percentile(d[zona], 97)), 1e-3))
    return float(max(anchos))


@dataclass
class Toque:
    toca: bool
    fraccion: float = 0.0
    objeto: ObjetoMovil | None = None
    motivo: str = ""


def toque_en_balon(res: ResultadoMovimiento | None, centro: np.ndarray, radio: float, margen: float = 1.35,
                   fraccion_min: float = 0.12) -> Toque:
    """¿Qué objeto válido cubre el disco del balón proyectado, y cuánto?"""
    if res is not None and res.cambio_luz:
        return Toque(False, motivo="cambio de luz")
    if res is None or res.etiquetas is None:
        return Toque(False, motivo="aprendiendo el fondo" if res is not None and res.calentando
                     else "sin sustracción de fondo")
    e = res.escala
    et = res.zona_golpe if res.zona_golpe is not None else res.etiquetas
    hp, wp = et.shape
    R = max(2.0, radio * margen * e)
    cx, cy = centro[0] * e, centro[1] * e
    x0, x1 = int(max(0, np.floor(cx - R))), int(min(wp, np.ceil(cx + R) + 1))
    y0, y1 = int(max(0, np.floor(cy - R))), int(min(hp, np.ceil(cy + R) + 1))
    if x0 >= x1 or y0 >= y1:
        return Toque(False, motivo="balón fuera de la imagen")
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disco = (xs - cx) ** 2 + (ys - cy) ** 2 <= R * R
    if not disco.any():
        return Toque(False, motivo="balón fuera de la imagen")
    dentro = et[y0:y1, x0:x1][disco]
    activos = dentro[dentro > 0]
    frac = activos.size / disco.sum()
    if activos.size == 0:
        return Toque(False, 0.0, motivo="nada se mueve ahí")
    obj = res.objeto(int(np.bincount(activos).argmax()))
    return Toque(frac >= fraccion_min, frac, obj, "" if frac >= fraccion_min else "apenas lo roza")


def cercania(mapa: np.ndarray, res: ResultadoMovimiento, obj: ObjetoMovil) -> float | None:
    """Cuánto más cerca de la cámara está el objeto que la escena (Depth Anything, 0 = igual, 1 = lo más cercano).

    Se usa la disparidad relativa (más alto = más cerca): mediana dentro del objeto
    menos la mediana de toda la escena, dividida por el rango (percentiles 5-95).
    """
    if mapa is None or res.etiquetas is None:
        return None
    hp, wp = res.etiquetas.shape
    m = cv2.resize(mapa.astype(np.float32), (wp, hp), interpolation=cv2.INTER_AREA)
    dentro = m[res.etiquetas == obj.etiqueta]
    if dentro.size == 0:
        return None
    p5, p95 = np.percentile(m, [5, 95])
    return float((np.median(dentro) - np.median(m)) / max(p95 - p5, 1e-6))
