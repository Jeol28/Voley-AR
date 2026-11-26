"""Pipeline completo de un frame (modo pantalla, v2): de la cámara a las dos manos en el plano de juego.

No hay cancha impresa: la imagen se divide en dos mitades, una por jugador, y la
red virtual queda en la columna central. Orden de las etapas (cada una es una
función pura de su módulo)::

    corrección de distorsión  (calibracion)
      → cámara virtual          (calibracion: encuadra la cancha virtual con la K real)
      → caras (Viola-Jones)     (segmentacion: se excluyen cara y cuello)
      → CLAHE + suavizado       (histogramas, filtros)
      → por cada mitad / jugador:
          retroproyección de SU piel (histogramas) → Otsu (umbralizacion)
          → apertura + cierre (morfologia) → contorno que entra por abajo (segmentacion)
          → corte mano-brazo / watershed (segmentacion)
          → casco, defectos, dedos, punta (caracterizacion)
          → distancia Z = f·W/w por el tamaño de la palma (calibracion)
          → punto de golpe en el plano de la red y velocidad (calibracion, caracterizacion)

El estado entre frames (velocidades, caras recientes) vive en ``EstadoPipeline``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import caracterizacion as car
from voleyar.core import config as cfgc
from voleyar.core import filtros as fil
from voleyar.core import histogramas as his
from voleyar.core import morfologia as mor
from voleyar.core import movimiento as movi
from voleyar.core import segmentacion as seg
from voleyar.core import umbralizacion as umb

LADOS = (-1, 1)  # jugador 1 (izquierda), jugador 2 (derecha)
NOMBRES_LADO = {-1: "Jugador 1", 1: "Jugador 2"}


@dataclass
class ConfigPipeline:
    usar_calibracion: bool = True
    espejo: bool = False  # voltear la cámara horizontalmente (como un espejo) antes de procesar
    ancho_proceso: int = 960  # el frame se reduce a este ancho para ir en tiempo real
    usar_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_celdas: int = 8
    suavizado: str = "gaussiano"
    k_suavizado: int = 5
    espacio_retro: str = "combinada"
    disco_retro: int = 7
    t_min_otsu: int = 30
    tam_apertura: int = 5
    tam_cierre: int = 5  # con 9 px se pegaban los dedos juntos (45/90 aciertos vs 90/90)
    forma_ee: str = "elipse"
    metodo_corte: str = "distancia"
    watershed_auto: bool = True
    k_dedo: float = 1.6
    excluir_caras: bool = True
    caras_cada: int = 3  # la cascada Haar corre cada N frames y se reutiliza el resultado
    memoria_caras: int = 90  # frames que se recuerda una cara perdida (la mano la tapa al jugar)
    seguimiento: bool = False  # presencia estable en el tiempo (histéresis + movimiento); el juego lo activa
    exigir_movimiento: bool = True  # con seguimiento: una mano nueva tiene que haberse movido
    golpe_por_movimiento: bool = False  # calcular siempre el primer plano en movimiento: con él golpea cualquier objeto
    area_min_frac: float = 0.004  # área mínima de la mano como fracción de la imagen
    # --- qué se mueve y qué toca el balón (voleyar/core/movimiento.py) ---
    umbral_luz: float = 0.30  # distancia de Bhattacharyya entre histogramas desde la que es un cambio de luz
    mov_t_min: int = 18  # mínimo del umbral de Otsu sobre |frame - fondo|
    mov_apertura: int = 3  # px (imagen de 320): quita el ruido suelto
    mov_cierre: int = 5  # px (imagen de 320): une el objeto
    mov_area_min: float = 0.0015  # área mínima de un objeto en movimiento (fracción de la imagen)
    ignorar_caras_golpe: bool = True  # la cara (Viola-Jones) no golpea el balón
    remate_vz: float = 60.0  # cm/s hacia la cámara (calibración: Z = f·W/w) desde los que es remate
    cercania_min: float = 0.10  # Depth Anything: lo que no esté más cerca que esto no golpea (0 = no filtrar)


@dataclass
class EstadoPipeline:
    velocidades: dict[int, car.EstimadorVelocidad] = field(
        default_factory=lambda: {lado: car.EstimadorVelocidad() for lado in LADOS})
    caras: list[tuple[int, int, int, int]] = field(default_factory=list)
    caras_por_lado: dict[int, tuple] = field(default_factory=dict)  # última cara de cada mitad y su frame
    n_frame: int = 0
    seguidores: dict[int, car.SeguidorMano] = field(default_factory=lambda: {lado: car.SeguidorMano() for lado in LADOS})
    movimiento: movi.AnalizadorMovimiento = field(default_factory=movi.AnalizadorMovimiento)


@dataclass
class ResultadoMano:
    lado: int
    zona: np.ndarray
    retro: np.ndarray | None = None
    T_otsu: int | None = None
    mascara_umbral: np.ndarray | None = None
    mascara_limpia: np.ndarray | None = None
    contorno: np.ndarray | None = None
    componente: np.ndarray | None = None
    uso_watershed: bool = False
    etiquetas_ws: np.ndarray | None = None
    mascara_mano: np.ndarray | None = None
    info_corte: seg.InfoCorte | None = None
    analisis: car.AnalisisMano | None = None
    descriptores: car.Descriptores | None = None
    punto_golpe_px: tuple[float, float] | None = None
    distancia_cm: float | None = None  # Z de la mano por el tamaño de la palma
    cm_por_px: float | None = None  # escala real en el plano de la mano (Z / f)
    punto_plano: np.ndarray | None = None  # (x, 0, z) en unidades de la cancha virtual
    rayo: tuple[np.ndarray, np.ndarray] | None = None
    velocidad: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (vx, vz) en unidades de cancha / s
    validacion: car.ValidacionMano | None = None  # ¿la región tiene forma de mano?
    movimiento: float | None = None  # fracción de la región en movimiento (sustracción de fondo)
    presente: bool | None = None  # con seguimiento: presencia estable en el tiempo
    dedos_estables: int | None = None  # gesto suavizado en el tiempo

    @property
    def hay_mano(self) -> bool:
        """Hay una mano jugable: región analizada, con forma de mano y (con seguimiento) presente."""
        if self.analisis is None or self.punto_plano is None:
            return False
        if self.validacion is not None and not self.validacion.valida:
            return False
        return self.presente is not False

    @property
    def gesto(self) -> tuple[str, float]:
        """Gesto y fuerza del golpe (suavizados si hay seguimiento)."""
        n = self.dedos_estables if self.dedos_estables is not None else (self.analisis.n_dedos if self.analisis else 0)
        return car.GESTOS[int(np.clip(n, 0, 5))]


@dataclass
class ResultadoFrame:
    original: np.ndarray
    imagen: np.ndarray  # imagen de trabajo (rectificada si hay calibración)
    modelo: cal.ModeloCamara  # modelo efectivo de ``imagen``
    pose: cal.Pose  # cámara virtual que encuadra la cancha
    caras: list[tuple[int, int, int, int]] = field(default_factory=list)
    mascara_caras: np.ndarray | None = None
    clahe: np.ndarray | None = None
    suavizada: np.ndarray | None = None
    manos: dict[int, ResultadoMano] = field(default_factory=dict)
    movimiento: np.ndarray | None = None  # objetos en movimiento (sin la cara): lo que no es el entorno
    mov: movi.ResultadoMovimiento | None = None  # todo el detalle: luz, Otsu, morfología, objetos
    tiempos_ms: dict[str, float] = field(default_factory=dict)

    def mano(self, lado: int) -> ResultadoMano | None:
        return self.manos.get(lado)

    def _union(self, atributo: str) -> np.ndarray | None:
        capas = [getattr(m, atributo) for m in self.manos.values() if getattr(m, atributo) is not None]
        if not capas:
            return None
        out = capas[0].copy()
        for c in capas[1:]:
            out = cv2.bitwise_or(out, c)
        return out

    @property
    def mascara_umbral(self) -> np.ndarray | None:
        return self._union("mascara_umbral")

    @property
    def mascara_limpia(self) -> np.ndarray | None:
        return self._union("mascara_limpia")

    @property
    def mascara_manos(self) -> np.ndarray | None:
        return self._union("mascara_mano")


def preparar_imagen(frame: np.ndarray, modelo_camara: cal.ModeloCamara | None,
                    config: ConfigPipeline) -> tuple[np.ndarray, cal.ModeloCamara]:
    """Voltea el frame si se juega en modo espejo, lo reduce al ancho de proceso y corrige la
    distorsión si hay calibración (con el modelo de cámara volteado igual que la imagen)."""
    if config.espejo:
        frame = cv2.flip(frame, 1)
        if modelo_camara is not None:
            modelo_camara = modelo_camara.espejo()
    h, w = frame.shape[:2]
    if config.ancho_proceso and w > config.ancho_proceso:
        e = config.ancho_proceso / w
        frame = cv2.resize(frame, (config.ancho_proceso, int(round(h * e))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    if config.usar_calibracion and modelo_camara is not None:
        return cal.corregir_distorsion(frame, modelo_camara.escalar_a(w, h))
    return frame, cal.modelo_aproximado(w, h)


def preprocesar(img: np.ndarray, config: ConfigPipeline) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE (luz) + suavizado (ruido). Se usa igual al calibrar la piel y al jugar."""
    clahe = his.aplicar_clahe(img, config.clahe_clip, config.clahe_celdas) if config.usar_clahe else img
    return clahe, fil.suavizar(clahe, config.suavizado, config.k_suavizado)


def calibrar_piel_en_frame(frame: np.ndarray, rect: tuple[int, int, int, int],
                           modelo_camara: cal.ModeloCamara | None, config: ConfigPipeline) -> his.ModeloPiel:
    """Calibra el color de la piel con el mismo preprocesado que usará el pipeline.

    ``rect`` está en coordenadas de la imagen de trabajo (tras ``preparar_imagen``).
    """
    img, _ = preparar_imagen(frame, modelo_camara, config)
    _, suav = preprocesar(img, config)
    return his.calibrar_piel(suav, rect)


def recuadro_piel(shape, lado: int, fraccion: float = 0.05) -> tuple[int, int, int, int]:
    """Recuadro donde cada jugador pone la palma para calibrar su piel (centro de su mitad)."""
    h, w = shape[:2]
    s = int(fraccion * w)
    # Abajo y hacia la red: lejos de la cara, para que los dedos no la toquen al calibrar
    cx = int(w * (0.36 if lado < 0 else 0.64))
    cy = int(h * 0.66)
    return (cx - s // 2, cy - s // 2, s, s)


def seleccionar_mano_pantalla(mascara: np.ndarray, abajo: np.ndarray, area_min: float,
                              caras: np.ndarray | None = None):
    """La mano es la región que entra por el borde inferior (con el antebrazo).

    - Se descartan las regiones que son casi solo cara (menos del 25 % fuera de una cara de Haar).
    - Si alguna región toca el borde inferior se elige solo entre esas: una cara o
      un cuello que se escaparon de la cascada Haar nunca tocan ese borde.
    - Si ninguna lo toca (mangas largas), se toma la mayor.
    """
    cs = seg.contornos(mascara, area_min)
    comp = np.zeros_like(mascara)
    candidatos, tocan = [], []
    for c in cs[:6]:
        m = np.zeros_like(mascara)
        cv2.drawContours(m, [c], -1, 255, -1)
        area = max(1, np.count_nonzero(m))
        toca = bool(np.count_nonzero(cv2.bitwise_and(m, abajo)))
        # Casi toda dentro de una cara: es una cara. Si buena parte queda fuera (mano, antebrazo), es mano + cara
        # y se separan después con watershed.
        if caras is not None and np.count_nonzero(cv2.bitwise_and(m, cv2.bitwise_not(caras))) < 0.25 * area:
            continue
        candidatos.append(c)
        if toca:
            tocan.append(c)
    if not candidatos:
        return None, comp
    elegido = max(tocan, key=cv2.contourArea) if tocan else candidatos[0]
    cv2.drawContours(comp, [elegido], -1, 255, -1)
    return elegido, cv2.bitwise_and(comp, mascara)


def _procesar_mano(lado: int, suav: np.ndarray, piel: his.ModeloPiel | None, prohibido: np.ndarray | None,
                   modelo: cal.ModeloCamara, pose: cal.Pose, config: ConfigPipeline, estado: EstadoPipeline,
                   t: float, movimiento: np.ndarray | None = None) -> ResultadoMano:
    # La cara NO se recorta de la zona: la mano pasa a menudo por delante de ella
    # (el balón vuela a esa altura). Se usa para descartar regiones y como marcador.
    zona = seg.mascara_mitad(suav.shape, lado)
    r = ResultadoMano(lado, zona)
    vel = estado.velocidades[lado]
    seguidor = estado.seguidores[lado] if config.seguimiento else None
    if piel is not None:
        r.retro = his.retroproyeccion(suav, piel, config.espacio_retro, config.disco_retro, zona)
        r.mascara_umbral, r.T_otsu = umb.umbral_piel_otsu(r.retro, zona, config.t_min_otsu)
    else:  # aún sin histograma aprendido de la mano: rango fijo en YCrCb
        r.mascara_umbral = cv2.bitwise_and(umb.umbral_ycrcb(suav), zona)
    r.mascara_limpia = mor.limpiar_mascara(r.mascara_umbral, config.tam_apertura, config.tam_cierre, config.forma_ee)
    area_min = config.area_min_frac * suav.shape[0] * suav.shape[1]
    abajo = seg.franja_inferior(suav.shape)
    r.contorno, r.componente = seleccionar_mano_pantalla(r.mascara_limpia, abajo, area_min, prohibido)
    if r.contorno is None:
        r.velocidad = vel.actualizar(None, t)
        if seguidor is not None:
            r.presente = seguidor.actualizar(False)
        return r
    if prohibido is not None and np.count_nonzero(cv2.bitwise_and(r.componente, prohibido)):
        sin_cara, r.etiquetas_ws = seg.separar_de_la_cara(suav, r.componente, prohibido)
        if np.count_nonzero(sin_cara) > area_min:
            r.componente, r.uso_watershed = sin_cara, True
    r.mascara_mano, r.info_corte = seg.separar_mano_brazo(r.componente, zona, config.metodo_corte)
    if config.watershed_auto and seg.hay_region_extra(r.mascara_mano, r.info_corte):
        comp_ws, r.etiquetas_ws, n = seg.watershed_mano(suav, r.componente, abajo)
        if n > 1 and comp_ws.any():
            r.uso_watershed = True
            r.componente = comp_ws
            r.mascara_mano, r.info_corte = seg.separar_mano_brazo(comp_ws, zona, config.metodo_corte)
    r.analisis = car.analizar_mano(r.mascara_mano, r.info_corte.direccion_brazo, k_dedo=config.k_dedo)
    if r.analisis is None:
        r.velocidad = vel.actualizar(None, t)
        if seguidor is not None:
            r.presente = seguidor.actualizar(False)
        return r
    a = r.analisis
    r.descriptores = car.descriptores_forma(r.mascara_mano, a.contorno)
    r.punto_golpe_px = tuple(float(v) for v in a.punto_golpe_px)
    f = float(modelo.K[0, 0] + modelo.K[1, 1]) / 2
    r.distancia_cm = cal.distancia_por_tamano(a.radio_palma, cfgc.PALMA_RADIO_CM, f)
    r.cm_por_px = r.distancia_cm / f
    r.punto_plano = cal.pixel_a_plano(r.punto_golpe_px, pose, modelo, eje=1, valor=0.0)
    r.rayo = cal.rayo_desde_pixel(r.punto_golpe_px, pose, modelo)
    # ¿Es de verdad una mano? Forma (caracterización) + piel; y en el juego, presencia estable en el tiempo
    prob = (float(r.retro[r.mascara_mano > 0].mean()) / 255.0) if r.retro is not None else None
    r.validacion = car.validar_mano(a, r.descriptores, f, prob, cfgc.PALMA_RADIO_CM)
    if seguidor is not None:
        mov = seg.fraccion_en_movimiento(r.mascara_mano, movimiento) if config.exigir_movimiento else None
        r.movimiento = mov
        r.presente = seguidor.actualizar(r.validacion.valida, a.n_dedos, mov if mov is not None else
                                         (None if not config.exigir_movimiento else 0.0))
        r.dedos_estables = seguidor.dedos_estables if r.presente else None
    ok = r.hay_mano
    r.velocidad = vel.actualizar(r.punto_plano[[0, 2]] if ok and r.punto_plano is not None else None, t)
    return r


def recordar_caras(nuevas: list, ancho: int, estado: EstadoPipeline, memoria: int) -> list:
    """Una cara por mitad: si Haar la pierde (p. ej. la mano la tapa), se usa la última vista."""
    for c in nuevas:
        lado = -1 if c[0] + c[2] / 2 < ancho / 2 else 1
        estado.caras_por_lado[lado] = (c, estado.n_frame)
    vigentes = [c for c, n in estado.caras_por_lado.values() if estado.n_frame - n <= memoria]
    return sorted(vigentes)


def procesar_frame(frame: np.ndarray, modelo_camara: cal.ModeloCamara | None,
                   pieles: dict | his.ModeloPiel | None = None, config: ConfigPipeline | None = None,
                   estado: EstadoPipeline | None = None, t: float | None = None, lados: tuple[int, ...] = LADOS,
                   caras: list[tuple[int, int, int, int]] | None = None) -> ResultadoFrame:
    """Ejecuta el pipeline completo sobre un frame BGR y devuelve una ``ResultadoMano`` por lado.

    ``pieles`` puede ser un modelo de piel por lado (``{-1: ..., 1: ...}``) o uno
    solo para ambos. ``caras`` permite pasar cajas conocidas (si no, Haar).
    """
    config = config or ConfigPipeline()
    estado = estado or EstadoPipeline()
    t = time.perf_counter() if t is None else t
    if not isinstance(pieles, dict):
        pieles = {lado: pieles for lado in LADOS}
    tiempos: dict[str, float] = {}
    t0 = time.perf_counter()

    def marca(nombre: str) -> None:
        nonlocal t0
        ahora = time.perf_counter()
        tiempos[nombre] = tiempos.get(nombre, 0.0) + (ahora - t0) * 1000
        t0 = ahora

    img, modelo = preparar_imagen(frame, modelo_camara, config)
    pose = cal.camara_virtual(modelo)
    marca("distorsión")
    res = ResultadoFrame(frame, img, modelo, pose, tiempos_ms=tiempos)

    if config.excluir_caras:
        if caras is not None:
            estado.caras = list(caras)
        elif estado.n_frame % max(1, config.caras_cada) == 0:
            estado.caras = recordar_caras(seg.detectar_caras(img), img.shape[1], estado, config.memoria_caras)
        res.caras = list(estado.caras)
        res.mascara_caras = seg.mascara_caras(img.shape, res.caras)
    estado.n_frame += 1
    marca("caras")
    res.clahe, res.suavizada = preprocesar(img, config)
    marca("clahe+suavizado")
    calcular = (config.seguimiento and config.exigir_movimiento) or config.golpe_por_movimiento
    res.mov = estado.movimiento.aplicar(img, config, res.caras) if calcular else None
    res.movimiento = res.mov.mascara if res.mov is not None else None
    marca("movimiento")
    mov_mano = res.movimiento if config.seguimiento and config.exigir_movimiento else None
    for lado in lados:
        res.manos[lado] = _procesar_mano(lado, res.suavizada, pieles.get(lado), res.mascara_caras, modelo, pose,
                                         config, estado, t, mov_mano)
        marca(f"mano {'J1' if lado < 0 else 'J2'}")
    return res


def dibujar_depuracion(res: ResultadoFrame, img: np.ndarray | None = None) -> np.ndarray:
    """Superpone mitades, caras excluidas, contornos, casco, puntas, corte y punto de golpe."""
    from voleyar.utils import visualizacion as vis

    out = (res.imagen if img is None else img).copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    for x, y, cw, ch in res.caras:
        cv2.rectangle(out, (x, y), (x + cw, y + ch), vis.BGR_ROJO, 2)
        vis.poner_texto(out, "cara excluida", (x, max(0, y - 22)), 16, vis.BGR_ROJO, fondo=vis.BGR_GRAFITO)
    for lado, r in res.manos.items():
        if r.contorno is not None:
            cv2.drawContours(out, [r.contorno], -1, (120, 120, 120), 1, cv2.LINE_AA)
        a = r.analisis
        if a is None:
            continue
        cv2.drawContours(out, [a.contorno], -1, vis.BGR_NARANJA, 2, cv2.LINE_AA)
        cv2.polylines(out, [a.casco.reshape(-1, 1, 2).astype(np.int32)], True, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
        for d in a.defectos:
            if d.valido:
                cv2.circle(out, d.valle, 5, vis.BGR_ROJO, -1, cv2.LINE_AA)
        for p in a.puntas:
            cv2.circle(out, p, 7, vis.BGR_VERDE, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(np.int32(a.centro_palma)), int(a.radio_palma), vis.BGR_NARANJA, 1, cv2.LINE_AA)
        if r.info_corte is not None and r.info_corte.corte is not None:
            cv2.line(out, r.info_corte.corte[0], r.info_corte.corte[1], vis.BGR_NEUTRO, 2, cv2.LINE_AA)
        cv2.drawMarker(out, tuple(np.int32(r.punto_golpe_px)), vis.BGR_NARANJA, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
        texto = f"{a.n_dedos} dedos · {a.gesto}" + (f" · {r.distancia_cm:.0f} cm" if r.distancia_cm else "")
        x0 = int(a.centro_palma[0] - 70)
        y0 = int(max(0, a.centro_palma[1] - a.radio_palma - 40))
        vis.poner_texto(out, texto, (max(0, x0), y0), 17, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
    return out


def dibujar_mascaras(res: ResultadoFrame) -> np.ndarray | None:
    """Máscaras de piel de ambas mitades: gris = descartado, naranja = brazo, blanco = mano."""
    from voleyar.utils import visualizacion as vis

    base = res.mascara_limpia
    if base is None:
        return None
    out = np.zeros((*base.shape, 3), np.uint8)
    out[base > 0] = (90, 90, 90)
    for r in res.manos.values():
        if r.componente is not None:
            out[r.componente > 0] = vis.BGR_NARANJA
        if r.mascara_mano is not None:
            out[r.mascara_mano > 0] = vis.BGR_NEUTRO
    if res.mascara_caras is not None:
        cs, _ = cv2.findContours(res.mascara_caras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, vis.BGR_ROJO, 2)
    cv2.line(out, (out.shape[1] // 2, 0), (out.shape[1] // 2, out.shape[0]), vis.BGR_NEUTRO, 1)
    return out
