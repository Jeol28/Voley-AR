"""Pipeline completo de un frame: de la imagen de la cámara al punto de golpe en cm.

Orden de las etapas (cada una es una función pura de su módulo)::

    corrección de distorsión  (calibracion)
      → pose de la cancha      (calibracion: ArUco + solvePnP)
      → CLAHE                  (histogramas)
      → suavizado              (filtros)
      → retroproyección piel   (histogramas)
      → Otsu                   (umbralizacion)
      → apertura + cierre      (morfologia)
      → zona + contorno mano   (segmentacion; watershed si hace falta)
      → vista cenital métrica  (calibracion: homografía del plano)
      → corte mano-brazo       (segmentacion)
      → dedos, casco, punta    (caracterizacion)
      → posición (cm) y velocidad (cm/s)

El estado entre frames (pose previa, velocidad) vive en ``EstadoPipeline`` y
se pasa explícitamente: ``procesar_frame`` no tiene variables globales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from voleyar.core import calibracion as cal
from voleyar.core import caracterizacion as car
from voleyar.core import filtros as fil
from voleyar.core import histogramas as his
from voleyar.core import morfologia as mor
from voleyar.core import segmentacion as seg
from voleyar.core import umbralizacion as umb


@dataclass
class ConfigPipeline:
    usar_calibracion: bool = True
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
    margen_zona_cm: float = 2.0
    metodo_corte: str = "distancia"
    watershed_auto: bool = True
    area_mano_cm2: float = 120.0
    factor_area_ws: float = 2.0
    ppcm_cenital: float = 10.0
    k_dedo: float = 1.6
    frames_memoria_pose: int = 20  # si la mano tapa los marcadores, se reutiliza la pose


@dataclass
class EstadoPipeline:
    pose_previa: cal.Pose | None = None
    frames_sin_pose: int = 0
    velocidad: car.EstimadorVelocidad = field(default_factory=car.EstimadorVelocidad)


@dataclass
class ResultadoFrame:
    original: np.ndarray
    imagen: np.ndarray  # imagen de trabajo (rectificada si hay calibración)
    modelo: cal.ModeloCamara  # modelo efectivo de ``imagen``
    marcadores: dict[int, np.ndarray]
    pose: cal.Pose | None
    pose_recordada: bool = False
    clahe: np.ndarray | None = None
    suavizada: np.ndarray | None = None
    zona: seg.ZonaBusqueda | None = None
    retro: np.ndarray | None = None
    T_otsu: int | None = None
    mascara_umbral: np.ndarray | None = None
    mascara_limpia: np.ndarray | None = None
    contorno: np.ndarray | None = None
    componente: np.ndarray | None = None
    uso_watershed: bool = False
    etiquetas_ws: np.ndarray | None = None
    vista: cal.VistaCenital | None = None
    componente_cenital: np.ndarray | None = None
    mano_cenital: np.ndarray | None = None
    info_corte: seg.InfoCorte | None = None
    analisis: car.AnalisisMano | None = None
    descriptores: car.Descriptores | None = None
    area_mano_cm2: float | None = None
    mascara_mano: np.ndarray | None = None  # mano sin brazo, en la imagen
    punto_golpe_px: tuple[float, float] | None = None
    puntas_px: np.ndarray | None = None
    centro_palma_px: tuple[float, float] | None = None
    casco_px: np.ndarray | None = None
    contorno_mano_px: np.ndarray | None = None
    punto_golpe_cm: np.ndarray | None = None
    rayo_golpe: tuple[np.ndarray, np.ndarray] | None = None
    velocidad_cm_s: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tiempos_ms: dict[str, float] = field(default_factory=dict)

    @property
    def hay_mano(self) -> bool:
        return self.analisis is not None and self.punto_golpe_cm is not None


def preparar_imagen(frame: np.ndarray, modelo_camara: cal.ModeloCamara | None,
                    config: ConfigPipeline) -> tuple[np.ndarray, cal.ModeloCamara]:
    """Reduce el frame al ancho de proceso y corrige la distorsión si hay calibración."""
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


def procesar_frame(frame: np.ndarray, modelo_camara: cal.ModeloCamara | None, modelo_piel: his.ModeloPiel | None,
                   config: ConfigPipeline | None = None, estado: EstadoPipeline | None = None,
                   t: float | None = None) -> ResultadoFrame:
    """Ejecuta el pipeline completo sobre un frame BGR."""
    config = config or ConfigPipeline()
    estado = estado or EstadoPipeline()
    t = time.perf_counter() if t is None else t
    tiempos: dict[str, float] = {}
    t0 = time.perf_counter()

    def marca(nombre: str) -> None:
        nonlocal t0
        ahora = time.perf_counter()
        tiempos[nombre] = (ahora - t0) * 1000
        t0 = ahora

    img, modelo = preparar_imagen(frame, modelo_camara, config)
    marca("distorsión")

    # --- Pose ---------------------------------------------------------------
    marcadores = cal.detectar_marcadores(img)
    pose = cal.estimar_pose_cancha(marcadores, modelo, estado.pose_previa)
    recordada = False
    if pose is None and estado.pose_previa is not None and estado.frames_sin_pose < config.frames_memoria_pose:
        pose, recordada = estado.pose_previa, True
        estado.frames_sin_pose += 1
    elif pose is not None:
        estado.pose_previa, estado.frames_sin_pose = pose, 0
    marca("pose")
    res = ResultadoFrame(frame, img, modelo, marcadores, pose, recordada, tiempos_ms=tiempos)

    # --- Preprocesado ----------------------------------------------------------
    res.clahe, res.suavizada = preprocesar(img, config)
    marca("clahe+suavizado")
    if pose is None:
        res.velocidad_cm_s = estado.velocidad.actualizar(None, t)
        return res

    res.zona = seg.zona_busqueda(img.shape, pose, modelo, config.margen_zona_cm)
    marca("zona")

    # --- Piel ------------------------------------------------------------------
    if modelo_piel is not None:
        res.retro = his.retroproyeccion(res.suavizada, modelo_piel, config.espacio_retro, config.disco_retro,
                                        res.zona.mascara)
        res.mascara_umbral, res.T_otsu = umb.umbral_piel_otsu(res.retro, res.zona.mascara, config.t_min_otsu)
    else:  # sin calibrar la piel: rango fijo en YCrCb
        res.mascara_umbral = cv2.bitwise_and(umb.umbral_ycrcb(res.suavizada), res.zona.mascara)
    marca("piel+umbral")
    res.mascara_limpia = mor.limpiar_mascara(res.mascara_umbral, config.tam_apertura, config.tam_cierre,
                                             config.forma_ee)
    marca("morfología")

    # --- Segmentación ------------------------------------------------------------
    s_px = cal.pixeles_por_cm(pose, modelo)
    res.contorno, res.componente = seg.seleccionar_mano(res.mascara_limpia, res.zona.franja,
                                                        area_min=max(300.0, 12.0 * s_px ** 2))
    if res.contorno is None:
        res.velocidad_cm_s = estado.velocidad.actualizar(None, t)
        marca("segmentación")
        return res
    res.vista = cal.vista_cenital(pose, modelo, config.ppcm_cenital)
    zona_c = res.vista.rectificar(res.zona.mascara)

    def cortar(componente: np.ndarray):
        comp_c = res.vista.rectificar(componente)
        mano_c, info = seg.separar_mano_brazo(comp_c, zona_c, config.metodo_corte)
        return comp_c, mano_c, info, np.count_nonzero(mano_c) / config.ppcm_cenital ** 2

    res.componente_cenital, res.mano_cenital, res.info_corte, area = cortar(res.componente)
    extra = seg.hay_region_extra(res.mano_cenital, res.info_corte)
    if config.watershed_auto and (extra or area > config.factor_area_ws * config.area_mano_cm2):
        comp_ws, res.etiquetas_ws, n = seg.watershed_mano(res.suavizada, res.componente, res.zona.franja)
        if n > 1 and comp_ws.any():
            res.uso_watershed = True
            res.componente = comp_ws
            res.componente_cenital, res.mano_cenital, res.info_corte, area = cortar(comp_ws)
    res.area_mano_cm2 = float(area)
    marca("segmentación")

    # --- Caracterización ------------------------------------------------------------
    res.analisis = car.analizar_mano(res.mano_cenital, res.info_corte.direccion_brazo, k_dedo=config.k_dedo)
    if res.analisis is None:
        res.velocidad_cm_s = estado.velocidad.actualizar(None, t)
        return res
    a, vc = res.analisis, res.vista
    res.descriptores = car.descriptores_forma(res.mano_cenital, a.contorno)
    res.punto_golpe_px = tuple(vc.a_imagen([a.punto_golpe_px])[0])
    res.centro_palma_px = tuple(vc.a_imagen([a.centro_palma])[0])
    res.puntas_px = vc.a_imagen(a.puntas) if a.puntas else np.zeros((0, 2))
    res.casco_px = vc.a_imagen(a.casco.reshape(-1, 2))
    res.contorno_mano_px = vc.a_imagen(a.contorno.reshape(-1, 2)[::2])
    res.mascara_mano = cv2.warpPerspective(res.mano_cenital, np.linalg.inv(vc.H), (img.shape[1], img.shape[0]),
                                           flags=cv2.INTER_NEAREST)
    # Punto de golpe en cm: rayo del píxel ∩ plano de la mesa (z = 0)
    res.punto_golpe_cm = car.posicion_en_mesa(res.punto_golpe_px, pose, modelo, z=0.0)
    res.rayo_golpe = cal.rayo_desde_pixel(res.punto_golpe_px, pose, modelo)
    res.velocidad_cm_s = estado.velocidad.actualizar(
        None if res.punto_golpe_cm is None else res.punto_golpe_cm[:2], t)
    marca("caracterización")
    return res


def dibujar_depuracion(res: ResultadoFrame, img: np.ndarray | None = None, ejes: bool = True) -> np.ndarray:
    """Superpone zona, contorno, casco, puntas, corte y punto de golpe sobre la imagen."""
    from voleyar.utils import visualizacion as vis

    out = (res.imagen if img is None else img).copy()
    if res.zona is not None:
        cs, _ = cv2.findContours(res.zona.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    if res.pose is not None and ejes:
        out = cal.dibujar_ejes(out, res.pose, res.modelo, 5.0, 2)
    if res.contorno_mano_px is not None:
        cv2.polylines(out, [np.round(res.contorno_mano_px).astype(np.int32)], True, vis.BGR_NARANJA, 2, cv2.LINE_AA)
    if res.casco_px is not None:
        cv2.polylines(out, [np.round(res.casco_px).astype(np.int32)], True, vis.BGR_NEUTRO, 1, cv2.LINE_AA)
    if res.analisis is not None and res.vista is not None:
        for d in res.analisis.defectos:
            if d.valido:
                p = res.vista.a_imagen([d.valle])[0]
                cv2.circle(out, tuple(np.int32(p)), 5, vis.BGR_ROJO, -1, cv2.LINE_AA)
        if res.info_corte is not None and res.info_corte.corte is not None:
            p1, p2 = res.vista.a_imagen(list(res.info_corte.corte))
            cv2.line(out, tuple(np.int32(p1)), tuple(np.int32(p2)), vis.BGR_NEUTRO, 2, cv2.LINE_AA)
    if res.puntas_px is not None:
        for p in res.puntas_px:
            cv2.circle(out, tuple(np.int32(p)), 7, vis.BGR_VERDE, 2, cv2.LINE_AA)
    if res.centro_palma_px is not None:
        cv2.circle(out, tuple(np.int32(res.centro_palma_px)), 5, vis.BGR_NEUTRO, -1, cv2.LINE_AA)
    if res.punto_golpe_px is not None:
        p = tuple(np.int32(res.punto_golpe_px))
        cv2.drawMarker(out, p, vis.BGR_NARANJA, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
    return out


def dibujar_cenital(res: ResultadoFrame) -> np.ndarray | None:
    """Vista cenital (cm) con la mano, el corte del brazo, el casco y los dedos."""
    from voleyar.utils import visualizacion as vis

    if res.componente_cenital is None:
        return None
    base = np.full((*res.componente_cenital.shape, 3), vis.BGR_GRAFITO, np.uint8)
    base[res.componente_cenital > 0] = (90, 90, 90)
    if res.mano_cenital is not None:
        base[res.mano_cenital > 0] = vis.BGR_NEUTRO
    vc = res.vista
    # Rejilla de la cancha cada 5 cm
    from voleyar.core import config as cfg

    def uv(x, y):
        return int(round((x - vc.x_min) * vc.ppcm)), int(round((vc.y_max - y) * vc.ppcm))

    cv2.rectangle(base, uv(-cfg.MEDIO_LARGO, cfg.MEDIO_ANCHO), uv(cfg.MEDIO_LARGO, -cfg.MEDIO_ANCHO), vis.BGR_NARANJA, 1)
    cv2.line(base, uv(0, cfg.MEDIO_ANCHO), uv(0, -cfg.MEDIO_ANCHO), vis.BGR_NARANJA, 2)
    a = res.analisis
    if a is not None:
        cv2.polylines(base, [a.casco.reshape(-1, 1, 2).astype(np.int32)], True, vis.BGR_NARANJA, 1, cv2.LINE_AA)
        cv2.circle(base, tuple(np.int32(a.centro_palma)), int(a.radio_palma), vis.BGR_NARANJA, 1, cv2.LINE_AA)
        cv2.circle(base, tuple(np.int32(a.centro_palma)), int(a.umbral_dedo), (120, 120, 120), 1, cv2.LINE_AA)
        for d in a.defectos:
            if d.valido:
                cv2.circle(base, d.valle, 4, vis.BGR_ROJO, -1)
        for p in a.puntas:
            cv2.circle(base, p, 6, vis.BGR_VERDE, 2)
        cv2.drawMarker(base, tuple(np.int32(a.punto_golpe_px)), vis.BGR_NARANJA, cv2.MARKER_CROSS, 16, 2)
    if res.info_corte is not None and res.info_corte.corte is not None:
        cv2.line(base, res.info_corte.corte[0], res.info_corte.corte[1], vis.BGR_ROJO, 2)
    return base
