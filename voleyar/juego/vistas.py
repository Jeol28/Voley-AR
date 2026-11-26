"""Vistas en vivo: "cómo te ve el juego", un tema del curso a la vez, sin salir de la partida.

Cada vista agrega un panel al lado del juego con lo que ese tema está haciendo en
este mismo frame (imágenes intermedias del pipeline y sus números) y permite
cambiar su parámetro principal mientras se juega (teclas ``[`` y ``]`` o los
botones −/+ de la web). Todo sale del ``ResultadoFrame`` que ya calculó el
pipeline: las vistas no recalculan la visión, solo la muestran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from voleyar.core import filtros as fil
from voleyar.core import pipeline as pl
from voleyar.core import umbralizacion as umb
from voleyar.utils import visualizacion as vis

VISTAS = ("juego", "calibracion", "filtros", "histogramas", "umbralizacion", "morfologia", "segmentacion",
          "caracterizacion", "transformers")
TITULOS = {
    "juego": "Juego",
    "calibracion": "Calibración de la cámara",
    "filtros": "Filtros",
    "histogramas": "Histogramas",
    "umbralizacion": "Umbralización",
    "morfologia": "Morfología",
    "segmentacion": "Segmentación",
    "caracterizacion": "Caracterización",
    "transformers": "Transformers",
}
QUE_HACE = {
    "calibracion": "K y la distorsión enderezan la imagen y miden a qué distancia está cada mano (Z = f·W/w).",
    "filtros": "El suavizado quita el ruido de la cámara antes de buscar la piel; Canny marca el borde de la mano.",
    "histogramas": "El histograma Cr-Cb de tu piel se retroproyecta sobre cada frame; CLAHE compensa la luz.",
    "umbralizacion": "Otsu elige solo el umbral que separa piel de fondo en la mitad de cada jugador.",
    "morfologia": "Apertura quita puntos sueltos, cierre rellena huecos sin pegar los dedos.",
    "segmentacion": "Una mano por mitad: la que entra por abajo; se separan el brazo y, si se tocan, la cara.",
    "caracterizacion": "Casco convexo y defectos cuentan los dedos: el gesto decide la fuerza del golpe.",
    "transformers": "Depth Anything confirma que la mano está delante del cuerpo; OWLv2 la busca por texto.",
}


# --------------------------------------------------------------------------- #
# Parámetro interactivo de cada vista
# --------------------------------------------------------------------------- #


@dataclass
class Parametro:
    nombre: str
    valores: tuple
    leer: Callable
    escribir: Callable
    formato: Callable = str


def _cfg(attr):
    return (lambda j: getattr(j.config, attr)), (lambda j, v: setattr(j.config, attr, v))


PARAMETROS: dict[str, Parametro] = {
    "calibracion": Parametro("usar la calibración", (True, False), lambda j: j.config.usar_calibracion,
                             lambda j, v: j.accion("calibracion") if v != j.config.usar_calibracion else None,
                             lambda v: "sí" if v else "no"),
    "filtros": Parametro("suavizado", ("gaussiano", "mediana", "bilateral", "ninguno"), *_cfg("suavizado")),
    "histogramas": Parametro("clip de CLAHE", (1.0, 2.0, 3.0, 4.0), *_cfg("clahe_clip"), lambda v: f"{v:.0f}"),
    "umbralizacion": Parametro("umbral mínimo de Otsu", (10, 20, 30, 50, 70, 90), *_cfg("t_min_otsu")),
    "morfologia": Parametro("tamaño del cierre", (3, 5, 7, 9), *_cfg("tam_cierre"), lambda v: f"{v} px"),
    "segmentacion": Parametro("corte mano-brazo", ("distancia", "ancho"), *_cfg("metodo_corte")),
    "caracterizacion": Parametro("un dedo sale a más de k·r, k =", (1.4, 1.5, 1.6, 1.7, 1.8, 2.0), *_cfg("k_dedo"),
                                 lambda v: f"{v:.1f}"),
    "transformers": Parametro("jugar con OWLv2", (False, True), lambda j: j.usar_owl,
                              lambda j, v: setattr(j, "usar_owl", v), lambda v: "sí" if v else "no"),
}


def cambiar_parametro(juego, vista: str, paso: int) -> None:
    p = PARAMETROS.get(vista)
    if p is None:
        return
    actual = p.leer(juego)
    i = p.valores.index(actual) if actual in p.valores else 0
    p.escribir(juego, p.valores[(i + paso) % len(p.valores)])


def texto_parametro(juego, vista: str) -> str:
    p = PARAMETROS.get(vista)
    return "" if p is None else f"{p.nombre}: {p.formato(p.leer(juego))}"


# --------------------------------------------------------------------------- #
# Utilidades de dibujo
# --------------------------------------------------------------------------- #


def _recorte(img: np.ndarray, mascara: np.ndarray | None, margen: int = 30) -> np.ndarray:
    if mascara is None or not mascara.any():
        return img
    ys, xs = np.nonzero(mascara)
    y0, y1 = max(0, ys.min() - margen), min(img.shape[0], ys.max() + margen)
    x0, x1 = max(0, xs.min() - margen), min(img.shape[1], xs.max() + margen)
    return img[y0:y1, x0:x1]


def _region_manos(res: pl.ResultadoFrame) -> np.ndarray | None:
    capas = [r.componente for r in res.manos.values() if r.componente is not None and r.componente.any()]
    return np.maximum.reduce(capas) if capas else None


def _ajustar(im: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    im = vis.a_bgr(im)
    e = min(ancho / im.shape[1], alto / im.shape[0])
    im = cv2.resize(im, (max(1, int(im.shape[1] * e)), max(1, int(im.shape[0] * e))), interpolation=cv2.INTER_AREA)
    c = np.full((alto, ancho, 3), vis.BGR_GRAFITO, np.uint8)
    y0, x0 = (alto - im.shape[0]) // 2, (ancho - im.shape[1]) // 2
    c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return c


def _lado_a_lado(a: np.ndarray, b: np.ndarray, ta: str, tb: str, ancho: int, alto: int) -> np.ndarray:
    m = ancho // 2
    ca, cb = _ajustar(a, m - 3, alto), _ajustar(b, ancho - m - 3, alto)
    for c, t in ((ca, ta), (cb, tb)):
        vis.poner_texto(c, t, (6, 4), max(12, alto // 16), vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
    sep = np.full((alto, 6, 3), vis.BGR_GRAFITO, np.uint8)
    return np.hstack([ca, sep, cb])[:, :ancho]


def _grafica(curvas: list[tuple[np.ndarray, tuple]], marcas: list[tuple[int, tuple]], ancho: int, alto: int,
             titulo: str = "") -> np.ndarray:
    """Gráfica simple con OpenCV (rápida, para dibujar en cada frame)."""
    g = np.full((alto, ancho, 3), (40, 32, 28), np.uint8)
    m = 8
    for y, color in curvas:
        y = np.asarray(y, float)
        if y.max() <= 0:
            continue
        xs = np.linspace(m, ancho - m, len(y))
        ys = alto - m - (y / y.max()) * (alto - 2 * m - 16)
        cv2.polylines(g, [np.column_stack([xs, ys]).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    for x, color in marcas:
        xp = int(m + x / 255 * (ancho - 2 * m))
        cv2.line(g, (xp, 18), (xp, alto - m), color, 2)
    if titulo:
        vis.poner_texto(g, titulo, (8, 2), max(12, alto // 10), vis.BGR_NEUTRO)
    return g


# --------------------------------------------------------------------------- #
# Una función por tema: devuelve (imagen del panel, líneas de datos)
# --------------------------------------------------------------------------- #


def _calibracion(juego, res, ancho, alto):
    modelo_cam = juego.modelo_camara
    lineas = []
    img = res.original.copy()
    if modelo_cam is not None:
        m = modelo_cam.escalar_a(img.shape[1], img.shape[0])
        # Rectas de la imagen corregida vistas a través de la lente: se curvan hacia los bordes
        h, w = img.shape[:2]
        for frac in np.linspace(0.05, 0.95, 9):
            for vertical in (True, False):
                t = np.linspace(0, 1, 40)
                u = np.full_like(t, frac * w) if vertical else t * w
                v = t * h if vertical else np.full_like(t, frac * h)
                xn = (u - m.K[0, 2]) / m.K[0, 0]
                yn = (v - m.K[1, 2]) / m.K[1, 1]
                pts, _ = cv2.projectPoints(np.column_stack([xn, yn, np.ones_like(xn)]), np.zeros(3), np.zeros(3),
                                           m.K, m.dist)
                cv2.polylines(img, [pts.reshape(-1, 2).astype(np.int32)], False, vis.BGR_NARANJA, 1, cv2.LINE_AA)
        vista = _lado_a_lado(img, res.imagen, "Lente (rectas curvadas)", "Corregida", ancho, alto)
        lineas.append(f"f = {res.modelo.K[0, 0]:.0f} px · centro ({res.modelo.K[0, 2]:.0f}, {res.modelo.K[1, 2]:.0f})"
                      f" · k1 = {modelo_cam.dist[0]:.3f}")
        if modelo_cam.error_rms is not None:
            lineas.append(f"Error de reproyección de tu calibración: {modelo_cam.error_rms:.2f} px")
    else:
        vista = _ajustar(img, ancho, alto)
        lineas.append("Cámara sin calibrar: K aproximada (f = ancho de la imagen, sin distorsión)")
    usando = "con" if juego.config.usar_calibracion and modelo_cam is not None else "sin"
    f_aprox = float(res.modelo.ancho)
    f_cal = float(modelo_cam.escalar_a(res.modelo.ancho, res.modelo.alto).K[0, 0]) if modelo_cam else f_aprox
    for lado, r in res.manos.items():
        if r.distancia_cm:
            otra = r.distancia_cm * (f_aprox / f_cal if usando == "con" else f_cal / f_aprox)
            lineas.append(f"{pl.NOMBRES_LADO[lado]}: Z = {r.distancia_cm:.0f} cm {usando} calibración"
                          f" ({otra:.0f} cm {'sin' if usando == 'con' else 'con'} ella)")
    return vista, lineas


def _filtros(juego, res, ancho, alto):
    region = _region_manos(res)
    antes = _recorte(res.clahe, region)
    despues = _recorte(res.suavizada, region).copy()
    bordes = _recorte(fil.canny(res.suavizada, 40, 120), region)
    despues[bordes > 0] = vis.BGR_NARANJA
    ruido = lambda im: float(np.std(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))  # noqa: E731
    vista = _lado_a_lado(antes, despues, "Antes", f"{juego.config.suavizado} + Canny", ancho, alto)
    return vista, [f"Suavizado {juego.config.suavizado} de {juego.config.k_suavizado}×{juego.config.k_suavizado}",
                   f"Ruido (σ del laplaciano): {ruido(antes):.1f} → {ruido(_recorte(res.suavizada, region)):.1f}",
                   f"Píxeles de borde (Canny) en la zona de las manos: {int(np.count_nonzero(bordes))}"]


def _histogramas(juego, res, ancho, alto):
    lineas = []
    mapas = []
    for lado in juego.lados:
        piel = juego.jugadores[lado].piel
        if piel is None:
            continue
        h = cv2.applyColorMap(vis.normalizar_u8(np.log1p(piel.hist_crcb)), cv2.COLORMAP_INFERNO)
        mapas.append(cv2.resize(h, (160, 160), interpolation=cv2.INTER_NEAREST))
        lineas.append(f"Piel de {pl.NOMBRES_LADO[lado]}: {piel.n_pixeles} píxeles en su histograma Cr-Cb")
    retros = [r.retro for r in res.manos.values() if r.retro is not None]
    retro = cv2.applyColorMap(np.maximum.reduce(retros), cv2.COLORMAP_INFERNO) if retros else res.imagen
    izq = np.vstack(mapas) if mapas else np.zeros((160, 160, 3), np.uint8)
    vista = _lado_a_lado(izq, retro, "Cr-Cb de cada piel", "Retroproyección", ancho, alto)
    g = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std())  # noqa: E731
    lineas.append(f"Contraste de la luz: {g(res.imagen):.1f} → {g(res.clahe):.1f} con CLAHE (clip "
                  f"{juego.config.clahe_clip:.0f})")
    return vista, lineas


def _umbralizacion(juego, res, ancho, alto):
    graficas, lineas = [], []
    for lado, r in res.manos.items():
        if r.retro is None:
            continue
        T, curva = umb.otsu_manual(r.retro, r.zona)
        hist = np.bincount(r.retro[r.zona > 0].ravel(), minlength=256).astype(float)
        hist[0] = 0  # el fondo (0) taparía todo lo demás
        graficas.append(_grafica([(hist, (180, 180, 180)), (curva, vis.BGR_NARANJA)],
                                 [(r.T_otsu or T, vis.BGR_VERDE)], ancho // 2 - 4, alto // 2 - 4,
                                 f"{pl.NOMBRES_LADO[lado]}: T = {r.T_otsu}"))
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: Otsu calcula T = {T} (se usa {r.T_otsu}, mínimo "
                      f"{juego.config.t_min_otsu})")
    curvas = np.vstack(graficas) if graficas else np.zeros((alto, ancho // 2, 3), np.uint8)
    mascara = res.mascara_umbral if res.mascara_umbral is not None else np.zeros(res.imagen.shape[:2], np.uint8)
    vista = _lado_a_lado(curvas, mascara, "σ²_B (naranja) e histograma", "Máscara binaria", ancho, alto)
    lineas.append("Verde: el umbral que maximiza la varianza entre clases")
    return vista, lineas


def _morfologia(juego, res, ancho, alto):
    region = _region_manos(res)
    antes = res.mascara_umbral if res.mascara_umbral is not None else np.zeros(res.imagen.shape[:2], np.uint8)
    despues = res.mascara_limpia if res.mascara_limpia is not None else antes
    zona = cv2.dilate(region, np.ones((61, 61), np.uint8)) if region is not None else None
    a, d = _recorte(antes, zona), _recorte(despues, zona)
    vista = _lado_a_lado(a, d, "Umbral", "Apertura + cierre", ancho, alto)
    n0 = cv2.connectedComponents(antes)[0] - 1
    n1 = cv2.connectedComponents(despues)[0] - 1
    quitados = int(np.count_nonzero(cv2.bitwise_and(antes, cv2.bitwise_not(despues))))
    rellenos = int(np.count_nonzero(cv2.bitwise_and(despues, cv2.bitwise_not(antes))))
    aviso = " (cuidado: puede pegar los dedos)" if juego.config.tam_cierre >= 9 else ""
    return vista, [f"Regiones: {n0} → {n1}", f"Píxeles de ruido quitados: {quitados} · huecos rellenados: {rellenos}",
                   f"Apertura {juego.config.tam_apertura} px · cierre {juego.config.tam_cierre} px{aviso}"]


def _segmentacion(juego, res, ancho, alto):
    vista = _ajustar(pl.dibujar_mascaras(res), ancho, alto)
    lineas = [f"Caras encontradas (Viola-Jones): {len(res.caras)}"]
    for lado, r in res.manos.items():
        if r.info_corte is None:
            lineas.append(f"{pl.NOMBRES_LADO[lado]}: sin mano en su mitad")
            continue
        corte = "corte por la muñeca" if r.info_corte.corte else "sin brazo visible"
        ws = " · watershed separó otra región" if r.uso_watershed else ""
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: {corte} ({juego.config.metodo_corte}){ws}")
    lineas.append("Blanco: mano · naranja: brazo · rojo: caras")
    return vista, lineas


def _caracterizacion(juego, res, ancho, alto):
    recortes, lineas = [], []
    for lado, r in res.manos.items():
        a = r.analisis
        if a is None:
            continue
        sil = np.zeros((*r.mascara_mano.shape, 3), np.uint8)
        if r.componente is not None:
            sil[r.componente > 0] = (70, 70, 70)
        sil[r.mascara_mano > 0] = vis.BGR_NEUTRO
        cv2.polylines(sil, [a.casco.reshape(-1, 1, 2).astype(np.int32)], True, vis.BGR_NARANJA, 2, cv2.LINE_AA)
        cv2.circle(sil, tuple(np.int32(a.centro_palma)), int(a.umbral_dedo), (120, 120, 120), 1, cv2.LINE_AA)
        for d in a.defectos:
            if d.valido:
                cv2.circle(sil, d.valle, 6, vis.BGR_ROJO, -1)
        for p in a.puntas:
            cv2.circle(sil, p, 9, vis.BGR_VERDE, 3)
        recortes.append(_recorte(sil, r.componente))
        d = r.descriptores
        e = r.cm_por_px or 0.0
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: {a.n_dedos} dedos → {a.gesto} · área {d.area * e ** 2:.0f} cm² · "
                      f"solidez {d.solidez:.2f} · circularidad {d.circularidad:.2f}")
    if not recortes:
        return _ajustar(res.imagen, ancho, alto), ["No hay manos que caracterizar en este frame"]
    vista = (_lado_a_lado(recortes[0], recortes[1], "Jugador 1", "Jugador 2", ancho, alto) if len(recortes) == 2
             else _ajustar(recortes[0], ancho, alto))
    lineas.append("Naranja: casco convexo · rojo: valles · verde: puntas")
    return vista, lineas


def _transformers(juego, res, ancho, alto):
    lineas = []
    prof, owl = juego.ultimo_prof, juego.ultimo_owl
    izq = (cv2.applyColorMap(vis.normalizar_u8(prof.mapa), cv2.COLORMAP_INFERNO) if prof is not None
           else np.zeros_like(res.imagen))
    der = res.imagen.copy()
    if owl is not None:
        for d in owl.detecciones:
            x0, y0, x1, y1 = (int(v) for v in d.caja)
            cv2.rectangle(der, (x0, y0), (x1, y1), vis.BGR_VERDE, 3)
        lineas.append(f"OWLv2 («a hand»): {len(owl.detecciones)} mano(s) en {owl.ms:.0f} ms")
    else:
        lineas.append("OWLv2: calculando…")
    if prof is not None:
        lineas.append(f"Depth Anything V2: {prof.ms:.0f} ms en {prof.dispositivo}")
        for lado in juego.lados:
            j = juego.jugadores[lado]
            if j.delante is not None:
                lineas.append(f"{pl.NOMBRES_LADO[lado]}: mano {'delante' if j.delante else 'detrás'} de su cara")
    else:
        lineas.append("Depth Anything V2: calculando…")
    lineas.append(f"Pipeline clásico (todo lo anterior): {sum(res.tiempos_ms.values()):.0f} ms")
    return _lado_a_lado(izq, der, "Profundidad (claro = cerca)", "OWLv2", ancho, alto), lineas


FUNCIONES = {"calibracion": _calibracion, "filtros": _filtros, "histogramas": _histogramas,
             "umbralizacion": _umbralizacion, "morfologia": _morfologia, "segmentacion": _segmentacion,
             "caracterizacion": _caracterizacion, "transformers": _transformers}


def componer(juego, res: pl.ResultadoFrame, img_juego: np.ndarray, vista: str) -> np.ndarray:
    """Juego a la izquierda y, a la derecha, el panel del tema elegido."""
    if vista not in FUNCIONES:
        return img_juego
    H, W = img_juego.shape[:2]
    ancho = int(W * 0.62)
    escala = H / 540
    t_titulo, t_texto = int(30 * escala), max(13, int(16 * escala))
    alto_vista = int(H * 0.56)
    panel = np.full((H, ancho, 3), vis.BGR_GRAFITO, np.uint8)
    imagen, lineas = FUNCIONES[vista](juego, res, ancho - 24, alto_vista)
    i = VISTAS.index(vista)
    vis.poner_texto(panel, f"{i}/{len(VISTAS) - 1} · {TITULOS[vista]}", (12, 8), t_titulo, vis.BGR_NEUTRO, negrita=True)
    vis.poner_texto(panel, QUE_HACE[vista], (12, 14 + int(t_titulo * 1.3)), t_texto - 1, vis.BGR_NARANJA)
    y0 = 24 + int(t_titulo * 1.3) + int(t_texto * 1.6)
    panel[y0:y0 + imagen.shape[0], 12:12 + imagen.shape[1]] = imagen[: H - y0]
    y = y0 + imagen.shape[0] + 10
    for linea in lineas:
        if y > H - 2 * t_texto - 10:
            break
        vis.poner_texto(panel, linea, (12, y), t_texto, vis.BGR_NEUTRO)
        y += int(t_texto * 1.45)
    parametro = texto_parametro(juego, vista)
    pista = ("[ ]" if juego.op.interfaz != "web" else "− / +") + f" cambia {parametro}" if parametro else ""
    pista += ("   ·   V: siguiente tema" if juego.op.interfaz != "web" else "")
    vis.poner_texto(panel, pista, (12, H - t_texto - 12), t_texto, (180, 182, 196))
    return np.hstack([img_juego, panel])
