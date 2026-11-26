"""Lentes del juego: cada tema del curso, visto **dentro** de la partida.

Elegir un tema cambia cómo se ve el mundo del juego, sin salir de él ni abrir
otro panel: la imagen de la cámara se reemplaza por lo que ese tema está
calculando en este mismo frame (la máscara de piel, los bordes, el mapa de
profundidad…). Encima se siguen dibujando la cancha, la red y el balón, y abajo
aparece una tarjeta translúcida con los números del tema y su ajuste, que se
cambia mientras se juega (teclas ``[`` y ``]`` o los botones −/+ de la web).

Todo sale del ``ResultadoFrame`` que ya calculó el pipeline: los lentes no
recalculan la visión, solo la muestran.
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


def _mezclar(img: np.ndarray, capa: np.ndarray, alfa: float, mascara: np.ndarray | None = None) -> np.ndarray:
    capa = vis.a_bgr(capa)
    mezcla = cv2.addWeighted(capa, alfa, img, 1 - alfa, 0)
    if mascara is None:
        return mezcla
    out = img.copy()
    out[mascara > 0] = mezcla[mascara > 0]
    return out


def _oscurecer(img: np.ndarray, cuanto: float = 0.55) -> np.ndarray:
    return (img.astype(np.float32) * (1 - cuanto)).astype(np.uint8)


def _insertar(img: np.ndarray, mini: np.ndarray, x: int, y: int, titulo: str = "") -> None:
    """Pequeña imagen enmarcada dentro de la escena (por ejemplo, el histograma de cada jugador)."""
    mini = vis.a_bgr(mini)
    h, w = mini.shape[:2]
    H, W = img.shape[:2]
    x, y = max(0, min(W - w, x)), max(0, min(H - h, y))
    img[y:y + h, x:x + w] = mini
    cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), vis.BGR_NARANJA, 1)
    if titulo:
        vis.poner_texto(img, titulo, (x + 4, y + 2), max(11, h // 9), vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)


def _grafica(curvas: list[tuple[np.ndarray, tuple]], marcas: list[tuple[int, tuple]], ancho: int, alto: int,
             titulo: str = "") -> np.ndarray:
    """Gráfica simple con OpenCV (rápida, para dibujar en cada frame)."""
    g = np.full((alto, ancho, 3), (40, 32, 28), np.uint8)
    m = 6
    for y, color in curvas:
        y = np.asarray(y, float)
        if y.max() <= 0:
            continue
        xs = np.linspace(m, ancho - m, len(y))
        ys = alto - m - (y / y.max()) * (alto - 2 * m - 14)
        cv2.polylines(g, [np.column_stack([xs, ys]).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    for x, color in marcas:
        xp = int(m + x / 255 * (ancho - 2 * m))
        cv2.line(g, (xp, 14), (xp, alto - m), color, 2)
    if titulo:
        vis.poner_texto(g, titulo, (6, 1), max(11, alto // 9), vis.BGR_NEUTRO)
    return g


def _etiqueta_mano(img, r, texto: str, color=vis.BGR_NEUTRO) -> None:
    if r.analisis is None:
        return
    c = r.analisis.centro_palma
    y = int(max(0, c[1] - r.analisis.radio_palma - 34))
    vis.poner_texto(img, texto, (int(max(0, c[0] - 80)), y), max(13, img.shape[0] // 34), color,
                    fondo=vis.BGR_GRAFITO)


# --------------------------------------------------------------------------- #
# Un lente por tema: devuelve (escena vista a través del tema, líneas de datos)
# --------------------------------------------------------------------------- #


def _calibracion(juego, res, img):
    modelo_cam = juego.modelo_camara
    lineas = []
    h, w = img.shape[:2]
    out = img.copy()
    if modelo_cam is not None and juego.config.usar_calibracion:
        m = modelo_cam.escalar_a(res.original.shape[1], res.original.shape[0])
        if juego.config.espejo:
            m = m.espejo()
        m = m.escalar_a(w, h)
        rect = res.modelo
        # Rectas de la imagen corregida (blanco) y cómo las vería la lente sin corregir (naranja)
        for frac in np.linspace(0.1, 0.9, 7):
            for vertical in (True, False):
                t = np.linspace(0, 1, 50)
                u = np.full_like(t, frac * w) if vertical else t * w
                v = t * h if vertical else np.full_like(t, frac * h)
                recta = np.column_stack([u, v]).astype(np.int32)
                cv2.polylines(out, [recta], False, (200, 200, 200), 1, cv2.LINE_AA)
                xn = (u - rect.K[0, 2]) / rect.K[0, 0]
                yn = (v - rect.K[1, 2]) / rect.K[1, 1]
                pts, _ = cv2.projectPoints(np.column_stack([xn, yn, np.ones_like(xn)]), np.zeros(3), np.zeros(3),
                                           m.K, m.dist)
                cv2.polylines(out, [pts.reshape(-1, 2).astype(np.int32)], False, vis.BGR_NARANJA, 2, cv2.LINE_AA)
        lineas.append(f"Blanco: rectas corregidas · naranja: cómo las curva tu lente (k1 = {modelo_cam.dist[0]:.3f})")
        if modelo_cam.error_rms is not None:
            lineas.append(f"f = {rect.K[0, 0]:.0f} px · error de reproyección de tu calibración {modelo_cam.error_rms:.2f} px")
    else:
        lineas.append("Sin calibración: K aproximada (f = ancho de la imagen, sin distorsión)")
    f_aprox = float(res.modelo.ancho)
    f_cal = (float(modelo_cam.escalar_a(res.modelo.ancho, res.modelo.alto).K[0, 0]) if modelo_cam is not None
             else f_aprox)
    usando = juego.config.usar_calibracion and modelo_cam is not None
    for lado, r in res.manos.items():
        if r.distancia_cm:
            otra = r.distancia_cm * (f_aprox / f_cal if usando else f_cal / f_aprox)
            _etiqueta_mano(out, r, f"{r.distancia_cm:.0f} cm (sin calibrar: {otra if usando else r.distancia_cm:.0f})"
                           if usando else f"{r.distancia_cm:.0f} cm (calibrada: {otra:.0f})", vis.BGR_NARANJA)
    lineas.append("Sobre cada mano: su distancia a la cámara, Z = f·W/w")
    return out, lineas


def _filtros(juego, res, img):
    bordes = fil.canny(res.suavizada, 40, 120)
    out = _oscurecer(res.suavizada, 0.25)
    out[bordes > 0] = vis.BGR_NARANJA
    ruido = lambda im: float(np.std(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))  # noqa: E731
    return out, [f"Ves la imagen suavizada ({juego.config.suavizado}, {juego.config.k_suavizado}×"
                 f"{juego.config.k_suavizado}) con los bordes de Canny en naranja",
                 f"Ruido (σ del laplaciano): {ruido(res.clahe):.1f} → {ruido(res.suavizada):.1f}"]


def _histogramas(juego, res, img):
    retros = [r.retro for r in res.manos.values() if r.retro is not None]
    out = img
    if retros:
        calor = cv2.applyColorMap(np.maximum.reduce(retros), cv2.COLORMAP_INFERNO)
        out = _mezclar(_oscurecer(img, 0.5), calor, 0.75, np.maximum.reduce(retros))
    h, w = out.shape[:2]
    lado_mini = h // 4
    lineas = []
    for lado in juego.lados:
        piel = juego.jugadores[lado].piel
        if piel is None:
            continue
        mapa = cv2.applyColorMap(vis.normalizar_u8(np.log1p(piel.hist_crcb)), cv2.COLORMAP_INFERNO)
        mapa = cv2.resize(mapa, (lado_mini, lado_mini), interpolation=cv2.INTER_NEAREST)
        x = 10 if lado < 0 else w - lado_mini - 10
        _insertar(out, mapa, x, int(h * 0.2), "Cr-Cb")
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: su piel en Cr-Cb ({piel.n_pixeles} px); la retroproyección pinta "
                      "lo que se le parece")
    g = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std())  # noqa: E731
    lineas.append(f"Contraste con CLAHE (clip {juego.config.clahe_clip:.0f}): {g(res.imagen):.1f} → {g(res.clahe):.1f}")
    return out, lineas


def _umbralizacion(juego, res, img):
    mascara = res.mascara_umbral if res.mascara_umbral is not None else np.zeros(img.shape[:2], np.uint8)
    out = _oscurecer(img, 0.7)
    out[mascara > 0] = vis.BGR_NEUTRO
    h, w = out.shape[:2]
    lineas = []
    for lado, r in res.manos.items():
        if r.retro is None:
            continue
        T, curva = umb.otsu_manual(r.retro, r.zona)
        hist = np.bincount(r.retro[r.zona > 0].ravel(), minlength=256).astype(float)
        hist[0] = 0
        g = _grafica([(hist, (180, 180, 180)), (curva, vis.BGR_NARANJA)], [(r.T_otsu or T, vis.BGR_VERDE)],
                     w // 4, h // 5, f"T = {r.T_otsu}")
        x = 10 if lado < 0 else w - w // 4 - 10
        _insertar(out, g, x, int(h * 0.2))
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: Otsu elige T = {T} (se usa {r.T_otsu}; mínimo {juego.config.t_min_otsu})")
    lineas.append("Blanco: lo que supera el umbral · gráfica: σ²_B (naranja), histograma y T (verde)")
    return out, lineas


def _morfologia(juego, res, img):
    antes = res.mascara_umbral if res.mascara_umbral is not None else np.zeros(img.shape[:2], np.uint8)
    despues = res.mascara_limpia if res.mascara_limpia is not None else antes
    out = _oscurecer(img, 0.6)
    out[despues > 0] = vis.BGR_NEUTRO
    quitados = cv2.bitwise_and(antes, cv2.bitwise_not(despues))
    rellenos = cv2.bitwise_and(despues, cv2.bitwise_not(antes))
    out[cv2.dilate(quitados, np.ones((3, 3), np.uint8)) > 0] = vis.BGR_ROJO
    out[rellenos > 0] = vis.BGR_VERDE
    n0 = cv2.connectedComponents(antes)[0] - 1
    n1 = cv2.connectedComponents(despues)[0] - 1
    aviso = " (cuidado: pega los dedos)" if juego.config.tam_cierre >= 9 else ""
    return out, [f"Rojo: ruido que quita la apertura ({int(np.count_nonzero(quitados))} px) · verde: huecos que "
                 f"rellena el cierre ({int(np.count_nonzero(rellenos))} px)",
                 f"Regiones: {n0} → {n1} · apertura {juego.config.tam_apertura} px · cierre "
                 f"{juego.config.tam_cierre} px{aviso}"]


def _segmentacion(juego, res, img):
    out = _oscurecer(img, 0.45)
    for r in res.manos.values():
        if r.componente is not None:
            out = _mezclar(out, np.full_like(out, vis.BGR_NARANJA), 0.7, r.componente)
        if r.mascara_mano is not None:
            out = _mezclar(out, np.full_like(out, vis.BGR_NEUTRO), 0.8, r.mascara_mano)
        if r.info_corte is not None and r.info_corte.corte is not None:
            cv2.line(out, r.info_corte.corte[0], r.info_corte.corte[1], vis.BGR_ROJO, 3, cv2.LINE_AA)
    for x, y, cw, ch in res.caras:
        cv2.rectangle(out, (x, y), (x + cw, y + ch), vis.BGR_ROJO, 2)
    lineas = [f"Blanco: mano · naranja: brazo · línea roja: corte por la muñeca ({juego.config.metodo_corte}) · "
              f"recuadros: {len(res.caras)} cara(s) de Viola-Jones"]
    ws = [pl.NOMBRES_LADO[l] for l, r in res.manos.items() if r.uso_watershed]
    if ws:
        lineas.append("Watershed separó otra región pegada a la mano de " + " y ".join(ws))
    return out, lineas


def _caracterizacion(juego, res, img):
    out = pl.dibujar_depuracion(res, img)
    lineas = []
    for lado, r in res.manos.items():
        if r.descriptores is None:
            continue
        d, e = r.descriptores, (r.cm_por_px or 0.0)
        lineas.append(f"{pl.NOMBRES_LADO[lado]}: {r.analisis.n_dedos} dedos → {r.analisis.gesto} · área "
                      f"{d.area * e ** 2:.0f} cm² · solidez {d.solidez:.2f} · circularidad {d.circularidad:.2f}")
    lineas.append("Blanco: casco convexo · rojo: valles · verde: puntas · cruz: punto de golpe")
    return out, lineas


def _transformers(juego, res, img):
    prof, owl = juego.ultimo_prof, juego.ultimo_owl
    out = img
    lineas = []
    if prof is not None and prof.mapa.shape[:2] == img.shape[:2]:
        out = _mezclar(img, cv2.applyColorMap(vis.normalizar_u8(prof.mapa), cv2.COLORMAP_INFERNO), 0.7)
        lineas.append(f"Depth Anything V2 ({prof.ms:.0f} ms): claro = cerca de la cámara")
    else:
        lineas.append("Depth Anything V2: calculando…")
    out = out.copy()
    if owl is not None:
        for d in owl.detecciones:
            x0, y0, x1, y1 = (int(v) for v in d.caja)
            cv2.rectangle(out, (x0, y0), (x1, y1), vis.BGR_VERDE, 3)
            vis.poner_texto(out, f"a hand {d.puntaje:.2f}", (x0 + 4, max(0, y0 - 22)), 15, vis.BGR_VERDE,
                            fondo=vis.BGR_GRAFITO)
        lineas.append(f"OWLv2 («a hand»): {len(owl.detecciones)} mano(s) en {owl.ms:.0f} ms · clásico "
                      f"{sum(res.tiempos_ms.values()):.0f} ms")
    for lado in juego.lados:
        j = juego.jugadores[lado]
        if j.delante is not None:
            _etiqueta_mano(out, res.manos[lado], "delante de su cara" if j.delante else "detrás de su cara",
                           vis.BGR_VERDE if j.delante else vis.BGR_ROJO)
    return out, lineas


LENTES = {"calibracion": _calibracion, "filtros": _filtros, "histogramas": _histogramas,
          "umbralizacion": _umbralizacion, "morfologia": _morfologia, "segmentacion": _segmentacion,
          "caracterizacion": _caracterizacion, "transformers": _transformers}


def lente(juego, res: pl.ResultadoFrame, img: np.ndarray, vista: str) -> tuple[np.ndarray, list[str]]:
    """La escena (antes de dibujar la cancha y el balón) vista a través del tema elegido."""
    if vista not in LENTES:
        return img, []
    return LENTES[vista](juego, res, img)


def tarjeta(juego, img: np.ndarray, vista: str, lineas: list[str]) -> np.ndarray:
    """Tarjeta translúcida abajo, como un HUD: el tema, sus datos y cómo cambiar su ajuste."""
    if vista not in LENTES:
        return img
    H, W = img.shape[:2]
    t = max(11, H // 44)
    parametro = texto_parametro(juego, vista)
    pista = (("[ ]" if juego.op.interfaz != "web" else "− / +") + f" cambia {parametro}") if parametro else ""
    filas = [f"{VISTAS.index(vista)} · {TITULOS[vista]} — {QUE_HACE[vista]}", *lineas[:2]] + ([pista] if pista else [])
    alto = int((len(filas) + 0.6) * t * 1.45)
    y0 = H - alto - int(t * 2.6)  # deja libre la línea de estado de abajo
    capa = img.copy()
    cv2.rectangle(capa, (8, y0), (W - 8, y0 + alto), vis.BGR_GRAFITO, -1)
    out = cv2.addWeighted(capa, 0.72, img, 0.28, 0)
    cv2.rectangle(out, (8, y0), (W - 8, y0 + alto), vis.BGR_NARANJA, 1)
    y = y0 + int(t * 0.4)
    for k, fila in enumerate(filas):
        color = vis.BGR_NARANJA if k == 0 else ((180, 182, 196) if fila is pista else vis.BGR_NEUTRO)
        vis.poner_texto(out, fila, (18, y), t if k else t + 1, color, negrita=(k == 0))
        y += int(t * 1.45)
    return out
