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

from voleyar.core import pipeline as pl
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
    "calibracion": "Con K se mide la distancia de la mano (Z = f·W/w): llevarla rápido hacia la cámara es un remate.",
    "filtros": "El suavizado quita el ruido de la cámara para que no parezca algo que se mueve y golpee solo.",
    "histogramas": "Si el histograma de la luz cambia de golpe es una lámpara, no un golpe: se reaprende el cuarto.",
    "umbralizacion": "Otsu decide en cada frame cuánto tiene que cambiar un píxel para contar como movimiento.",
    "morfologia": "Apertura quita el ruido suelto y cierre une el objeto que se mueve en una sola pieza.",
    "segmentacion": "Cada cosa que se mueve es un objeto; la cara no cuenta, así la cabeza no golpea sin querer.",
    "caracterizacion": "La forma de lo que golpea decide el golpe: puño fuerte, mano abierta suave, objeto grande bloqueo.",
    "transformers": "Depth Anything dice qué está cerca: alguien que pasa por detrás no devuelve el balón.",
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
    "histogramas": Parametro("cambio de luz desde", (0.15, 0.2, 0.3, 0.45), *_cfg("umbral_luz"), lambda v: f"{v:.2f}"),
    "umbralizacion": Parametro("umbral mínimo del movimiento", (8, 12, 18, 25, 35, 50), *_cfg("mov_t_min")),
    "morfologia": Parametro("apertura del movimiento", (1, 3, 5, 7), *_cfg("mov_apertura"), lambda v: f"{v} px"),
    "segmentacion": Parametro("la cara golpea", (False, True), lambda j: not j.config.ignorar_caras_golpe,
                              lambda j, v: setattr(j.config, "ignorar_caras_golpe", not v),
                              lambda v: "sí" if v else "no"),
    "caracterizacion": Parametro("un dedo sale a más de k·r, k =", (1.4, 1.5, 1.6, 1.7, 1.8, 2.0), *_cfg("k_dedo"),
                                 lambda v: f"{v:.1f}"),
    "transformers": Parametro("cercanía mínima para golpear", (0.0, 0.05, 0.1, 0.2, 0.3), *_cfg("cercania_min"),
                              lambda v: "no se filtra" if v <= 0 else f"{v:.2f}"),
}


def cambiar_parametro(juego, vista: str, paso: int) -> None:
    p = PARAMETROS.get(vista)
    if p is None:
        return
    actual = p.leer(juego)
    i = p.valores.index(actual) if actual in p.valores else 0
    p.escribir(juego, p.valores[(i + paso) % len(p.valores)])


def parametro_disponible(juego, vista: str) -> bool:
    """¿Se puede cambiar ahora el ajuste de esta vista? (sin calibración no hay nada que encender)."""
    if vista not in PARAMETROS or juego is None:
        return False
    if vista == "calibracion":
        return juego.modelo_camara is not None
    return True


def texto_parametro(juego, vista: str) -> str:
    p = PARAMETROS.get(vista)
    if p is None:
        return ""
    if not parametro_disponible(juego, vista):
        return "nada todavía: calibra la cámara desde el inicio (Calibrar cámara) para comparar con y sin calibración"
    return f"{p.nombre}: {p.formato(p.leer(juego))}"


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


def _grande(m, img):
    """Una máscara de la imagen reducida del movimiento, al tamaño de la escena."""
    return cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


def _toques(juego) -> dict:
    """Lo que toca el balón ahora en cada mitad (movimiento.Toque), si algo lo toca o lo roza."""
    out = {}
    for lado, j in getattr(juego, "jugadores", {}).items():
        c = getattr(j, "contacto", None)
        if c is not None and c.objeto is not None:
            out[lado] = c
    return out


def _calibracion(juego, res, img):
    modelo_cam = juego.modelo_camara
    h, w = img.shape[:2]
    out = img.copy()
    usando = juego.config.usar_calibracion and modelo_cam is not None
    if usando:
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
    for lado, r in res.manos.items():
        if r.distancia_cm and r.hay_mano:
            vz = juego.jugadores[lado].vz
            rapido = vz is not None and vz <= -juego.config.remate_vz
            texto = f"{r.distancia_cm:.0f} cm" + (f" · {vz:+.0f} cm/s" if vz is not None else "")
            _etiqueta_mano(out, r, texto + (" ¡remate!" if rapido else ""),
                           vis.BGR_VERDE if rapido else vis.BGR_NARANJA)
    lineas = [f"Remate: mano hacia la cámara a más de {juego.config.remate_vz:.0f} cm/s · remates en la partida: "
              f"{juego.stats['remates']}"]
    if usando:
        lineas.append(f"Blanco: rectas corregidas · naranja: cómo las curva tu lente (k1 = {modelo_cam.dist[0]:.3f}) · "
                      f"f = {res.modelo.K[0, 0]:.0f} px")
    else:
        lineas.append("Sin calibración: f aproximada = ancho de la imagen, así que Z y su velocidad salen mal "
                      "(y el remate se dispara de más o de menos)")
    return out, lineas


def _filtros(juego, res, img):
    out = _oscurecer(res.suavizada, 0.35)
    m = res.mov
    lineas = []
    if m is not None and m.cruda is not None:
        cruda, limpia = _grande(m.cruda, img), _grande(m.limpia, img)
        sueltos = cv2.bitwise_and(cruda, cv2.bitwise_not(limpia))
        out[limpia > 0] = vis.BGR_NEUTRO
        out[cv2.dilate(sueltos, np.ones((3, 3), np.uint8)) > 0] = vis.BGR_ROJO
        lineas.append(f"Puntos de ruido que «se movieron»: {m.puntos_sueltos} (rojo) · con «ninguno» suben, y si "
                      "crecen golpean el balón solos")
    else:
        lineas.append("Aprendiendo el fondo…")
    ruido = lambda im: float(np.std(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F)))  # noqa: E731
    lineas.append(f"Suavizado {juego.config.suavizado} {juego.config.k_suavizado}×{juego.config.k_suavizado}: ruido "
                  f"(σ del laplaciano) {ruido(res.clahe):.1f} → {ruido(res.suavizada):.1f} · blanco: lo que sí se mueve")
    return out, lineas


def _histogramas(juego, res, img):
    out = _oscurecer(img, 0.35)
    h, w = out.shape[:2]
    m = res.mov
    lineas = []
    if m is not None and m.hist_luz is not None:
        g = _grafica([(m.hist_ref, (180, 180, 180)), (m.hist_luz, vis.BGR_NARANJA)], [], w // 3, h // 4,
                     "luz: referencia (gris) y ahora (naranja)")
        _insertar(out, g, w // 2 - w // 6, int(h * 0.2))
        cambio = " ¡cambio de luz!" if m.cambio_luz else ""
        lineas.append(f"Bhattacharyya entre histogramas: {m.distancia_luz:.2f} (desde {juego.config.umbral_luz:.2f} es "
                      f"la luz y se reaprende el cuarto) · cambios de luz: {juego.estado_pipe.movimiento.cambios_luz}"
                      f"{cambio}")
    lado_mini = h // 5
    piel_txt = []
    for lado in juego.lados:
        piel = juego.jugadores[lado].piel
        if piel is None:
            piel_txt.append(f"{pl.NOMBRES_LADO[lado]}: piel por rango fijo YCrCb")
            continue
        mapa = cv2.applyColorMap(vis.normalizar_u8(np.log1p(piel.hist_crcb)), cv2.COLORMAP_INFERNO)
        mapa = cv2.resize(mapa, (lado_mini, lado_mini), interpolation=cv2.INTER_NEAREST)
        _insertar(out, mapa, 10 if lado < 0 else w - lado_mini - 10, int(h * 0.2), "piel Cr-Cb")
        piel_txt.append(f"{pl.NOMBRES_LADO[lado]}: piel aprendida sola ({juego.jugadores[lado].n_aprendizajes} veces)")
    lineas.append(" · ".join(piel_txt) + " · la retroproyección de ese histograma busca la mano")
    return out, lineas


def _umbralizacion(juego, res, img):
    out = _oscurecer(img, 0.7)
    h, w = out.shape[:2]
    m = res.mov
    lineas = []
    if m is not None and m.diferencia is not None:
        dif = _grande(m.diferencia, img)
        out = _mezclar(out, cv2.applyColorMap(dif, cv2.COLORMAP_INFERNO), 0.8, (dif > 6).astype(np.uint8))
        out[_grande(m.cruda, img) > 0] = vis.BGR_NEUTRO
        hist = np.bincount(m.diferencia.ravel(), minlength=256).astype(float)
        hist[:3] = 0
        g = _grafica([(np.log1p(hist), (180, 180, 180)), (m.curva_otsu, vis.BGR_NARANJA)],
                     [(m.T_otsu, vis.BGR_NARANJA), (m.T, vis.BGR_VERDE)], w // 3, h // 4, f"|frame − fondo|, T = {m.T}")
        _insertar(out, g, w // 2 - w // 6, int(h * 0.2))
        lineas.append(f"Otsu sobre |frame − fondo| elige T = {m.T_otsu} → se usa {m.T} (mínimo "
                      f"{juego.config.mov_t_min}) · blanco: lo que lo supera, se está moviendo")
    else:
        lineas.append("Aprendiendo el fondo…")
    ts = [f"{pl.NOMBRES_LADO[l]} T = {r.T_otsu}" for l, r in res.manos.items() if r.T_otsu is not None]
    lineas.append("Otsu también separa la piel de la mano: " + (" · ".join(ts) if ts else "aún sin histograma de piel"))
    return out, lineas


def _morfologia(juego, res, img):
    out = _oscurecer(img, 0.6)
    m = res.mov
    if m is None or m.cruda is None:
        return out, ["Aprendiendo el fondo…"]
    antes, despues = _grande(m.cruda, img), _grande(m.limpia, img)
    out[despues > 0] = vis.BGR_NEUTRO
    quitados = cv2.bitwise_and(antes, cv2.bitwise_not(despues))
    rellenos = cv2.bitwise_and(despues, cv2.bitwise_not(antes))
    out[cv2.dilate(quitados, np.ones((3, 3), np.uint8)) > 0] = vis.BGR_ROJO
    out[rellenos > 0] = vis.BGR_VERDE
    n0 = cv2.connectedComponents(m.cruda)[0] - 1
    n1 = cv2.connectedComponents(m.limpia)[0] - 1
    return out, [f"Rojo: ruido que quita la apertura ({m.px_quitados} px) · verde: lo que une el cierre · "
                 f"regiones {n0} → {n1}: menos pedazos sueltos que puedan tocar el balón",
                 f"Apertura {juego.config.mov_apertura} px · cierre {juego.config.mov_cierre} px (en la imagen de "
                 f"320 px) · la máscara de la mano usa apertura {juego.config.tam_apertura} y cierre "
                 f"{juego.config.tam_cierre}"]


def _segmentacion(juego, res, img):
    out = _oscurecer(img, 0.45)
    m = res.mov
    toques = _toques(juego)
    que_toca = {c.objeto.etiqueta for c in toques.values() if c.toca}
    if m is not None and m.etiquetas is not None:
        for o in m.objetos:
            color = vis.BGR_VERDE if o.etiqueta in que_toca else vis.BGR_NARANJA
            out = _mezclar(out, np.full_like(out, color), 0.6, _grande((m.etiquetas == o.etiqueta).astype(np.uint8), img))
            cv2.drawContours(out, [o.contorno], -1, color, 2, cv2.LINE_AA)
        for o in m.descartados_cara:
            cv2.drawContours(out, [o.contorno], -1, vis.BGR_ROJO, 2, cv2.LINE_AA)
    for x, y, cw, ch in res.caras:
        cv2.rectangle(out, (x, y), (x + cw, y + ch), vis.BGR_ROJO, 2)
    n = len(m.objetos) if m is not None else 0
    nc = len(m.descartados_cara) if m is not None else 0
    cara = "no golpea" if juego.config.ignorar_caras_golpe else "SÍ golpea"
    linea = f"{n} objeto(s) en movimiento (naranja; verde = el que toca el balón) · {nc} descartado(s) por ser la " \
            f"cara (rojo): la cara {cara}"
    t = [f"{pl.NOMBRES_LADO[l]} {c.fraccion * 100:.0f} % del balón" for l, c in toques.items()]
    lineas = [linea, ("Sobre el balón: " + " · ".join(t) + " (desde 12 % lo devuelve)") if t else
              f"Recuadros rojos: {len(res.caras)} cara(s) de Viola-Jones · el fondo lo modela MOG2"]
    return out, lineas


def _caracterizacion(juego, res, img):
    from voleyar.core.caracterizacion import TIPOS_GOLPE

    out = pl.dibujar_depuracion(res, img)
    tipos = []
    for lado, c in _toques(juego).items():
        o, tipo = c.objeto, juego.jugadores[lado].tipo
        cv2.drawContours(out, [o.contorno], -1, vis.BGR_VERDE, 2, cv2.LINE_AA)
        x, y, _, _ = o.caja
        nombre = TIPOS_GOLPE.get(tipo, "—")
        vis.poner_texto(out, nombre, (x, max(0, y - 26)), max(13, img.shape[0] // 30), vis.BGR_VERDE,
                        fondo=vis.BGR_GRAFITO)
        tipos.append(f"{pl.NOMBRES_LADO[lado]}: {nombre} (solidez {o.solidez:.2f})")
    for lado, r in res.manos.items():
        v = r.validacion
        if r.descriptores is not None and v is not None and not v.valida:
            cv2.drawContours(out, [r.analisis.contorno], -1, vis.BGR_ROJO, 3, cv2.LINE_AA)
            _etiqueta_mano(out, r, "no es una mano: " + v.motivos[0], vis.BGR_ROJO)
        elif r.hay_mano:
            n = r.dedos_estables if r.dedos_estables is not None else r.analisis.n_dedos
            _etiqueta_mano(out, r, f"{n} dedos · {r.gesto[0]}")
    hechos = ", ".join(f"{TIPOS_GOLPE[t]} {n}" for t, n in juego.stats["tipos"].items()) or "ninguno aún"
    lineas = [("Ahora: " + " · ".join(tipos)) if tipos else
              "Puño = golpe fuerte · mano abierta = toque suave · objeto grande y compacto = bloqueo",
              f"Golpes de la partida: {hechos} · verde: lo que toca el balón · contorno rojo: forma descartada"]
    return out, lineas


def _transformers(juego, res, img):
    prof, owl = juego.ultimo_prof, juego.ultimo_owl
    if juego.prof is not None and juego.prof.resultado is not None:
        prof = juego.prof.resultado
    out = img
    lineas = []
    cmin = juego.config.cercania_min
    filtro = "no se filtra" if cmin <= 0 else f"cercanía ≥ {cmin:.2f}"
    lineas.append(f"Solo golpea lo que está cerca ({filtro}) · descartados por estar lejos: {juego.stats['lejos']}")
    if prof is not None and prof.mapa.shape[:2] == img.shape[:2]:
        out = _mezclar(img, cv2.applyColorMap(vis.normalizar_u8(prof.mapa), cv2.COLORMAP_INFERNO), 0.7)
        extra = f" · OWLv2: {len(owl.detecciones)} mano(s) en {owl.ms:.0f} ms" if owl is not None else ""
        lineas.append(f"Depth Anything V2 ({prof.ms:.0f} ms, {prof.dispositivo}): claro = cerca de la cámara{extra}")
    else:
        lineas.append("Depth Anything V2: calculando…")
    out = out.copy()
    for lado, c in _toques(juego).items():
        j = juego.jugadores[lado]
        if j.cercania is None:
            continue
        x, y, _, _ = c.objeto.caja
        cerca = not j.lejos
        cv2.drawContours(out, [c.objeto.contorno], -1, vis.BGR_VERDE if cerca else vis.BGR_ROJO, 2, cv2.LINE_AA)
        vis.poner_texto(out, f"{'cerca' if cerca else 'lejos'} {j.cercania:+.2f}", (x, max(0, y - 26)),
                        max(13, img.shape[0] // 30), vis.BGR_VERDE if cerca else vis.BGR_ROJO, fondo=vis.BGR_GRAFITO)
    if owl is not None:
        for d in owl.detecciones:
            x0, y0, x1, y1 = (int(v) for v in d.caja)
            cv2.rectangle(out, (x0, y0), (x1, y1), vis.BGR_VERDE, 3)
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
    if parametro and not parametro_disponible(juego, vista):
        pista = "Sin calibración de la cámara: calíbrala desde el inicio para compararla"
    else:
        pista = (("[ ]" if juego.op.interfaz != "web" else "− / +") + f" cambia {parametro}") if parametro else ""
    cabe = max(30, int((W - 40) / (t * 0.52)))  # caracteres por línea (aprox.) a este tamaño de letra
    filas, estilos = [], []
    bloques = [(f"{VISTAS.index(vista)} · {TITULOS[vista]} — {QUE_HACE[vista]}", vis.BGR_NARANJA, True),
               *[(x, vis.BGR_NEUTRO, False) for x in lineas[:2]],
               *([(pista, (180, 182, 196), False)] if pista else [])]
    for texto, color, titulo in bloques:  # el texto salta de línea en vez de salirse de la imagen
        for parte in _partir(texto, int(cabe * 0.88) if titulo else cabe):  # la negrita es más ancha
            filas.append(parte)
            estilos.append((color, titulo))
    alto = int((len(filas) + 0.6) * t * 1.45)
    y0 = H - alto - int(t * 2.6)  # deja libre la línea de estado de abajo
    capa = img.copy()
    cv2.rectangle(capa, (8, y0), (W - 8, y0 + alto), vis.BGR_GRAFITO, -1)
    out = cv2.addWeighted(capa, 0.72, img, 0.28, 0)
    cv2.rectangle(out, (8, y0), (W - 8, y0 + alto), vis.BGR_NARANJA, 1)
    y = y0 + int(t * 0.4)
    for fila, (color, titulo) in zip(filas, estilos):
        vis.poner_texto(out, fila, (18, y), t + 1 if titulo else t, color, negrita=titulo)
        y += int(t * 1.45)
    return out


def _partir(texto: str, cabe: int) -> list[str]:
    """Parte un texto en líneas de como mucho ``cabe`` caracteres, sin cortar palabras."""
    filas, actual = [], ""
    for palabra in texto.split(" "):
        if actual and len(actual) + 1 + len(palabra) > cabe:
            filas.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}" if actual else palabra
    return filas + ([actual] if actual else [])
