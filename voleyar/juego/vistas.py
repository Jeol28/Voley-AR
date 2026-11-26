"""Lentes del juego: cada tema del curso, visto **dentro** de la partida.

Elegir un tema cambia cómo se ve el mundo del juego, sin salir de él ni abrir
otro panel: la imagen de la cámara se reemplaza por lo que ese tema está
calculando en este mismo frame (la máscara de piel, los bordes, el mapa de
profundidad…). Encima se siguen dibujando la cancha, la red y el balón. Los números
y las gráficas del tema van en un panel FUERA de la cancha (una franja debajo del
video en la ventana, la columna izquierda en la web), para no tapar nunca el balón.
El ajuste se cambia mientras se juega (``[`` y ``]`` o los botones −/+ de la web).

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
    "calibracion": "Calibración",
    "filtros": "Filtros",
    "histogramas": "Histogramas",
    "umbralizacion": "Umbralización",
    "morfologia": "Morfología",
    "segmentacion": "Segmentación",
    "caracterizacion": "Caracterización",
    "transformers": "Transformers",
}
# El nombre del tema de la clase y, al lado, lo que hace en palabras de juego
APODOS = {  # cortos: tienen que caber en el menú
    "calibracion": "tu distancia",
    "filtros": "velocidad",
    "histogramas": "la luz",
    "umbralizacion": "¿qué se movió?",
    "morfologia": "zona de golpe",
    "segmentacion": "cada cosa",
    "caracterizacion": "¿qué golpe?",
    "transformers": "cerca o lejos",
}
QUE_HACE = {
    "calibracion": "El juego mide qué tan cerca está tu mano. ¡Acércala rápido a la cámara para hacer un remate!",
    "filtros": "Si mueves la mano muy rápido, la cámara la ve borrosa. Un filtro lo nota, y el golpe sale más fuerte.",
    "histogramas": "Si alguien prende o apaga una luz, el juego se da cuenta y no lo confunde con un golpe.",
    "umbralizacion": "El juego decide qué cambió lo suficiente para contar como movimiento. Lo blanco se está moviendo.",
    "morfologia": "El juego agranda lo que se mueve: con una zona de golpe más grande es más fácil alcanzar el balón.",
    "segmentacion": "Cada cosa que se mueve sale de un color. Tu cara no cuenta: así no golpeas con la cabeza sin querer.",
    "caracterizacion": "La forma decide el golpe: puño = golpe fuerte, mano abierta = toque suave, cuaderno = bloqueo.",
    "transformers": "Una inteligencia artificial ve qué está cerca. Lo que pasa lejos, por detrás, no golpea el balón.",
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


def _nombres(*pares):
    """Muestra cada valor técnico con una palabra sencilla."""
    d = dict(pares)
    return lambda v: d.get(v, str(v))


_SI_NO = _nombres((True, "sí"), (False, "no"))

PARAMETROS: dict[str, Parametro] = {
    "calibracion": Parametro("Usar la calibración", (True, False), lambda j: j.config.usar_calibracion,
                             lambda j, v: j.accion("calibracion") if v != j.config.usar_calibracion else None, _SI_NO),
    "filtros": Parametro("Fuerza por velocidad", (1.0, 2.0, 0.0), *_cfg("potencia_borroso"),
                         _nombres((0.0, "apagada"), (1.0, "normal"), (2.0, "turbo"))),
    "histogramas": Parametro("Atento a la luz", (0.15, 0.2, 0.3, 0.45), *_cfg("umbral_luz"),
                             _nombres((0.15, "muchísimo"), (0.2, "mucho"), (0.3, "normal"), (0.45, "poco"))),
    "umbralizacion": Parametro("Sensible al movimiento", (8, 12, 18, 25, 35, 50), *_cfg("mov_t_min"),
                               _nombres((8, "al máximo"), (12, "mucho"), (18, "normal"), (25, "poco"),
                                        (35, "muy poco"), (50, "casi nada"))),
    "morfologia": Parametro("Zona de golpe", (0, 4, 8, 14), *_cfg("mov_alcance"),
                            _nombres((0, "normal"), (4, "grande"), (8, "muy grande"), (14, "gigante"))),
    "segmentacion": Parametro("La cabeza golpea", (False, True), lambda j: not j.config.ignorar_caras_golpe,
                              lambda j, v: setattr(j.config, "ignorar_caras_golpe", not v), _SI_NO),
    "caracterizacion": Parametro("Contar dedos", (1.4, 1.5, 1.6, 1.7, 1.8, 2.0), *_cfg("k_dedo"),
                                 _nombres((1.4, "muy fácil"), (1.5, "fácil"), (1.6, "normal"), (1.7, "exigente"),
                                          (1.8, "muy exigente"), (2.0, "solo dedos bien abiertos"))),
    "transformers": Parametro("Solo golpea lo cercano", (0.0, 0.05, 0.1, 0.2, 0.3), *_cfg("cercania_min"),
                              _nombres((0.0, "apagado"), (0.05, "un poco"), (0.1, "normal"), (0.2, "más"),
                                       (0.3, "muy cerca"))),
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
        return "primero calibra la cámara desde el inicio (botón Calibrar cámara)"
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


def _al_panel(juego, grafica: np.ndarray, titulo: str = "") -> None:
    """Las gráficas van al panel de datos, fuera de la cancha: nunca tapan el balón."""
    if titulo:
        grafica = grafica.copy()
        vis.poner_texto(grafica, titulo, (4, 2), max(11, grafica.shape[0] // 9), vis.BGR_NEUTRO,
                        fondo=vis.BGR_GRAFITO)
    juego.graficas_tema.append(grafica)


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
            texto = f"a {r.distancia_cm:.0f} cm de la cámara"
            _etiqueta_mano(out, r, "¡remate!" if rapido else texto,
                           vis.BGR_VERDE if rapido else vis.BGR_NARANJA)
    lineas = [f"Remates en esta partida: {juego.stats['remates']}"]
    if usando:
        lineas.append("Cámara calibrada: las líneas naranjas muestran cómo dobla la imagen tu cámara, y el juego "
                      "la endereza")
    else:
        lineas.append("Cámara sin calibrar: las distancias son aproximadas, y el remate puede fallar "
                      "(y el remate se dispara de más o de menos)")
    return out, lineas


def _filtros(juego, res, img):
    # Bordes de Sobel en naranja tenue: lo nítido tiene bordes finos, lo que se mueve rápido los tiene anchos
    g = cv2.cvtColor(res.imagen, cv2.COLOR_BGR2GRAY)
    borde = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    borde = vis.normalizar_u8(np.minimum(borde, np.percentile(borde, 99)))
    out = _mezclar(_oscurecer(img, 0.45), np.full_like(img, vis.BGR_NARANJA), 0.6, (borde > 60).astype(np.uint8))
    m = res.mov
    if m is not None and m.etiquetas is not None:
        w = img.shape[1]
        for o in m.objetos:
            lado = -1 if o.centroide[0] < w / 2 else 1
            rapido = lado in juego.jugadores and juego.borrosidad(lado, o) >= 0.5
            color = vis.BGR_VERDE if rapido else vis.BGR_NEUTRO
            cv2.drawContours(out, [o.contorno], -1, color, 2, cv2.LINE_AA)
            x, y, _, _ = o.caja
            vis.poner_texto(out, "borroso: ¡más fuerza!" if rapido else "nítido", (x, max(0, y - 26)),
                            max(13, img.shape[0] // 30), color, fondo=vis.BGR_GRAFITO)
    lineas = [f"Golpes rapidísimos en esta partida: {juego.stats['rapidos']}",
              "Mueve la mano tan rápido que se vea borrosa: el balón sale más fuerte y más lejos"]
    return out, lineas


def _histogramas(juego, res, img):
    out = _oscurecer(img, 0.35)
    h, w = out.shape[:2]
    m = res.mov
    lineas = []
    if m is not None and m.hist_luz is not None:
        g = _grafica([(m.hist_ref, (180, 180, 180)), (m.hist_luz, vis.BGR_NARANJA)], [], w // 3, h // 4,
                     "luz: antes (gris) y ahora (naranja)")
        _al_panel(juego, g)
        estado = "¡Cambió la luz! El juego vuelve a mirar el cuarto" if m.cambio_luz else "La luz está estable"
        lineas.append(f"{estado} · cambios de luz: {juego.estado_pipe.movimiento.cambios_luz}")
    lado_mini = h // 5
    piel_txt = []
    for lado in juego.lados:
        piel = juego.jugadores[lado].piel
        if piel is None:
            piel_txt.append(f"{pl.NOMBRES_LADO[lado]}: aún aprendiendo su color de piel")
            continue
        mapa = cv2.applyColorMap(vis.normalizar_u8(np.log1p(piel.hist_crcb)), cv2.COLORMAP_INFERNO)
        mapa = cv2.resize(mapa, (lado_mini, lado_mini), interpolation=cv2.INTER_NEAREST)
        _al_panel(juego, mapa, f"color de piel {pl.NOMBRES_LADO[lado]}")
        piel_txt.append(f"{pl.NOMBRES_LADO[lado]}: ya conoce su color de piel")
    lineas.append(" · ".join(piel_txt))
    return out, lineas


def _umbralizacion(juego, res, img):
    out = _oscurecer(img, 0.45)
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
                     [(m.T_otsu, vis.BGR_NARANJA), (m.T, vis.BGR_VERDE)], w // 3, h // 4, "cuánto cambió cada punto")
        _al_panel(juego, g)
        lineas.append(f"Lo blanco se está moviendo. Para contar, algo tiene que cambiar al menos {m.T} "
                      "(el juego elige ese número solo en cada momento)")
    else:
        lineas.append("Mirando el cuarto… quédate quieto un segundo")
    lineas.append("La línea verde de la gráfica es el punto donde el juego corta: a la derecha, movimiento")
    return out, lineas


def _morfologia(juego, res, img):
    out = _oscurecer(img, 0.45)
    m = res.mov
    if m is None or m.cruda is None:
        return out, ["Mirando el cuarto… quédate quieto un segundo"]
    antes, despues = _grande(m.cruda, img), _grande(m.limpia, img)
    objetos = _grande((m.etiquetas > 0).astype(np.uint8) * 255, img)
    if m.zona_golpe is not None:  # la zona de golpe (dilatación), en naranja alrededor de cada objeto
        zona = _grande((m.zona_golpe > 0).astype(np.uint8) * 255, img)
        out = _mezclar(out, np.full_like(out, vis.BGR_NARANJA), 0.55, zona)
    out[objetos > 0] = vis.BGR_NEUTRO
    quitados = cv2.bitwise_and(antes, cv2.bitwise_not(despues))
    out[cv2.dilate(quitados, np.ones((3, 3), np.uint8)) > 0] = vis.BGR_ROJO
    tam = PARAMETROS["morfologia"].formato(juego.config.mov_alcance)
    return out, [f"Blanco: lo que se mueve · naranja: tu zona de golpe ({tam}) · rojo: manchitas borradas",
                 "Si al balón lo toca lo naranja, cuenta como golpe"]


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
    cara = "no golpea" if juego.config.ignorar_caras_golpe else "sí golpea"
    linea = f"Cosas moviéndose: {n} · caras que no cuentan: {nc} (la cabeza {cara})"
    lineas = [linea, "Naranja: cosas que se mueven · verde: la que toca el balón · rojo: tu cara"]
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
        tipos.append(f"{pl.NOMBRES_LADO[lado]}: {nombre}")
    for lado, r in res.manos.items():
        v = r.validacion
        if r.descriptores is not None and v is not None and not v.valida:
            cv2.drawContours(out, [r.analisis.contorno], -1, vis.BGR_ROJO, 3, cv2.LINE_AA)
            _etiqueta_mano(out, r, "esto no es una mano", vis.BGR_ROJO)
        elif r.hay_mano:
            n = r.dedos_estables if r.dedos_estables is not None else r.analisis.n_dedos
            _etiqueta_mano(out, r, f"{n} dedos")
    hechos = ", ".join(f"{TIPOS_GOLPE[t]} {n}" for t, n in juego.stats["tipos"].items()) or "todavía ninguno"
    lineas = [("Ahora: " + " · ".join(tipos)) if tipos else
              "Prueba con el puño, con la mano abierta o con un cuaderno",
              f"Golpes de esta partida: {hechos}"]
    return out, lineas


def _transformers(juego, res, img):
    prof, owl = juego.ultimo_prof, juego.ultimo_owl
    if juego.prof is not None and juego.prof.resultado is not None:
        prof = juego.prof.resultado
    out = img
    lineas = []
    cmin = juego.config.cercania_min
    lineas.append(f"Cosas lejanas que no pudieron golpear: {juego.stats['lejos']}" +
                  (" (el filtro está apagado)" if cmin <= 0 else ""))
    if prof is not None:
        mapa = cv2.resize(prof.mapa.astype(np.float32), (img.shape[1], img.shape[0]))  # pudo cambiar la resolución
        out = _mezclar(img, cv2.applyColorMap(vis.normalizar_u8(mapa), cv2.COLORMAP_INFERNO), 0.7)
        lineas.append("Colores claros = cerca de la cámara · oscuros = lejos")
    else:
        lineas.append("Calculando qué está cerca…")
    out = out.copy()
    for lado, c in _toques(juego).items():
        j = juego.jugadores[lado]
        if j.cercania is None:
            continue
        x, y, _, _ = c.objeto.caja
        cerca = not j.lejos
        cv2.drawContours(out, [c.objeto.contorno], -1, vis.BGR_VERDE if cerca else vis.BGR_ROJO, 2, cv2.LINE_AA)
        vis.poner_texto(out, "cerca" if cerca else "lejos: no cuenta", (x, max(0, y - 26)),
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
    juego.graficas_tema = []
    if vista not in LENTES:
        return img, []
    return LENTES[vista](juego, res, img)


def panel(juego, vista: str, lineas: list[str], ancho: int, alto: int) -> np.ndarray | None:
    """Panel de datos del tema, **fuera** de la cancha (debajo del video o en la columna de la web).

    En la escena solo queda la capa del tema sobre la imagen de la cámara; el balón,
    la red y el marcador se dibujan encima, así que nada tapa el juego.
    """
    if vista not in LENTES:
        return None
    out = np.full((alto, ancho, 3), vis.BGR_GRAFITO, np.uint8)
    t = max(12, min(22, ancho // 46))
    parametro = texto_parametro(juego, vista)
    if parametro and not parametro_disponible(juego, vista):
        pista = "Calibra la cámara desde el inicio para poder cambiar esto"
    else:
        pista = (("Teclas [ ]" if juego.op.interfaz != "web" else "Botones − / +") + f": {parametro}") if parametro else ""
    cabe = max(24, int((ancho - 24) / (t * 0.52)))
    bloques = [(f"{TITULOS[vista]}: {APODOS[vista]}", vis.BGR_NARANJA, True), (QUE_HACE[vista], vis.BGR_NEUTRO, False),
               *[(x, vis.BGR_NEUTRO, False) for x in lineas[:2]],
               *([(pista, (180, 182, 196), False)] if pista else [])]
    y = int(t * 0.5)
    for texto, color, titulo in bloques:
        for parte in _partir(texto, int(cabe * 0.88) if titulo else cabe):  # la negrita es más ancha
            if y + t > alto:
                break
            vis.poner_texto(out, parte, (12, y), t + 1 if titulo else t, color, negrita=titulo)
            y += int(t * 1.4)
    graficas = getattr(juego, "graficas_tema", [])
    libre = alto - y - 8
    if graficas and libre > 40:  # las gráficas, en fila, en lo que queda del panel
        ancho_g = (ancho - 12 * (len(graficas) + 1)) // len(graficas)
        x = 12
        for g in graficas:
            e = min(libre / g.shape[0], ancho_g / g.shape[1])
            g2 = cv2.resize(g, (max(1, int(g.shape[1] * e)), max(1, int(g.shape[0] * e))), interpolation=cv2.INTER_AREA)
            out[y:y + g2.shape[0], x:x + g2.shape[1]] = g2
            cv2.rectangle(out, (x - 1, y - 1), (x + g2.shape[1], y + g2.shape[0]), vis.BGR_NARANJA, 1)
            x += g2.shape[1] + 12
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
