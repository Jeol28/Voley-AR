"""Genera los imprimibles de Voley AR en PDF tamaño carta, con medidas reales.

- ``docs/imprimibles/tablero.pdf``: tablero de ajedrez 10 × 7 cuadros
  (9 × 6 esquinas interiores) para la calibración intrínseca.
- ``docs/imprimibles/cancha.pdf``: mini cancha azul con 4 marcadores ArUco
  (DICT_4X4_50, IDs 0-3), línea central y medidas en cm.

Todo se dibuja en vectores (incluidos los marcadores, celda por celda) para
que al imprimir **al 100 %, sin ajustar a la página**, las medidas sean exactas.

Uso::

    python scripts/generar_imprimibles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.colors import Color, white, black
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voleyar.core import config as cfg  # noqa: E402
from voleyar.utils import sinteticos  # noqa: E402

AZUL = Color(22 / 255, 92 / 255, 176 / 255)
GRAFITO = Color(26 / 255, 35 / 255, 50 / 255)
NARANJA = Color(1.0, 107 / 255, 53 / 255)
ANCHO_PAG, ALTO_PAG = landscape(letter)  # 27.94 × 21.59 cm


def regla(c: canvas.Canvas, x: float, y: float, largo_cm: int = 10) -> None:
    """Regla de verificación con marcas cada milímetro (en cm de página)."""
    c.setStrokeColor(black)
    c.setLineWidth(0.6)
    c.line(x * cm, y * cm, (x + largo_cm) * cm, y * cm)
    for mm in range(largo_cm * 10 + 1):
        alto = 0.45 if mm % 10 == 0 else (0.3 if mm % 5 == 0 else 0.18)
        xm = (x + mm / 10) * cm
        c.line(xm, y * cm, xm, (y + alto) * cm)
        if mm % 10 == 0:
            c.setFont("Helvetica", 6.5)
            c.drawCentredString(xm, (y - 0.35) * cm, str(mm // 10))
    c.setFont("Helvetica", 7.5)
    c.drawString((x + largo_cm + 0.3) * cm, (y - 0.05) * cm, f"{largo_cm} cm: mídela para verificar la escala")


def generar_tablero(ruta: Path) -> None:
    c = canvas.Canvas(str(ruta), pagesize=landscape(letter))
    c.setTitle("Voley AR – Tablero de calibración")
    c.setAuthor("Jorge Enrique Olaya Liévano")
    cols, filas = cfg.TABLERO_ESQUINAS[0] + 1, cfg.TABLERO_ESQUINAS[1] + 1
    q = cfg.TABLERO_CUADRO_CM
    ancho, alto = cols * q, filas * q
    x0 = (ANCHO_PAG / cm - ancho) / 2
    y0 = 2.0  # deja espacio abajo para la regla (fuera del margen no imprimible)
    c.setFillColor(black)
    for j in range(filas):
        for i in range(cols):
            if (i + j) % 2 == 0:
                c.rect((x0 + i * q) * cm, (y0 + j * q) * cm, q * cm, q * cm, stroke=0, fill=1)
    c.setFillColor(GRAFITO)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0 * cm, (y0 + alto + 0.85) * cm,
                 f"Voley AR · Tablero de calibración · {cols} × {filas} cuadros "
                 f"({cfg.TABLERO_ESQUINAS[0]} × {cfg.TABLERO_ESQUINAS[1]} esquinas interiores)")
    c.setFont("Helvetica", 8.5)
    c.drawString(x0 * cm, (y0 + alto + 0.3) * cm,
                 f"Cuadro nominal: {q * 10:.0f} mm. Imprime al 100 % (sin ajustar a la página), mide un cuadro "
                 "con una regla y usa ese valor al calibrar. Pégalo sobre cartón: debe quedar plano.")
    regla(c, x0, 1.0)
    c.showPage()
    c.save()


def _marcador_vectorial(c: canvas.Canvas, id_marcador: int, cx: float, cy: float, lado: float) -> None:
    """Dibuja un marcador ArUco celda por celda (cx, cy en cm de página)."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.DICCIONARIO_ARUCO))
    n = dic.markerSize + 2  # bits + borde negro
    bits = cv2.aruco.generateImageMarker(dic, id_marcador, n)  # 1 px por celda
    celda = lado / n
    x0, y0 = cx - lado / 2, cy + lado / 2
    c.setFillColor(black)
    eps = 0.004  # leve solape para que el visor no deje líneas finas entre celdas
    for f in range(n):
        k = 0
        while k < n:  # agrupa celdas negras consecutivas de la fila en un solo rectángulo
            if bits[f, k] < 128:
                k0 = k
                while k < n and bits[f, k] < 128:
                    k += 1
                c.rect((x0 + k0 * celda - eps) * cm, (y0 - (f + 1) * celda - eps) * cm,
                       ((k - k0) * celda + 2 * eps) * cm, (celda + 2 * eps) * cm, stroke=0, fill=1)
            else:
                k += 1


def generar_cancha(ruta: Path) -> None:
    c = canvas.Canvas(str(ruta), pagesize=landscape(letter))
    c.setTitle("Voley AR – Mini cancha")
    c.setAuthor("Jorge Enrique Olaya Liévano")
    ox, oy = ANCHO_PAG / cm / 2, ALTO_PAG / cm / 2  # origen de la mesa = centro de la hoja

    def P(x: float, y: float) -> tuple[float, float]:
        return (ox + x) * cm, (oy + y) * cm

    L, A = cfg.MEDIO_LARGO, cfg.MEDIO_ANCHO
    c.setFillColor(AZUL)
    c.rect(*P(-L, -A), 2 * L * cm, 2 * A * cm, stroke=0, fill=1)
    # Líneas blancas
    c.setStrokeColor(white)
    c.setLineWidth(0.25 * cm)
    c.rect(*P(-L + 0.3, -A + 0.3), (2 * L - 0.6) * cm, (2 * A - 0.6) * cm, stroke=1, fill=0)
    c.line(*P(0, -A), *P(0, A))
    c.setLineWidth(0.12 * cm)
    c.setStrokeColor(Color(0.92, 0.92, 0.92))
    for xa in (-4.2, 4.2):
        c.line(*P(xa, -A + 0.3), *P(xa, A - 0.3))
    # Marcadores con su zona blanca
    for i, (cx, cy) in cfg.CENTROS_MARCADORES.items():
        m = cfg.LADO_MARCADOR / 2 + cfg.MARGEN_BLANCO_MARCADOR
        c.setFillColor(white)
        c.rect(*P(cx - m, cy - m), 2 * m * cm, 2 * m * cm, stroke=0, fill=1)
        _marcador_vectorial(c, i, ox + cx, oy + cy, cfg.LADO_MARCADOR)
    # Textos (blanco sobre azul)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(*P(-L / 2 - 0.6, -0.3), "TU LADO")
    c.drawCentredString(*P(L / 2 + 0.6, -0.3), "RIVAL")
    c.setFont("Helvetica", 8.5)
    c.drawCentredString(*P(-L / 2 - 0.6, -1.0), "(este borde hacia ti)")
    c.saveState()
    c.translate(*P(0.45, -A + 2.0))
    c.rotate(90)
    c.drawString(0, 0, f"RED (x = 0) · altura virtual {cfg.ALTURA_RED:.0f} cm")
    c.restoreState()
    # Cotas
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(*P(0 - 4.5, A - 1.05), f"← {2 * L:.0f} cm →")
    c.saveState()
    c.translate(*P(-L + 1.0, 0))
    c.rotate(90)
    c.drawCentredString(0, 0, f"← {2 * A:.0f} cm →")
    c.restoreState()
    c.setFont("Helvetica", 7)
    for i, (cx, cy) in cfg.CENTROS_MARCADORES.items():
        dy = -(cfg.LADO_MARCADOR / 2 + cfg.MARGEN_BLANCO_MARCADOR + 0.35) if cy > 0 else \
            (cfg.LADO_MARCADOR / 2 + cfg.MARGEN_BLANCO_MARCADOR + 0.15)
        c.drawCentredString(*P(cx, cy + dy), f"ID {i} · {cfg.LADO_MARCADOR:g} cm")
    # Pie mínimo fuera del campo con la instrucción de impresión
    c.setFillColor(GRAFITO)
    c.setFont("Helvetica", 7.5)
    c.drawString(*P(-L, -A - 0.55), "Imprime al 100 % (sin escalar). Comprueba con una regla: "
                 f"campo {2 * L:.0f} × {2 * A:.0f} cm, marcadores de {cfg.LADO_MARCADOR:g} cm.")
    c.showPage()
    c.save()


def main() -> None:
    destino = cfg.RUTA_IMPRIMIBLES
    destino.mkdir(parents=True, exist_ok=True)
    generar_tablero(destino / "tablero.pdf")
    generar_cancha(destino / "cancha.pdf")
    # Vistas previas PNG para el README
    tex, _, _ = sinteticos.textura_cancha(30.0)
    cv2.imwrite(str(destino / "cancha_preview.png"), tex)
    tab, _, _ = sinteticos.textura_tablero(20.0)
    cv2.imwrite(str(destino / "tablero_preview.png"), tab)
    print(f"Imprimibles generados en {destino}")
    for f in sorted(destino.iterdir()):
        print("  ·", f.name)


if __name__ == "__main__":
    main()
