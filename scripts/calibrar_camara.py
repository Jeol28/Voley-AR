"""Calibración intrínseca guiada de la cámara web (ventana de OpenCV).

Uso::

    python scripts/calibrar_camara.py --cuadro 2.5 --vistas 15

Teclas:
    ESPACIO  capturar la vista actual (si el tablero está detectado y es nueva)
    A        activar/desactivar captura automática de vistas nuevas
    C        calibrar con las vistas capturadas (mínimo 6, ideal 15+)
    R        reiniciar las vistas
    Q / ESC  salir

Al calibrar se guarda ``config/camara.json`` y se muestra la imagen original
junto a la corregida para comprobar que las rectas quedan rectas.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voleyar.core import calibracion as cal  # noqa: E402
from voleyar.core import config as cfg  # noqa: E402
from voleyar.utils.camara import FuenteCamara  # noqa: E402
from voleyar.utils.io import guardar_modelo_camara  # noqa: E402
from voleyar.utils import visualizacion as vis  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--cuadro", type=float, default=cfg.TABLERO_CUADRO_CM, help="lado del cuadro medido (cm)")
    ap.add_argument("--vistas", type=int, default=15)
    args = ap.parse_args()

    fuente = FuenteCamara(args.camara, args.ancho, args.alto)
    ok, frame = fuente.leer()
    if not ok:
        raise SystemExit("La cámara no entregó imágenes.")
    h, w = frame.shape[:2]
    gestor = cal.GestorCapturaCalibracion((w, h), objetivo=args.vistas)
    auto = False
    ultimo_auto = 0.0
    mensaje, t_mensaje = "", 0.0
    resultado = None
    ventana = "Voley AR · Calibración de cámara"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = fuente.leer()
        if not ok:
            break
        vista = frame.copy()
        detectado, pts = cal.detectar_tablero(frame, rapido=True)
        if detectado:
            cv2.drawChessboardCorners(vista, cfg.TABLERO_ESQUINAS, pts, True)
            if auto and time.time() - ultimo_auto > 1.0 and gestor.agregar(pts):
                ultimo_auto = time.time()
                mensaje, t_mensaje = f"Vista {len(gestor.vistas)} capturada", time.time()
        # Vistas ya capturadas: puntos tenues
        for v in gestor.vistas:
            for p in v.reshape(-1, 2)[::4]:
                cv2.circle(vista, tuple(np.int32(p)), 2, vis.BGR_NARANJA, -1)
        estado = "Tablero detectado" if detectado else "Buscando el tablero…"
        color = vis.BGR_VERDE if detectado else vis.BGR_ROJO
        vis.poner_texto(vista, estado, (20, 16), 26, color, negrita=True, fondo=vis.BGR_GRAFITO)
        vis.poner_texto(vista, f"Vistas: {len(gestor.vistas)}/{args.vistas}   ·   auto: {'sí' if auto else 'no'}",
                        (20, 56), 22, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        vis.poner_texto(vista, gestor.sugerencia(), (20, 90), 22, vis.BGR_NARANJA, fondo=vis.BGR_GRAFITO)
        vis.poner_texto(vista, "ESPACIO capturar · A auto · C calibrar · R reiniciar · Q salir",
                        (20, h - 40), 20, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
        if mensaje and time.time() - t_mensaje < 1.5:
            vis.poner_texto(vista, mensaje, (w // 2 - 120, h // 2), 32, vis.BGR_VERDE, True, fondo=vis.BGR_GRAFITO)
        if resultado is not None:
            m = resultado.modelo
            vis.poner_texto(vista, f"RMS = {m.error_rms:.3f} px · fx = {m.K[0, 0]:.1f} · fy = {m.K[1, 1]:.1f}",
                            (20, 124), 22, vis.BGR_VERDE, fondo=vis.BGR_GRAFITO)
        cv2.imshow(ventana, vista)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" ") and detectado:
            if gestor.agregar(pts):
                mensaje = f"Vista {len(gestor.vistas)} capturada"
            else:
                mensaje = "Vista muy parecida a otra: mueve o inclina el tablero"
            t_mensaje = time.time()
        elif k == ord("a"):
            auto = not auto
        elif k == ord("r"):
            gestor.reiniciar()
            resultado = None
        elif k == ord("c"):
            if len(gestor.vistas) < 6:
                mensaje, t_mensaje = "Necesitas al menos 6 vistas", time.time()
                continue
            resultado = cal.calibrar_camara(gestor.vistas, (w, h), tam_cuadro=args.cuadro)
            ruta = guardar_modelo_camara(resultado.modelo)
            m = resultado.modelo
            print("\n=== Calibración ===")
            print("K =\n", np.array2string(m.K, precision=2, suppress_small=True))
            print("dist [k1 k2 p1 p2 k3] =", np.array2string(m.dist, precision=4))
            print(f"Error de reproyección RMS = {m.error_rms:.4f} px con {m.n_vistas} vistas")
            print("Error por vista:", " ".join(f"{e:.2f}" for e in resultado.errores_por_vista))
            print(f"Guardado en {ruta}")
            corregida, _ = cal.corregir_distorsion(frame, m, alpha=1.0)
            comp = np.hstack([frame, corregida])
            vis.poner_texto(comp, "Original", (20, 16), 28, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            vis.poner_texto(comp, "Sin distorsión", (w + 20, 16), 28, vis.BGR_NEUTRO, fondo=vis.BGR_GRAFITO)
            cv2.imshow("Comparación (cualquier tecla para seguir)", comp)
            cv2.waitKey(0)
            cv2.destroyWindow("Comparación (cualquier tecla para seguir)")
    fuente.liberar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
