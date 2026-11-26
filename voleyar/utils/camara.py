"""Acceso a la cámara web (AVFoundation en macOS)."""

from __future__ import annotations

import platform

import cv2
import numpy as np


class FuenteCamara:
    """Envoltura de ``cv2.VideoCapture`` con la misma interfaz que ``FuenteSintetica``."""

    def __init__(self, indice: int = 0, ancho: int = 1280, alto: int = 720, espejo: bool = False):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(indice, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        self.espejo = espejo
        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara {indice}. En macOS revisa Ajustes del Sistema → "
                "Privacidad y seguridad → Cámara y da permiso a la Terminal / VS Code."
            )

    @property
    def tam(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def leer(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        if ok and self.espejo:
            # Ojo: el espejo invierte la geometría; solo usarlo para visualizar.
            frame = cv2.flip(frame, 1)
        return ok, frame

    def liberar(self) -> None:
        self.cap.release()
