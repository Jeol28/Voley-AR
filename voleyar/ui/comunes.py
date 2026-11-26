"""Piezas comunes de la interfaz web: el estado de la sesión y la concurrencia de los streams."""

from __future__ import annotations

import gradio as gr

from voleyar.entorno import EN_SPACE


# En un Space varios visitantes transmiten a la vez; en local basta un flujo por evento.
CONCURRENCIA_VIVO = 4 if EN_SPACE else 1
_SESION: gr.State | None = None


def sesion_componente() -> gr.State:
    """``gr.State`` con el estado de la sesión (calibración, piel). Lo crea ``crear_app``."""
    global _SESION
    if _SESION is None:
        _SESION = gr.State(None)
    return _SESION


def nueva_sesion_componente() -> gr.State:
    global _SESION
    _SESION = gr.State(None)
    return _SESION
