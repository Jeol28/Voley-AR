"""Entrada/salida: lectura y escritura de JSON y del modelo de cámara."""

from __future__ import annotations

import json
from pathlib import Path

from voleyar.core import config as cfg
from voleyar.core.calibracion import ModeloCamara


def guardar_json(datos: dict, ruta: Path | str) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar_json(ruta: Path | str) -> dict | None:
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_modelo_camara(modelo: ModeloCamara, ruta: Path | str = cfg.RUTA_CAMARA_JSON) -> Path:
    return guardar_json(modelo.a_dict(), ruta)


def cargar_modelo_camara(ruta: Path | str = cfg.RUTA_CAMARA_JSON) -> ModeloCamara | None:
    """Carga ``config/camara.json``. Devuelve ``None`` si aún no se ha calibrado."""
    datos = cargar_json(ruta)
    if datos is None:
        return None
    return ModeloCamara.desde_dict(datos)
