"""Configuración común de pytest: añade la raíz del proyecto al path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

# Las pruebas no cargan Depth Anything en un hilo (lento y no determinista); sus pruebas lo llaman a propósito
os.environ.setdefault("VOLEYAR_PROFUNDIDAD", "0")
