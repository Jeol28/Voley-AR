"""Voley AR: voleibol en realidad aumentada sobre la mesa.

Paquete organizado en capas:

- ``voleyar.core``   → funciones puras de visión artificial (imagen in → imagen/datos out).
- ``voleyar.juego``  → física, IA rival, render AR y bucle de juego (usa ``core``).
- ``voleyar.ui``     → el juego en el navegador con Gradio (usa ``core`` y ``juego``).
- ``voleyar.utils``  → cámara, E/S, visualización y escenas sintéticas.
"""

__version__ = "2.5.0"
