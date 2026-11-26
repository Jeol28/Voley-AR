"""Tema visual de la UI: grafito de base, naranja de acento, neutro para el texto.

Verde y rojo se reservan para datos (golpe válido/inválido, aciertos/errores).
Sin etiquetas pequeñas sobre los títulos y sin pie de página.
"""

from __future__ import annotations

import gradio as gr

GRAFITO = "#1A2332"
GRAFITO_2 = "#222D3F"
GRAFITO_3 = "#2C3A50"
NARANJA = "#FF6B35"
NEUTRO = "#F5F5F7"
NEUTRO_TENUE = "#AEB6C4"


def crear_tema() -> gr.themes.Base:
    naranja = gr.themes.Color(c50="#FFF1EB", c100="#FFE0D2", c200="#FFC2A6", c300="#FFA37A", c400="#FF8555",
                              c500=NARANJA, c600="#E85A26", c700="#C2481C", c800="#9A3815", c900="#73290F",
                              c950="#4D1B0A")
    grafito = gr.themes.Color(c50=NEUTRO, c100="#E4E7EC", c200="#C9CFD8", c300=NEUTRO_TENUE, c400="#7F8A9C",
                              c500="#5A6678", c600="#3A4658", c700=GRAFITO_3, c800=GRAFITO_2, c900=GRAFITO,
                              c950="#121926")
    tema = gr.themes.Base(primary_hue=naranja, secondary_hue=naranja, neutral_hue=grafito,
                          font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                          font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
                          radius_size=gr.themes.sizes.radius_md)
    valores = dict(
        body_background_fill=GRAFITO,
        body_text_color=NEUTRO,
        body_text_color_subdued=NEUTRO_TENUE,
        background_fill_primary=GRAFITO_2,
        background_fill_secondary=GRAFITO,
        block_background_fill=GRAFITO_2,
        block_border_color=GRAFITO_3,
        block_label_background_fill=GRAFITO_3,
        block_label_text_color=NEUTRO,
        block_title_text_color=NEUTRO,
        border_color_primary=GRAFITO_3,
        border_color_accent=NARANJA,
        color_accent=NARANJA,
        color_accent_soft=GRAFITO_3,
        input_background_fill=GRAFITO,
        input_border_color=GRAFITO_3,
        input_placeholder_color=NEUTRO_TENUE,
        button_primary_background_fill=NARANJA,
        button_primary_background_fill_hover="#E85A26",
        button_primary_text_color=GRAFITO,
        button_secondary_background_fill=GRAFITO_3,
        button_secondary_background_fill_hover="#3A4658",
        button_secondary_text_color=NEUTRO,
        slider_color=NARANJA,
        checkbox_background_color=GRAFITO,
        checkbox_background_color_selected=NARANJA,
        checkbox_border_color=GRAFITO_3,
        table_even_background_fill=GRAFITO_2,
        table_odd_background_fill=GRAFITO,
        table_border_color=GRAFITO_3,
        panel_background_fill=GRAFITO_2,
        panel_border_color=GRAFITO_3,
        link_text_color=NARANJA,
        link_text_color_hover="#FF8555",
        code_background_fill=GRAFITO,
    )
    # Mismos valores en modo claro y oscuro: la app siempre se ve en grafito
    import inspect

    aceptados = set(inspect.signature(tema.set).parameters)
    valores.update({f"{k}_dark": v for k, v in list(valores.items()) if f"{k}_dark" in aceptados})
    return tema.set(**{k: v for k, v in valores.items() if k in aceptados})


CSS = f"""
footer {{ display: none !important; }}
.gradio-container {{ max-width: 1500px !important; }}
h1 {{ color: {NEUTRO}; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 0.1em; }}
h2, h3 {{ color: {NEUTRO}; }}
.acento {{ color: {NARANJA}; }}
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {{ color: {NARANJA} !important;
    border-color: {NARANJA} !important; }}
.dato-ok {{ color: #00C896; font-weight: 600; }}
.dato-mal {{ color: #FF4C4C; font-weight: 600; }}
.prose table td, .prose table th {{ padding: 4px 10px; }}
"""

CSS += f"""
/* --- Interfaz de juego --- */
.nav-holder {{ display: none !important; }}  /* el juego es una sola página */
.pantalla {{ max-width: 900px; margin: 0 auto !important; }}
.pantalla-juego {{ max-width: 1400px; margin: 0 auto !important; }}
.hero {{ text-align: center; padding: 56px 0 18px 0; }}
.hero h1 {{ font-size: 5.2rem !important; font-weight: 800; letter-spacing: -0.03em; margin: 0;
           color: {NEUTRO}; line-height: 1; }}
.hero h1::after {{ content: ""; display: block; width: 120px; height: 8px; background: {NARANJA};
                  border-radius: 4px; margin: 22px auto 0 auto; }}
.hero .lema {{ font-size: 1.3rem; color: {NEUTRO_TENUE}; margin-top: 22px; }}
.fila-botones {{ justify-content: center; gap: 18px !important; margin-top: 14px; }}
.boton-grande {{ font-size: 1.55rem !important; font-weight: 700 !important; padding: 26px 18px !important;
                border-radius: 14px !important; min-height: 92px; }}
.boton-medio {{ font-size: 1.1rem !important; padding: 14px !important; border-radius: 12px !important; }}
.chip {{ text-align: center; color: {NEUTRO_TENUE}; margin-top: 26px; font-size: 0.98rem; }}
.chip.ok {{ color: #00C896; }}
.nota {{ color: {NEUTRO_TENUE}; }}
"""
