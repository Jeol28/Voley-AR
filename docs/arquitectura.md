# Arquitectura de Voley AR

Este documento amplía la sección de arquitectura del [README](../README.md): sistemas de coordenadas, flujo de
datos, concurrencia y las decisiones que se tomaron a partir de las pruebas.

## 1. Capas

| Capa | Paquete | Regla |
|---|---|---|
| Presentación | `voleyar/ui`, `voleyar/juego/bucle.py`, `app.py`, `jugar.py` | Maneja eventos, ventanas y widgets. No implementa visión. |
| Dominio del juego | `voleyar/juego` (`fisica`, `ia_rival`, `partida`, `render_ar`) | Reglas y física en cm. `partida.py` no conoce la cámara. |
| Núcleo de visión | `voleyar/core` | Funciones puras: imagen in → imagen/datos out. Sin Gradio ni `imshow`. |
| Infraestructura | `voleyar/utils` | Cámara, JSON, texto con tildes, escenas sintéticas. |

La dependencia siempre va hacia abajo. La UI y el juego llaman a la **misma** función,
`core.pipeline.procesar_frame`, así que lo que se ve en la pestaña Pipeline es lo que usa el juego.

## 2. Sistemas de coordenadas

- **Mesa** (cm): origen en el centro de la cancha impresa; X hacia el lado rival, Y hacia el borde superior de la
  hoja y Z hacia arriba. Tu lado es `x < 0` y la red es el plano `x = 0` con `z ∈ [0, 5]`.
- **Cámara**: `P_c = R P_w + t` (OpenCV: x a la derecha, y hacia abajo, z hacia adelante).
- **Imagen** (px): `s [u v 1]ᵀ = K P_c`, con distorsión radial/tangencial.
- **Vista cenital** (px): `u = (X − x_min)·ppcm`, `v = (y_max − Y)·ppcm`, con `ppcm = 10`. Se obtiene con la
  homografía `H = A · (K [r₁ r₂ t])⁻¹`.

Todas las medidas de la cancha viven en `core/config.py` y las usan por igual el generador de PDF, el render
sintético y la estimación de pose.

## 3. Flujo de un frame

```
frame (1280×720)
 └─ preparar_imagen ── reducción a 960 px; si hay calibración: remap(K, d) → imagen rectificada, K', d = 0
     └─ detectar_marcadores + estimar_pose_cancha ── 16 esquinas ArUco, IPPE → refinamiento iterativo
         (si no hay marcadores: se reutiliza la pose anterior hasta 20 frames)
     └─ preprocesar ── CLAHE en L (Lab) + suavizado gaussiano 5×5
     └─ zona_busqueda ── casco de la caja de la cancha (12 cm de alto) ∪ franja del brazo
     └─ retroproyeccion ── √(B_HS · B_CrCb), disco de 7 px, solo dentro de la zona
     └─ umbral_piel_otsu ── Otsu manual con el histograma de la zona (T ≥ 30)
     └─ limpiar_mascara ── apertura 5 → cierre 5 → relleno de huecos
     └─ seleccionar_mano ── componente que entra por la franja del brazo
     └─ vista_cenital.rectificar ── máscara en cm
     └─ separar_mano_brazo ── centro y radio de la palma (transformada de distancia), corte a 1.2 r
         (si queda una región gruesa pegada → watershed y se repite)
     └─ analizar_mano ── tramos del contorno a > 1.6 r = dedos; casco y defectos; gesto
     └─ punto de golpe → imagen (H⁻¹) → rayo → ∩ z = 0 (cm); EstimadorVelocidad (cm/s)
```

`ResultadoFrame` guarda cada resultado intermedio. Así la UI y el panel de depuración los muestran sin recalcular,
y `tiempos_ms` registra cuánto tardó cada etapa.

## 4. Concurrencia en el juego

```
hilo principal (≈ 25 FPS)            hilo "profundidad"                 hilo "owl" (tecla O)
─────────────────────────            ──────────────────                 ────────────────────
leer frame                           espera tarea
procesar_frame (≈ 12 ms)
cada 8 frames: enviar(img, máscaras) ─► Depth Anything (~110 ms)
partida.actualizar(dt, entrada)        mano_cerca_de_mesa → resultado
render + HUD                          ◄─ resultado (si edad < 1 s)
imshow / waitKey
```

`TrabajadorAsincrono` acepta una tarea solo si está libre: si el modelo va atrasado se descartan frames en vez de
acumular retraso. Si un modelo falla (sin internet, sin memoria), el juego sigue sin esa información: `cerca = None`
cuenta como golpe válido.

## 5. Física y reglas

- Euler semi-implícito en subpasos de 1/240 s. La gravedad de juego es de 140 cm/s², porque con 981 cm/s² una jugada
  en 25 cm duraría décimas de segundo.
- Rebote en la mesa con restitución de 0.62 y fricción horizontal de 0.85; en la red, `v_x ← −0.25 v_x`.
- Golpe: tiro parabólico con un ápice que depende del gesto (remate: 7.5 cm, toque: 15 cm) y un objetivo que depende
  de la velocidad de la mano. Si la parábola no pasa la red con 1 cm de holgura, se sube el ápice.
- Colisión mano-balón: distancia 3D entre el **rayo** del punto de golpe y el centro del balón, menor que
  `r + 2.6 cm`, y el punto más cercano del rayo por debajo de 8 cm. No depende de suponer la altura de la mano.

## 6. Render AR y oclusión

- Radio aparente del balón: `f · R / z_c`. Sprite sombreado (Lambert + Blinn-Phong) con caché por radio.
- Orden balón/red por profundidad: se compara `z_c` del balón con la de la red a su misma altura.
- Oclusión por la mano: para cada píxel de la mano se calcula la profundidad del plano `z = 1.5 cm` en ese rayo y se
  compara con la del balón (constante) y la de la red (plano `x = 0`). Donde la mano está delante, se devuelven los
  píxeles reales.

## 7. Calibración y resolución

`config/camara.json` guarda `K`, `dist`, el ancho, el alto, el error RMS, el número de vistas y el tamaño del cuadro.
`ModeloCamara.escalar_a` adapta `K` si el frame tiene otra resolución con la **misma** relación de aspecto. Por eso
conviene calibrar con `scripts/calibrar_camara.py`, que captura igual que el juego. En la UI, la cámara del navegador
pide 1280×720 con `WebcamOptions(mirror=False)`; una imagen en espejo invertiría la geometría.

## 8. Rendimiento medido (escenas sintéticas, Apple Silicon)

| Etapa | ms |
|---|---|
| Corrección de distorsión (remap, 960 px) | ~1-6 |
| ArUco + solvePnP | ~1-4 |
| CLAHE + suavizado | ~1-11 |
| Retroproyección + Otsu | ~3-9 |
| Morfología | ~1-2 |
| Segmentación (incluye vista cenital) | ~1-18 |
| Caracterización | ~1-5 |
| **Pipeline completo** | **≈ 10-15** en el juego |
| Depth Anything V2 Small (MPS) | ≈ 110 |
| OWLv2 base (MPS) | ≈ 400-600 |
