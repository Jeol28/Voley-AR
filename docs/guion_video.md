# Guion del video (8-10 minutos)

**Voley AR – Proyecto final de Visión Artificial · Jorge Enrique Olaya Liévano**

Preparación antes de grabar:
- Imprime `tablero.pdf` al 100 % y pégalo en cartón. Ten una regla a mano.
- Consigue un segundo jugador para la partida de dos (o usa el modo contra la IA).
- Buena luz blanca y mangas largas oscuras.
- Deja abiertos en VS Code los archivos que se van a mostrar (lista en cada bloque).
- Ten la UI corriendo (`python app.py`) y el juego listo (`python jugar.py --modo 2j`).

---

## 0:00-0:40 · Introducción

**En pantalla:** unos segundos de partida de dos jugadores, cada uno en su mitad.

> "Este es Voley AR: voleibol en realidad aumentada para dos jugadores frente a una sola cámara web. No hay nada
> impreso ni ningún objeto: la cancha, la red y el balón se dibujan en la pantalla y lo único real son nuestras manos.
> Cada tema del curso cumple una función en el juego, y la calibración de la cámara es la que une lo real con lo
> virtual."

## 0:40-1:40 · Arquitectura

**En pantalla:** README → *Arquitectura* (diagrama de capas y del pipeline) y la estructura de carpetas.

> "El código está en capas. `core` tiene un módulo por tema, todos como funciones puras. `juego` tiene la física, las
> reglas y el render. La UI de Gradio, la ventana de OpenCV y el juego en el navegador son tres presentaciones del mismo
> pipeline: `core/pipeline.py`, función `procesar_frame`, que procesa una mano por cada mitad de la imagen."

- Mostrar `voleyar/core/pipeline.py` → `procesar_frame` y `_procesar_mano`.
- Mostrar la estrategia de ramas: diagrama `gitGraph` y `git log --graph --oneline` (ramas `feature/` por tema, merges
  `--no-ff`, tags v1.0.0, v1.1.0 y v2.0.0, y la rama `feature/multijugador`).

## 1:40-3:00 · Calibración en vivo con el tablero

**En pantalla:** `python scripts/calibrar_camara.py --cuadro 2.5`.

1. Medir el cuadro con la regla frente a la cámara.
2. Mover el tablero: el aviso "Tablero detectado", el contador de vistas y la sugerencia. Activar la captura
   automática con `A`.
3. Pulsar `C`: mostrar `K`, la distorsión, el **error de reproyección** (< 0.5 px) y la ventana sin distorsión.

**Código a mostrar:** `voleyar/core/calibracion.py` → docstring (pinhole, intrínsecos/extrínsecos, distorsión),
`calibrar_camara`, `camara_virtual` y `distancia_por_tamano`.

## 3:00-3:50 · Con calibración vs. sin calibración

**En pantalla:** UI → **Calibración → Distancia de la mano: con y sin calibración**, con la cámara en vivo y la mano
abierta.

> "Sin cancha impresa, ¿para qué la calibración? Para tres cosas: corregir la distorsión, construir la cámara virtual
> con la que se dibuja la cancha y, sobre todo, medir a qué distancia está cada mano: Z = f por el tamaño de la palma
> dividido por su tamaño en píxeles. Con la K calibrada la mano está a unos 60 cm; con la K aproximada el juego cree
> que está a más de 75. Por eso un golpe solo vale si la mano está en su plano de juego."

- Mostrar la escena de ejemplo (56/61 cm frente a 77/83 cm, con reales de 60/64) y luego tu mano en vivo.
- Mencionar la subpestaña **Pose con ArUco (demo de la v1)**: `solvePnP` sigue en el código.

## 3:50-6:30 · Recorrido por los temas (código + UI)

| Tiempo | Pestaña | Qué mostrar | Archivo / función |
|---|---|---|---|
| 3:50 | **Filtros** | Ejemplo *Poca luz*: ruido, gaussiano/mediana/bilateral, Sobel y Canny; un núcleo propio con "Diferencia máxima ≈ 1e-6". | `core/filtros.py` → `convolucion_manual` |
| 4:15 | **Histogramas** | Recuadro sobre la palma del jugador 1, histogramas H-S y Cr-Cb, retroproyección; "¿De quién es esta piel?" → Jugador 1, luego Jugador 2. CLAHE. | `core/histogramas.py` → `calibrar_piel`, `retroproyeccion` |
| 4:40 | **Umbralización** | Mitad de cada jugador: rangos HSV vs. YCrCb, curva de Otsu con T*, "coinciden", adaptativo y tabla de IoU. | `core/umbralizacion.py` → `otsu_manual` |
| 5:00 | **Morfología** | Ruido sal y pimienta → apertura (miles de componentes → pocas), cierre, las 5 operaciones. | `core/morfologia.py` → `limpiar_mascara` |
| 5:20 | **Segmentación** | Ejemplo *Mano delante de la cara*: mitades, cara de Haar, contorno elegido, corte por la muñeca y watershed separando la mano de la cara. | `core/segmentacion.py` → `detectar_caras`, `separar_de_la_cara`, `separar_mano_brazo` |
| 5:45 | **Caracterización** | Jugador 1 y 2: casco convexo, valles, puntas, descriptores (cm² con Z/f) y gesto → golpe. | `core/caracterizacion.py` → `analizar_mano` |
| 6:10 | **Pipeline** | Los 9 pasos sobre el mismo frame y la tabla de tiempos. | `core/pipeline.py` |

## 6:30-7:30 · Transformers: clásico vs. OWLv2 y profundidad

**En pantalla:** UI → **Transformers**, con la cámara en vivo y los dos jugadores.

1. Depth Anything: mapa de profundidad y, por jugador, "mano delante del cuerpo: golpe válido".
2. Comparación por mitad: cajas clásicas (naranja) vs. OWLv2 (verde), latencias (~25 ms vs. ~500 ms), IoU y
   diferencia en cm.

**Código a mostrar:** `core/transformers_models.py` → `mano_delante_de_la_cara` y `estimar_profundidad` (decorador
`gpu` de `voleyar/entorno.py` para ZeroGPU).

## 7:30-9:30 · Partida de dos jugadores con el panel de depuración

**En pantalla:** `python jugar.py --modo 2j`.

1. Calibración de la piel: cada jugador cubre su recuadro naranja y se pulsa `ESPACIO`.
2. Señalar arriba de cada mitad "midiendo…" → "plano a 60 cm" (la distancia de referencia de cada jugador).
3. Activar `D`: máscaras (mano blanca, brazo naranja, caras rojas), retroproyección de cada jugador, profundidad,
   distancias y tiempos.
4. Jugar un par de puntos: un **remate** con la mano abierta, un **toque** con un dedo. Mostrar un golpe inválido
   acercando mucho la mano a la cámara (punto rojo).
5. Pulsar `C`: sin calibración la distancia medida cambia y los golpes empiezan a salir inválidos. Volver a `C`.
6. Mostrar 20 s del juego en el Space (pestaña **Jugar → En el navegador**).

**Código a mostrar (breve):** `juego/partida.py` → `_intentar_golpe` y `juego/bucle.py` → `_validar`.

## 9:30-10:00 · Cierre

**En pantalla:** README → *Limitaciones* y `pytest` en verde.

> "Las pruebas usan escenas sintéticas de verdad conocida: dos personas a distancias conocidas, caras detectables y
> cámaras con distorsión. La calibración recupera K, la distancia de la mano sale con ±4 % de error calibrada y con
> ~30 % sin calibrar. Las limitaciones principales son la luz, la ropa de color piel y que Haar detecta peor las caras
> de piel oscura. Gracias."
