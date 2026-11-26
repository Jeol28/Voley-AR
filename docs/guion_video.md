# Guion del video (8-10 minutos)

**Voley AR – Proyecto final de Visión Artificial · Jorge Enrique Olaya Liévano**

Preparación antes de grabar:
- Imprime `cancha.pdf` y `tablero.pdf` al 100 %. Ten a mano el tablero pegado en cartón y una regla.
- Deja abiertos en VS Code los archivos que se van a mostrar (lista en cada bloque).
- Ten la UI corriendo (`python app.py`) y el juego listo para lanzar (`python jugar.py`).
- Graba la pantalla y una toma de la mesa si puedes (con el celular, desde arriba).

---

## 0:00-0:40 · Introducción

**En pantalla:** la cancha impresa sobre la mesa y el juego corriendo unos segundos.

> "Este es Voley AR: un minijuego de voleibol en realidad aumentada sobre la mesa. La cámara reconoce esta cancha
> impresa por sus marcadores ArUco y dibuja encima la red y un balón en 3D. Juego con mi mano, sin ningún objeto,
> contra una IA. La idea central es que la calibración de la cámara es indispensable, y que cada tema del curso
> cumple una función real en el juego."

## 0:40-1:40 · Arquitectura

**En pantalla:** README en GitHub, sección *Arquitectura* (diagrama de capas y del pipeline) y la estructura de carpetas.

> "El código está en capas. `core` tiene un módulo por tema, todos como funciones puras: entra una imagen y salen
> una imagen o datos. `juego` tiene la física, la IA y el render. La UI de Gradio y la ventana del juego son dos
> presentaciones del mismo pipeline: `core/pipeline.py`, función `procesar_frame`."

- Mostrar `voleyar/core/pipeline.py` → `procesar_frame` (recorrer las etapas con el cursor).
- Mencionar la estrategia de ramas: diagrama `gitGraph` del README y `git log --graph --oneline` en la terminal
  (main, develop, una rama `feature/` por tema, merges `--no-ff`, tag `v1.0.0`).

## 1:40-3:00 · Calibración en vivo con el tablero

**En pantalla:** terminal con `python scripts/calibrar_camara.py --cuadro 2.5`.

1. Medir el cuadro del tablero con la regla frente a la cámara (25 mm).
2. Mover el tablero: mostrar el aviso "Tablero detectado", el contador de vistas y la sugerencia ("inclina el
   tablero…", "llévalo a la esquina…"). Activar la captura automática con `A`.
3. Pulsar `C`: mostrar `K`, los coeficientes de distorsión y el **error de reproyección** (< 0.5 px), y la ventana
   original vs. sin distorsión.

> "La calibración de Zhang usa varias vistas del tablero plano para despejar la matriz K y la distorsión. Por eso se
> piden ángulos variados: vistas paralelas no dicen nada sobre la focal."

**Código a mostrar:** `voleyar/core/calibracion.py` → docstring del módulo (pinhole, intrínsecos/extrínsecos,
distorsión), `calibrar_camara`, `GestorCapturaCalibracion.sugerencia`, `estimar_pose_cancha` y `pixel_a_mesa`.

## 3:00-3:50 · Con calibración vs. sin calibración

**En pantalla:** UI → pestaña **Calibración → Cancha: con y sin calibración**, con la cámara en vivo sobre la cancha.

1. Modo *Ambos*: el contorno naranja (calibrado) cae sobre las líneas impresas; el blanco (sin calibrar) se corre,
   sobre todo el borde de la red y el balón, que tienen altura.
2. Señalar la tabla con el error de reproyección de las esquinas ArUco en ambos modos.
3. Subir el slider de altura del balón: la diferencia crece con la altura.

> "Sin calibrar, solvePnP compensa la K equivocada con una pose falsa: lo plano casi encaja, pero lo que tiene
> altura flota. Por eso el juego necesita la K de mi cámara."

## 3:50-6:30 · Recorrido por los temas (código + UI)

Por cada tema: unos 20-25 s en la UI y unos 5-10 s en el código.

| Tiempo | Pestaña de la UI | Qué mostrar | Archivo / función |
|---|---|---|---|
| 3:50 | **Filtros** | Ejemplo *Poca luz*: subir el ruido, comparar gaussiano/mediana/bilateral (tabla de ruido antes/después); Sobel y Canny; escribir un núcleo propio y ver "Diferencia máxima manual vs. filter2D ≈ 1e-6". | `core/filtros.py` → `convolucion_manual` (por qué se voltea el núcleo) |
| 4:15 | **Histogramas** | Mover el recuadro sobre la palma, ver los histogramas H-S y Cr-Cb y la retroproyección; pulsar "Usar este recuadro como modelo de piel". Ecualización vs. CLAHE con sus histogramas. | `core/histogramas.py` → `calibrar_piel`, `retroproyeccion`, `aplicar_clahe` |
| 4:40 | **Umbralización** | Rangos HSV vs. YCrCb (tabla de IoU), curva de varianza entre clases con T*, "Otsu manual = OpenCV: coinciden", adaptativo. | `core/umbralizacion.py` → `otsu_manual` |
| 5:00 | **Morfología** | Ruido sal y pimienta; apertura (componentes: miles → 1); cierre; mosaico de las 5 operaciones; cambiar forma/tamaño del elemento. | `core/morfologia.py` → `limpiar_mascara`, `erosion_manual` |
| 5:20 | **Segmentación** | Ejemplo *Mano tocando un objeto color piel*: zona por la pose, contornos, corte por la muñeca en la vista cenital, transformada de distancia y watershed separando la mano del objeto. | `core/segmentacion.py` → `zona_busqueda`, `separar_mano_brazo`, `watershed_mano` |
| 5:45 | **Caracterización** | Ejemplos *Un dedo*, *Dos dedos*, *Puño*: casco convexo, valles, puntas, tabla de descriptores (cm², Hu) y gesto → golpe. Con la cámara en vivo, mostrar la velocidad. | `core/caracterizacion.py` → `analizar_mano`, `descriptores_forma` |
| 6:10 | **Pipeline** | Los 9 pasos sobre el mismo frame y la tabla de tiempos (~10-15 ms). | `core/pipeline.py` |

## 6:30-7:30 · Transformers: clásico vs. OWLv2 y profundidad

**En pantalla:** UI → pestaña **Transformers**, con la cámara en vivo.

1. Depth Anything: mapa de profundidad y puntaje de altura. Apoyar la mano (verde, "golpe válido") y levantarla
   (rojo, "mano levantada").
2. Comparación sobre el mismo frame: caja clásica (naranja) vs. OWLv2 (verde), tabla de latencias (~12 ms vs.
   ~500 ms), IoU y diferencia en cm.

> "Uso los transformers solo en inferencia, con pesos públicos. Depth Anything es lento por frame, así que en el juego
> corre en un hilo aparte cada 8 frames: la altura de la mano cambia más despacio que su posición. OWLv2 no necesita
> calibrar la piel, pero es unas 40 veces más lento; por eso el detector principal es el clásico."

**Código a mostrar:** `core/transformers_models.py` → `mano_cerca_de_mesa` (la fórmula del puntaje) y
`DetectorManoOWL.detectar`; `juego/asincrono.py` → `TrabajadorAsincrono`.

## 7:30-9:30 · Partida con el panel de depuración

**En pantalla:** `python jugar.py` (o el botón de la pestaña **Jugar**).

1. Calibración de la piel: cubrir el recuadro naranja y pulsar `ESPACIO`.
2. Activar `D`: señalar la máscara de piel (brazo en naranja), la vista cenital con los dedos, el mapa de
   profundidad, la pose (ejes), los FPS y los tiempos.
3. Jugar un par de puntos: un **remate** con la mano abierta y un **toque** con un dedo; mostrar el anillo verde del
   golpe válido y uno rojo levantando la mano.
4. Pulsar `C` durante la partida: sin calibración la red y el balón se desalinean. Volver a pulsar `C`.
5. Pulsar `O` unos segundos para jugar con OWLv2 y notar la latencia.

**Código a mostrar (breve):** `juego/partida.py` → `_intentar_golpe` (rayo vs. balón y profundidad),
`juego/render_ar.py` → `ocluir_con_mano`.

## 9:30-10:00 · Cierre

**En pantalla:** README → *Limitaciones* y *Pruebas* (`pytest` en verde en la terminal).

> "Las pruebas usan escenas sintéticas de verdad conocida: la calibración recupera K, solvePnP la pose, Otsu y la
> convolución coinciden con OpenCV, los descriptores con scikit-image, y los dedos se cuentan bien. Las limitaciones
> principales son la luz, las mesas de madera y que la mano no tape todos los marcadores. Gracias."
