# Guion del video (8-10 minutos)

**Voley AR – Proyecto final de Visión Artificial · Jorge Enrique Olaya Liévano**

Todo se muestra desde el juego: los temas del curso se ven en vivo con **¿Qué quieres ver?** (o `V` en la ventana) en
plena partida.

Preparación antes de grabar:
- Imprime `tablero.pdf` al 100 % y pégalo en cartón. Ten una regla a mano.
- Consigue un segundo jugador (o usa el modo contra la IA).
- Buena luz blanca y mangas largas oscuras.
- Deja abiertos en VS Code los archivos que se van a mostrar (lista en cada bloque).
- Ten el juego listo: `python jugar.py` (ventana) y `python app.py` (navegador).

---

## 0:00-0:40 · Introducción

**En pantalla:** el menú de inicio, "2 jugadores" y unos segundos de partida.

> "Este es Voley AR: voleibol en realidad aumentada para dos jugadores frente a una sola cámara web. No hay nada
> impreso ni ningún objeto: la cancha, la red y el balón se dibujan en la pantalla y lo único real son nuestras manos.
> Y dentro del mismo juego se puede ver, tema por tema, cómo usa la visión artificial."

## 0:40-1:30 · Arquitectura

**En pantalla:** README → *Arquitectura* (diagrama de capas y del pipeline) y la estructura de carpetas.

> "`core` tiene un módulo por tema, todos como funciones puras. `juego` tiene la física, las reglas, el render, los
> menús y las vistas por tema. La ventana y la página web son dos presentaciones del mismo pipeline:
> `core/pipeline.py`, que procesa una mano por cada mitad de la imagen."

- Mostrar `voleyar/core/pipeline.py` → `procesar_frame` y `_procesar_mano`.
- Mostrar el `gitGraph` y `git log --graph --oneline` (ramas `feature/`, merges `--no-ff`, tags de v1.0.0 a v2.2.0).

## 1:30-2:40 · Calibración de la cámara

**En pantalla:** navegador → **Calibrar cámara** (o `python scripts/calibrar_camara.py`).

1. Medir el cuadro del tablero con la regla.
2. Mover el tablero: la barra se llena y aparecen sugerencias de ángulo.
3. **Calibrar** → "Calibración excelente" y, en *Detalles técnicos*, `K`, la distorsión y el error por vista.

**Código a mostrar:** `voleyar/core/calibracion.py` → docstring (pinhole, intrínsecos/extrínsecos, distorsión),
`calibrar_camara`, `camara_virtual`, `distancia_por_tamano`.

## 2:40-7:30 · Los temas, en vivo, en plena partida

**En pantalla:** una partida de dos jugadores. Ir pasando "¿Qué quieres ver?" (o `V`) y en cada tema pulsar **−/+**
(o `[`/`]`) para mostrar el efecto en vivo.

| Tiempo | Vista | Qué mostrar y decir | Ajuste a probar | Archivo / función |
|---|---|---|---|---|
| 2:40 | 1 · Calibración | Rectas curvadas por la lente vs. imagen corregida; f y k1; "Z = 58 cm con calibración (75 sin ella)". | Apagar la calibración: la distancia salta y los golpes empiezan a salir inválidos. | `core/calibracion.py` → `corregir_distorsion`, `distancia_por_tamano` |
| 3:15 | 2 · Filtros | Zona de las manos antes y después del suavizado, bordes de Canny; el ruido baja. | Cambiar a mediana o bilateral, y a "ninguno" para ver el ruido. | `core/filtros.py` → `suavizar`, `canny`, `convolucion_manual` |
| 3:45 | 3 · Histogramas | El histograma Cr-Cb de la piel de cada jugador (son distintos) y la retroproyección; contraste con CLAHE. | Subir el clip de CLAHE. | `core/histogramas.py` → `calibrar_piel`, `retroproyeccion` |
| 4:15 | 4 · Umbralización | La curva de Otsu de cada mitad con su T; la máscara binaria. | Subir el umbral mínimo hasta que la mano se rompa. | `core/umbralizacion.py` → `otsu_manual` |
| 4:45 | 5 · Morfología | Máscara antes/después: regiones de decenas a pocas. | Cierre de 9 px: los dedos se pegan (y el conteo falla). | `core/morfologia.py` → `limpiar_mascara` |
| 5:20 | 6 · Segmentación | Mano blanca, brazo naranja, cara roja (Viola-Jones); pasar la mano por delante de la cara y ver el watershed. | Corte por distancia vs. por ancho. | `core/segmentacion.py` → `detectar_caras`, `separar_de_la_cara`, `separar_mano_brazo` |
| 6:00 | 7 · Caracterización | Casco convexo, valles y puntas; cambiar de mano abierta (remate) a un dedo (toque). | Cambiar k del umbral de los dedos. | `core/caracterizacion.py` → `analizar_mano` |
| 6:40 | 8 · Transformers | Mapa de profundidad (la mano más clara que la cara) y cajas de OWLv2; latencias (~25 ms el clásico vs. ~500 ms OWLv2). | Jugar unos segundos con OWLv2 para notar la latencia. | `core/transformers_models.py` → `mano_delante_de_la_cara`, `detectar_manos` |

> "Nada de esto es una demo aparte: es exactamente lo que el juego está calculando en ese frame para decidir si golpeé
> el balón."

## 7:30-9:30 · Partida completa

**En pantalla:** volver a "Solo el juego" y jugar un par de puntos.

1. Un **remate** con la mano abierta, un **toque** con un dedo.
2. Acercar mucho la mano a la cámara: golpe inválido (punto rojo, "mano fuera del plano").
3. `ESC` → menú de pausa; terminar la partida → **Revancha**.
4. 20 s del juego en el Space (navegador).

**Código a mostrar (breve):** `juego/partida.py` → `_intentar_golpe`, `juego/bucle.py` → `_validar` y
`juego/vistas.py` → `PARAMETROS`.

## 9:30-10:00 · Cierre

**En pantalla:** README → *Limitaciones* y `pytest` en verde.

> "Las pruebas usan escenas sintéticas de verdad conocida: la calibración recupera K, la distancia de la mano sale
> con ±4 % de error calibrada y ~30 % sin calibrar, y cada vista del juego se prueba con su ajuste. Las limitaciones
> principales son la luz, la ropa de color piel y que la cascada Haar detecta peor las caras de piel oscura. Gracias."
