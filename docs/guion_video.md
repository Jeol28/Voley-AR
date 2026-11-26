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
| 2:40 | 1 · Calibración | Distancia de la mano y su velocidad: acercarla rápido a la cámara da "¡Remate!". | Apagar la calibración: Z y su velocidad cambian y el remate sale de más o de menos. | `core/calibracion.py` → `distancia_por_tamano`; `juego/bucle.py` → `_velocidad_z` |
| 3:15 | 2 · Filtros | Los puntitos de la mano y sus flechas; mover la mano lento ("lento") y muy rápido ("¡rápido: más fuerza!") y ver que el balón sale más lejos. | "Fuerza por velocidad": apagada vs. turbo. | `core/movimiento.py` → `rapidez_por_puntos`; `juego/bucle.py` → `rapidez` |
| 3:45 | 3 · Histogramas | El histograma de tonos del cuarto; poner algo grande del color del balón delante de la cámara y ver que el balón cambia de color. | Fijar el balón en un color parecido al cuarto: se pierde de vista. | `core/histogramas.py` → `histograma_tono`, `color_de_contraste` |
| 4:15 | 4 · Umbralización | |frame − fondo| en colores, la curva de Otsu y el T que se usa. | Subir el mínimo hasta que un objeto oscuro deje de golpear. | `core/umbralizacion.py` → `otsu_manual` |
| 4:45 | 5 · Morfología | La zona de golpe (naranja) alrededor de la mano: es una dilatación; en rojo, el ruido que borra la apertura. | Zona "gigante": pasar la mano cerca del balón, sin tocarlo, y verlo volver. | `core/morfologia.py` → `apertura`, `cierre`; `cv2.dilate` en `core/movimiento.py` |
| 5:20 | 6 · Segmentación | Cada objeto en movimiento; mover la cabeza: sale en rojo y no golpea. | "La cara golpea: sí" y cabecear el balón. | `core/movimiento.py`; `core/segmentacion.py` → `detectar_caras` |
| 6:00 | 7 · Caracterización | Puño = golpe fuerte, mano abierta = toque suave, un cuaderno = bloqueo; el nombre aparece en el golpe. | Cambiar k del umbral de los dedos. | `core/caracterizacion.py` → `analizar_mano`, `tipo_de_golpe` |
| 6:40 | 8 · Transformers | Mapa de profundidad; alguien pasa por detrás y su contorno sale en rojo "lejos". | Cercanía 0: ahora el que pasa por detrás devuelve el balón. | `core/movimiento.py` → `cercania`; `core/transformers_models.py` |

> "Nada de esto es una demo aparte: cada tema decide algo de la partida, y esto es exactamente lo que el juego está
> calculando en ese frame."

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
