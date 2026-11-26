# Arquitectura de Voley AR (v2, modo pantalla)

Este documento amplía la sección de arquitectura del [README](../README.md): sistemas de coordenadas, flujo de
datos, concurrencia y las decisiones que se tomaron a partir de las pruebas.

## 1. Capas

| Capa | Paquete | Regla |
|---|---|---|
| Presentación | `voleyar/ui/juego_web.py` (el juego en la web: inicio, ajustes, calibración, partida), `voleyar/juego/bucle.py` + `menus.py` (ventana con menús), `voleyar/juego/vistas.py` (un panel en vivo por tema del curso), `voleyar/juego/navegador.py`, `app.py`, `jugar.py` | Maneja eventos, menús, vistas y widgets. No implementa visión: las vistas solo muestran lo que el pipeline ya calculó. |
| Dominio del juego | `voleyar/juego` (`fisica`, `ia_rival`, `partida`, `render_ar`) | Reglas y física en unidades de la cancha. `partida.py` no conoce la cámara. |
| Núcleo de visión | `voleyar/core` | Funciones puras: imagen in → imagen/datos out. Sin Gradio ni `imshow`. |
| Infraestructura | `voleyar/utils`, `voleyar/entorno.py` | Cámara, JSON, texto con tildes, escenas sintéticas, detección de ZeroGPU. |

La dependencia siempre va hacia abajo. La UI, la ventana y el juego del navegador llaman a la **misma** función,
`core.pipeline.procesar_frame`.

## 2. Sistemas de coordenadas

- **Cámara real** (cm): `x` a la derecha, `y` hacia abajo, `z` hacia adelante. Las manos de los jugadores están a
  `Z ≈ 50-75 cm`.
- **Imagen** (px): `s [u v 1]ᵀ = K P_c`, con distorsión radial y tangencial. El pipeline trabaja sobre la imagen ya
  corregida (`K'`, `d = 0`).
- **Cancha virtual** (unidades de la cancha de la v1, 25 × 18 con red de 5): `x` de la mitad del jugador 1 (`x < 0`) a
  la del jugador 2 (`x > 0`), `y` = profundidad (el balón siempre en `y = 0`, el plano de la red) y `z` hacia arriba
  (piso en `z = 0`).
- **Cámara virtual:** con la **misma** `K` mira horizontalmente la red desde `y = −D`
  (`calibracion.camara_virtual`):
  - `D = f_x · 2L / (0.92 · W)`, para que la cancha ocupe el 92 % del ancho;
  - altura `h = (0.86·H − c_y) · D / f_y`, para que el piso quede en la fila del 86 %;
  - desplazamiento lateral `−(W/2 − c_x)·D/f_x`, para que la red quede en la columna central.

  Así, un píxel de la mano y el balón virtual comparten geometría: el rayo de cada píxel se intersecta con el plano
  `y = 0`.

## 3. Flujo de un frame

```
frame (1280×720)
 └─ preparar_imagen ── reducción a 960 px; si hay calibración: remap(K, d) → imagen rectificada, K', d = 0
 └─ camara_virtual(K') ── pose de la cámara que encuadra la cancha
 └─ detectar_caras (cada 3 frames) ── Viola-Jones a 640 px; una cara por mitad (≥ 20 % de la altura);
     recordar_caras: si se pierde (la mano la tapa) se usa la última vista durante 90 frames
 └─ preprocesar ── CLAHE en L (Lab) + suavizado gaussiano 5×5
 └─ por cada mitad (jugador 1 = izquierda, jugador 2 = derecha):
     ├─ retroproyeccion con la piel de ESE jugador (√(B_HS · B_CrCb), disco de 7 px)
     ├─ umbral_piel_otsu (Otsu manual con el histograma de la mitad, T ≥ 30)
     ├─ limpiar_mascara (apertura 5 → cierre 5 → relleno de huecos)
     ├─ seleccionar_mano_pantalla: se descartan las regiones casi todo-cara; se prefiere la que toca el borde
     │   inferior (antebrazo)
     ├─ separar_de_la_cara (si toca una cara): watershed con marcadores cara / picos de la mano
     ├─ separar_mano_brazo (transformada de distancia, corte a 1.2 r); watershed si queda una región gruesa
     ├─ analizar_mano: tramos del contorno a > 1.6 r = dedos; casco y defectos; gesto y punto de golpe
     ├─ distancia_por_tamano: Z = f · 3.5 cm / r_px
     └─ pixel_a_plano(punto de golpe, cámara virtual) → (x, 0, z); EstimadorVelocidad → (vx, vz)
```

`ResultadoFrame` guarda los resultados comunes (imagen, modelo, cámara virtual, caras, tiempos) y una `ResultadoMano`
por lado con cada paso intermedio.

## 4. Del pipeline a la partida

Por cada mano se arma una `EntradaJugador`:

- **rayo:** el rayo de la cámara virtual que pasa por el punto de golpe. La colisión es la distancia 3D entre ese rayo
  y el centro del balón, menor que `r + 2.6`.
- **punto y velocidad:** el punto en el plano de la red y su velocidad.
- **fuerza y gesto:** salen del número de dedos.
- **valida:** la mano está en el plano de juego. `|Z − Z_ref| ≤ max(10 cm, 25 % · Z_ref)`, donde `Z_ref` es la
  mediana de las primeras 15 mediciones de ese jugador. Además, si Depth Anything tiene un resultado reciente, la mano
  debe estar delante de su cara.

`Partida` (modo `2j` o `ia`) solo acepta golpes de cada jugador en su mitad, con un toque por lado. En el saque solo
golpea quien saca.

## 5. Concurrencia (escritorio)

```
hilo principal (≈ 20-30 FPS)          hilo "profundidad"                 hilo "owl" (tecla O)
────────────────────────────          ──────────────────                 ────────────────────
leer frame
procesar_frame, 2 manos (≈ 22 ms)
cada 8 frames: enviar(imagen) ───────► Depth Anything (~110 ms)
partida.actualizar(dt, J1, J2)         mapa de disparidad
render + HUD                           ◄─ resultado (si edad < 1 s)
```

`TrabajadorAsincrono` acepta una tarea solo si está libre, así que nunca se acumula retraso.

En el navegador (y en el Space) la cámara llega como **video por WebRTC** (fastrtc):

- **El video:** `VideoStreamHandler(frame_juego, fps=30, skip_frames=True)` recibe cada frame en BGR y devuelve el
  frame del juego, que vuelve por el mismo canal. Con `skip_frames` se procesa siempre el frame más reciente.
- **El juego compartido:** el manejador de video y los botones de la página comparten el `Juego` de cada visitante a
  través de `ui/sesiones.py`. Es un registro con clave aleatoria por pestaña, con un candado por sesión porque el
  video y los botones corren en hilos distintos.
- **Los modelos:** no hay hilos de modelos en la web; la vista de Transformers llama a los modelos como mucho cada
  2 s.

## 6. Física y reglas

- Euler semi-implícito en subpasos de 1/240 s. La gravedad de juego es de 85 unidades/s², más baja que en la v1 para
  compensar la latencia de la cámara.
- Rebote en el piso con restitución de 0.6 y fricción de 0.85; en la red, `v_x ← −0.25 v_x`.
- Golpe: tiro parabólico con un ápice que depende del gesto (remate 7.5, toque 11.5) y un objetivo que depende de la
  velocidad de la mano. El ápice se sube si la parábola no pasa la red.

## 7. Render AR

- **Piso:** el plano `z = 0` en perspectiva, con `|y| ≤ 3`, una mitad por jugador y líneas blancas.
- **Red:** el plano `x = 0` se ve de canto desde la cámara virtual, así que se dibuja una franja de malla frontal
  (`|x| ≤ 0.6`) con postes y banda superior.
- **Balón:** sprite sombreado (Lambert + Blinn-Phong) de radio `f · R / z_c`, con sombra en el piso que se difumina con
  la altura.

## 8. Rendimiento medido (escenas sintéticas, Apple Silicon)

| Etapa | ms |
|---|---|
| Corrección de distorsión (remap, 960 px) | ~1-6 |
| Caras (Haar a 640 px, cada 3 frames) | ~5 |
| CLAHE + suavizado | ~1-11 |
| Una mano (piel → dedos → distancia) | ~9-10 |
| **Pipeline completo, 2 manos** | **≈ 20-25** |
| Depth Anything V2 Small (MPS) | ≈ 110 |
| OWLv2 base (MPS) | ≈ 400-600 |

## 9. Precisión medida (escenas sintéticas)

| Medida | Resultado |
|---|---|
| Distancia de la mano, con calibración | error de -4 % a +4 % (60-79 cm) |
| Distancia de la mano, sin calibración | ~+30 % (77 cm para una mano a 60) |
| Conteo de dedos, ejemplos de la UI | 12/13 (falla con la mano sobre la cara) |
| Conteo de dedos, barrido 0-5 dedos a 60-79 cm | 19/24 (4 de los 5 fallos a 79 cm) |
| Mano tocando la cara | Z = 64 cm (real 60) con watershed |

## 10. Vistas por tema dentro de la partida

`voleyar/juego/vistas.py` define una vista por tema del curso:

- **Contenido:** cada vista recibe el `ResultadoFrame` del frame actual y devuelve una imagen (los pasos intermedios
  de ese tema) y unas líneas con sus números reales.
- **Ajuste:** cada vista tiene un `Parametro` (por ejemplo `config.tam_cierre`) que `[`/`]` o −/+ recorren. Como el
  pipeline lee `config` en cada frame, el cambio se ve en el frame siguiente.
- **Costo:** las vistas no recalculan la visión, salvo Otsu (256 bins, para dibujar su curva) y Canny en la vista de
  filtros.
- **Transformers:** su vista dispara Depth Anything y OWLv2. En escritorio van en hilos; en la web, una llamada cada
  2 s y solo mientras se mira esa vista.
