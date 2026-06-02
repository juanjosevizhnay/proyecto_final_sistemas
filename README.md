# Restauración de Imágenes de Placas Vehiculares

**Proyecto Final — Sistemas Lineales y Señales**

Sistema que modela la degradación de imágenes como convolución y aplica deconvolución para restaurar placas vehiculares borrosas, con OCR para medir la mejora en legibilidad.

---

## Modelo Matemático

```
g(x, y) = f(x, y) ∗ h(x, y) + n(x, y)
```

| Símbolo | Significado |
|---------|-------------|
| `f(x, y)` | Imagen original |
| `h(x, y)` | PSF del sistema (kernel de degradación) |
| `n(x, y)` | Ruido aditivo |
| `g(x, y)` | Imagen degradada |
| `∗` | Convolución 2D |

En frecuencia: `G = F · H + N` → restaurar implica estimar `F` a partir de `G` y `H`.

---

## Núcleo del Código — Convolución Manual

La convolución está implementada **a mano con bucles explícitos** en `src/convolution.py`:

```python
# Voltear kernel (definición matemática de convolución, no correlación)
kernel_flipped = kernel[::-1, ::-1]

for m in range(M):
    for n in range(N):
        region = padded[m:m + kH, n:n + kW]
        output[m, n] = np.sum(region * kernel_flipped)
```

Esto implementa directamente: `g[m,n] = Σ_i Σ_j f[m-i, n-j] · h[i,j]`

---

## Estructura

```
proyecto_sistemas/
├── app.py              # Interfaz gráfica (CustomTkinter)
├── main.py             # Pipeline de consola
├── requirements.txt
├── images/             # Coloca aquí tus imágenes de placas
├── output/             # Resultados generados automáticamente
└── src/
    ├── convolution.py  # ← NÚCLEO: convolución 2D manual con loops
    ├── kernels.py      # Kernels: Gaussiano, motion blur, Laplaciano, Sobel
    ├── degradation.py  # Modelo g = f*h + n
    ├── restoration.py  # Filtro inverso, Wiener, Richardson-Lucy
    ├── postprocessing.py
    ├── ocr.py          # PaddleOCR via ONNX Runtime
    ├── metrics.py      # MSE, PSNR, SSIM manuales
    ├── video.py        # Pipeline de video: detección + restauración por frames
    ├── autotune.py     # Auto-ajuste de K/kernel + votación (mejoras de accuracy)
    └── visualization.py
```

---

## Instalación

**Requisitos previos:** [Python 3.8+](https://www.python.org/downloads/) instalado en el sistema.

```bash
# Clonar o descargar el proyecto, luego instalar dependencias:
pip install -r requirements.txt
```

> El OCR usa `rapidocr-onnxruntime` (PaddleOCR sobre ONNX). Los modelos (~15 MB) se descargan automáticamente la primera vez. No requiere instalar PaddlePaddle.

---

## Ejecución

### Interfaz gráfica (recomendado)

```bash
python app.py
```

- Carga una imagen desde cualquier carpeta
- Ajusta parámetros con sliders (blur, ruido, Wiener K, iteraciones R-L)
- Muestra resultados en pestañas: Resultados, Métodos, Métricas, OCR, Espectro
- Botón **Exportar** guarda todas las gráficas en `output/`

### Pipeline de consola

```bash
python main.py
```

Selecciona una imagen de la carpeta `images/` y ejecuta el pipeline completo imprimiendo logs y métricas en terminal.

---

## Modo Video

Además de imágenes, el sistema procesa videos completos para detectar y restaurar placas vehiculares a través de los frames.

### Uso en la GUI

1. Ejecuta `python app.py`
2. En la sidebar, sección **VIDEO**, presiona **Cargar Video** (formatos `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
3. Ajusta el slider **Frame step** (1 = analizar todos los frames, 3 = uno de cada tres)
4. Configura el kernel y `K` de Wiener como en imagen
5. Presiona **PROCESAR VIDEO**
6. Los resultados aparecen en la pestaña **Video** con un grid de placas (recorte original vs restaurado + OCR antes/después)

### Pipeline interno (`src/video.py`)

1. **Detección por frame** — `detect_text_regions()` corre OCR sobre cada frame seleccionado, filtrando con `looks_like_plate()` (longitud alfanumérica + presencia de dígitos)
2. **Agregación** — `aggregate_detections()` agrupa la misma placa que aparece en N frames usando `difflib.SequenceMatcher` sobre los textos normalizados
3. **Restauración** — solo del frame con mayor confianza por grupo, aplicando **el mismo** `wiener_filter` + `postprocess_pipeline` del modo imagen
4. **Re-OCR** — sobre la versión restaurada para medir la mejora

### Archivos de salida (`output/video_<nombre>/`)

| Archivo | Contenido |
|---------|-----------|
| `summary.txt` | Reporte legible: fps, duración, placas únicas, OCR antes/después |
| `plates.json` | Datos estructurados de todas las placas detectadas |
| `plate_NN_orig.png` | Recorte original de cada placa |
| `plate_NN_restored.png` | Recorte tras Wiener + post-proceso |
| `plate_NN_comparison.png` | Original y restaurada lado a lado |

### Mejoras de eficacia (accuracy)

El modo video incluye tres técnicas para subir el acierto cuando no se conoce el blur real (caso típico de cámaras de tránsito). Se activan con el checkbox **Auto-ajuste** de la sidebar (`src/autotune.py`):

| Mejora | Qué hace |
|--------|----------|
| **(a) Barrido de K de Wiener** | Prueba varios valores de `K` (regularización) y se queda con el de mejor lectura, en vez de fijarlo a mano |
| **(b) Búsqueda de kernel** | Genera varias PSF candidatas (Gaussiano + motion blur en distintos ángulos/longitudes) y elige la que mejor restaura — deconvolución "ciega" simplificada |
| **(d) Votación por carácter** | Combina las lecturas OCR de la **misma placa en todos los frames** votando carácter por carácter, corrigiendo errores puntuales que no se repiten |

El "mejor" resultado se mide con la **confianza del OCR** (caso real, sin imagen de referencia) o con **SSIM** cuando sí hay referencia (`auto_restore(..., reference=original)`).

### Notas

- El modo video reutiliza exactamente la misma matemática del proyecto (`g = f*h + n`), aplicada ahora a frames secuenciales
- Solo se restaura **una vez por placa única**, no por cada frame en que aparece — mantiene el tiempo de procesamiento manejable
- Con **Auto-ajuste** activo el procesamiento es más lento (prueba varias combinaciones por placa); desactívalo para usar el kernel/K fijos de la sidebar
- Si OCR detecta 0 placas, prueba bajar `Frame step` a 1-2 o usar un video de mayor resolución

---

## Métodos de Restauración

| Método | Fórmula | Uso |
|--------|---------|-----|
| Filtro Inverso | `F̂ = G / H` | Didáctico — falla con ruido |
| **Filtro de Wiener** | `F̂ = [H* / (\|H\|² + K)] · G` | **Principal** |
| Richardson-Lucy | `f_{k+1} = f_k · [(g/(f_k∗h)) ∗ h^T]` | Iterativo, ML |

---

## Referencias

- Gonzalez & Woods — *Digital Image Processing* (4th ed.)
- Oppenheim & Willsky — *Signals and Systems* (2nd ed.)
- Wang et al. (2004) — *Image Quality Assessment: SSIM*
- Richardson (1972), Lucy (1974) — Método iterativo de restauración
