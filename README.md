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
