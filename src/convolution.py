"""
Modulo de Convolucion 2D Manual
================================
Implementacion explicita de la operacion de convolucion 2D con bucles,
demostrando la definicion matematica de sistemas lineales:

    g[m, n] = SUM_i SUM_j { f[m - i, n - j] * h[i, j] }

donde:
    f : senal de entrada (imagen)
    h : respuesta al impulso (kernel)
    g : senal de salida (imagen convolucionada)

La convolucion implica voltear (flip) el kernel en ambos ejes antes
de deslizarlo sobre la imagen. Esto la diferencia de la correlacion
cruzada, que no voltea el kernel.
"""

import numpy as np


def _pad_image(image: np.ndarray, pad_h: int, pad_w: int,
               mode: str = 'zero') -> np.ndarray:
    """
    Aplica padding a la imagen segun el modo indicado.

    Parametros
    ----------
    image : ndarray (M, N)
    pad_h : pixeles de padding vertical (arriba y abajo)
    pad_w : pixeles de padding horizontal (izquierda y derecha)
    mode  : 'zero' | 'reflect' | 'replicate'
    """
    if mode == 'zero':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='constant', constant_values=0)
    elif mode == 'reflect':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='reflect')
    elif mode == 'replicate':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                       mode='edge')
    else:
        raise ValueError(f"Modo de padding no soportado: {mode}")


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray,
                      padding: str = 'zero') -> np.ndarray:
    """
    Convolucion 2D implementada con bucles explicitos.

    Aplica la definicion matematica estricta:
        g[m, n] = SUM_i SUM_j { f[m - i, n - j] * h[i, j] }

    lo cual equivale a voltear h y luego correlacionar.

    Parametros
    ----------
    image   : ndarray (M, N), imagen en escala de grises, rango [0, 1]
    kernel  : ndarray (kH, kW), kernel / respuesta al impulso
    padding : 'zero' | 'reflect' | 'replicate'

    Retorna
    -------
    output : ndarray (M, N), imagen convolucionada
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Voltear el kernel en ambos ejes (definicion de convolucion)
    kernel_flipped = kernel[::-1, ::-1]

    kH, kW = kernel_flipped.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = _pad_image(image, pad_h, pad_w, mode=padding)

    M, N = image.shape
    output = np.zeros((M, N), dtype=np.float64)

    for m in range(M):
        for n in range(N):
            region = padded[m:m + kH, n:n + kW]
            output[m, n] = np.sum(region * kernel_flipped)

    return output


def correlate2d_manual(image: np.ndarray, kernel: np.ndarray,
                       padding: str = 'zero') -> np.ndarray:
    """
    Correlacion cruzada 2D (sin voltear el kernel).

    A diferencia de la convolucion, aqui el kernel se desliza
    directamente sin flip:
        g[m, n] = SUM_i SUM_j { f[m + i, n + j] * h[i, j] }

    Muchas librerias usan correlacion internamente y la llaman
    'convolucion'. Esta funcion existe para demostrar la diferencia.
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = _pad_image(image, pad_h, pad_w, mode=padding)

    M, N = image.shape
    output = np.zeros((M, N), dtype=np.float64)

    for m in range(M):
        for n in range(N):
            region = padded[m:m + kH, n:n + kW]
            output[m, n] = np.sum(region * kernel)

    return output


def convolve2d_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolucion 2D via FFT para validacion.

    Aprovecha el teorema de convolucion:
        f * h  <-->  F(u,v) . H(u,v)

    La convolucion en el dominio espacial equivale a la multiplicacion
    punto a punto en el dominio de frecuencia.

    NOTA: Esta funcion NO es la implementacion principal del proyecto.
    Existe unicamente para validar que convolve2d_manual produce
    resultados correctos.
    """
    M, N = image.shape
    kH, kW = kernel.shape

    # Tamano del resultado con padding para evitar aliasing circular
    fft_rows = M + kH - 1
    fft_cols = N + kW - 1

    F = np.fft.fft2(image, s=(fft_rows, fft_cols))
    H = np.fft.fft2(kernel, s=(fft_rows, fft_cols))

    G = F * H
    g = np.fft.ifft2(G).real

    # Recortar al tamano original (same convolution)
    row_start = kH // 2
    col_start = kW // 2
    return g[row_start:row_start + M, col_start:col_start + N]
