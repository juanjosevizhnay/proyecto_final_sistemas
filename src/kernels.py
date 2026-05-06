"""
Modulo de Construccion de Kernels
==================================
Genera kernels (respuestas al impulso) usados para modelar la
degradacion y el procesamiento de imagenes.

En el contexto de sistemas lineales, el kernel h(x,y) representa
la respuesta al impulso del sistema. La salida del sistema se obtiene
mediante la convolucion de la entrada con h:

    g(x,y) = f(x,y) * h(x,y)

Todos los kernels se construyen con la matematica explicita.
"""

import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Construye un kernel Gaussiano 2D.

    Formula:
        h(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    El kernel se normaliza para que la suma de todos sus elementos sea 1,
    preservando la energia de la imagen al convolucionarla.

    Parametros
    ----------
    size  : dimension del kernel (debe ser impar)
    sigma : desviacion estandar de la Gaussiana

    Retorna
    -------
    kernel : ndarray (size, size), normalizado (suma = 1)
    """
    if size % 2 == 0:
        raise ValueError("El tamano del kernel debe ser impar.")

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    # Normalizar: la constante 1/(2*pi*sigma^2) se absorbe en la normalizacion
    kernel /= kernel.sum()
    return kernel


def motion_blur_kernel(size: int, angle: float = 0.0) -> np.ndarray:
    """
    Construye un kernel de desenfoque por movimiento (motion blur).

    Modela el efecto de movimiento relativo entre la camara y el objeto
    durante la exposicion. La PSF (Point Spread Function) es una linea
    en la direccion del movimiento.

    Parametros
    ----------
    size  : longitud del kernel en pixeles (debe ser impar)
    angle : angulo del movimiento en grados (0 = horizontal)

    Retorna
    -------
    kernel : ndarray (size, size), normalizado
    """
    if size % 2 == 0:
        raise ValueError("El tamano del kernel debe ser impar.")

    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    angle_rad = np.deg2rad(angle)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    for i in range(size):
        offset = i - center
        x = center + round(offset * cos_a)
        y = center + round(offset * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    # Asegurar al menos un pixel activo
    if kernel.sum() == 0:
        kernel[center, center] = 1.0

    kernel /= kernel.sum()
    return kernel


def sharpening_kernel() -> np.ndarray:
    """
    Kernel de enfoque (sharpening) basado en el Laplaciano.

    Realza bordes restando el Laplaciano de la imagen original:
        sharp = original - Laplaciano(original)

    Esto equivale a convolucion con:
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]]

    El centro (5) es el pixel original, y los -1 restan las contribuciones
    del Laplaciano (segunda derivada discreta).
    """
    return np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)


def laplacian_kernel() -> np.ndarray:
    """
    Kernel Laplaciano para deteccion de bordes.

    Aproxima la segunda derivada discreta:
        nabla^2 f = d^2f/dx^2 + d^2f/dy^2

    En forma discreta:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    """
    return np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64)


def sobel_kernels() -> tuple:
    """
    Kernels de Sobel para deteccion de bordes en X e Y.

    Aproximan la primera derivada parcial:
        Gx = df/dx ,  Gy = df/dy

    La magnitud del gradiente |G| = sqrt(Gx^2 + Gy^2) indica
    la intensidad del borde.

    Retorna
    -------
    (sobel_x, sobel_y) : tupla de ndarray (3, 3)
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    return sobel_x, sobel_y


def identity_kernel(size: int = 3) -> np.ndarray:
    """
    Kernel identidad (delta de Dirac discreta).

    La convolucion de cualquier senal con delta produce la misma senal:
        f * delta = f

    Util para verificar que la implementacion de convolucion es correcta.
    """
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    kernel[center, center] = 1.0
    return kernel
