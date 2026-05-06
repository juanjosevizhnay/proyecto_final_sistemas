"""
Modulo de Degradacion de Imagenes
==================================
Simula el modelo clasico de degradacion de un sistema de adquisicion:

    g(x, y) = f(x, y) * h(x, y) + n(x, y)

donde:
    f(x, y) : imagen original (senal de entrada)
    h(x, y) : PSF del sistema (respuesta al impulso / kernel)
    n(x, y) : ruido aditivo
    g(x, y) : imagen degradada (senal de salida)
    *        : operador de convolucion 2D

Este modulo permite simular condiciones reales de camaras de transito:
movimiento del vehiculo, desenfoque optico y ruido del sensor.
"""

import numpy as np
from src.convolution import convolve2d_manual


def apply_blur(image: np.ndarray, kernel: np.ndarray,
               padding: str = 'zero') -> np.ndarray:
    """
    Aplica desenfoque a la imagen mediante convolucion con el kernel dado.

    Esto modela el termino  f(x,y) * h(x,y)  de la ecuacion de degradacion,
    usando nuestra implementacion manual de convolucion 2D.

    Parametros
    ----------
    image   : ndarray (M, N), rango [0, 1]
    kernel  : ndarray (kH, kW), kernel de blur (Gaussiano, movimiento, etc.)
    padding : tipo de padding para la convolucion

    Retorna
    -------
    blurred : ndarray (M, N), imagen desenfocada
    """
    blurred = convolve2d_manual(image, kernel, padding=padding)
    return np.clip(blurred, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0,
                       std: float = 0.01) -> np.ndarray:
    """
    Agrega ruido Gaussiano aditivo a la imagen.

    Modela el ruido termico del sensor de la camara:
        g(x, y) = f(x, y) + n(x, y)
    donde n ~ N(mean, std^2)

    Parametros
    ----------
    image : ndarray (M, N), rango [0, 1]
    mean  : media del ruido (normalmente 0)
    std   : desviacion estandar del ruido

    Retorna
    -------
    noisy : ndarray (M, N), imagen con ruido, recortada a [0, 1]
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


def add_salt_pepper_noise(image: np.ndarray,
                          amount: float = 0.02) -> np.ndarray:
    """
    Agrega ruido de sal y pimienta (impulsivo).

    Modela defectos puntuales del sensor o errores de transmision.
    Un porcentaje 'amount' de pixeles se fuerza a 0 (pimienta) o 1 (sal).

    Parametros
    ----------
    image  : ndarray (M, N), rango [0, 1]
    amount : fraccion de pixeles afectados (ej. 0.02 = 2%)

    Retorna
    -------
    noisy : ndarray (M, N), imagen con ruido impulsivo
    """
    noisy = image.copy()
    total_pixels = image.size
    n_salt = int(total_pixels * amount / 2)
    n_pepper = int(total_pixels * amount / 2)

    # Sal (pixeles blancos)
    salt_coords = (
        np.random.randint(0, image.shape[0], n_salt),
        np.random.randint(0, image.shape[1], n_salt)
    )
    noisy[salt_coords] = 1.0

    # Pimienta (pixeles negros)
    pepper_coords = (
        np.random.randint(0, image.shape[0], n_pepper),
        np.random.randint(0, image.shape[1], n_pepper)
    )
    noisy[pepper_coords] = 0.0

    return noisy


def degrade_image(image: np.ndarray, kernel: np.ndarray,
                  noise_type: str = 'gaussian',
                  noise_params: dict = None,
                  padding: str = 'zero') -> np.ndarray:
    """
    Pipeline completo de degradacion: blur + ruido.

    Implementa el modelo:
        g(x, y) = f(x, y) * h(x, y) + n(x, y)

    Primero aplica la convolucion con el kernel (blur), luego
    agrega el ruido especificado.

    Parametros
    ----------
    image        : ndarray (M, N), imagen original, rango [0, 1]
    kernel       : ndarray (kH, kW), PSF del sistema
    noise_type   : 'gaussian' | 'salt_pepper' | 'none'
    noise_params : dict con parametros del ruido
                   - gaussian: {'mean': 0.0, 'std': 0.01}
                   - salt_pepper: {'amount': 0.02}
    padding      : tipo de padding para la convolucion

    Retorna
    -------
    degraded : ndarray (M, N), imagen degradada
    """
    if noise_params is None:
        noise_params = {}

    # Paso 1: Convolucion  f * h
    blurred = apply_blur(image, kernel, padding=padding)

    # Paso 2: Adicion de ruido  + n
    if noise_type == 'gaussian':
        mean = noise_params.get('mean', 0.0)
        std = noise_params.get('std', 0.01)
        degraded = add_gaussian_noise(blurred, mean=mean, std=std)
    elif noise_type == 'salt_pepper':
        amount = noise_params.get('amount', 0.02)
        degraded = add_salt_pepper_noise(blurred, amount=amount)
    elif noise_type == 'none':
        degraded = blurred
    else:
        raise ValueError(f"Tipo de ruido no soportado: {noise_type}")

    return degraded
