"""
Modulo de Post-Procesamiento
==============================
Aplica mejoras adicionales a la imagen restaurada para maximizar
la legibilidad de la placa vehicular.

Tecnicas implementadas:
    - Unsharp Masking (enfoque via convolucion manual)
    - Mejora de contraste (estiramiento de histograma y CLAHE)
    - Resaltado de bordes (Laplaciano superpuesto)
"""

import numpy as np
import cv2
from src.convolution import convolve2d_manual
from src.kernels import gaussian_kernel, laplacian_kernel


def unsharp_mask(image: np.ndarray, sigma: float = 1.0,
                 alpha: float = 1.5, kernel_size: int = 5) -> np.ndarray:
    """
    Enfoque por mascara de desenfoque (Unsharp Masking).

    Formula:
        sharp(x, y) = f(x, y) + alpha * [f(x, y) - blur(f)(x, y)]

    El termino entre corchetes es la "mascara de desenfoque": contiene
    los detalles de alta frecuencia que el blur elimino. Al sumarlos
    amplificados, se realzan bordes y texturas.

    Parametros
    ----------
    image       : ndarray (M, N), rango [0, 1]
    sigma       : sigma del kernel Gaussiano para el blur
    alpha       : factor de amplificacion de los detalles
    kernel_size : tamano del kernel Gaussiano

    Retorna
    -------
    sharpened : ndarray (M, N), imagen enfocada
    """
    gauss = gaussian_kernel(kernel_size, sigma)
    blurred = convolve2d_manual(image, gauss)

    # Mascara de detalles (alta frecuencia)
    detail_mask = image - blurred

    sharpened = image + alpha * detail_mask
    return np.clip(sharpened, 0.0, 1.0)


def histogram_stretch(image: np.ndarray) -> np.ndarray:
    """
    Estiramiento de histograma (contrast stretching).

    Mapea linealmente el rango [min, max] de la imagen al rango [0, 1]:
        f_out = (f_in - f_min) / (f_max - f_min)

    Maximiza el uso del rango dinamico disponible.
    """
    f_min = image.min()
    f_max = image.max()

    if f_max - f_min < 1e-10:
        return image.copy()

    stretched = (image - f_min) / (f_max - f_min)
    return stretched


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Ecualización adaptativa de histograma con limitacion de contraste (CLAHE).

    Usa OpenCV para esta operacion (permitido como post-proceso,
    no como parte de la logica central de convolucion/deconvolucion).

    CLAHE divide la imagen en bloques y ecualiza localmente, evitando
    la sobre-amplificacion de ruido que produce la ecualizacion global.

    Parametros
    ----------
    image      : ndarray (M, N), rango [0, 1]
    clip_limit : limite de amplificacion del contraste
    tile_size  : tamano de los bloques

    Retorna
    -------
    enhanced : ndarray (M, N), rango [0, 1]
    """
    img_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img_uint8)
    return enhanced.astype(np.float64) / 255.0


def edge_enhancement(image: np.ndarray,
                     weight: float = 0.3) -> np.ndarray:
    """
    Resaltado de bordes mediante superposicion del Laplaciano.

    Formula:
        enhanced = f(x, y) - weight * Laplaciano(f)

    El signo negativo es porque el Laplaciano con centro -4 produce
    valores negativos en los bordes; restarlos realza los bordes.

    Parametros
    ----------
    image  : ndarray (M, N), rango [0, 1]
    weight : peso del componente de bordes

    Retorna
    -------
    enhanced : ndarray (M, N), imagen con bordes realzados
    """
    lap_kernel = laplacian_kernel()
    edges = convolve2d_manual(image, lap_kernel)

    enhanced = image - weight * edges
    return np.clip(enhanced, 0.0, 1.0)


def postprocess_pipeline(image: np.ndarray,
                         sharpen: bool = True,
                         contrast: str = 'clahe',
                         edge_enhance: bool = True,
                         sharpen_params: dict = None,
                         clahe_params: dict = None,
                         edge_params: dict = None) -> np.ndarray:
    """
    Pipeline completo de post-procesamiento.

    Orden de aplicacion:
        1. Sharpening (unsharp mask)
        2. Mejora de contraste (CLAHE o estiramiento)
        3. Resaltado de bordes

    Parametros
    ----------
    image           : ndarray (M, N), imagen restaurada, rango [0, 1]
    sharpen         : aplicar unsharp mask
    contrast        : 'clahe' | 'stretch' | 'none'
    edge_enhance    : aplicar resaltado de bordes
    sharpen_params  : dict con parametros para unsharp_mask
    clahe_params    : dict con parametros para CLAHE
    edge_params     : dict con parametros para edge_enhancement

    Retorna
    -------
    result : ndarray (M, N), imagen post-procesada
    """
    if sharpen_params is None:
        sharpen_params = {}
    if clahe_params is None:
        clahe_params = {}
    if edge_params is None:
        edge_params = {}

    result = image.copy()

    if sharpen:
        result = unsharp_mask(result, **sharpen_params)

    if contrast == 'clahe':
        result = apply_clahe(result, **clahe_params)
    elif contrast == 'stretch':
        result = histogram_stretch(result)

    if edge_enhance:
        result = edge_enhancement(result, **edge_params)

    return result
