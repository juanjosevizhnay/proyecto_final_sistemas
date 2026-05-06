"""
Modulo de Metricas de Calidad de Imagen
=========================================
Implementa metricas cuantitativas para evaluar la calidad de la
restauracion comparando la imagen restaurada con la original.

Metricas implementadas manualmente:
    - MSE  : Error Cuadratico Medio
    - PSNR : Relacion Senal-Ruido Pico
    - SSIM : Indice de Similitud Estructural (simplificado)

Opcionalmente compara con scikit-image para validacion.
"""

import numpy as np


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Error Cuadratico Medio (Mean Squared Error).

    Formula:
        MSE = (1 / M*N) * SUM_x SUM_y [f(x,y) - g(x,y)]^2

    Un MSE de 0 indica imagenes identicas.

    Parametros
    ----------
    image1 : ndarray (M, N), imagen de referencia
    image2 : ndarray (M, N), imagen a evaluar

    Retorna
    -------
    mse_val : float
    """
    return float(np.mean((image1.astype(np.float64)
                          - image2.astype(np.float64)) ** 2))


def psnr(image1: np.ndarray, image2: np.ndarray,
         max_val: float = 1.0) -> float:
    """
    Relacion Senal-Ruido Pico (Peak Signal-to-Noise Ratio).

    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)

    donde MAX es el valor maximo posible de la senal (1.0 para
    imagenes normalizadas, 255 para uint8).

    Un PSNR mayor indica mejor calidad. Tipicamente:
        > 30 dB : buena calidad
        > 40 dB : excelente calidad

    Parametros
    ----------
    image1  : ndarray (M, N), imagen de referencia
    image2  : ndarray (M, N), imagen a evaluar
    max_val : valor maximo de la senal

    Retorna
    -------
    psnr_val : float (en dB), inf si las imagenes son identicas
    """
    mse_val = mse(image1, image2)
    if mse_val < 1e-10:
        return float('inf')
    return float(10.0 * np.log10(max_val ** 2 / mse_val))


def ssim(image1: np.ndarray, image2: np.ndarray,
         window_size: int = 7, K1: float = 0.01,
         K2: float = 0.03, L: float = 1.0) -> float:
    """
    Indice de Similitud Estructural (Structural Similarity Index).

    Implementacion manual simplificada basada en la formulacion original
    de Wang et al. (2004):

        SSIM(x, y) = [(2*mu_x*mu_y + C1)(2*sigma_xy + C2)] /
                     [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]

    donde:
        mu_x, mu_y       : medias locales
        sigma_x, sigma_y : desviaciones estandar locales
        sigma_xy          : covarianza local
        C1 = (K1 * L)^2  : constante de estabilizacion
        C2 = (K2 * L)^2  : constante de estabilizacion
        L                 : rango dinamico de la senal

    Se calcula SSIM local en ventanas y se promedia (MSSIM).

    Parametros
    ----------
    image1      : ndarray (M, N), imagen de referencia
    image2      : ndarray (M, N), imagen a evaluar
    window_size : tamano de la ventana deslizante
    K1, K2      : constantes de estabilidad
    L           : rango dinamico

    Retorna
    -------
    mssim : float, promedio de SSIM local (rango [-1, 1], 1 = identicas)
    """
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    M, N = img1.shape
    pad = window_size // 2

    ssim_map = []

    for i in range(pad, M - pad):
        for j in range(pad, N - pad):
            w1 = img1[i - pad:i + pad + 1, j - pad:j + pad + 1]
            w2 = img2[i - pad:i + pad + 1, j - pad:j + pad + 1]

            mu1 = np.mean(w1)
            mu2 = np.mean(w2)

            sigma1_sq = np.var(w1)
            sigma2_sq = np.var(w2)
            sigma12 = np.mean((w1 - mu1) * (w2 - mu2))

            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = ((mu1 ** 2 + mu2 ** 2 + C1)
                           * (sigma1_sq + sigma2_sq + C2))

            ssim_map.append(numerator / denominator)

    return float(np.mean(ssim_map))


def compute_all_metrics(original: np.ndarray,
                        degraded: np.ndarray,
                        restored: np.ndarray) -> dict:
    """
    Calcula todas las metricas para las imagenes degradada y restaurada
    respecto a la original.

    Retorna
    -------
    metrics : dict con estructura:
        {
            'degraded': {'mse': ..., 'psnr': ..., 'ssim': ...},
            'restored': {'mse': ..., 'psnr': ..., 'ssim': ...}
        }
    """
    metrics = {
        'degraded': {
            'mse': mse(original, degraded),
            'psnr': psnr(original, degraded),
            'ssim': ssim(original, degraded),
        },
        'restored': {
            'mse': mse(original, restored),
            'psnr': psnr(original, restored),
            'ssim': ssim(original, restored),
        }
    }
    return metrics


def print_metrics_report(metrics: dict):
    """
    Imprime un reporte formateado de las metricas de calidad.
    """
    print("\n" + "=" * 60)
    print("          METRICAS DE CALIDAD DE IMAGEN")
    print("=" * 60)
    print(f"{'Metrica':<12} {'Degradada':>14} {'Restaurada':>14} {'Mejora':>10}")
    print("-" * 60)

    for name in ['mse', 'psnr', 'ssim']:
        val_d = metrics['degraded'][name]
        val_r = metrics['restored'][name]

        if name == 'mse':
            improvement = f"{val_d - val_r:+.6f}"
            fmt = '.6f'
        elif name == 'psnr':
            improvement = f"{val_r - val_d:+.2f} dB"
            fmt = '.2f'
        else:
            improvement = f"{val_r - val_d:+.4f}"
            fmt = '.4f'

        print(f"{name.upper():<12} {val_d:>14{fmt}} {val_r:>14{fmt}} {improvement:>10}")

    print("=" * 60)
