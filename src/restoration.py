"""
Modulo de Restauracion / Deconvolucion
========================================
Implementa tres estrategias de deconvolucion para recuperar la imagen
original a partir de la imagen degradada, conociendo (o estimando)
la PSF del sistema.

Modelo de degradacion en frecuencia:
    G(u, v) = F(u, v) . H(u, v) + N(u, v)

El objetivo es estimar F_hat(u, v) a partir de G(u, v) y H(u, v).

Estrategias implementadas:
    1. Filtro Inverso Naive       - didactico, falla con ruido
    2. Filtro de Wiener           - principal, robusto con regularizacion
    3. Richardson-Lucy Iterativo  - iterativo, no requiere dominio frecuencia

Se usa np.fft para la DFT (es un algoritmo de calculo numerico, no una
funcion de restauracion "magica"). La logica de restauracion esta
implementada manualmente operacion por operacion.
"""

import numpy as np
from src.convolution import convolve2d_manual


def _psf_to_otf(kernel: np.ndarray, shape) -> np.ndarray:
    """
    Convierte la PSF h(x,y) a su funcion de transferencia optica H(u,v)
    del tamano de la imagen, CENTRANDO la PSF antes de la FFT.

    Esto es indispensable: la degradacion del proyecto usa convolucion
    espacial *centrada* (el kernel se alinea con su centro en cada pixel,
    ver convolve2d_manual). Si aqui colocaramos la PSF anclada en la
    esquina [0, 0], la imagen restaurada saldria desplazada (kH//2, kW//2)
    pixeles respecto a la original, arruinando metricas y alineacion.

    Por eso desplazamos circularmente la PSF para que su centro quede en
    el origen [0, 0], que es la convencion que asume la DFT.
    """
    M, N = shape
    kH, kW = kernel.shape
    psf = np.zeros((M, N), dtype=np.float64)
    # Si el kernel es mas grande que la imagen (p.ej. recortes pequenos de
    # placas en video), lo recortamos para que quepa sin romper.
    kh, kw = min(kH, M), min(kW, N)
    psf[:kh, :kw] = kernel[:kh, :kw]
    # Llevar el centro de la PSF (kH//2, kW//2) al origen [0, 0]
    psf = np.roll(psf, (-(kH // 2), -(kW // 2)), axis=(0, 1))
    return np.fft.fft2(psf)


def _to_freq_domain(image: np.ndarray, kernel: np.ndarray):
    """
    Transforma imagen y kernel al dominio de frecuencia con el mismo
    tamano, necesario para operar punto a punto.

    La PSF se centra con _psf_to_otf para ser consistente con la
    convolucion espacial centrada usada en la degradacion.

    Retorna G y H.
    """
    M, N = image.shape
    G = np.fft.fft2(image, s=(M, N))
    H = _psf_to_otf(kernel, (M, N))
    return G, H


def inverse_filter(degraded: np.ndarray, kernel: np.ndarray,
                   threshold: float = 1e-3) -> np.ndarray:
    """
    Filtro Inverso Naive.

    Formula:
        F_hat(u, v) = G(u, v) / H(u, v)

    Problema: cuando |H(u,v)| es pequeno, el ruido N/H se amplifica
    enormemente. Por eso se usa un umbral para evitar division por
    valores cercanos a cero.

    NOTA: Este filtro se incluye con proposito DIDACTICO para demostrar
    por que se necesita regularizacion. En presencia de ruido, los
    resultados son muy malos.

    Parametros
    ----------
    degraded  : ndarray (M, N), imagen degradada g(x,y)
    kernel    : ndarray (kH, kW), PSF conocida h(x,y)
    threshold : umbral minimo para |H| (evita division por ~0)

    Retorna
    -------
    restored : ndarray (M, N), imagen restaurada (recortada a [0, 1])
    """
    G, H = _to_freq_domain(degraded, kernel)

    # Evitar division por valores muy pequenos
    H_safe = np.where(np.abs(H) < threshold, threshold, H)

    # F_hat = G / H
    F_hat = G / H_safe

    restored = np.fft.ifft2(F_hat).real
    return np.clip(restored, 0.0, 1.0)


def wiener_filter(degraded: np.ndarray, kernel: np.ndarray,
                  K: float = 0.01) -> np.ndarray:
    """
    Filtro de Wiener (implementacion manual).

    Formula:
        F_hat(u, v) = [ H*(u,v) / (|H(u,v)|^2 + K) ] . G(u,v)

    donde:
        H*     : conjugado complejo de H
        |H|^2  : H . H* (potencia espectral de la PSF)
        K      : parametro de regularizacion, aproxima SNR^{-1}
                 (relacion inversa senal-ruido)

    Cuando K = 0, se reduce al filtro inverso.
    Cuando K es grande, se suaviza mas (menos ruido, mas blur residual).

    Este es el metodo principal de restauracion del proyecto.

    Parametros
    ----------
    degraded : ndarray (M, N), imagen degradada
    kernel   : ndarray (kH, kW), PSF conocida
    K        : parametro de regularizacion (float > 0)

    Retorna
    -------
    restored : ndarray (M, N), imagen restaurada
    """
    G, H = _to_freq_domain(degraded, kernel)

    # Conjugado complejo de H
    H_conj = np.conj(H)

    # |H|^2 = H * H*
    H_power = np.abs(H) ** 2

    # Filtro de Wiener: H* / (|H|^2 + K)
    W = H_conj / (H_power + K)

    # Estimacion: F_hat = W . G
    F_hat = W * G

    restored = np.fft.ifft2(F_hat).real
    return np.clip(restored, 0.0, 1.0)


def richardson_lucy(degraded: np.ndarray, kernel: np.ndarray,
                    iterations: int = 30,
                    use_fft: bool = True) -> np.ndarray:
    """
    Deconvolucion de Richardson-Lucy (implementacion manual iterativa).

    Algoritmo:
        f_{k+1} = f_k . [ (g / (f_k * h)) * h^T ]

    donde:
        f_k   : estimacion actual de la imagen
        g     : imagen degradada observada
        h     : PSF del sistema
        h^T   : PSF transpuesta (rotada 180 grados)
        *     : convolucion 2D
        .     : multiplicacion punto a punto
        /     : division punto a punto

    Este metodo:
    - Converge a la estimacion de maxima verosimilitud (ML)
    - Preserva la positividad de la imagen
    - No requiere transformar al dominio de frecuencia
    - Puede usar convolucion manual o FFT (configurable)

    Parametros
    ----------
    degraded   : ndarray (M, N), imagen degradada
    kernel     : ndarray (kH, kW), PSF conocida
    iterations : numero de iteraciones
    use_fft    : si True, usa FFT para las convoluciones internas
                 (mas rapido, matematicamente equivalente)

    Retorna
    -------
    estimate : ndarray (M, N), imagen restaurada
    """
    # PSF transpuesta (rotada 180 grados)
    kernel_t = kernel[::-1, ::-1]

    # Inicializacion: la imagen degradada como primera estimacion
    estimate = degraded.copy().astype(np.float64)
    estimate = np.maximum(estimate, 1e-10)

    epsilon = 1e-10

    def _convolve(img, kern):
        if use_fft:
            M, N = img.shape
            F = np.fft.fft2(img, s=(M, N))
            # PSF centrada (misma convencion que la degradacion centrada),
            # evita que la estimacion se desplace en cada iteracion
            K = _psf_to_otf(kern, (M, N))
            return np.fft.ifft2(F * K).real
        else:
            return convolve2d_manual(img, kern)

    for _ in range(iterations):
        # Paso 1: f_k * h  (estimacion de la imagen degradada)
        estimate_blurred = _convolve(estimate, kernel)
        estimate_blurred = np.maximum(estimate_blurred, epsilon)

        # Paso 2: g / (f_k * h)  (ratio observado/estimado)
        ratio = degraded / estimate_blurred

        # Paso 3: ratio * h^T  (correccion por correlacion con PSF)
        correction = _convolve(ratio, kernel_t)

        # Paso 4: f_{k+1} = f_k . correction  (actualizacion multiplicativa)
        estimate = estimate * correction
        estimate = np.maximum(estimate, epsilon)

    return np.clip(estimate, 0.0, 1.0)
