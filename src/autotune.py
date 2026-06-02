"""
Modulo de Auto-Ajuste de la Restauracion
==========================================
Mejora la eficacia de la restauracion cuando NO conocemos con exactitud
ni el kernel de degradacion h(x,y) ni el nivel de ruido (caso real).

Implementa dos estrategias complementarias:

  (a) Barrido del parametro K de Wiener
      K aproxima la relacion ruido/senal. Con poco ruido conviene K
      pequeno (recupera mas detalle); con mucho ruido, K grande (evita
      amplificar el ruido). En vez de fijarlo a mano, probamos varios
      valores y nos quedamos con el mejor segun una metrica.

  (b) Busqueda del kernel (deconvolucion "ciega" simplificada)
      Como el blur real puede ser Gaussiano o de movimiento con cierto
      angulo/longitud, generamos un conjunto de PSF candidatas, restauramos
      con cada una y elegimos la que da el mejor resultado.

El "mejor resultado" se decide con un *scorer*:
  - Si tenemos imagen de referencia  -> SSIM (objetivo).
  - Si no (caso real)                -> confianza del OCR sobre el
                                          recorte restaurado (la placa
                                          mas legible gana).

Reutiliza la matematica del proyecto: wiener_filter + postprocess_pipeline.
"""

import numpy as np

from src.kernels import gaussian_kernel, motion_blur_kernel
from src.restoration import wiener_filter
from src.postprocessing import postprocess_pipeline
from src.metrics import ssim
from src.ocr import detect_text_regions, OCR_AVAILABLE


# ---------------------------------------------------------------------------
#   CANDIDATOS DE KERNEL  (mejora b)
# ---------------------------------------------------------------------------

def default_kernel_candidates() -> list:
    """
    Conjunto amplio de PSF candidatas: Gaussianas de varios tamanos/sigmas
    y motion blur en varias longitudes y angulos.

    Retorna lista de tuplas (nombre, kernel).
    """
    cands = []
    for size, sigma in [(5, 1.5), (9, 2.5), (13, 3.0)]:
        cands.append((f"gauss_{size}_s{sigma}", gaussian_kernel(size, sigma)))
    for length in (9, 15):
        for angle in (0, 45, 90, 135):
            cands.append((f"motion_{length}_a{angle}",
                          motion_blur_kernel(length, angle)))
    return cands


def quick_kernel_candidates() -> list:
    """
    Conjunto reducido (mas rapido) para usar por placa en video, donde
    se restauran muchos recortes. Cubre Gaussiano + movimiento en las
    cuatro orientaciones principales.
    """
    return [
        ("gauss_9_s2.5", gaussian_kernel(9, 2.5)),
        ("motion_11_a0", motion_blur_kernel(11, 0)),
        ("motion_11_a90", motion_blur_kernel(11, 90)),
        ("motion_15_a45", motion_blur_kernel(15, 45)),
    ]


# ---------------------------------------------------------------------------
#   SCORERS
# ---------------------------------------------------------------------------

def ocr_confidence_score(image: np.ndarray) -> float:
    """
    Puntua una restauracion por la confianza del OCR (0-100).

    Usa la deteccion mas fuerte del recorte: una placa bien restaurada
    produce una lectura de alta confianza. No requiere imagen de referencia,
    por eso sirve para el caso real.
    """
    if not OCR_AVAILABLE:
        return 0.0
    dets = detect_text_regions(image)
    if not dets:
        return 0.0
    return max(d['score'] for d in dets) * 100.0


def _post(restored, apply_postprocess):
    if not apply_postprocess:
        return restored
    return postprocess_pipeline(
        restored,
        sharpen=True, contrast='clahe', edge_enhance=True,
        sharpen_params={'sigma': 1.0, 'alpha': 1.5, 'kernel_size': 5},
        clahe_params={'clip_limit': 2.0, 'tile_size': 8},
        edge_params={'weight': 0.2},
    )


# ---------------------------------------------------------------------------
#   BUSQUEDA PRINCIPAL  (combina a + b)
# ---------------------------------------------------------------------------

def auto_restore(degraded: np.ndarray,
                 kernels: list = None,
                 k_values: list = None,
                 apply_postprocess: bool = True,
                 reference: np.ndarray = None,
                 scorer=None) -> dict:
    """
    Restaura barriendo (kernel candidato) x (valor de K) y devuelve la
    mejor combinacion segun el scorer.

    Parametros
    ----------
    degraded          : ndarray (M, N), recorte/imagen degradada [0, 1]
    kernels           : lista de (nombre, kernel). Default = quick_kernel_candidates()
    k_values          : lista de K para Wiener. Default = [0.005, 0.02, 0.08]
    apply_postprocess : aplicar sharpening/CLAHE/bordes tras Wiener
    reference         : imagen original (si se conoce) -> puntua con SSIM
    scorer            : funcion(img)->float. Si None y no hay reference,
                        usa ocr_confidence_score

    Retorna
    -------
    dict con:
        'restored'     : mejor imagen restaurada (post-procesada)
        'kernel'       : kernel ganador (ndarray)
        'kernel_name'  : nombre del kernel ganador
        'K'            : K ganador
        'score'        : puntuacion del ganador
        'history'      : lista de (kernel_name, K, score) de todo el barrido
    """
    if kernels is None:
        kernels = quick_kernel_candidates()
    if k_values is None:
        k_values = [0.005, 0.02, 0.08]

    use_ref = reference is not None
    if scorer is None and not use_ref:
        scorer = ocr_confidence_score

    best = None
    history = []

    for kname, kernel in kernels:
        for K in k_values:
            restored = wiener_filter(degraded, kernel, K=K)
            restored = _post(restored, apply_postprocess)

            if use_ref:
                # SSIM necesita mismo tamano que la referencia
                score = ssim(reference, restored)
            else:
                score = scorer(restored)

            history.append((kname, K, score))

            if best is None or score > best['score']:
                best = {
                    'restored': restored,
                    'kernel': kernel,
                    'kernel_name': kname,
                    'K': K,
                    'score': score,
                }

    best['history'] = history
    return best


def auto_wiener_k(degraded: np.ndarray,
                  kernel: np.ndarray,
                  k_values: list = None,
                  apply_postprocess: bool = True,
                  reference: np.ndarray = None,
                  scorer=None) -> dict:
    """
    Version mas ligera de la mejora (a) sola: kernel fijo conocido,
    solo se busca el mejor K de Wiener.

    Mismos parametros/salida que auto_restore (con un unico kernel).
    """
    return auto_restore(
        degraded,
        kernels=[("fijo", kernel)],
        k_values=k_values,
        apply_postprocess=apply_postprocess,
        reference=reference,
        scorer=scorer,
    )
