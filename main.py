"""
============================================================================
  RESTAURACION DE IMAGENES DE PLACAS VEHICULARES
  Mediante Convolucion y Deconvolucion
  -----------------------------------------------
  Proyecto Final - Sistemas Lineales y Senales
============================================================================

Pipeline principal que demuestra:
  1. Que la degradacion de imagen puede modelarse como convolucion
  2. Que la restauracion puede abordarse mediante deconvolucion
  3. Aplicacion real en control de transito y legibilidad de placas

Modelo matematico:
  g(x,y) = f(x,y) * h(x,y) + n(x,y)

  Donde:
    f(x,y) : imagen original
    h(x,y) : respuesta al impulso (PSF) del sistema degradante
    n(x,y) : ruido aditivo
    g(x,y) : imagen degradada
    *       : convolucion 2D
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')

from src.kernels import gaussian_kernel, motion_blur_kernel
from src.degradation import degrade_image
from src.restoration import inverse_filter, wiener_filter, richardson_lucy
from src.postprocessing import postprocess_pipeline
from src.ocr import compare_ocr_results, print_ocr_report
from src.metrics import compute_all_metrics, print_metrics_report
from src.visualization import (
    show_comparison,
    show_frequency_spectrum,
    show_kernel,
    show_difference_map,
    show_restoration_pipeline,
    show_ocr_comparison,
    show_metrics_bar_chart,
)


# ============================================================================
#   CONFIGURACION
# ============================================================================

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
SUPPORTED_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')

# --- Parametros de degradacion ---
BLUR_TYPE = 'gaussian'          # 'gaussian' o 'motion'
GAUSSIAN_KERNEL_SIZE = 11       # Tamano del kernel Gaussiano (impar)
GAUSSIAN_SIGMA = 2.0            # Sigma del blur Gaussiano
MOTION_KERNEL_SIZE = 15         # Tamano del kernel de movimiento (impar)
MOTION_ANGLE = 0.0              # Angulo del movimiento en grados

NOISE_TYPE = 'gaussian'         # 'gaussian', 'salt_pepper', o 'none'
NOISE_STD = 0.01                # Desviacion estandar del ruido Gaussiano
SALT_PEPPER_AMOUNT = 0.02       # Proporcion de pixeles afectados

# --- Parametros de restauracion ---
WIENER_K = 0.01                 # Parametro de regularizacion del filtro Wiener
INVERSE_THRESHOLD = 1e-3        # Umbral del filtro inverso
RL_ITERATIONS = 30              # Iteraciones para Richardson-Lucy

# --- OCR ---
GROUND_TRUTH = None             # Texto real de la placa (None si no se conoce)


# ============================================================================
#   FUNCIONES AUXILIARES
# ============================================================================

def load_image(path: str) -> np.ndarray:
    """Carga una imagen, la convierte a escala de grises y normaliza a [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img.astype(np.float64) / 255.0


def find_images() -> list:
    """Busca todas las imagenes en el directorio de imagenes."""
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    return sorted(image_files)


def select_image(image_files: list) -> str:
    """Permite al usuario seleccionar una imagen del directorio."""
    if not image_files:
        print(f"\nNo se encontraron imagenes en '{IMAGES_DIR}'.")
        print("Coloca imagenes de placas vehiculares en esa carpeta.")
        sys.exit(1)

    if len(image_files) == 1:
        print(f"\nImagen encontrada: {os.path.basename(image_files[0])}")
        return image_files[0]

    print(f"\nImagenes disponibles en '{IMAGES_DIR}':")
    for i, f in enumerate(image_files):
        print(f"  [{i + 1}] {os.path.basename(f)}")

    while True:
        try:
            choice = int(input(f"\nSelecciona una imagen (1-{len(image_files)}): "))
            if 1 <= choice <= len(image_files):
                return image_files[choice - 1]
        except (ValueError, EOFError):
            pass
        print("Opcion invalida.")


def build_kernel():
    """Construye el kernel de degradacion segun la configuracion."""
    if BLUR_TYPE == 'gaussian':
        kernel = gaussian_kernel(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        kernel_name = f'Gaussiano ({GAUSSIAN_KERNEL_SIZE}x{GAUSSIAN_KERNEL_SIZE}, sigma={GAUSSIAN_SIGMA})'
    elif BLUR_TYPE == 'motion':
        kernel = motion_blur_kernel(MOTION_KERNEL_SIZE, MOTION_ANGLE)
        kernel_name = f'Movimiento ({MOTION_KERNEL_SIZE}px, angulo={MOTION_ANGLE} grados)'
    else:
        raise ValueError(f"Tipo de blur no soportado: {BLUR_TYPE}")
    return kernel, kernel_name


def build_noise_params() -> dict:
    """Construye los parametros de ruido segun la configuracion."""
    if NOISE_TYPE == 'gaussian':
        return {'mean': 0.0, 'std': NOISE_STD}
    elif NOISE_TYPE == 'salt_pepper':
        return {'amount': SALT_PEPPER_AMOUNT}
    return {}


# ============================================================================
#   PIPELINE PRINCIPAL
# ============================================================================

def run_pipeline(image_path: str):
    """
    Ejecuta el pipeline completo de degradacion y restauracion.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE DE RESTAURACION DE PLACAS VEHICULARES")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PASO 1: Cargar imagen original
    # ------------------------------------------------------------------
    print("\n[1/10] Cargando imagen original...")
    original = load_image(image_path)
    print(f"  Imagen: {os.path.basename(image_path)}")
    print(f"  Tamano: {original.shape[1]}x{original.shape[0]} pixeles")

    # ------------------------------------------------------------------
    # PASO 2: OCR sobre imagen original (baseline)
    # ------------------------------------------------------------------
    print("\n[2/10] OCR sobre imagen original (baseline)...")
    from src.ocr import read_plate
    ocr_original = read_plate(original)
    print(f"  Texto detectado: '{ocr_original['text']}'")
    print(f"  Confianza: {ocr_original['confidence']:.1f}%")

    # ------------------------------------------------------------------
    # PASO 3: Construir kernel de degradacion
    # ------------------------------------------------------------------
    print("\n[3/10] Construyendo kernel de degradacion...")
    kernel, kernel_name = build_kernel()
    print(f"  Kernel: {kernel_name}")
    show_kernel(kernel, title=f'PSF: {kernel_name}', save_name='01_kernel')

    # ------------------------------------------------------------------
    # PASO 4: Degradar imagen (convolucion manual + ruido)
    # ------------------------------------------------------------------
    print("\n[4/10] Degradando imagen (convolucion manual + ruido)...")
    print(f"  Tipo de ruido: {NOISE_TYPE}")
    noise_params = build_noise_params()
    degraded = degrade_image(original, kernel,
                             noise_type=NOISE_TYPE,
                             noise_params=noise_params)
    print("  Degradacion completa.")

    # ------------------------------------------------------------------
    # PASO 5: OCR sobre imagen degradada
    # ------------------------------------------------------------------
    print("\n[5/10] OCR sobre imagen degradada...")
    ocr_degraded = read_plate(degraded)
    print(f"  Texto detectado: '{ocr_degraded['text']}'")
    print(f"  Confianza: {ocr_degraded['confidence']:.1f}%")

    # Mostrar comparacion original vs degradada
    show_comparison(
        [original, degraded],
        ['Original', 'Degradada'],
        suptitle='Efecto de la Degradacion',
        save_name='02_original_vs_degradada'
    )

    # Espectros de frecuencia
    show_frequency_spectrum(original, 'Imagen Original',
                            save_name='03_espectro_original')
    show_frequency_spectrum(degraded, 'Imagen Degradada',
                            save_name='04_espectro_degradada')

    # ------------------------------------------------------------------
    # PASO 6: Restauracion con filtro inverso (didactico)
    # ------------------------------------------------------------------
    print("\n[6/10] Restaurando con filtro inverso (demostracion didactica)...")
    restored_inverse = inverse_filter(degraded, kernel,
                                      threshold=INVERSE_THRESHOLD)
    print("  NOTA: El filtro inverso amplifica el ruido. Esto es esperado.")

    # ------------------------------------------------------------------
    # PASO 7: Restauracion con filtro de Wiener (principal)
    # ------------------------------------------------------------------
    print("\n[7/10] Restaurando con filtro de Wiener (metodo principal)...")
    restored_wiener = wiener_filter(degraded, kernel, K=WIENER_K)
    print(f"  Parametro K = {WIENER_K}")

    # ------------------------------------------------------------------
    # PASO 8: Restauracion con Richardson-Lucy (comparacion)
    # ------------------------------------------------------------------
    print("\n[8/10] Restaurando con Richardson-Lucy iterativo...")
    restored_rl = richardson_lucy(degraded, kernel,
                                  iterations=RL_ITERATIONS, use_fft=True)
    print(f"  Iteraciones: {RL_ITERATIONS}")

    # Comparar los tres metodos
    show_comparison(
        [restored_inverse, restored_wiener, restored_rl],
        ['Filtro Inverso', 'Filtro de Wiener', 'Richardson-Lucy'],
        suptitle='Comparacion de Metodos de Restauracion',
        save_name='05_comparacion_restauracion'
    )

    # ------------------------------------------------------------------
    # PASO 9: Post-procesamiento
    # ------------------------------------------------------------------
    print("\n[9/10] Aplicando post-procesamiento...")
    restored_final = postprocess_pipeline(
        restored_wiener,
        sharpen=True,
        contrast='clahe',
        edge_enhance=True,
        sharpen_params={'sigma': 1.0, 'alpha': 1.5, 'kernel_size': 5},
        clahe_params={'clip_limit': 2.0, 'tile_size': 8},
        edge_params={'weight': 0.2},
    )
    print("  Sharpening + CLAHE + Realce de bordes aplicados.")

    # ------------------------------------------------------------------
    # PASO 10: OCR sobre imagen restaurada
    # ------------------------------------------------------------------
    print("\n[10/10] OCR sobre imagen restaurada...")
    ocr_restored = read_plate(restored_final)
    print(f"  Texto detectado: '{ocr_restored['text']}'")
    print(f"  Confianza: {ocr_restored['confidence']:.1f}%")

    # ==================================================================
    #   RESULTADOS FINALES
    # ==================================================================

    # Pipeline completo visual
    pipeline_images = {
        '1. Original': original,
        '2. Degradada': degraded,
        '3. Filtro Inverso': restored_inverse,
        '4. Wiener': restored_wiener,
        '5. Richardson-Lucy': restored_rl,
        '6. Post-procesada': restored_final,
    }
    show_restoration_pipeline(pipeline_images,
                              save_name='06_pipeline_completo')

    # Mapa de diferencia
    show_difference_map(degraded, restored_final,
                        title='|Degradada - Restaurada|',
                        save_name='07_diferencia')

    # Metricas cuantitativas
    metrics = compute_all_metrics(original, degraded, restored_final)
    print_metrics_report(metrics)
    show_metrics_bar_chart(metrics, save_name='08_metricas')

    # Comparacion OCR completa
    ocr_results = compare_ocr_results(original, degraded, restored_final,
                                      ground_truth=GROUND_TRUTH)
    print_ocr_report(ocr_results, ground_truth=GROUND_TRUTH)

    show_ocr_comparison(
        {'original': original, 'degraded': degraded,
         'restored': restored_final},
        ocr_results,
        ground_truth=GROUND_TRUTH,
        save_name='09_comparacion_ocr'
    )

    # Comparacion final lado a lado
    show_comparison(
        [original, degraded, restored_final],
        ['Original', 'Degradada', 'Restaurada (Final)'],
        suptitle='Resultado Final del Pipeline de Restauracion',
        save_name='10_resultado_final'
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print(f"  Las graficas se guardaron en: {os.path.abspath('output')}")
    print("=" * 60 + "\n")


# ============================================================================
#   EJECUCION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  RESTAURACION DE IMAGENES DE PLACAS VEHICULARES")
    print("  Proyecto Final - Sistemas Lineales y Senales")
    print("=" * 60)

    image_files = find_images()
    image_path = select_image(image_files)
    run_pipeline(image_path)
