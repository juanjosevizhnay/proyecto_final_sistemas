"""
Modulo de OCR (Reconocimiento Optico de Caracteres)
=====================================================
Utiliza RapidOCR (PaddleOCR sobre ONNX Runtime) para leer el texto
de las placas vehiculares en diferentes etapas del pipeline:

    1. Imagen original   -> lectura baseline (debe leer bien)
    2. Imagen degradada  -> lectura mala (demuestra el problema)
    3. Imagen restaurada -> lectura mejorada (demuestra la solucion)

La comparacion de resultados OCR es la validacion practica de que
la deconvolucion realmente mejora la legibilidad de las placas.

Motor OCR: PaddleOCR via rapidocr-onnxruntime (no requiere PaddlePaddle).
"""

import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    _engine = RapidOCR()
    OCR_AVAILABLE = True
    print("[OCR] RapidOCR (PaddleOCR ONNX) cargado correctamente")
except ImportError:
    _engine = None
    OCR_AVAILABLE = False
    print("[OCR] ADVERTENCIA: rapidocr-onnxruntime no instalado. "
          "Ejecuta: pip install rapidocr-onnxruntime")
except Exception as _e:
    _engine = None
    OCR_AVAILABLE = False
    print(f"[OCR] Error al inicializar RapidOCR: {_e}")


def read_plate(image: np.ndarray) -> dict:
    """
    Lee el texto de una imagen de placa vehicular usando RapidOCR.

    Parametros
    ----------
    image : ndarray (M, N), imagen en escala de grises, rango [0, 1]

    Retorna
    -------
    result : dict con claves:
        'text'       : texto detectado (limpio)
        'raw_text'   : texto crudo (todos los bloques)
        'confidence' : confianza promedio (0-100)
    """
    if not OCR_AVAILABLE or _engine is None:
        print("[OCR] Motor no disponible, retornando vacio")
        return {
            'text': '[OCR no disponible]',
            'raw_text': '',
            'confidence': 0.0
        }

    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    print(f"[OCR] Procesando imagen {img_uint8.shape[1]}x{img_uint8.shape[0]}, "
          f"rango [{img_uint8.min()}-{img_uint8.max()}]")

    try:
        result, elapse = _engine(img_uint8)
    except Exception as e:
        print(f"[OCR] Error en el motor: {e}")
        return {
            'text': '[Error en OCR]',
            'raw_text': '',
            'confidence': 0.0
        }

    if result is None or len(result) == 0:
        print("[OCR] No se detecto ningun texto")
        return {
            'text': '',
            'raw_text': '',
            'confidence': 0.0
        }

    # result es una lista de [bbox, texto, score]
    texts = []
    scores = []
    for detection in result:
        bbox, text, score = detection
        score = float(score) if score is not None else 0.0
        texts.append(text)
        scores.append(score)
        print(f"[OCR]   Deteccion: '{text}' (confianza: {score * 100:.1f}%)")

    full_text = ' '.join(texts)
    avg_confidence = float(np.mean(scores) * 100) if scores else 0.0

    print(f"[OCR] Resultado final: '{full_text}' "
          f"(confianza promedio: {avg_confidence:.1f}%)")

    return {
        'text': full_text,
        'raw_text': '\n'.join(texts),
        'confidence': avg_confidence
    }


def character_accuracy(detected: str, ground_truth: str) -> float:
    """
    Calcula la tasa de acierto de caracteres entre el texto detectado
    y el texto real de la placa.

    Metodo: compara caracter por caracter (alineado por posicion).
    Ignora espacios y diferencias de mayusculas/minusculas.

    Retorna un valor entre 0.0 (nada correcto) y 1.0 (todo correcto).
    """
    det = detected.upper().replace(' ', '').replace('-', '')
    gt = ground_truth.upper().replace(' ', '').replace('-', '')

    if len(gt) == 0:
        return 1.0 if len(det) == 0 else 0.0

    max_len = max(len(det), len(gt))
    matches = 0
    for i in range(min(len(det), len(gt))):
        if det[i] == gt[i]:
            matches += 1

    return matches / max_len


def compare_ocr_results(original: np.ndarray,
                        degraded: np.ndarray,
                        restored: np.ndarray,
                        ground_truth: str = None) -> dict:
    """
    Ejecuta OCR en las tres imagenes y genera un reporte comparativo.

    Parametros
    ----------
    original     : imagen original (limpia)
    degraded     : imagen degradada
    restored     : imagen restaurada
    ground_truth : texto real de la placa (opcional, para calcular accuracy)

    Retorna
    -------
    results : dict con resultados de cada etapa y comparacion
    """
    print("\n[OCR] === Leyendo imagen ORIGINAL ===")
    res_original = read_plate(original)

    print("\n[OCR] === Leyendo imagen DEGRADADA ===")
    res_degraded = read_plate(degraded)

    print("\n[OCR] === Leyendo imagen RESTAURADA ===")
    res_restored = read_plate(restored)

    results = {
        'original': res_original,
        'degraded': res_degraded,
        'restored': res_restored,
    }

    if ground_truth:
        results['accuracy'] = {
            'original': character_accuracy(res_original['text'], ground_truth),
            'degraded': character_accuracy(res_degraded['text'], ground_truth),
            'restored': character_accuracy(res_restored['text'], ground_truth),
        }
        print(f"\n[OCR] Ground truth: '{ground_truth}'")
        for stage in ('original', 'degraded', 'restored'):
            print(f"[OCR]   {stage}: accuracy={results['accuracy'][stage]:.0%}")

    return results


def print_ocr_report(results: dict, ground_truth: str = None):
    """
    Imprime un reporte formateado de los resultados OCR.
    """
    print("\n" + "=" * 60)
    print("       REPORTE DE RECONOCIMIENTO OCR (PaddleOCR ONNX)")
    print("=" * 60)

    for stage, label in [('original', 'IMAGEN ORIGINAL'),
                         ('degraded', 'IMAGEN DEGRADADA'),
                         ('restored', 'IMAGEN RESTAURADA')]:
        r = results[stage]
        print(f"\n  {label}:")
        print(f"    Texto detectado : '{r['text']}'")
        print(f"    Confianza       : {r['confidence']:.1f}%")

        if ground_truth and 'accuracy' in results:
            acc = results['accuracy'][stage]
            print(f"    Precision chars : {acc:.1%}")

    if ground_truth:
        print(f"\n  Texto real de la placa: '{ground_truth}'")

    print("\n" + "=" * 60)
