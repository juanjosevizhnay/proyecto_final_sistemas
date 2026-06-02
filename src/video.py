"""
Modulo de Procesamiento de Video
==================================
Extiende el pipeline de restauracion a videos cortos de trafico.

Idea general:
    1. Iterar frame por frame (con un paso configurable para no analizar todos)
    2. En cada frame, ejecutar el detector OCR para localizar regiones
       que parezcan placas vehiculares (heuristica alfanumerica)
    3. Recortar cada region detectada y agruparla con las demas
       detecciones del mismo plato a traves de los frames (similitud de
       texto) para evitar duplicar la misma placa
    4. Para cada placa unica, tomar el frame de mayor confianza y
       aplicar el mismo pipeline de restauracion del proyecto:
           Wiener (con la PSF h(x,y) elegida por el usuario)
            + post-procesado (sharpening / CLAHE / realce de bordes)
    5. Re-ejecutar OCR sobre la region restaurada para demostrar la
       ganancia de legibilidad y guardar todo en disco.

Este modulo NO reimplementa la matematica del proyecto; reutiliza:
    - src.restoration.wiener_filter           (deconvolucion principal)
    - src.postprocessing.postprocess_pipeline (sharpening + CLAHE)
    - src.ocr.detect_text_regions             (PaddleOCR via ONNX)

Por lo tanto, todo el reconocimiento de placas en video es una
APLICACION REAL del mismo modelo  g = f * h + n  estudiado en el
proyecto, ahora corriendo sobre frames secuenciales de un video.
"""

import os
import re
import json
import cv2
import numpy as np
from difflib import SequenceMatcher
from collections import Counter

from src.restoration import wiener_filter
from src.postprocessing import postprocess_pipeline
from src.ocr import detect_text_regions, OCR_AVAILABLE
from src.autotune import auto_restore, quick_kernel_candidates


# ---------------------------------------------------------------------------
#   HEURISTICA DE PLACAS
# ---------------------------------------------------------------------------

_PLATE_NORMALIZE_RE = re.compile(r'[^A-Z0-9]')


def normalize_plate(text: str) -> str:
    """Deja solo letras/digitos en mayusculas (ABC-1234 -> ABC1234)."""
    return _PLATE_NORMALIZE_RE.sub('', text.upper())


def looks_like_plate(text: str,
                     min_len: int = 4,
                     max_len: int = 8) -> bool:
    """
    Heuristica simple para decidir si un texto detectado parece una
    placa vehicular.

    Reglas:
      - Longitud alfanumerica entre [min_len, max_len]
      - Debe contener al menos un digito
      - Solo letras/digitos (sin signos de puntuacion adicionales tras
        la normalizacion)
    """
    if not text:
        return False
    clean = normalize_plate(text)
    if not (min_len <= len(clean) <= max_len):
        return False
    if not any(c.isdigit() for c in clean):
        return False
    return True


# ---------------------------------------------------------------------------
#   UTILIDADES DE FRAME
# ---------------------------------------------------------------------------

def _frame_to_gray_float(frame_bgr: np.ndarray) -> np.ndarray:
    """Convierte un frame BGR de OpenCV a escala de grises [0, 1]."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float64) / 255.0


def crop_bbox(image: np.ndarray, bbox, padding: float = 0.15):
    """
    Recorta una region rectangular alrededor del cuadrilatero detectado
    por OCR, con un padding relativo para no cortar caracteres.

    Parametros
    ----------
    image   : ndarray (H, W), imagen en escala de grises [0, 1]
    bbox    : lista de 4 puntos [(x, y), ...] del cuadrilatero
    padding : fraccion del ancho/alto a expandir hacia afuera

    Retorna
    -------
    crop      : ndarray con la region recortada (puede estar vacia)
    box_xyxy  : tupla (x0, y0, x1, y1) en coordenadas del frame original
    """
    h, w = image.shape
    xs = [int(round(p[0])) for p in bbox]
    ys = [int(round(p[1])) for p in bbox]

    x0 = max(0, min(xs))
    x1 = min(w, max(xs))
    y0 = max(0, min(ys))
    y1 = min(h, max(ys))

    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)

    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
#   AGREGACION DE DETECCIONES
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    """Similitud entre placas normalizadas (0.0 - 1.0)."""
    return SequenceMatcher(None, normalize_plate(a),
                           normalize_plate(b)).ratio()


def aggregate_detections(detections: list,
                         similarity_threshold: float = 0.6) -> list:
    """
    Agrupa detecciones que probablemente corresponden a la misma placa
    fisica, basado en similitud de texto.

    Una placa puede aparecer en multiples frames con pequenas variaciones
    en el texto leido por OCR (p.ej. 'ABC123', 'ABCI23', 'ABC-123').
    Esta funcion las une en un solo grupo.

    Retorna
    -------
    groups : lista de listas de detecciones
    """
    groups = []

    # Ordenar de mayor a menor confianza para que el "ancla" del grupo
    # sea la deteccion mas confiable
    sorted_dets = sorted(detections, key=lambda d: -d['confidence'])

    for det in sorted_dets:
        placed = False
        for grp in groups:
            anchor = grp[0]  # mejor deteccion del grupo
            if _text_similarity(det['text'], anchor['text']) >= similarity_threshold:
                grp.append(det)
                placed = True
                break
        if not placed:
            groups.append([det])

    return groups


def vote_plate_text(texts: list) -> str:
    """
    Votacion por caracter entre varias lecturas de la MISMA placa
    (mejora de eficacia en video).

    Una placa aparece en N frames; el OCR puede equivocarse en un
    caracter distinto en cada frame (p.ej. 'ABC123', 'ABG123', 'ABC723').
    En vez de quedarnos con una sola lectura, votamos posicion por
    posicion y nos quedamos con el caracter mas frecuente. Esto corrige
    errores puntuales que no se repiten entre frames.

    Estrategia:
        1. Normalizar todas las lecturas (solo A-Z0-9)
        2. Tomar la longitud mas frecuente L (la mayoria acierta el largo)
        3. Para cada posicion 0..L-1, elegir el caracter mas votado
           entre las lecturas de longitud L

    Retorna la placa consensuada (cadena), o '' si no hay lecturas.
    """
    norm = [normalize_plate(t) for t in texts]
    norm = [t for t in norm if t]
    if not norm:
        return ''

    # Longitud mas frecuente
    length_votes = Counter(len(t) for t in norm)
    target_len = length_votes.most_common(1)[0][0]

    same_len = [t for t in norm if len(t) == target_len]
    if not same_len:
        return norm[0]

    voted_chars = []
    for i in range(target_len):
        char_votes = Counter(t[i] for t in same_len)
        voted_chars.append(char_votes.most_common(1)[0][0])

    return ''.join(voted_chars)


# ---------------------------------------------------------------------------
#   PIPELINE DE VIDEO
# ---------------------------------------------------------------------------

def _restore_crop(crop: np.ndarray,
                  kernel: np.ndarray,
                  wiener_k: float,
                  apply_postprocess: bool = True) -> np.ndarray:
    """
    Aplica el pipeline de restauracion del proyecto a una region recortada.

    Pasos:
        1. Filtro de Wiener  (mismo H(u,v) del proyecto)
        2. Post-procesado    (unsharp mask + CLAHE + realce de bordes)
    """
    if crop.size == 0 or min(crop.shape) < 3:
        return crop

    restored = wiener_filter(crop, kernel, K=wiener_k)

    if apply_postprocess:
        restored = postprocess_pipeline(
            restored,
            sharpen=True, contrast='clahe', edge_enhance=True,
            sharpen_params={'sigma': 1.0, 'alpha': 1.5, 'kernel_size': 5},
            clahe_params={'clip_limit': 2.0, 'tile_size': 8},
            edge_params={'weight': 0.2},
        )

    return restored


def _ocr_text_from_detections(dets: list) -> tuple:
    """Resume una lista de detecciones OCR en (texto, confianza %)."""
    if not dets:
        return '', 0.0
    text = ' '.join(d['text'] for d in dets).upper().strip()
    confidence = float(np.mean([d['score'] for d in dets])) * 100.0
    return text, confidence


def process_video(video_path: str,
                  kernel: np.ndarray,
                  output_dir: str,
                  frame_step: int = 3,
                  wiener_k: float = 0.01,
                  apply_postprocess: bool = True,
                  similarity_threshold: float = 0.6,
                  auto_tune: bool = True,
                  max_frames: int = None,
                  progress_callback=None,
                  status_callback=None) -> dict:
    """
    Procesa un video completo y devuelve la lista de placas detectadas
    junto con sus versiones restauradas.

    Parametros
    ----------
    video_path           : ruta al archivo de video
    kernel               : ndarray (kH, kW), PSF asumida para la restauracion
    output_dir           : carpeta donde guardar resultados
    frame_step           : analizar 1 de cada 'frame_step' frames
    wiener_k             : parametro de regularizacion del filtro Wiener
    apply_postprocess    : aplicar sharpening / CLAHE / bordes tras Wiener
    similarity_threshold : umbral de similitud para agrupar detecciones
    auto_tune            : si True, por cada placa busca el mejor kernel y
                           el mejor K de Wiener (mejoras a + b) maximizando
                           la confianza del OCR. Si False, usa kernel/K fijos.
    max_frames           : limite de frames a leer (None = todo el video)
    progress_callback    : funcion(fraction: float) llamada periodicamente
    status_callback      : funcion(msg: str) para reportar estado

    Retorna
    -------
    results : dict con claves:
        'video_path'        : ruta del video
        'fps'               : frames por segundo
        'duration'          : duracion en segundos
        'total_frames'      : total de frames del video
        'analyzed_frames'   : frames realmente analizados
        'raw_detections'    : numero total de detecciones (antes de agrupar)
        'plates'            : lista de placas unicas detectadas, cada una:
            {
                'id'                  : int (1-indexed)
                'text'                : texto despues de restauracion
                'original_text'       : texto OCR sin restaurar
                'confidence'          : confianza despues (%)
                'original_confidence' : confianza antes (%)
                'frame'               : indice del mejor frame
                'timestamp'           : tiempo en segundos
                'detections_count'    : veces que aparecio
                'crop'                : ndarray con el recorte original
                'restored'            : ndarray con el recorte restaurado
                'box_xyxy'            : (x0, y0, x1, y1) en el frame
            }
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR no disponible. Instala rapidocr-onnxruntime."
        )

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"No se encontro el video: {video_path}")

    def _log(msg):
        if status_callback:
            status_callback(msg)

    def _progress(p):
        if progress_callback:
            progress_callback(max(0.0, min(1.0, p)))

    # ----------------------------------------------------------------------
    # Apertura del video
    # ----------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    _log(f"Video abierto: {total_frames} frames @ {fps:.1f}fps")

    # ----------------------------------------------------------------------
    # FASE 1: deteccion frame por frame
    # ----------------------------------------------------------------------
    all_detections = []
    analyzed = 0
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        if frame_idx % frame_step == 0:
            gray = _frame_to_gray_float(frame_bgr)
            regions = detect_text_regions(gray)

            for det in regions:
                if not looks_like_plate(det['text']):
                    continue

                crop, box = crop_bbox(gray, det['bbox'], padding=0.15)
                if crop.size == 0 or min(crop.shape) < 5:
                    continue

                all_detections.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0.0,
                    'text': det['text'],
                    'confidence': det['score'] * 100.0,
                    'crop': crop,
                    'box_xyxy': box,
                })

            analyzed += 1
            if analyzed % 5 == 0:
                _log(f"Frame {frame_idx}/{total_frames} - "
                     f"{len(all_detections)} detecciones")

            _progress(frame_idx / max(total_frames, 1) * 0.7)

        frame_idx += 1

    cap.release()
    _log(f"Fase 1 lista: {len(all_detections)} detecciones brutas en "
         f"{analyzed} frames analizados")

    # ----------------------------------------------------------------------
    # FASE 2: agregacion por similitud de texto
    # ----------------------------------------------------------------------
    _progress(0.72)
    groups = aggregate_detections(all_detections,
                                  similarity_threshold=similarity_threshold)
    _log(f"Fase 2 lista: {len(groups)} placas unicas tras agrupar")

    # ----------------------------------------------------------------------
    # FASE 3: restauracion + re-OCR solo del mejor frame de cada placa
    # ----------------------------------------------------------------------
    kernel_candidates = quick_kernel_candidates() if auto_tune else None

    plates = []
    for i, grp in enumerate(groups):
        _progress(0.75 + (i / max(len(groups), 1)) * 0.20)
        _log(f"Restaurando placa {i + 1}/{len(groups)}"
             + (" (auto-ajuste)" if auto_tune else ""))

        best = max(grp, key=lambda d: d['confidence'])

        if auto_tune:
            # Mejoras (a)+(b): buscar el mejor kernel y K maximizando la
            # confianza del OCR sobre el recorte restaurado
            res = auto_restore(best['crop'],
                               kernels=kernel_candidates,
                               apply_postprocess=apply_postprocess)
            restored = res['restored']
            used_kernel_name = res['kernel_name']
            used_k = res['K']
        else:
            restored = _restore_crop(best['crop'], kernel, wiener_k,
                                     apply_postprocess=apply_postprocess)
            used_kernel_name = 'fijo'
            used_k = wiener_k

        # Re-ejecutar OCR sobre la version restaurada
        rest_dets = detect_text_regions(restored)
        rest_text, rest_conf = _ocr_text_from_detections(rest_dets)

        # Mejora (d): votacion por caracter entre todas las lecturas de
        # esta placa (las de cada frame + la de la version restaurada)
        all_texts = [d['text'] for d in grp]
        if rest_text:
            all_texts.append(rest_text)
        voted_text = vote_plate_text(all_texts)

        # El texto final prioriza la votacion; si no hay, cae a la lectura
        # restaurada y, en ultimo caso, a la original
        final_text = voted_text or rest_text or best['text']

        # La confianza reportada es la del OCR restaurado (o la original
        # si la restauracion no produjo lectura)
        final_conf = rest_conf if rest_text else best['confidence']

        plates.append({
            'id': i + 1,
            'text': final_text,
            'voted_text': voted_text,
            'restored_text': rest_text,
            'original_text': best['text'],
            'confidence': final_conf,
            'original_confidence': best['confidence'],
            'frame': best['frame'],
            'timestamp': best['timestamp'],
            'detections_count': len(grp),
            'kernel_used': used_kernel_name,
            'k_used': used_k,
            'crop': best['crop'],
            'restored': restored,
            'box_xyxy': best['box_xyxy'],
        })

    # Ordenar por timestamp del mejor frame
    plates.sort(key=lambda p: p['timestamp'])
    for new_id, p in enumerate(plates, start=1):
        p['id'] = new_id

    # ----------------------------------------------------------------------
    # FASE 4: guardar resultados en disco
    # ----------------------------------------------------------------------
    _progress(0.96)
    _log("Guardando resultados...")
    os.makedirs(output_dir, exist_ok=True)
    _save_results(video_path, plates, output_dir,
                  fps=fps, duration=duration,
                  total_frames=total_frames, analyzed=analyzed,
                  raw_detections=len(all_detections))

    _progress(1.0)
    _log(f"Completado. {len(plates)} placas guardadas en {output_dir}")

    return {
        'video_path': video_path,
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames,
        'analyzed_frames': analyzed,
        'raw_detections': len(all_detections),
        'plates': plates,
        'output_dir': output_dir,
    }


# ---------------------------------------------------------------------------
#   GUARDADO EN DISCO
# ---------------------------------------------------------------------------

def _save_results(video_path, plates, output_dir,
                  fps, duration, total_frames, analyzed, raw_detections):
    """
    Persiste los resultados:
      - summary.txt   : reporte legible
      - plates.json   : datos estructurados (sin imagenes)
      - plate_NN_orig.png      : recorte original
      - plate_NN_restored.png  : recorte restaurado
      - plate_NN_comparison.png : original vs restaurado lado a lado
    """
    # ---- Reporte de texto ------------------------------------------------
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 64 + "\n")
        f.write("  REPORTE DE RECONOCIMIENTO DE PLACAS EN VIDEO\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Video             : {os.path.basename(video_path)}\n")
        f.write(f"Duracion          : {duration:.2f} s\n")
        f.write(f"FPS               : {fps:.1f}\n")
        f.write(f"Frames totales    : {total_frames}\n")
        f.write(f"Frames analizados : {analyzed}\n")
        f.write(f"Detecciones brutas: {raw_detections}\n")
        f.write(f"Placas unicas     : {len(plates)}\n")
        f.write("-" * 64 + "\n\n")

        for p in plates:
            f.write(f"Placa #{p['id']}: {p['text']}\n")
            f.write(f"  Mejor frame     : {p['frame']}  "
                    f"(t = {p['timestamp']:.2f}s)\n")
            f.write(f"  Apariciones     : {p['detections_count']}\n")
            f.write(f"  Antes (OCR)     : '{p['original_text']}'  "
                    f"({p['original_confidence']:.1f}%)\n")
            f.write(f"  Restaurada (OCR): '{p.get('restored_text', '')}'  "
                    f"({p['confidence']:.1f}%)\n")
            f.write(f"  Votada (final)  : '{p.get('voted_text', '')}'\n")
            f.write(f"  Kernel/K usados : {p.get('kernel_used', '-')}  "
                    f"K={p.get('k_used', '-')}\n")
            f.write("\n")

    # ---- JSON estructurado ------------------------------------------------
    json_path = os.path.join(output_dir, 'plates.json')
    serializable = {
        'video': os.path.basename(video_path),
        'fps': fps,
        'duration_seconds': duration,
        'total_frames': total_frames,
        'analyzed_frames': analyzed,
        'raw_detections': raw_detections,
        'plates': [
            {
                'id': p['id'],
                'text': p['text'],
                'voted_text': p.get('voted_text', ''),
                'restored_text': p.get('restored_text', ''),
                'original_text': p['original_text'],
                'confidence': round(p['confidence'], 2),
                'original_confidence': round(p['original_confidence'], 2),
                'frame': p['frame'],
                'timestamp': round(p['timestamp'], 3),
                'detections_count': p['detections_count'],
                'kernel_used': p.get('kernel_used', ''),
                'k_used': p.get('k_used', ''),
                'box_xyxy': list(p['box_xyxy']),
            }
            for p in plates
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    # ---- Imagenes por placa ----------------------------------------------
    for p in plates:
        prefix = f"plate_{p['id']:02d}"
        crop_uint8 = (np.clip(p['crop'], 0, 1) * 255).astype(np.uint8)
        rest_uint8 = (np.clip(p['restored'], 0, 1) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"{prefix}_orig.png"),
                    crop_uint8)
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_restored.png"),
                    rest_uint8)

        # Comparacion lado a lado (mismo alto, sumando anchos)
        h = max(crop_uint8.shape[0], rest_uint8.shape[0])
        c1 = _pad_to_height(crop_uint8, h)
        c2 = _pad_to_height(rest_uint8, h)
        separator = np.full((h, 4), 80, dtype=np.uint8)
        combo = np.hstack([c1, separator, c2])
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_comparison.png"),
                    combo)


def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Centra verticalmente una imagen rellenando con negro."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    pad_total = target_h - h
    pad_top = pad_total // 2
    pad_bot = pad_total - pad_top
    return cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0,
                              cv2.BORDER_CONSTANT, value=0)


# ---------------------------------------------------------------------------
#   PREVIEW
# ---------------------------------------------------------------------------

def read_first_frame(video_path: str) -> np.ndarray:
    """Lee el primer frame del video como imagen en grises [0, 1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame del video")
    return _frame_to_gray_float(frame)


def get_video_info(video_path: str) -> dict:
    """Metadata basica del video (sin leer todos los frames)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = (info['frame_count'] / info['fps']
                        if info['fps'] > 0 else 0.0)
    cap.release()
    return info
