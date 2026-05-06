"""
Modulo de Visualizacion
========================
Funciones de graficado para mostrar resultados del pipeline de
restauracion de imagenes. Genera visualizaciones comparativas
para el analisis academico.

Todas las funciones aceptan un parametro return_fig (default False).
Si return_fig=True, retornan la figura matplotlib sin llamar plt.show(),
permitiendo embeber la figura en una GUI con FigureCanvasTkAgg.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def show_comparison(images: list, titles: list,
                    suptitle: str = '', save_name: str = None,
                    return_fig: bool = False):
    """
    Muestra varias imagenes lado a lado para comparacion.

    Parametros
    ----------
    images     : lista de ndarray (M, N)
    titles     : lista de strings con titulos
    suptitle   : titulo general de la figura
    save_name  : nombre del archivo para guardar (sin extension)
    return_fig : si True, retorna la figura sin mostrarla
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_frequency_spectrum(image: np.ndarray, title: str = 'Espectro',
                            save_name: str = None,
                            return_fig: bool = False):
    """
    Muestra la magnitud del espectro de Fourier en escala logaritmica.

    El espectro se centra usando fftshift para que las bajas frecuencias
    queden en el centro (convencion estandar).

    Visualizacion: 20 * log10(1 + |F(u,v)|) para comprimir el rango.
    """
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    magnitude = 20 * np.log10(1 + np.abs(F_shifted))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Imagen Espacial', fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(magnitude, cmap='inferno')
    axes[1].set_title(f'|F(u,v)| - {title}', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_kernel(kernel: np.ndarray, title: str = 'Kernel',
                save_name: str = None, return_fig: bool = False):
    """
    Visualiza un kernel como heatmap con valores numericos.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    kH, kW = kernel.shape
    if kH <= 15 and kW <= 15:
        for i in range(kH):
            for j in range(kW):
                val = kernel[i, j]
                color = 'white' if val < kernel.max() * 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color)

    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_difference_map(img1: np.ndarray, img2: np.ndarray,
                        title: str = 'Mapa de Diferencia',
                        save_name: str = None,
                        return_fig: bool = False):
    """
    Muestra el mapa de diferencia absoluta entre dos imagenes.

    |img1 - img2| amplificado para visibilidad.
    """
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img1, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Imagen 1', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Imagen 2', fontsize=11)
    axes[1].axis('off')

    diff_display = diff / (diff.max() + 1e-10)
    im = axes[2].imshow(diff_display, cmap='hot')
    axes[2].set_title(title, fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_restoration_pipeline(images_dict: dict,
                              save_name: str = None,
                              return_fig: bool = False):
    """
    Muestra el pipeline completo de restauracion paso a paso.

    Parametros
    ----------
    images_dict : dict ordenado con {nombre_etapa: imagen}
                  Ej: {'Original': img, 'Degradada': img_deg, ...}
    save_name   : nombre del archivo para guardar
    return_fig  : si True, retorna la figura sin mostrarla
    """
    n = len(images_dict)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, img) in enumerate(images_dict.items()):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r, c].set_title(name, fontsize=11, fontweight='bold')
        axes[r, c].axis('off')

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    fig.suptitle('Pipeline de Restauracion de Imagen',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_ocr_comparison(images: dict, ocr_results: dict,
                        ground_truth: str = None,
                        save_name: str = None,
                        return_fig: bool = False):
    """
    Visualiza las imagenes con los textos OCR detectados anotados.

    Parametros
    ----------
    images      : dict con {'original': img, 'degraded': img, 'restored': img}
    ocr_results : dict retornado por compare_ocr_results()
    ground_truth: texto real de la placa
    save_name   : nombre del archivo para guardar
    return_fig  : si True, retorna la figura sin mostrarla
    """
    stages = ['original', 'degraded', 'restored']
    labels = ['Original', 'Degradada', 'Restaurada']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, stage, label in zip(axes, stages, labels):
        ax.imshow(images[stage], cmap='gray', vmin=0, vmax=1)

        ocr = ocr_results[stage]
        text = ocr['text'] if ocr['text'] else '(no detectado)'
        conf = ocr['confidence']

        caption = f"OCR: \"{text}\"\nConfianza: {conf:.1f}%"
        if ground_truth and 'accuracy' in ocr_results:
            acc = ocr_results['accuracy'][stage]
            caption += f"\nPrecision: {acc:.0%}"

        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.12, caption, transform=ax.transAxes,
                fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                          alpha=0.8))
        ax.axis('off')

    if ground_truth:
        fig.suptitle(f'Comparacion OCR  |  Placa real: "{ground_truth}"',
                     fontsize=13, fontweight='bold', y=1.05)
    else:
        fig.suptitle('Comparacion de Resultados OCR',
                     fontsize=13, fontweight='bold', y=1.05)

    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()


def show_metrics_bar_chart(metrics: dict, save_name: str = None,
                           return_fig: bool = False):
    """
    Grafico de barras comparando metricas entre degradada y restaurada.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    categories = ['Degradada', 'Restaurada']
    colors = ['#e74c3c', '#2ecc71']

    for ax, metric_name, ylabel in zip(
            axes,
            ['mse', 'psnr', 'ssim'],
            ['MSE (menor = mejor)', 'PSNR dB (mayor = mejor)',
             'SSIM (mayor = mejor)']):

        vals = [metrics['degraded'][metric_name],
                metrics['restored'][metric_name]]

        bars = ax.bar(categories, vals, color=colors, width=0.5,
                      edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

        ax.set_title(metric_name.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle('Metricas de Calidad: Degradada vs Restaurada',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_name:
        _ensure_output_dir()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{save_name}.png'),
                    dpi=150, bbox_inches='tight')

    if return_fig:
        return fig
    plt.show()
