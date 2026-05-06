"""
============================================================================
  RESTAURACION DE IMAGENES DE PLACAS VEHICULARES
  Interfaz Grafica - CustomTkinter
  -----------------------------------------------
  Proyecto Final - Sistemas Lineales y Senales
============================================================================
"""

import os
import sys
import threading
import traceback
import time
import numpy as np
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.kernels import gaussian_kernel, motion_blur_kernel
from src.degradation import degrade_image
from src.restoration import inverse_filter, wiener_filter, richardson_lucy
from src.postprocessing import postprocess_pipeline
from src.ocr import read_plate, compare_ocr_results
from src.metrics import compute_all_metrics
from src.visualization import (
    show_comparison,
    show_frequency_spectrum,
    show_kernel,
    show_difference_map,
    show_metrics_bar_chart,
)

# ---------------------------------------------------------------------------
#  Tema y colores
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DARK = "#0a0a14"
BG_PANEL = "#10101c"
BG_CARD = "#161625"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#2563eb"
TEXT_PRIMARY = "#e0e6ed"
TEXT_SECONDARY = "#8892a0"
TEXT_MUTED = "#505868"
SUCCESS = "#22c55e"
DANGER = "#ef4444"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Restauracion de Placas Vehiculares  |  Sistemas Lineales y Senales")
        self.geometry("1320x780")
        self.minsize(1100, 650)
        self.configure(fg_color=BG_DARK)

        self.original = None
        self.degraded = None
        self.restored_inverse = None
        self.restored_wiener = None
        self.restored_rl = None
        self.restored_final = None
        self.kernel = None
        self.metrics = None
        self.ocr_results = None
        self.image_path = None

        self._build_layout()

    # ======================================================================
    #  Layout
    # ======================================================================
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_sidebar()
        self._build_main_area()

    # --- Header -----------------------------------------------------------
    def _build_header(self):
        header = ctk.CTkFrame(self, height=54, fg_color=BG_PANEL,
                              corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="  RESTAURACION DE PLACAS VEHICULARES",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=16)

        ctk.CTkLabel(
            header,
            text="Proyecto Final  —  Sistemas Lineales y Senales   ",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY
        ).pack(side="right", padx=16)

    # --- Sidebar ----------------------------------------------------------
    def _build_sidebar(self):
        sidebar = ctk.CTkScrollableFrame(
            self, width=280, fg_color=BG_PANEL, corner_radius=0
        )
        sidebar.grid(row=1, column=0, sticky="ns")

        # -- Imagen --------------------------------------------------------
        self._section_label(sidebar, "IMAGEN")

        self.btn_load = ctk.CTkButton(
            sidebar, text="Cargar Imagen", command=self._load_image,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, height=36
        )
        self.btn_load.pack(padx=12, pady=(0, 6), fill="x")

        self.preview_frame = ctk.CTkFrame(sidebar, height=150,
                                           fg_color=BG_CARD, corner_radius=8)
        self.preview_frame.pack(padx=12, pady=(0, 4), fill="x")
        self.preview_frame.pack_propagate(False)

        self.lbl_preview = ctk.CTkLabel(
            self.preview_frame, text="Sin imagen",
            text_color=TEXT_MUTED, font=ctk.CTkFont(size=11)
        )
        self.lbl_preview.pack(expand=True)

        self.lbl_img_info = ctk.CTkLabel(
            sidebar, text="", font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY
        )
        self.lbl_img_info.pack(padx=12, pady=(0, 8))

        # -- Degradacion ---------------------------------------------------
        self._section_label(sidebar, "DEGRADACION")

        ctk.CTkLabel(sidebar, text="Tipo de blur",
                     text_color=TEXT_SECONDARY,
                     font=ctk.CTkFont(size=11)).pack(padx=14, anchor="w")
        self.var_blur = ctk.StringVar(value="gaussian")
        self.opt_blur = ctk.CTkOptionMenu(
            sidebar, variable=self.var_blur,
            values=["gaussian", "motion"],
            fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, width=250
        )
        self.opt_blur.pack(padx=12, pady=(0, 6), fill="x")

        self.lbl_sigma = ctk.CTkLabel(
            sidebar, text="Sigma / Tamano kernel: 2.0",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11)
        )
        self.lbl_sigma.pack(padx=14, anchor="w")
        self.slider_sigma = ctk.CTkSlider(
            sidebar, from_=0.5, to=5.0, number_of_steps=18,
            command=self._on_sigma_change,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT
        )
        self.slider_sigma.set(2.0)
        self.slider_sigma.pack(padx=12, pady=(0, 4), fill="x")

        self.lbl_ksize = ctk.CTkLabel(
            sidebar, text="Tamano kernel: 11",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11)
        )
        self.lbl_ksize.pack(padx=14, anchor="w")
        self.slider_ksize = ctk.CTkSlider(
            sidebar, from_=3, to=21, number_of_steps=9,
            command=self._on_ksize_change,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT
        )
        self.slider_ksize.set(11)
        self.slider_ksize.pack(padx=12, pady=(0, 8), fill="x")

        # -- Ruido ---------------------------------------------------------
        self._section_label(sidebar, "RUIDO")

        ctk.CTkLabel(sidebar, text="Tipo de ruido",
                     text_color=TEXT_SECONDARY,
                     font=ctk.CTkFont(size=11)).pack(padx=14, anchor="w")
        self.var_noise = ctk.StringVar(value="gaussian")
        self.opt_noise = ctk.CTkOptionMenu(
            sidebar, variable=self.var_noise,
            values=["gaussian", "salt_pepper", "none"],
            fg_color=BG_CARD, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, width=250
        )
        self.opt_noise.pack(padx=12, pady=(0, 6), fill="x")

        self.lbl_noise_std = ctk.CTkLabel(
            sidebar, text="Intensidad ruido: 0.010",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11)
        )
        self.lbl_noise_std.pack(padx=14, anchor="w")
        self.slider_noise = ctk.CTkSlider(
            sidebar, from_=0.001, to=0.08, number_of_steps=20,
            command=self._on_noise_change,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT
        )
        self.slider_noise.set(0.01)
        self.slider_noise.pack(padx=12, pady=(0, 8), fill="x")

        # -- Restauracion --------------------------------------------------
        self._section_label(sidebar, "RESTAURACION")

        self.lbl_wiener_k = ctk.CTkLabel(
            sidebar, text="Wiener K: 0.010",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11)
        )
        self.lbl_wiener_k.pack(padx=14, anchor="w")
        self.slider_wiener = ctk.CTkSlider(
            sidebar, from_=0.001, to=0.1, number_of_steps=20,
            command=self._on_wiener_change,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT
        )
        self.slider_wiener.set(0.01)
        self.slider_wiener.pack(padx=12, pady=(0, 4), fill="x")

        self.lbl_rl_iter = ctk.CTkLabel(
            sidebar, text="R-L Iteraciones: 30",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11)
        )
        self.lbl_rl_iter.pack(padx=14, anchor="w")
        self.slider_rl = ctk.CTkSlider(
            sidebar, from_=5, to=80, number_of_steps=15,
            command=self._on_rl_change,
            button_color=ACCENT, button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT
        )
        self.slider_rl.set(30)
        self.slider_rl.pack(padx=12, pady=(0, 8), fill="x")

        # -- OCR -----------------------------------------------------------
        self._section_label(sidebar, "OCR")

        ctk.CTkLabel(sidebar, text="Ground truth (opcional)",
                     text_color=TEXT_SECONDARY,
                     font=ctk.CTkFont(size=11)).pack(padx=14, anchor="w")
        self.entry_gt = ctk.CTkEntry(
            sidebar, placeholder_text="Ej: ABC-1234",
            fg_color=BG_CARD, border_color=TEXT_MUTED, height=32
        )
        self.entry_gt.pack(padx=12, pady=(0, 12), fill="x")

        # -- Boton procesar ------------------------------------------------
        self.btn_run = ctk.CTkButton(
            sidebar, text="PROCESAR", command=self._run_pipeline,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            height=42, font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_run.pack(padx=12, pady=(4, 4), fill="x")

        self.progress = ctk.CTkProgressBar(
            sidebar, progress_color=ACCENT, height=6
        )
        self.progress.pack(padx=12, pady=(4, 4), fill="x")
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(
            sidebar, text="Listo", text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=10)
        )
        self.lbl_status.pack(padx=12, pady=(0, 4))

        # -- Exportar ------------------------------------------------------
        self.btn_export = ctk.CTkButton(
            sidebar, text="Exportar resultados", command=self._export,
            fg_color=BG_CARD, hover_color="#1e1e30",
            border_width=1, border_color=TEXT_MUTED,
            height=34, font=ctk.CTkFont(size=11),
            state="disabled"
        )
        self.btn_export.pack(padx=12, pady=(4, 16), fill="x")

    # --- Main area (tabs) -------------------------------------------------
    def _build_main_area(self):
        self.tabs = ctk.CTkTabview(
            self, fg_color=BG_DARK,
            segmented_button_fg_color=BG_PANEL,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=BG_CARD,
            segmented_button_unselected_hover_color="#1a1a2e"
        )
        self.tabs.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        self.tab_results = self.tabs.add("Resultados")
        self.tab_methods = self.tabs.add("Metodos")
        self.tab_metrics = self.tabs.add("Metricas")
        self.tab_ocr = self.tabs.add("OCR")
        self.tab_spectrum = self.tabs.add("Espectro")

        for tab in (self.tab_results, self.tab_methods, self.tab_metrics,
                    self.tab_ocr, self.tab_spectrum):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self._placeholder_labels = {}
        for tab, text in [
            (self.tab_results, "Carga una imagen y presiona PROCESAR para ver resultados"),
            (self.tab_methods, "Comparacion de metodos de restauracion"),
            (self.tab_metrics, "Metricas de calidad de imagen"),
            (self.tab_ocr, "Resultados de reconocimiento OCR"),
            (self.tab_spectrum, "Espectros de frecuencia"),
        ]:
            lbl = ctk.CTkLabel(tab, text=text, text_color=TEXT_MUTED,
                               font=ctk.CTkFont(size=13))
            lbl.grid(row=0, column=0)
            self._placeholder_labels[tab] = lbl

    # ======================================================================
    #  Helpers
    # ======================================================================
    def _section_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=ACCENT
        ).pack(padx=14, pady=(12, 4), anchor="w")

    def _on_sigma_change(self, val):
        self.lbl_sigma.configure(text=f"Sigma / Tamano kernel: {val:.1f}")

    def _on_ksize_change(self, val):
        v = int(val)
        if v % 2 == 0:
            v += 1
        self.lbl_ksize.configure(text=f"Tamano kernel: {v}")

    def _on_noise_change(self, val):
        self.lbl_noise_std.configure(text=f"Intensidad ruido: {val:.3f}")

    def _on_wiener_change(self, val):
        self.lbl_wiener_k.configure(text=f"Wiener K: {val:.3f}")

    def _on_rl_change(self, val):
        self.lbl_rl_iter.configure(text=f"R-L Iteraciones: {int(val)}")

    def _set_status(self, text, color=TEXT_MUTED):
        self.lbl_status.configure(text=text, text_color=color)

    def _set_progress(self, value):
        self.progress.set(value)

    def _np_to_ctk_image(self, img_array, size=(240, 140)):
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).convert('L')
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                            size=size)

    def _embed_figure(self, parent, fig):
        for child in parent.winfo_children():
            child.destroy()

        fig.patch.set_facecolor('#0f0f1a')
        for ax in fig.get_axes():
            ax.set_facecolor('#0f0f1a')
            ax.tick_params(colors='#8892a0')
            ax.xaxis.label.set_color('#8892a0')
            ax.yaxis.label.set_color('#8892a0')
            ax.title.set_color('#e0e6ed')
            for spine in ax.spines.values():
                spine.set_edgecolor('#303040')
        if fig._suptitle:
            fig._suptitle.set_color('#e0e6ed')

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.configure(bg='#0f0f1a', highlightthickness=0)
        widget.grid(row=0, column=0, sticky="nsew")
        return canvas

    # ======================================================================
    #  Load image
    # ======================================================================
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen de placa",
            filetypes=[
                ("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Todos", "*.*")
            ]
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._set_status("Error: no se pudo cargar", DANGER)
            return

        self.image_path = path
        self.original = img.astype(np.float64) / 255.0

        ctk_img = self._np_to_ctk_image(self.original, size=(256, 150))
        self.lbl_preview.configure(image=ctk_img, text="")
        self.lbl_preview._ctk_image = ctk_img

        name = os.path.basename(path)
        h, w = self.original.shape
        self.lbl_img_info.configure(text=f"{name}  ({w}x{h} px)")
        self._set_status("Imagen cargada", SUCCESS)

    # ======================================================================
    #  Pipeline
    # ======================================================================
    def _run_pipeline(self):
        if self.original is None:
            self._set_status("Carga una imagen primero", DANGER)
            return

        self.btn_run.configure(state="disabled")
        self.btn_export.configure(state="disabled")
        self._set_status("Procesando...", ACCENT)
        self._set_progress(0)

        thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        thread.start()

    def _pipeline_worker(self):
        try:
            self._pipeline_logic()
        except Exception as e:
            msg = str(e)
            print("\n" + "=" * 60)
            print("ERROR EN EL PIPELINE:")
            print(traceback.format_exc())
            print("=" * 60)
            self.after(0, lambda m=msg: self._set_status(f"Error: {m}", DANGER))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal"))

    def _pipeline_logic(self):
        original = self.original
        pipeline_start = time.time()

        # --- Read params ---
        blur_type = self.var_blur.get()
        sigma = self.slider_sigma.get()
        ksize = int(self.slider_ksize.get())
        if ksize % 2 == 0:
            ksize += 1
        noise_type = self.var_noise.get()
        noise_intensity = self.slider_noise.get()
        wiener_k = self.slider_wiener.get()
        rl_iters = int(self.slider_rl.get())
        gt = self.entry_gt.get().strip() or None

        h, w = original.shape
        print("\n" + "=" * 60)
        print("  INICIO DEL PIPELINE")
        print("=" * 60)
        print(f"  Imagen: {self.image_path}")
        print(f"  Tamano: {w}x{h} px")
        print(f"  Parametros:")
        print(f"    Blur: {blur_type}, sigma={sigma:.1f}, kernel={ksize}x{ksize}")
        print(f"    Ruido: {noise_type}, intensidad={noise_intensity:.3f}")
        print(f"    Wiener K: {wiener_k:.3f}")
        print(f"    R-L iteraciones: {rl_iters}")
        print(f"    Ground truth: {gt or '(no definido)'}")
        print("-" * 60)

        # --- Step 1: Build kernel -----------------------------------------
        self.after(0, lambda: self._set_status("Construyendo kernel...", ACCENT))
        self.after(0, lambda: self._set_progress(0.05))
        t0 = time.time()

        if blur_type == 'gaussian':
            self.kernel = gaussian_kernel(ksize, sigma)
        else:
            self.kernel = motion_blur_kernel(ksize, 0.0)

        print(f"\n[1/8] Kernel construido ({time.time()-t0:.3f}s)")
        print(f"  Tipo: {blur_type}, forma: {self.kernel.shape}")
        print(f"  Suma: {self.kernel.sum():.6f}, max: {self.kernel.max():.6f}")

        # --- Step 2: Degrade ----------------------------------------------
        self.after(0, lambda: self._set_status("Degradando imagen (convolucion manual)...", ACCENT))
        self.after(0, lambda: self._set_progress(0.10))
        t0 = time.time()

        if noise_type == 'gaussian':
            noise_params = {'mean': 0.0, 'std': noise_intensity}
        elif noise_type == 'salt_pepper':
            noise_params = {'amount': noise_intensity}
        else:
            noise_params = {}

        self.degraded = degrade_image(
            original, self.kernel,
            noise_type=noise_type,
            noise_params=noise_params
        )

        print(f"\n[2/8] Imagen degradada ({time.time()-t0:.3f}s)")
        print(f"  Rango: [{self.degraded.min():.4f}, {self.degraded.max():.4f}]")
        print(f"  Media: {self.degraded.mean():.4f}, Std: {self.degraded.std():.4f}")

        # --- Step 3: Inverse filter (didactic) ----------------------------
        self.after(0, lambda: self._set_status("Filtro inverso (didactico)...", ACCENT))
        self.after(0, lambda: self._set_progress(0.35))
        t0 = time.time()

        self.restored_inverse = inverse_filter(self.degraded, self.kernel)

        print(f"\n[3/8] Filtro inverso ({time.time()-t0:.3f}s)")
        print(f"  Rango: [{self.restored_inverse.min():.4f}, {self.restored_inverse.max():.4f}]")

        # --- Step 4: Wiener filter ----------------------------------------
        self.after(0, lambda: self._set_status("Filtro de Wiener...", ACCENT))
        self.after(0, lambda: self._set_progress(0.45))
        t0 = time.time()

        self.restored_wiener = wiener_filter(self.degraded, self.kernel,
                                             K=wiener_k)

        print(f"\n[4/8] Filtro de Wiener ({time.time()-t0:.3f}s)")
        print(f"  K={wiener_k:.3f}")
        print(f"  Rango: [{self.restored_wiener.min():.4f}, {self.restored_wiener.max():.4f}]")

        # --- Step 5: Richardson-Lucy --------------------------------------
        self.after(0, lambda: self._set_status("Richardson-Lucy iterativo...", ACCENT))
        self.after(0, lambda: self._set_progress(0.55))
        t0 = time.time()

        self.restored_rl = richardson_lucy(self.degraded, self.kernel,
                                           iterations=rl_iters, use_fft=True)

        print(f"\n[5/8] Richardson-Lucy ({time.time()-t0:.3f}s)")
        print(f"  Iteraciones: {rl_iters}")
        print(f"  Rango: [{self.restored_rl.min():.4f}, {self.restored_rl.max():.4f}]")

        # --- Step 6: Post-processing --------------------------------------
        self.after(0, lambda: self._set_status("Post-procesamiento...", ACCENT))
        self.after(0, lambda: self._set_progress(0.70))
        t0 = time.time()

        self.restored_final = postprocess_pipeline(
            self.restored_wiener,
            sharpen=True, contrast='clahe', edge_enhance=True,
            sharpen_params={'sigma': 1.0, 'alpha': 1.5, 'kernel_size': 5},
            clahe_params={'clip_limit': 2.0, 'tile_size': 8},
            edge_params={'weight': 0.2},
        )

        print(f"\n[6/8] Post-procesamiento ({time.time()-t0:.3f}s)")
        print(f"  Sharpening + CLAHE + Bordes")
        print(f"  Rango: [{self.restored_final.min():.4f}, {self.restored_final.max():.4f}]")

        # --- Step 7: Metrics ----------------------------------------------
        self.after(0, lambda: self._set_status("Calculando metricas...", ACCENT))
        self.after(0, lambda: self._set_progress(0.80))
        t0 = time.time()

        self.metrics = compute_all_metrics(original, self.degraded,
                                           self.restored_final)

        m = self.metrics
        print(f"\n[7/8] Metricas calculadas ({time.time()-t0:.3f}s)")
        print(f"  {'':12s} {'Degradada':>12s} {'Restaurada':>12s}")
        print(f"  {'MSE':12s} {m['degraded']['mse']:12.6f} {m['restored']['mse']:12.6f}")
        print(f"  {'PSNR (dB)':12s} {m['degraded']['psnr']:12.2f} {m['restored']['psnr']:12.2f}")
        print(f"  {'SSIM':12s} {m['degraded']['ssim']:12.4f} {m['restored']['ssim']:12.4f}")

        # --- Step 8: OCR --------------------------------------------------
        self.after(0, lambda: self._set_status("Ejecutando OCR (PaddleOCR)...", ACCENT))
        self.after(0, lambda: self._set_progress(0.90))
        t0 = time.time()

        self.ocr_results = compare_ocr_results(
            original, self.degraded, self.restored_final,
            ground_truth=gt
        )

        print(f"\n[8/8] OCR completado ({time.time()-t0:.3f}s)")
        for stage, label in [('original', 'Original'),
                             ('degraded', 'Degradada'),
                             ('restored', 'Restaurada')]:
            r = self.ocr_results[stage]
            print(f"  {label:12s}: '{r['text']}' ({r['confidence']:.1f}%)")

        total = time.time() - pipeline_start
        print("\n" + "=" * 60)
        print(f"  PIPELINE COMPLETADO en {total:.2f}s")
        print("=" * 60 + "\n")

        # --- Done -- update UI on main thread -----------------------------
        self.after(0, lambda: self._set_progress(1.0))
        self.after(0, lambda: self._display_results(gt))

    # ======================================================================
    #  Display results in tabs
    # ======================================================================
    def _display_results(self, ground_truth):
        self._set_status("Completado", SUCCESS)
        self.btn_export.configure(state="normal")

        self._display_tab_results()
        self._display_tab_methods()
        self._display_tab_metrics()
        self._display_tab_ocr(ground_truth)
        self._display_tab_spectrum()

        self.tabs.set("Resultados")

    def _display_tab_results(self):
        fig = show_comparison(
            [self.original, self.degraded, self.restored_final],
            ['Original', 'Degradada', 'Restaurada (Final)'],
            suptitle='Resultado del Pipeline de Restauracion',
            return_fig=True
        )
        self._embed_figure(self.tab_results, fig)

    def _display_tab_methods(self):
        fig = show_comparison(
            [self.restored_inverse, self.restored_wiener,
             self.restored_rl, self.restored_final],
            ['Filtro Inverso', 'Filtro de Wiener',
             'Richardson-Lucy', 'Post-procesada'],
            suptitle='Comparacion de Metodos de Restauracion',
            return_fig=True
        )
        self._embed_figure(self.tab_methods, fig)

    def _display_tab_metrics(self):
        tab = self.tab_metrics
        for child in tab.winfo_children():
            child.destroy()

        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=1)

        # --- Metrics table card ---
        card = ctk.CTkFrame(tab, fg_color=BG_CARD, corner_radius=10)
        card.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        headers = ["Metrica", "Degradada", "Restaurada", "Mejora"]
        for col_idx, h in enumerate(headers):
            ctk.CTkLabel(
                card, text=h,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=ACCENT
            ).grid(row=0, column=col_idx, padx=20, pady=(10, 4))

        rows_data = []
        m = self.metrics
        for name, fmt, improvement_fn in [
            ('MSE', '.6f', lambda d, r: f"{d - r:+.6f}"),
            ('PSNR', '.2f', lambda d, r: f"{r - d:+.2f} dB"),
            ('SSIM', '.4f', lambda d, r: f"{r - d:+.4f}"),
        ]:
            val_d = m['degraded'][name.lower()]
            val_r = m['restored'][name.lower()]
            rows_data.append((name, f"{val_d:{fmt}}", f"{val_r:{fmt}}",
                              improvement_fn(val_d, val_r)))

        for row_idx, (metric, deg, rest, imp) in enumerate(rows_data, start=1):
            ctk.CTkLabel(card, text=metric,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=TEXT_PRIMARY
                         ).grid(row=row_idx, column=0, padx=20, pady=4)
            ctk.CTkLabel(card, text=deg,
                         font=ctk.CTkFont(size=12),
                         text_color=DANGER
                         ).grid(row=row_idx, column=1, padx=20, pady=4)
            ctk.CTkLabel(card, text=rest,
                         font=ctk.CTkFont(size=12),
                         text_color=SUCCESS
                         ).grid(row=row_idx, column=2, padx=20, pady=4)
            ctk.CTkLabel(card, text=imp,
                         font=ctk.CTkFont(size=12),
                         text_color=TEXT_SECONDARY
                         ).grid(row=row_idx, column=3, padx=20, pady=(4, 10))

        # --- Bar chart ---
        chart_frame = ctk.CTkFrame(tab, fg_color=BG_DARK, corner_radius=0)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        fig = show_metrics_bar_chart(self.metrics, return_fig=True)
        self._embed_figure(chart_frame, fig)

    def _display_tab_ocr(self, ground_truth):
        tab = self.tab_ocr
        for child in tab.winfo_children():
            child.destroy()

        tab.grid_rowconfigure(0, weight=0)
        tab.grid_rowconfigure(1, weight=1)

        # --- OCR results cards ---
        cards_frame = ctk.CTkFrame(tab, fg_color=BG_DARK)
        cards_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1)

        stages = [
            ('original', 'ORIGINAL', self.original),
            ('degraded', 'DEGRADADA', self.degraded),
            ('restored', 'RESTAURADA', self.restored_final),
        ]

        for col_idx, (key, title, img) in enumerate(stages):
            card = ctk.CTkFrame(cards_frame, fg_color=BG_CARD,
                                corner_radius=10)
            card.grid(row=0, column=col_idx, padx=6, pady=4, sticky="nsew")

            # Title
            title_color = TEXT_SECONDARY
            if key == 'restored':
                title_color = SUCCESS
            elif key == 'degraded':
                title_color = DANGER

            ctk.CTkLabel(
                card, text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=title_color
            ).pack(padx=10, pady=(10, 6))

            # Preview image
            ctk_img = self._np_to_ctk_image(img, size=(200, 100))
            img_lbl = ctk.CTkLabel(card, image=ctk_img, text="")
            img_lbl._ctk_image = ctk_img
            img_lbl.pack(padx=10, pady=4)

            # OCR text
            ocr = self.ocr_results[key]
            text = ocr['text'] if ocr['text'] else '(no detectado)'
            conf = ocr['confidence']

            ctk.CTkLabel(
                card, text=f'OCR: "{text}"',
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=TEXT_PRIMARY, wraplength=200
            ).pack(padx=10, pady=(6, 2))

            ctk.CTkLabel(
                card, text=f"Confianza: {conf:.1f}%",
                font=ctk.CTkFont(size=11),
                text_color=TEXT_SECONDARY
            ).pack(padx=10, pady=(0, 2))

            if ground_truth and 'accuracy' in self.ocr_results:
                acc = self.ocr_results['accuracy'][key]
                acc_color = SUCCESS if acc > 0.7 else (
                    "#eab308" if acc > 0.3 else DANGER)
                ctk.CTkLabel(
                    card, text=f"Precision: {acc:.0%}",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=acc_color
                ).pack(padx=10, pady=(0, 10))
            else:
                ctk.CTkLabel(card, text="").pack(pady=(0, 10))

        # Ground truth banner
        if ground_truth:
            ctk.CTkLabel(
                tab, text=f'Texto real de la placa: "{ground_truth}"',
                font=ctk.CTkFont(size=12),
                text_color=TEXT_SECONDARY
            ).grid(row=1, column=0, pady=10)

    def _display_tab_spectrum(self):
        tab = self.tab_spectrum
        for child in tab.winfo_children():
            child.destroy()

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        frame_orig = ctk.CTkFrame(tab, fg_color=BG_DARK)
        frame_orig.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        frame_orig.grid_columnconfigure(0, weight=1)
        frame_orig.grid_rowconfigure(0, weight=1)

        fig1 = show_frequency_spectrum(self.original, 'Original',
                                       return_fig=True)
        self._embed_figure(frame_orig, fig1)

        frame_deg = ctk.CTkFrame(tab, fg_color=BG_DARK)
        frame_deg.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        frame_deg.grid_columnconfigure(0, weight=1)
        frame_deg.grid_rowconfigure(0, weight=1)

        fig2 = show_frequency_spectrum(self.degraded, 'Degradada',
                                       return_fig=True)
        self._embed_figure(frame_deg, fig2)

    # ======================================================================
    #  Export
    # ======================================================================
    def _export(self):
        if self.original is None or self.restored_final is None:
            return

        out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(out_dir, exist_ok=True)

        show_comparison(
            [self.original, self.degraded, self.restored_final],
            ['Original', 'Degradada', 'Restaurada'],
            suptitle='Resultado Final',
            save_name='10_resultado_final', return_fig=True
        )
        show_comparison(
            [self.restored_inverse, self.restored_wiener,
             self.restored_rl, self.restored_final],
            ['Inverso', 'Wiener', 'R-L', 'Final'],
            suptitle='Metodos', save_name='05_comparacion_restauracion',
            return_fig=True
        )
        show_metrics_bar_chart(self.metrics, save_name='08_metricas',
                               return_fig=True)
        show_frequency_spectrum(self.original, 'Original',
                                save_name='03_espectro_original',
                                return_fig=True)
        show_frequency_spectrum(self.degraded, 'Degradada',
                                save_name='04_espectro_degradada',
                                return_fig=True)
        show_kernel(self.kernel, 'Kernel de Degradacion',
                    save_name='01_kernel', return_fig=True)
        show_difference_map(self.degraded, self.restored_final,
                            save_name='07_diferencia', return_fig=True)

        plt.close('all')

        self._set_status(f"Exportado a {os.path.abspath(out_dir)}", SUCCESS)


# ======================================================================
#  Entry point
# ======================================================================
if __name__ == '__main__':
    app = App()
    app.mainloop()
