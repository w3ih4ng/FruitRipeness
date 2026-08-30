"""Local desktop comparison UI. Launch: python -m fruitripeness.ui"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .processing import process_image
from .pdf_export import (
    METRICS_PDF_FIELDS,
    PREDICTION_PDF_FIELDS,
    SURFACE_PDF_FIELDS,
    export_report_pdf,
)
from .media import (
    VIDEO_EXTENSIONS,
    VIDEO_RESULT_FIELDS,
    process_video,
    run_live_camera,
)

from .ui_core import (
    LABELS,
    CLASSIFIER_LABELS,
    METHODS,
    HYBRID_VARIANTS,
    SURFACE_RESULT_FIELDS,
    choice_labels,
    run_label,
    comparison_methods,
    METRIC_FIELDS,
    RESULT_FIELDS,
    collect_inputs,
    comparison_sheet,
    discover_runs,
    evaluation_rows,
    evaluation_sheet,
    export_csv,
    export_png,
    check_destination,
    infer,
    load_model,
    prediction_card,
    run_batch,
    run_surface_batch,
)


class NamedCombobox(ttk.Combobox):
    """Show friendly labels while the application keeps exact, stable IDs."""

    def __init__(
        self,
        parent,
        *,
        textvariable,
        values=(),
        label=str,
        **kwargs,
    ):
        self.key_variable = textvariable
        self.display_variable = tk.StringVar(master=parent)
        self.format_label = label
        self.labels = {}
        self.syncing = False

        super().__init__(
            parent,
            textvariable=self.display_variable,
            **kwargs,
        )

        self.key_trace = self.key_variable.trace_add(
            "write",
            self.show_key,
        )

        self.display_trace = self.display_variable.trace_add(
            "write",
            self.select_key,
        )

        self.set_choices(values)

    def set_choices(self, keys):
        self.labels = choice_labels(
            keys,
            self.format_label,
        )

        self.configure(
            values=list(self.labels.values())
        )

        self.show_key()

    def show_key(self, *_):
        self.syncing = True

        try:
            self.display_variable.set(
                self.labels.get(
                    self.key_variable.get(),
                    "",
                )
            )
        finally:
            self.syncing = False

    def select_key(self, *_):
        if not self.syncing:
            key = next(
                (
                    key
                    for key, value in self.labels.items()
                    if value == self.display_variable.get()
                ),
                "",
            )

            if key:
                self.key_variable.set(key)

    def destroy(self):
        try:
            self.key_variable.trace_remove(
                "write",
                self.key_trace,
            )

            self.display_variable.trace_remove(
                "write",
                self.display_trace,
            )
        except Exception:
            pass

        super().destroy()


class App:

    def __init__(self, root: tk.Tk, outputs: Path):

        self.root = root
        self.busy = False
        self.closing = False
        self.worker_error = ""

        self.cancel = threading.Event()
        self.events = queue.Queue(maxsize=16)

        # =========================================================
        # Input / results
        # =========================================================

        self.inputs = []
        self.protected = []

        self.result_protected = []
        self.rows = []
        self.cards = {}
        self.card_source = ""
        self.preview_redraw_after = None
        self.results_expanded = False

        # =========================================================
        # Validation
        # =========================================================

        self.eval_rows = []
        self.eval_image = None
        self.eval_protected = []

        # =========================================================
        # Final test
        # =========================================================

        self.test_runs = []
        self.test_rows = []
        self.test_image = None
        self.test_protected = []

        # =========================================================
        # Surface analysis
        # =========================================================

        self.surface_rows = []
        self.surface_preview = None
        self.surface_source = ""

        # =========================================================
        # Camera / video
        # =========================================================

        self.camera_running = False
        self.camera_latest = None
        self.camera_latest_row = None
        self.video_source = None
        self.video_output = None
        self.video_rows = []
        self.video_preview = None
        self.video_protected = []
        self.task_kind = ""

        # =========================================================
        # Saved models
        # =========================================================

        self.found = {}
        self.discovery_warnings = []

        # =========================================================
        # Controls
        # =========================================================

        self.buttons = []

        self.outputs = Path(outputs).resolve()

        self.status = tk.StringVar(
            value="Choose images or a folder to begin."
        )

        self.input_note = tk.StringVar(
            value="No images selected"
        )

        self.recursive = tk.BooleanVar(
            value=False
        )

        self.method = tk.StringVar(
            value="hybrid"
        )

        self.hybrid_variant = tk.StringVar(
            value="hybrid"
        )

        self.backend = tk.StringVar(
            value="shared_cnn"
        )

        self.cnn_run = tk.StringVar()

        self.scope = tk.StringVar(
            value="overall"
        )

        self.test_scope = tk.StringVar(
            value="overall"
        )

        self.test_note = tk.StringVar(
            value=(
                "Open a completed outputs/final_test/<timestamp> "
                "report. This tab reads results only; it does not "
                "evaluate images."
            )
        )

        # =========================================================
        # Surface Analysis has its OWN segmentation selection.
        # =========================================================

        self.surface_method = tk.StringVar(
            value="kmeans"
        )

        self.media_method = tk.StringVar(
            value="hybrid_refined"
        )

        self.camera_index = tk.StringVar(
            value="0"
        )

        self.video_note = tk.StringVar(
            value="No video selected"
        )

        self.media_summary = tk.StringVar(
            value="Camera and video processing are idle."
        )

        # =========================================================
        # IMPORTANT FIX
        #
        # refresh_runs(), selected_runs(), update_scopes(),
        # change_backend(), and select_cnn_run() all use
        # self.run_vars.
        #
        # The original code did not create run_vars before
        # refresh_runs() was called.
        # =========================================================

        self.run_vars = {
            method: tk.StringVar()
            for method in LABELS
        }

        # =========================================================
        # Window
        # =========================================================

        self.root.title(
            "Fruit Ripeness | Comparison Studio"
        )

        width = min(
            1280,
            max(
                800,
                root.winfo_screenwidth() - 80,
            ),
        )

        height = min(
            850,
            max(
                600,
                root.winfo_screenheight() - 100,
            ),
        )

        self.root.geometry(
            f"{width}x{height}"
        )

        self.root.minsize(
            min(1000, width),
            min(700, height),
        )

        self.root.configure(
            bg="#edf2ee"
        )

        # =========================================================
        # Style
        # =========================================================

        style = ttk.Style(root)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            ".",
            font=("Segoe UI", 10),
        )

        style.configure(
            "TFrame",
            background="#f6f8f6",
        )

        style.configure(
            "TLabel",
            background="#f6f8f6",
            foreground="#243b30",
        )

        style.configure(
            "TButton",
            padding=(12, 7),
        )

        style.configure(
            "Accent.TButton",
            background="#2f6547",
            foreground="white",
        )

        style.configure(
            "Treeview",
            rowheight=27,
        )

        # =========================================================
        # Header
        # =========================================================

        header = tk.Frame(
            root,
            bg="#214737",
            padx=18,
            pady=10,
        )

        header.pack(
            fill="x"
        )

        tk.Label(
            header,
            text="FRUIT RIPENESS",
            font=("Segoe UI", 19, "bold"),
            bg="#214737",
            fg="white",
        ).pack(
            side="left"
        )

        tk.Label(
            header,
            text="Comparison Studio  /  Mode A",
            font=("Segoe UI", 12),
            bg="#214737",
            fg="#cee3d5",
        ).pack(
            side="right"
        )

        # =========================================================
        # Classifier bar
        # =========================================================

        classifier_bar = ttk.Frame(
            root,
            padding=(12, 4),
        )

        classifier_bar.pack(
            fill="x"
        )

        ttk.Label(
            classifier_bar,
            text="Classifier:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.backend_box = NamedCombobox(
            classifier_bar,
            textvariable=self.backend,
            values=CLASSIFIER_LABELS,
            label=CLASSIFIER_LABELS.__getitem__,
            state="readonly",
            width=34,
        )

        self.backend_box.pack(
            side="left",
            padx=(0, 12),
        )

        self.backend_box.bind(
            "<<ComboboxSelected>>",
            self.change_backend,
        )

        ttk.Label(
            classifier_bar,
            text="CNN trained on:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.cnn_box = NamedCombobox(
            classifier_bar,
            textvariable=self.cnn_run,
            label=run_label,
            state="disabled",
            width=34,
        )

        self.cnn_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.cnn_box.bind(
            "<<ComboboxSelected>>",
            self.select_cnn_run,
        )

        self.button(
            classifier_bar,
            "Refresh models",
            self.refresh_runs,
        )

        # =========================================================
        # Notebook
        # =========================================================

        self.tabs = ttk.Notebook(
            root
        )

        self.tabs.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=6,
        )

        self.input_tab = ttk.Frame(
            self.tabs,
            padding=8,
        )

        self.eval_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.test_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.surface_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.media_tab = ttk.Frame(
            self.tabs,
            padding=12,
        )

        self.tabs.add(
            self.input_tab,
            text="  Images & folders  ",
        )

        self.tabs.add(
            self.eval_tab,
            text="  Saved validation  ",
        )

        self.tabs.add(
            self.test_tab,
            text="  Saved final Test  ",
        )

        self.tabs.add(
            self.surface_tab,
            text="  Surface analysis  ",
        )

        self.tabs.add(
            self.media_tab,
            text="  Camera & video  ",
        )

        # =========================================================
        # Build tabs
        # =========================================================

        self.build_inputs()
        self.build_evaluation()
        self.build_test_evaluation()
        self.build_surface_analysis()
        self.build_media()

        # =========================================================
        # Footer
        # =========================================================

        footer = ttk.Frame(
            root,
            padding=(15, 6),
        )

        footer.pack(
            fill="x"
        )

        self.progress = ttk.Progressbar(
            footer,
            mode="determinate",
            length=190,
        )

        self.progress.pack(
            side="right"
        )

        ttk.Label(
            footer,
            textvariable=self.status,
            wraplength=950,
        ).pack(
            side="left"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

        # IMPORTANT:
        # run_vars now exists before refresh_runs().
        self.refresh_runs(
            fallback=True
        )

        self.root.after(
            100,
            self.poll,
        )

    # =============================================================
    # GENERAL UI HELPERS
    # =============================================================

    def button(
        self,
        parent,
        text,
        command,
        *,
        accent=False,
    ):
        button = ttk.Button(
            parent,
            text=text,
            command=command,
            style=(
                "Accent.TButton"
                if accent
                else "TButton"
            ),
        )

        button.pack(
            side="left",
            padx=(0, 7),
            pady=3,
        )

        self.buttons.append(
            button
        )

        return button

    def scroll_image(self, parent):

        frame = ttk.Frame(parent)

        canvas = tk.Canvas(
            frame,
            background="#edf2ee",
            highlightthickness=0,
        )

        ybar = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=canvas.yview,
        )

        xbar = ttk.Scrollbar(
            frame,
            orient="horizontal",
            command=canvas.xview,
        )

        canvas.configure(
            yscrollcommand=ybar.set,
            xscrollcommand=xbar.set,
        )

        frame.columnconfigure(
            0,
            weight=1,
        )

        frame.rowconfigure(
            0,
            weight=1,
        )

        canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        ybar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        xbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        return frame, canvas

    # =============================================================
    # INPUT TAB
    # =============================================================

    def build_inputs(self):

        controls = ttk.Frame(
            self.input_tab
        )

        controls.pack(
            fill="x"
        )

        self.button(
            controls,
            "Choose image(s)",
            self.choose_files,
        )

        self.button(
            controls,
            "Choose folder",
            self.choose_folder,
        )

        self.recurse_check = ttk.Checkbutton(
            controls,
            text="Include subfolders when choosing a folder",
            variable=self.recursive,
        )

        self.recurse_check.pack(
            side="left",
            padx=10,
        )

        ttk.Label(
            self.input_tab,
            textvariable=self.input_note,
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(3, 8),
        )

        actions = ttk.Frame(
            self.input_tab
        )

        actions.pack(
            fill="x"
        )

        self.method_box = NamedCombobox(
            actions,
            textvariable=self.method,
            values=["baseline", *METHODS],
            label=LABELS.__getitem__,
            state="readonly",
            width=35,
        )

        self.method_box.pack(
            side="left",
            padx=(0, 8),
        )

        self.button(
            actions,
            "Run selected method",
            lambda: self.start(
                [self.method.get()]
            ),
            accent=True,
        )

        self.button(
            actions,
            "Compare all six",
            lambda: self.start(
                comparison_methods(
                    self.hybrid_variant.get()
                )
            ),
            accent=True,
        )

        self.cancel_button = ttk.Button(
            actions,
            text="Cancel",
            command=self.cancel_work,
            state="disabled",
        )

        self.cancel_button.pack(
            side="left"
        )

        variants = ttk.Frame(
            self.input_tab
        )

        variants.pack(
            fill="x",
            pady=(3, 0),
        )

        ttk.Label(
            variants,
            text="Hybrid used by Compare all six:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.hybrid_box = NamedCombobox(
            variants,
            textvariable=self.hybrid_variant,
            values=HYBRID_VARIANTS,
            label=LABELS.__getitem__,
            state="readonly",
            width=35,
        )

        self.hybrid_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.button(
            variants,
            "Compare hybrid versions",
            lambda: self.start(
                list(HYBRID_VARIANTS)
            ),
        )

        ttk.Label(
            self.input_tab,
            text=(
                "No ground truth is assumed for these inputs. "
                "Scores are uncalibrated; masks are heuristics. "
                "Original-Test content is blocked."
            ),
            wraplength=900,
        ).pack(
            anchor="w",
            pady=(8, 7),
        )

        panes = ttk.Panedwindow(
            self.input_tab,
            orient="vertical",
        )

        self.input_panes = panes

        panes.pack(
            fill="both",
            expand=True,
        )

        preview_frame, self.preview_canvas = (
            self.scroll_image(panes)
        )

        panes.add(
            preview_frame,
            weight=8,
        )

        self.preview_canvas.bind(
            "<Configure>",
            self.schedule_preview_redraw,
        )

        table_frame = ttk.Frame(
            panes
        )

        panes.add(
            table_frame,
            weight=0,
        )

        self.results_table_frame = table_frame

        bar = ttk.Frame(
            table_frame
        )

        bar.pack(
            fill="x"
        )

        self.button(
            bar,
            "Preview selected image",
            self.preview_selected,
        )

        self.button(
            bar,
            "Export predictions CSV",
            self.save_results,
        )

        self.button(
            bar,
            "Export preview PNG",
            self.save_preview,
        )

        self.button(
            bar,
            "Export report PDF",
            self.save_prediction_pdf,
        )

        self.button(
            bar,
            "Expand preview",
            self.open_preview_window,
        )

        self.results_toggle_button = self.button(
            bar,
            "Show result details",
            self.toggle_results,
        )

        table_area = ttk.Frame(
            table_frame
        )

        self.results_table_area = table_area

        table_area.pack(
            fill="both",
            expand=True,
        )

        columns = [
            "source",
            "method",
            "status",
            "predicted_stage",
            "processing_ms",
            "prediction_ms",
            "error",
        ]

        headings = [
            "Source",
            "Method",
            "Status",
            "Prediction",
            "Process ms",
            "Predict ms",
            "Error",
        ]

        self.table = ttk.Treeview(
            table_area,
            columns=columns,
            show="headings",
            height=5,
            selectmode="browse",
        )

        for name, heading in zip(
            columns,
            headings,
        ):
            self.table.heading(
                name,
                text=heading,
            )

            if name in ["source", "error"]:
                width = 330
            elif name == "method":
                width = 265
            else:
                width = 100

            self.table.column(
                name,
                width=width,
                stretch=False,
            )

        ybar = ttk.Scrollbar(
            table_area,
            orient="vertical",
            command=self.table.yview,
        )

        xbar = ttk.Scrollbar(
            table_area,
            orient="horizontal",
            command=self.table.xview,
        )

        self.table.configure(
            yscrollcommand=ybar.set,
            xscrollcommand=xbar.set,
        )

        table_area.columnconfigure(
            0,
            weight=1,
        )

        table_area.rowconfigure(
            0,
            weight=1,
        )

        self.table.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        ybar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        xbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.table.bind(
            "<<TreeviewSelect>>",
            self.show_row_detail,
        )

        self.table.bind(
            "<Double-1>",
            lambda event: self.preview_selected(),
        )

        # Keep detailed rows available without allowing them to consume most
        # of the image workspace on startup.
        self.results_table_area.pack_forget()

    # =============================================================
    # VALIDATION TAB
    # =============================================================

    def build_evaluation(self):

        controls = ttk.Frame(
            self.eval_tab
        )

        controls.pack(
            fill="x"
        )

        ttk.Label(
            controls,
            text="Fruit scope:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.scope_box = ttk.Combobox(
            controls,
            textvariable=self.scope,
            values=["overall"],
            state="readonly",
            width=16,
        )

        self.scope_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.button(
            controls,
            "Load selected runs",
            self.load_evaluation,
            accent=True,
        )

        self.button(
            controls,
            "Export metrics CSV",
            self.save_metrics,
        )

        self.button(
            controls,
            "Export comparison PNG",
            self.save_evaluation,
        )

        self.button(
            controls,
            "Export validation PDF",
            self.save_validation_pdf,
        )

        ttk.Label(
            self.eval_tab,
            text=(
                "Reads saved validation metrics only. "
                "Baseline is a reference; hybrid versions are "
                "separate experiments. No retraining or Test evaluation."
            ),
            wraplength=900,
        ).pack(
            anchor="w",
            pady=10,
        )

        frame, self.eval_canvas = (
            self.scroll_image(self.eval_tab)
        )

        frame.pack(
            fill="both",
            expand=True,
        )

    # =============================================================
    # FINAL TEST TAB
    # =============================================================

    def build_test_evaluation(self):

        controls = ttk.Frame(
            self.test_tab
        )

        controls.pack(
            fill="x"
        )

        self.button(
            controls,
            "Open Test report",
            self.open_test_report,
            accent=True,
        )

        ttk.Label(
            controls,
            text="Fruit scope:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.test_scope_box = ttk.Combobox(
            controls,
            textvariable=self.test_scope,
            values=["overall"],
            state="readonly",
            width=14,
        )

        self.test_scope_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.button(
            controls,
            "Load scope",
            self.load_test_view,
        )

        exports = ttk.Frame(
            self.test_tab
        )

        exports.pack(
            fill="x"
        )

        self.button(
            exports,
            "Export Test metrics CSV",
            self.save_test_metrics,
        )

        self.button(
            exports,
            "Export Test comparison PNG",
            self.save_test_image,
        )

        self.button(
            exports,
            "Export Test PDF",
            self.save_test_pdf,
        )

        ttk.Label(
            self.test_tab,
            textvariable=self.test_note,
            wraplength=1100,
        ).pack(
            anchor="w",
            pady=8,
        )

        frame, self.test_canvas = (
            self.scroll_image(self.test_tab)
        )

        frame.pack(
            fill="both",
            expand=True,
        )

    # =============================================================
    # SURFACE ANALYSIS TAB
    # =============================================================

    def build_surface_analysis(self):

        ttk.Label(
            self.surface_tab,
            text=(
                'Uses the images selected on the "Images & folders" tab. '
                "Preprocessing (denoise + contrast stretch), "
                "blemish/damage quantification, heuristic surface-quality "
                "grading, object detection, "
                "and pixel-based sizing & no classifier or label involved."
            ),
            wraplength=1100,
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        controls = ttk.Frame(
            self.surface_tab
        )

        controls.pack(
            fill="x"
        )

        ttk.Label(
            controls,
            text="Segmentation method:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.surface_method_box = NamedCombobox(
            controls,
            textvariable=self.surface_method,
            values=[
                "hsv",
                "otsu",
                "kmeans",
                "grabcut",
                "watershed",
                "hybrid",
                "hybrid_refined",
            ],
            label=LABELS.__getitem__,
            state="readonly",
            width=35,
        )

        self.surface_method_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.button(
            controls,
            "Analyze blemishes & size",
            self.start_surface,
            accent=True,
        )

        self.button(
            controls,
            "Export surface report CSV",
            self.save_surface_results,
        )

        self.button(
            controls,
            "Export surface preview PNG",
            self.save_surface_preview,
        )

        self.button(
            controls,
            "Export surface PDF",
            self.save_surface_pdf,
        )

        panes = ttk.Panedwindow(
            self.surface_tab,
            orient="vertical",
        )

        panes.pack(
            fill="both",
            expand=True,
            pady=(8, 0),
        )

        preview_frame, self.surface_canvas = (
            self.scroll_image(panes)
        )

        panes.add(
            preview_frame,
            weight=3,
        )

        table_frame = ttk.Frame(
            panes
        )

        panes.add(
            table_frame,
            weight=2,
        )

        columns = [
            "source",
            "segmentation_method",
            "status",
            "blemish_fraction",
            "blemish_status",
            "quality_grade",
            "objects_detected",
            "equivalent_diameter_px",
            "calibrated",
            "error",
        ]

        headings = [
            "Source",
            "Segmentation",
            "Status",
            "Blemish %",
            "Blemish status",
            "Quality grade",
            "Objects",
            "Equiv. diameter (px)",
            "Calibrated",
            "Error",
        ]

        self.surface_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=6,
            selectmode="browse",
        )

        for name, heading in zip(
            columns,
            headings,
        ):
            self.surface_table.heading(
                name,
                text=heading,
            )

            if name in ["source", "error"]:
                width = 330
            elif name == "segmentation_method":
                width = 180
            else:
                width = 120

            self.surface_table.column(
                name,
                width=width,
                stretch=False,
            )

        ybar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.surface_table.yview,
        )

        xbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.surface_table.xview,
        )

        self.surface_table.configure(
            yscrollcommand=ybar.set,
            xscrollcommand=xbar.set,
        )

        table_frame.columnconfigure(
            0,
            weight=1,
        )

        table_frame.rowconfigure(
            0,
            weight=1,
        )

        self.surface_table.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        ybar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        xbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self.surface_table.bind(
            "<<TreeviewSelect>>",
            self.show_surface_row_detail,
        )

    # =============================================================
    # CAMERA & VIDEO TAB
    # =============================================================

    def build_media(self):

        ttk.Label(
            self.media_tab,
            text=(
                "Uses the selected saved classifier and one processing method. "
                "Camera snapshots can be sent to Images & folders for a full "
                "six-method comparison. Uploaded videos are analysed frame by "
                "frame and exported as a new annotated MP4 or AVI."
            ),
            wraplength=1150,
        ).pack(
            anchor="w",
            pady=(0, 7),
        )

        method_controls = ttk.Frame(
            self.media_tab
        )

        method_controls.pack(
            fill="x"
        )

        ttk.Label(
            method_controls,
            text="Processing method:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.media_method_box = NamedCombobox(
            method_controls,
            textvariable=self.media_method,
            values=["baseline", *METHODS],
            label=LABELS.__getitem__,
            state="readonly",
            width=35,
        )

        self.media_method_box.pack(
            side="left",
            padx=(0, 12),
        )

        ttk.Label(
            method_controls,
            text=(
                "Prediction is for the complete frame; object boxes come "
                "from the selected segmentation mask."
            ),
            wraplength=650,
        ).pack(
            side="left",
        )

        camera_controls = ttk.Frame(
            self.media_tab
        )

        camera_controls.pack(
            fill="x",
            pady=(5, 0),
        )

        ttk.Label(
            camera_controls,
            text="Camera index:",
        ).pack(
            side="left",
            padx=(0, 8),
        )

        self.camera_index_box = ttk.Spinbox(
            camera_controls,
            from_=0,
            to=9,
            textvariable=self.camera_index,
            width=5,
        )

        self.camera_index_box.pack(
            side="left",
            padx=(0, 10),
        )

        self.camera_start_button = self.button(
            camera_controls,
            "Start live camera",
            self.start_camera,
            accent=True,
        )

        self.camera_stop_button = self.button(
            camera_controls,
            "Stop camera",
            self.stop_camera,
        )

        self.camera_snapshot_button = self.button(
            camera_controls,
            "Capture snapshot for comparison",
            self.capture_camera_snapshot,
        )

        video_controls = ttk.Frame(
            self.media_tab
        )

        video_controls.pack(
            fill="x",
            pady=(3, 0),
        )

        self.video_choose_button = self.button(
            video_controls,
            "Choose video",
            self.choose_video,
        )

        self.video_process_button = self.button(
            video_controls,
            "Process and export annotated video",
            self.start_video,
            accent=True,
        )

        self.video_csv_button = self.button(
            video_controls,
            "Export frame results CSV",
            self.save_video_csv,
        )

        ttk.Label(
            self.media_tab,
            textvariable=self.video_note,
            wraplength=1120,
        ).pack(
            anchor="w",
            pady=(2, 2),
        )

        ttk.Label(
            self.media_tab,
            textvariable=self.media_summary,
            wraplength=1120,
        ).pack(
            anchor="w",
            pady=(0, 5),
        )

        panes = ttk.Panedwindow(
            self.media_tab,
            orient="vertical",
        )

        panes.pack(
            fill="both",
            expand=True,
        )

        preview_frame, self.media_canvas = (
            self.scroll_image(panes)
        )

        panes.add(
            preview_frame,
            weight=3,
        )

        table_frame = ttk.Frame(
            panes
        )

        panes.add(
            table_frame,
            weight=2,
        )

        columns = [
            "frame_index",
            "timestamp_seconds",
            "predicted_stage",
            "score_unripe",
            "score_ripe",
            "score_overripe",
            "objects_detected",
            "processing_ms",
        ]

        headings = [
            "Frame",
            "Time (s)",
            "Prediction",
            "Unripe",
            "Ripe",
            "Overripe",
            "Objects",
            "Process ms",
        ]

        self.video_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=5,
            selectmode="browse",
        )

        for name, heading in zip(columns, headings):
            self.video_table.heading(name, text=heading)
            self.video_table.column(name, width=120, stretch=False)

        ybar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.video_table.yview,
        )

        xbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.video_table.xview,
        )

        self.video_table.configure(
            yscrollcommand=ybar.set,
            xscrollcommand=xbar.set,
        )

        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.video_table.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        self.sync_media_controls()

    # =============================================================
    # SURFACE ANALYSIS
    # =============================================================

    def start_surface(self):

        if self.busy:
            return

        try:

            if not self.inputs:
                raise ValueError(
                    "Choose at least one supported image first "
                    "(Images & folders tab)."
                )

            paths = list(
                self.inputs
            )

            self.surface_rows = []
            self.surface_preview = None
            self.surface_source = ""

            self.surface_canvas.delete(
                "all"
            )

            self.surface_table.delete(
                *self.surface_table.get_children()
            )

            self.progress.configure(
                value=0,
                maximum=len(paths),
            )

            method = self.surface_method.get()

            self.status.set(
                f"Starting surface analysis using "
                f"{LABELS.get(method, method)}..."
            )

            self.launch(
                lambda emit: run_surface_batch(
                    paths,
                    method=method,
                    cancel=self.cancel,
                    emit=lambda kind, value: emit(
                        ("surface_" + kind, value)
                    ),
                ),
                kind="surface",
            )

        except Exception as exc:
            self.error(exc)

    def show_surface_row_detail(
        self,
        event=None,
    ):

        selection = self.surface_table.selection()

        if selection and not self.busy:

            index = int(selection[0])

            if index >= len(self.surface_rows):
                return

            row = self.surface_rows[index]

            self.status.set(
                f"{row['source']} | "
                f"{row.get('segmentation_method', '')} | "
                f"{row.get('error') or row.get('blemish_status', '')}"
            )

    def save_surface_results(self):

        try:

            if not self.surface_rows:
                raise ValueError(
                    "Run surface analysis first."
                )

            path = self.destination(
                "Export surface report; choose a NEW filename outside input folders",
                ".csv",
            )

            if path:

                export_csv(
                    Path(path),
                    self.surface_rows,
                    SURFACE_RESULT_FIELDS,
                    self.protected,
                )

                self.status.set(
                    f"Exported surface report: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_surface_preview(self):

        try:

            if self.surface_preview is None:
                raise ValueError(
                    "Run surface analysis first; "
                    "no preview to export yet."
                )

            path = self.destination(
                "Export displayed surface preview; choose a NEW filename",
                ".png",
            )

            if path:

                export_png(
                    Path(path),
                    self.surface_preview,
                    self.protected,
                )

                self.status.set(
                    f"Exported surface preview: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_surface_pdf(self):

        try:

            if not self.surface_rows:
                raise ValueError(
                    "Run surface analysis first."
                )

            path = self.destination(
                "Export surface-analysis PDF; choose a NEW filename",
                ".pdf",
            )

            if path:

                export_report_pdf(
                    Path(path),
                    title="Fruit Surface Analysis Report",
                    rows=self.surface_rows,
                    fields=SURFACE_PDF_FIELDS,
                    image=self.surface_preview,
                    protected=self.protected,
                    labels=LABELS,
                    notes=[
                        "Quality grade is a project heuristic: Good up to 5%, "
                        "Acceptable above 5% up to 15%, and Poor above 15% "
                        "detected blemish area.",
                        "Blemish masks are not ground-truth annotations. "
                        "Fruit size is reported in pixels only.",
                    ],
                )

                self.status.set(
                    f"Exported surface-analysis PDF: {path}"
                )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # CAMERA & VIDEO
    # =============================================================

    def sync_media_controls(self):

        if not hasattr(self, "camera_start_button"):
            return

        if self.camera_running:
            self.camera_start_button.configure(state="disabled")
            self.camera_stop_button.configure(state="normal")
            self.camera_snapshot_button.configure(
                state=(
                    "normal"
                    if self.camera_latest is not None
                    else "disabled"
                )
            )
            self.video_choose_button.configure(state="disabled")
            self.video_process_button.configure(state="disabled")
            self.video_csv_button.configure(state="disabled")
            self.camera_index_box.configure(state="disabled")
            self.media_method_box.configure(state="disabled")
            return

        if self.busy:
            for button in [
                self.camera_start_button,
                self.camera_stop_button,
                self.camera_snapshot_button,
                self.video_choose_button,
                self.video_process_button,
                self.video_csv_button,
            ]:
                button.configure(state="disabled")
            self.camera_index_box.configure(state="disabled")
            self.media_method_box.configure(state="disabled")
            return

        self.camera_start_button.configure(state="normal")
        self.camera_stop_button.configure(state="disabled")
        self.camera_snapshot_button.configure(state="disabled")
        self.video_choose_button.configure(state="normal")
        self.video_process_button.configure(
            state=("normal" if self.video_source else "disabled")
        )
        self.video_csv_button.configure(
            state=("normal" if self.video_rows else "disabled")
        )
        self.camera_index_box.configure(state="normal")
        self.media_method_box.configure(state="readonly")

    def _media_prediction_task(self, run):

        bundle, _ = load_model(
            run,
            trusted=True,
        )

        return lambda image: infer(
            image,
            bundle,
        )

    def start_camera(self):

        if self.busy:
            return

        try:
            camera_index = int(self.camera_index.get())
            if camera_index < 0:
                raise ValueError(
                    "Camera index must be zero or greater."
                )

            method = self.media_method.get()
            run = self.selected_runs([method])[0]
            self.camera_latest = None
            self.camera_latest_row = None
            self.camera_running = True
            self.media_summary.set(
                "Opening camera and loading the selected model..."
            )
            self.media_canvas.delete("all")

            def task(emit):
                predict = self._media_prediction_task(run)
                run_live_camera(
                    camera_index,
                    predict=predict,
                    method=method,
                    method_label=LABELS[method],
                    cancel=self.cancel,
                    emit=lambda kind, value: emit((kind, value)),
                )

            self.launch(
                task,
                kind="camera",
            )
            self.sync_media_controls()

        except Exception as exc:
            self.camera_running = False
            self.sync_media_controls()
            self.error(exc)

    def stop_camera(self):

        if self.camera_running:
            self.cancel.set()
            self.status.set(
                "Stopping camera after the current frame..."
            )

    def capture_camera_snapshot(self):

        try:
            if self.camera_latest is None:
                raise ValueError(
                    "Wait for the first analysed camera frame."
                )

            path = self.destination(
                "Save camera snapshot for image comparison",
                ".png",
            )

            if path:
                target = Path(path)
                export_png(
                    target,
                    self.camera_latest,
                    [],
                )
                self.set_inputs([target])
                self.tabs.select(self.input_tab)
                self.cancel.set()
                self.status.set(
                    "Snapshot saved and selected. Wait for the camera "
                    "to stop, then choose Compare all six."
                )

        except Exception as exc:
            self.error(exc)

    def choose_video(self):

        if self.busy:
            return

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a video",
            filetypes=[
                ("Supported videos", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if path:
            selected = Path(path).resolve()
            if (
                not selected.is_file()
                or selected.suffix.lower() not in VIDEO_EXTENSIONS
            ):
                self.error(
                    "Choose an existing MP4, AVI, MOV or MKV video."
                )
                return
            self.video_source = selected
            self.video_note.set(
                f"Selected video: {selected}"
            )
            self.video_rows = []
            self.video_output = None
            self.video_preview = None
            self.video_table.delete(
                *self.video_table.get_children()
            )
            self.sync_media_controls()

    def start_video(self):

        if self.busy:
            return

        try:
            if not self.video_source:
                raise ValueError(
                    "Choose a video first."
                )

            method = self.media_method.get()
            run = self.selected_runs([method])[0]
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=(
                    "Export annotated video; choose a NEW filename "
                    "outside the source and model folders"
                ),
                defaultextension=".mp4",
                filetypes=[
                    ("MP4 video", "*.mp4"),
                    ("AVI video", "*.avi"),
                ],
            )

            if not path:
                return

            output = check_destination(
                Path(path),
                [
                    self.video_source.parent,
                    run.path,
                ],
            )

            self.video_rows = []
            self.video_output = None
            self.video_preview = None
            self.video_protected = [
                self.video_source.parent,
                run.path,
            ]
            self.video_table.delete(
                *self.video_table.get_children()
            )
            self.media_canvas.delete("all")
            self.progress.configure(value=0, maximum=1)
            self.media_summary.set(
                "Loading the selected model and opening the video..."
            )

            def task(emit):
                predict = self._media_prediction_task(run)
                process_video(
                    self.video_source,
                    output,
                    predict=predict,
                    method=method,
                    method_label=LABELS[method],
                    cancel=self.cancel,
                    emit=lambda kind, value: emit((kind, value)),
                )

            self.launch(
                task,
                kind="video",
            )

        except Exception as exc:
            self.error(exc)

    def save_video_csv(self):

        try:
            if not self.video_rows:
                raise ValueError(
                    "Process a video before exporting frame results."
                )

            path = self.destination(
                "Export frame-level video results CSV",
                ".csv",
            )

            if path:
                export_csv(
                    Path(path),
                    self.video_rows,
                    VIDEO_RESULT_FIELDS,
                    self.video_protected,
                )
                self.status.set(
                    f"Exported frame-level video results: {path}"
                )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # TEST REPORT
    # =============================================================

    def open_test_report(self):

        path = filedialog.askdirectory(
            parent=self.root,
            title=(
                "Choose completed "
                "outputs/final_test/<timestamp> folder"
            ),
            mustexist=True,
        )

        if path:
            self.load_test_report(
                Path(path)
            )

    def load_test_report(
        self,
        path,
    ):

        try:

            from .final_test import read_report

            runs = read_report(
                path
            )

            if not runs:
                raise ValueError(
                    "The selected Test report contains no runs."
                )

            rows = evaluation_rows(
                runs,
                split="test",
            )

            image = evaluation_sheet(
                runs,
                split="test",
            )

            self.test_runs = runs
            self.test_rows = rows
            self.test_image = image

            self.test_scope.set(
                "overall"
            )

            fruits = sorted(
                runs[0].metrics["test"]["per_fruit"]
            )

            self.test_scope_box.configure(
                values=[
                    "overall",
                    *fruits,
                ]
            )

            self.test_protected = [
                Path(path).resolve(),
                Path(
                    runs[0].metadata["source_run"]
                ),
            ]

            self.test_note.set(
                "FINAL TEST / MobileNetV2 CNN trained "
                f"{run_label(Path(runs[0].metadata['source_run']).name)} "
                "| Report: "
                f"{run_label(Path(path).name)}. "
                "Independent of the top classifier selector; "
                "no tuning or retraining."
            )

            self.display(
                self.test_canvas,
                image,
                1180,
            )

            self.status.set(
                f"Loaded all {len(runs)} variants "
                f"from final-Test report: {path}"
            )

        except Exception as exc:
            self.error(exc)

    def load_test_view(self):

        try:

            if not self.test_runs:
                raise ValueError(
                    "Open a completed final-Test report first."
                )

            rows = evaluation_rows(
                self.test_runs,
                self.test_scope.get(),
                split="test",
            )

            image = evaluation_sheet(
                self.test_runs,
                self.test_scope.get(),
                split="test",
            )

            self.test_rows = rows
            self.test_image = image

            self.display(
                self.test_canvas,
                image,
                1180,
            )

            self.status.set(
                f"Loaded saved final Test / "
                f"{self.test_scope.get()} / "
                f"{len(rows)} methods"
            )

        except Exception as exc:
            self.error(exc)

    def save_test_metrics(self):

        try:

            if not self.test_rows:
                raise ValueError(
                    "Open a completed final-Test report first"
                )

            path = self.destination(
                "Export displayed final-Test metrics",
                ".csv",
            )

            if path:

                export_csv(
                    Path(path),
                    self.test_rows,
                    METRIC_FIELDS,
                    [
                        *self.protected,
                        *self.test_protected,
                    ],
                )

                self.status.set(
                    f"Exported final-Test metrics: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_test_image(self):

        try:

            if self.test_image is None:
                raise ValueError(
                    "Open a completed final-Test report first"
                )

            path = self.destination(
                "Export displayed final-Test comparison",
                ".png",
            )

            if path:

                export_png(
                    Path(path),
                    self.test_image,
                    [
                        *self.protected,
                        *self.test_protected,
                    ],
                )

                self.status.set(
                    f"Exported final-Test comparison: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_test_pdf(self):

        try:

            if not self.test_rows or self.test_image is None:
                raise ValueError(
                    "Open a completed final-Test report first"
                )

            path = self.destination(
                "Export displayed final-Test PDF",
                ".pdf",
            )

            if path:

                export_report_pdf(
                    Path(path),
                    title=(
                        "Frozen Final-Test Comparison - "
                        f"{self.test_scope.get()}"
                    ),
                    rows=self.test_rows,
                    fields=METRICS_PDF_FIELDS,
                    image=self.test_image,
                    protected=[
                        *self.protected,
                        *self.test_protected,
                    ],
                    labels=LABELS,
                    notes=[
                        "All displayed methods use the checkpoint recorded "
                        "in the completed final-Test report.",
                        "This export reads saved measurements only; it does "
                        "not retrain or evaluate the model.",
                    ],
                )

                self.status.set(
                    f"Exported final-Test PDF: {path}"
                )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # MODEL DISCOVERY
    # =============================================================

    def selected_runs(
        self,
        methods,
        required=True,
    ):

        runs = []

        for method in methods:

            run = next(
                (
                    run
                    for run in self.found.get(
                        method,
                        []
                    )
                    if run.path.name
                    == self.run_vars[method].get()
                ),
                None,
            )

            if run is None and required:
                raise ValueError(
                    f"{LABELS.get(method, method)}: "
                    f"no compatible saved model was found "
                    f"in {self.outputs}."
                )

            if run:
                runs.append(run)

        return runs

    def update_scopes(self):

        try:
            selected = self.selected_runs(
                LABELS,
                required=False,
            )
        except Exception:
            selected = []

        scopes = sorted(
            {
                fruit
                for run in selected
                for fruit in run.metrics[
                    "validation"
                ].get(
                    "per_fruit",
                    {},
                )
            }
        )

        self.scope_box.configure(
            values=[
                "overall",
                *scopes,
            ]
        )

        if self.scope.get() not in [
            "overall",
            *scopes,
        ]:
            self.scope.set(
                "overall"
            )

    def refresh_runs(
        self,
        fallback=False,
    ):

        if self.busy:
            return

        # Safety guard in case this method is ever called before
        # initialization is completed.
        if not hasattr(self, "run_vars"):
            self.run_vars = {
                method: tk.StringVar()
                for method in LABELS
            }

        self.found, warnings = discover_runs(
            self.outputs,
            self.backend.get(),
        )

        # Make sure every expected method exists in found.
        for method in LABELS:
            self.found.setdefault(
                method,
                []
            )

        if (
            fallback
            and self.backend.get() == "shared_cnn"
            and not self.found["baseline"]
        ):

            self.backend.set(
                "random_forest"
            )

            self.found, fallback_warnings = (
                discover_runs(
                    self.outputs,
                    "random_forest",
                )
            )

            for method in LABELS:
                self.found.setdefault(
                    method,
                    []
                )

            warnings.extend(
                fallback_warnings
            )

        for method, runs in self.found.items():

            if method not in self.run_vars:
                self.run_vars[method] = tk.StringVar()

            names = [
                run.path.name
                for run in runs
            ]

            old = self.run_vars[
                method
            ].get()

            self.run_vars[
                method
            ].set(
                old
                if old in names
                else (
                    names[0]
                    if names
                    else ""
                )
            )

        if self.backend.get() == "shared_cnn":

            names = [
                run.path.name
                for run in self.found["baseline"]
            ]

            self.cnn_box.set_choices(
                names
            )

            current = self.cnn_run.get()

            self.cnn_run.set(
                current
                if current in names
                else (
                    names[0]
                    if names
                    else ""
                )
            )

            # One shared CNN model is used by every segmentation
            # method.
            for method in LABELS:
                self.run_vars[
                    method
                ].set(
                    self.cnn_run.get()
                )

        else:

            self.cnn_box.set_choices(
                []
            )

            self.cnn_run.set(
                ""
            )

        self.sync_classifier_controls()

        self.update_scopes()

        self.discovery_warnings = warnings

        count = sum(
            bool(runs)
            for runs in self.found.values()
        )

        warning_note = (
            f" | {len(warnings)} incompatible/incomplete "
            "run(s) ignored"
            if warnings
            else ""
        )

        self.status.set(
            f"{CLASSIFIER_LABELS[self.backend.get()]}: "
            f"{count}/{len(LABELS)} model groups ready"
            f"{warning_note}."
        )

    def sync_classifier_controls(self):

        self.backend_box.configure(
            state=(
                "disabled"
                if self.busy
                else "readonly"
            )
        )

        self.cnn_box.configure(
            state=(
                "readonly"
                if (
                    not self.busy
                    and self.backend.get()
                    == "shared_cnn"
                )
                else "disabled"
            )
        )

    def change_backend(
        self,
        event=None,
    ):

        if self.busy:
            return

        backend = self.backend.get()

        self.found, warnings = discover_runs(
            self.outputs,
            backend,
        )

        for method in LABELS:
            self.found.setdefault(
                method,
                []
            )

        self.discovery_warnings = warnings

        for method, runs in self.found.items():

            if method not in self.run_vars:
                self.run_vars[method] = tk.StringVar()

            names = [
                run.path.name
                for run in runs
            ]

            old = self.run_vars[
                method
            ].get()

            self.run_vars[
                method
            ].set(
                old
                if old in names
                else (
                    names[0]
                    if names
                    else ""
                )
            )

        if backend == "shared_cnn":

            names = [
                run.path.name
                for run in self.found[
                    "baseline"
                ]
            ]

            self.cnn_box.set_choices(
                names
            )

            current = self.cnn_run.get()

            self.cnn_run.set(
                current
                if current in names
                else (
                    names[0]
                    if names
                    else ""
                )
            )

            for method in LABELS:
                self.run_vars[
                    method
                ].set(
                    self.cnn_run.get()
                )

        else:

            self.cnn_box.set_choices(
                []
            )

            self.cnn_run.set(
                ""
            )

        self.sync_classifier_controls()

        self.update_scopes()

        count = sum(
            bool(runs)
            for runs in self.found.values()
        )

        warning_note = (
            f" | {len(warnings)} incompatible/incomplete "
            "run(s) ignored"
            if warnings
            else ""
        )

        self.status.set(
            f"{CLASSIFIER_LABELS[backend]}: "
            f"{count}/{len(LABELS)} model groups ready"
            f"{warning_note}."
        )

    def select_cnn_run(
        self,
        event=None,
    ):

        if self.busy:
            return

        for method in LABELS:
            self.run_vars[
                method
            ].set(
                self.cnn_run.get()
            )

        self.update_scopes()

        self.status.set(
            "One shared CNN run selected automatically "
            "for every method. Reload validation if needed."
        )

    # =============================================================
    # INPUT SELECTION
    # =============================================================

    def choose_files(self):

        files = filedialog.askopenfilenames(
            parent=self.root,
            title="Choose one or more images",
            filetypes=[
                (
                    "Supported images",
                    "*.png *.jpg *.jpeg *.webp *.bmp",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if files:
            self.set_inputs(
                [
                    Path(p)
                    for p in files
                ],
                recursive=False,
            )

    def choose_folder(self):

        folder = filedialog.askdirectory(
            parent=self.root,
            title="Choose a folder of images",
            mustexist=True,
        )

        if folder:
            self.set_inputs(
                [
                    Path(folder)
                ],
                recursive=self.recursive.get(),
            )

    def set_inputs(
        self,
        paths,
        recursive=False,
    ):

        try:

            self.inputs, warnings = collect_inputs(
                paths,
                recursive,
            )

            self.protected = [
                p.resolve()
                if p.is_dir()
                else p.resolve().parent
                for p in paths
            ]

            self.input_note.set(
                f"{len(self.inputs)} image(s) selected | "
                f"{len(warnings)} skipped entries | "
                f"{paths[0]}"
            )

            self.status.set(
                "Inputs selected. Previous results remain "
                "until the next run; "
                f"{len(warnings)} unsupported/invalid entries skipped."
            )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # RUN CONTROL
    # =============================================================

    def set_busy(
        self,
        busy,
    ):

        self.busy = busy

        for button in self.buttons:
            button.configure(
                state=(
                    "disabled"
                    if busy
                    else "normal"
                )
            )

        for box in [
            self.method_box,
            self.hybrid_box,
            self.scope_box,
            self.test_scope_box,
            self.surface_method_box,
            self.media_method_box,
        ]:
            box.configure(
                state=(
                    "disabled"
                    if busy
                    else "readonly"
                )
            )

        self.recurse_check.configure(
            state=(
                "disabled"
                if busy
                else "normal"
            )
        )

        self.cancel_button.configure(
            state=(
                "normal"
                if busy
                else "disabled"
            )
        )

        self.sync_classifier_controls()
        self.sync_media_controls()

    def launch(
        self,
        task,
        *,
        kind="prediction",
    ):

        self.cancel.clear()
        self.worker_error = ""
        self.task_kind = kind

        self.set_busy(
            True
        )

        def worker():

            try:

                task(
                    self.events.put
                )

            except Exception as exc:

                self.events.put(
                    (
                        "error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )

            finally:

                self.events.put(
                    (
                        "done",
                        None,
                    )
                )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def start(
        self,
        methods,
    ):

        if self.busy:
            return

        try:

            if not self.inputs:
                raise ValueError(
                    "Choose at least one supported image first."
                )

            runs = self.selected_runs(
                methods
            )

            from .ui_core import ensure_compatible

            ensure_compatible(
                runs
            )

            paths = list(
                self.inputs
            )

            self.rows = []
            self.cards = {}
            self.card_source = ""

            self.preview_canvas.delete(
                "all"
            )

            self.table.delete(
                *self.table.get_children()
            )

            self.result_protected = [
                *self.protected,
                *[
                    r.path
                    for r in runs
                ],
            ]

            self.progress.configure(
                value=0,
                maximum=len(paths),
            )

            self.status.set(
                "Starting local prediction..."
            )

            self.launch(
                lambda emit: run_batch(
                    paths,
                    runs,
                    trusted=True,
                    cancel=self.cancel,
                    emit=lambda kind, value: emit(
                        (
                            kind,
                            value,
                        )
                    ),
                )
            )

        except Exception as exc:
            self.error(exc)

    def cancel_work(self):

        self.cancel.set()

        if self.task_kind == "camera":
            message = "Stopping camera after the current frame..."
        elif self.task_kind == "video":
            message = (
                "Cancelling after the current frame; the incomplete video "
                "will be discarded and completed frame rows retained."
            )
        else:
            message = (
                "Cancelling after the current model/image operation; "
                "completed rows will remain exportable."
            )

        self.status.set(message)

    # =============================================================
    # EVENT LOOP
    # =============================================================

    def poll(self):

        try:

            for _ in range(24):

                kind, value = (
                    self.events.get_nowait()
                )

                # -------------------------------------------------
                # Normal prediction result
                # -------------------------------------------------

                if kind == "result":

                    row, card = value

                    self.rows.append(
                        row
                    )

                    values = [
                        row.get(
                            key,
                            "",
                        )
                        for key in [
                            "source",
                            "method",
                            "status",
                            "predicted_stage",
                            "processing_ms",
                            "prediction_ms",
                            "error",
                        ]
                    ]

                    values[1] = LABELS.get(
                        values[1],
                        values[1],
                    )

                    values[4:6] = [
                        f"{v:.1f}"
                        if isinstance(v, float)
                        else v
                        for v in values[4:6]
                    ]

                    self.table.insert(
                        "",
                        "end",
                        iid=str(
                            len(self.rows) - 1
                        ),
                        values=values,
                    )

                    if (
                        row["source"]
                        != self.card_source
                    ):

                        self.cards = {}
                        self.card_source = (
                            row["source"]
                        )

                    if card:
                        self.cards[
                            row["method"]
                        ] = card

                    self.draw_preview()

                # -------------------------------------------------
                # Preview
                # -------------------------------------------------

                elif kind == "preview":

                    self.card_source, self.cards = (
                        value
                    )

                    self.draw_preview()

                # -------------------------------------------------
                # Surface result
                # -------------------------------------------------

                elif kind == "surface_result":

                    row, card = value

                    self.surface_rows.append(
                        row
                    )

                    values = [
                        row.get(
                            key,
                            "",
                        )
                        for key in [
                            "source",
                            "segmentation_method",
                            "status",
                            "blemish_fraction",
                            "blemish_status",
                            "quality_grade",
                            "objects_detected",
                            "equivalent_diameter_px",
                            "calibrated",
                            "error",
                        ]
                    ]

                    values[1] = LABELS.get(
                        values[1],
                        values[1],
                    )

                    if isinstance(
                        values[3],
                        float,
                    ):
                        values[3] = (
                            f"{values[3]:.1%}"
                        )

                    if isinstance(
                        values[7],
                        float,
                    ):
                        values[7] = (
                            f"{values[7]:.1f}"
                        )

                    self.surface_table.insert(
                        "",
                        "end",
                        iid=str(
                            len(
                                self.surface_rows
                            ) - 1
                        ),
                        values=values,
                    )

                    if card:

                        self.surface_source = (
                            row["source"]
                        )

                        self.surface_preview = (
                            card
                        )

                        width = min(
                            1180,
                            max(
                                900,
                                self.surface_canvas.winfo_width()
                                - 20,
                            ),
                        )

                        self.display(
                            self.surface_canvas,
                            card,
                            width,
                        )

                # -------------------------------------------------
                # Surface progress
                # -------------------------------------------------

                elif kind == "surface_progress":

                    completed, total = value

                    self.progress.configure(
                        value=completed,
                        maximum=total,
                    )

                    failed = sum(
                        r["status"] != "ok"
                        for r in self.surface_rows
                    )

                    self.status.set(
                        f"{completed}/{total} images analyzed | "
                        f"{len(self.surface_rows) - failed} ok | "
                        f"{failed} errors"
                    )

                # -------------------------------------------------
                # Live camera frame
                # -------------------------------------------------

                elif kind == "camera_frame":

                    self.camera_latest = value["original"]
                    self.camera_latest_row = value

                    width = min(
                        1180,
                        max(700, self.media_canvas.winfo_width() - 20),
                    )

                    self.display(
                        self.media_canvas,
                        value["annotated"],
                        width,
                    )

                    self.media_summary.set(
                        f"Live frame {value['frame_index']} | "
                        f"{value['predicted_stage'].upper()} | "
                        f"unripe {value['score_unripe']:.1%}, "
                        f"ripe {value['score_ripe']:.1%}, "
                        f"overripe {value['score_overripe']:.1%} | "
                        f"{value['objects_detected']} object(s)"
                    )
                    self.status.set(
                        "Live camera is running. Capture the current "
                        "original frame or stop the camera."
                    )
                    self.sync_media_controls()

                # -------------------------------------------------
                # Uploaded-video frame
                # -------------------------------------------------

                elif kind == "video_frame":

                    row, annotated, total = value
                    self.video_rows.append(row)
                    self.video_preview = annotated

                    values = [
                        row["frame_index"],
                        f"{row['timestamp_seconds']:.2f}",
                        row["predicted_stage"],
                        f"{row['score_unripe']:.1%}",
                        f"{row['score_ripe']:.1%}",
                        f"{row['score_overripe']:.1%}",
                        row["objects_detected"],
                        f"{row['processing_ms']:.1f}",
                    ]

                    self.video_table.insert(
                        "",
                        "end",
                        iid=str(len(self.video_rows) - 1),
                        values=values,
                    )
                    self.video_table.see(str(len(self.video_rows) - 1))

                    width = min(
                        1180,
                        max(700, self.media_canvas.winfo_width() - 20),
                    )
                    self.display(self.media_canvas, annotated, width)

                    completed = row["frame_index"]
                    maximum = total if total > 0 else completed
                    self.progress.configure(
                        value=completed,
                        maximum=max(1, maximum),
                    )
                    total_text = str(total) if total > 0 else "unknown"
                    self.media_summary.set(
                        f"Processed frame {completed}/{total_text} | "
                        f"{row['predicted_stage'].upper()} | "
                        f"{row['objects_detected']} object(s)"
                    )

                elif kind == "video_completed":

                    self.video_output = Path(value)
                    self.video_note.set(
                        f"Annotated video saved: {self.video_output}"
                    )

                # -------------------------------------------------
                # General status
                # -------------------------------------------------

                elif kind == "status":

                    self.status.set(
                        value
                    )

                # -------------------------------------------------
                # Normal progress
                # -------------------------------------------------

                elif kind == "progress":

                    completed, total = value

                    self.progress.configure(
                        value=completed,
                        maximum=total,
                    )

                    failed = sum(
                        r["status"] != "ok"
                        for r in self.rows
                    )

                    self.status.set(
                        f"{completed}/{total} images processed | "
                        f"{len(self.rows) - failed} successful "
                        f"method results | {failed} errors/blocked"
                    )

                # -------------------------------------------------
                # Error
                # -------------------------------------------------

                elif kind == "error":

                    self.worker_error = value

                    self.error(
                        value
                    )

                # -------------------------------------------------
                # Done
                # -------------------------------------------------

                elif kind == "done":

                    completed_kind = self.task_kind
                    self.task_kind = ""

                    if completed_kind == "camera":
                        self.camera_running = False

                    self.set_busy(
                        False
                    )

                    if self.worker_error:

                        self.status.set(
                            f"Failed: {self.worker_error}"
                        )

                    elif completed_kind == "camera":

                        self.status.set(
                            "Camera stopped. The latest captured frame "
                            "remains available until another media run."
                        )

                    elif completed_kind == "video":

                        if self.cancel.is_set():
                            self.status.set(
                                "Video processing cancelled; the incomplete "
                                "video was discarded. Completed frame rows "
                                "remain exportable."
                            )
                        else:
                            self.status.set(
                                f"Finished video processing | "
                                f"{len(self.video_rows)} frames | "
                                f"saved to {self.video_output}"
                            )

                    elif completed_kind == "surface":

                        failed = sum(
                            r["status"] != "ok"
                            for r in self.surface_rows
                        )
                        self.status.set(
                            f"{'Cancelled; partial results' if self.cancel.is_set() else 'Finished'} | "
                            f"{len(self.surface_rows) - failed} successful "
                            f"surface results | {failed} errors."
                        )

                    else:

                        failed = sum(
                            r["status"] != "ok"
                            for r in self.rows
                        )

                        self.status.set(
                            f"{'Cancelled; partial results' if self.cancel.is_set() else 'Finished'} | "
                            f"{len(self.rows) - failed} successful "
                            f"method results | "
                            f"{failed} errors/blocked. "
                            "Select a row to inspect."
                        )

                    self.sync_media_controls()

                    if self.closing:

                        self.root.destroy()

                        return

        except queue.Empty:
            pass

        if not self.closing:
            self.root.after(
                100,
                self.poll,
            )

    # =============================================================
    # PREVIEW
    # =============================================================

    def display(
        self,
        canvas,
        image,
        width,
        *,
        center=False,
    ):

        canvas.delete(
            "all"
        )

        if image is None:
            return

        if image.width <= 0:
            return

        width = max(
            1,
            int(width),
        )

        display = image.resize(
            (
                width,
                round(
                    image.height
                    * width
                    / image.width
                ),
            ),
            Image.Resampling.LANCZOS,
        )

        photo = ImageTk.PhotoImage(
            display,
            master=self.root,
        )

        canvas.photo = photo

        canvas_width = max(
            1,
            canvas.winfo_width(),
        )

        canvas_height = max(
            1,
            canvas.winfo_height(),
        )

        x = (
            max(0, (canvas_width - display.width) // 2)
            if center
            else 0
        )

        y = (
            max(0, (canvas_height - display.height) // 2)
            if center
            else 0
        )

        canvas.create_image(
            x,
            y,
            anchor="nw",
            image=photo,
        )

        canvas.configure(
            scrollregion=(
                0,
                0,
                max(canvas_width, x + display.width),
                max(canvas_height, y + display.height),
            )
        )

        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

    def preview_width(
        self,
        canvas,
        image,
        *,
        fit_height,
    ):

        available_width = max(
            320,
            canvas.winfo_width() - 20,
        )

        width = min(
            image.width,
            available_width,
        )

        if fit_height:
            available_height = max(
                220,
                canvas.winfo_height() - 20,
            )

            width = min(
                width,
                image.width * available_height / image.height,
            )

        return max(
            1,
            round(width),
        )

    def schedule_preview_redraw(
        self,
        event=None,
    ):

        if self.preview_redraw_after is not None:
            try:
                self.root.after_cancel(
                    self.preview_redraw_after
                )
            except tk.TclError:
                pass

        self.preview_redraw_after = self.root.after(
            80,
            self.draw_preview,
        )

    def toggle_results(self):

        self.results_expanded = not self.results_expanded

        if self.results_expanded:
            self.results_table_area.pack(
                fill="both",
                expand=True,
            )

            self.input_panes.pane(
                self.results_table_frame,
                weight=2,
            )

            self.results_toggle_button.configure(
                text="Hide result details"
            )
        else:
            self.results_table_area.pack_forget()

            self.input_panes.pane(
                self.results_table_frame,
                weight=0,
            )

            self.results_toggle_button.configure(
                text="Show result details"
            )

        self.root.after_idle(
            self.draw_preview
        )

    def open_preview_window(self):

        try:
            if not self.cards:
                raise ValueError(
                    "No preview is available to expand."
                )

            sheet = comparison_sheet(
                self.card_source,
                list(self.cards.values()),
            )

            window = tk.Toplevel(
                self.root
            )

            window.title(
                "Fruit Ripeness | Expanded preview"
            )

            width = min(
                1500,
                max(900, self.root.winfo_screenwidth() - 80),
            )

            height = min(
                950,
                max(650, self.root.winfo_screenheight() - 100),
            )

            window.geometry(
                f"{width}x{height}"
            )

            frame, canvas = self.scroll_image(
                window
            )

            frame.pack(
                fill="both",
                expand=True,
            )

            def redraw(event=None):
                self.display(
                    canvas,
                    sheet,
                    self.preview_width(
                        canvas,
                        sheet,
                        fit_height=len(self.cards) == 1,
                    ),
                    center=len(self.cards) == 1,
                )

            canvas.bind(
                "<Configure>",
                redraw,
            )

            window.after_idle(
                redraw
            )

        except Exception as exc:
            self.error(exc)

    def draw_preview(self):

        self.preview_redraw_after = None

        if self.cards:

            sheet = comparison_sheet(
                self.card_source,
                list(
                    self.cards.values()
                ),
            )

            self.display(
                self.preview_canvas,
                sheet,
                self.preview_width(
                    self.preview_canvas,
                    sheet,
                    fit_height=len(self.cards) == 1,
                ),
                center=len(self.cards) == 1,
            )

        else:

            self.preview_canvas.delete(
                "all"
            )

            self.preview_canvas.create_text(
                25,
                30,
                anchor="nw",
                text=(
                    "No successful preview for this image. "
                    "See the result/error row."
                ),
                fill="#555555",
            )

    def show_row_detail(
        self,
        event=None,
    ):

        selection = self.table.selection()

        if selection and not self.busy:

            index = int(selection[0])

            if index >= len(self.rows):
                return

            row = self.rows[index]

            self.status.set(
                f"{row['source']} | "
                f"{LABELS.get(row['method'], row['method'])} | "
                f"{row.get('error') or row.get('predicted_stage', '')}"
            )

    def preview_selected(self):

        if self.busy:
            return

        selection = self.table.selection()

        if not selection:

            self.error(
                "Select a result row first."
            )

            return

        source = self.rows[
            int(selection[0])
        ]["source"]

        rows = [
            r.copy()
            for r in self.rows
            if (
                r["source"] == source
                and r["status"] == "ok"
            )
        ]

        if not rows:

            self.error(
                "This image has no successful predictions; "
                "inspect its error row."
            )

            return

        def task(emit):

            blob = Path(
                source
            ).read_bytes()

            if any(
                hashlib.sha256(
                    blob
                ).hexdigest()
                != r["sha256"]
                for r in rows
            ):
                raise ValueError(
                    "Source image changed since prediction. "
                    "Run it again; old scores cannot label new pixels."
                )

            cards = {}

            with Image.open(
                io.BytesIO(blob)
            ) as image:

                image.load()

                for row in rows:

                    if self.cancel.is_set():
                        return

                    cards[
                        row["method"]
                    ] = prediction_card(
                        row,
                        process_image(
                            image,
                            row["method"],
                        ),
                    )

            emit(
                (
                    "preview",
                    (
                        source,
                        cards,
                    ),
                )
            )

        self.status.set(
            "Reconstructing preview from unchanged source; "
            "preserving recorded scores and timings..."
        )

        self.launch(
            task
        )

    # =============================================================
    # VALIDATION
    # =============================================================

    def load_evaluation(self):

        try:

            runs = self.selected_runs(
                LABELS,
                required=False,
            )

            scope = self.scope.get()

            rows = evaluation_rows(
                runs,
                scope,
            )

            image = evaluation_sheet(
                runs,
                scope,
            )

            self.eval_rows = rows
            self.eval_image = image

            self.eval_protected = [
                r.path
                for r in runs
            ]

            self.display(
                self.eval_canvas,
                image,
                1180,
            )

            self.status.set(
                f"Loaded saved validation: "
                f"{len(runs)}/{len(LABELS)} selected runs | "
                f"{scope}. Missing methods are omitted, not invented."
            )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # EXPORT
    # =============================================================

    def destination(
        self,
        title,
        extension,
    ):

        return filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            defaultextension=extension,
            filetypes=[
                (
                    extension.upper()[1:] + " file",
                    "*" + extension,
                )
            ],
        )

    def save_results(self):

        try:

            if not self.rows:
                raise ValueError(
                    "No prediction rows to export."
                )

            path = self.destination(
                "Export predictions; choose a NEW filename outside input folders",
                ".csv",
            )

            if path:

                export_csv(
                    Path(path),
                    self.rows,
                    RESULT_FIELDS,
                    self.result_protected,
                )

                self.status.set(
                    f"Exported predictions: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_preview(self):

        try:

            if not self.cards:
                raise ValueError(
                    "No preview is available to export."
                )

            image = comparison_sheet(
                self.card_source,
                list(
                    self.cards.values()
                ),
            )

            path = self.destination(
                "Export displayed image comparison; choose a NEW filename",
                ".png",
            )

            if path:

                export_png(
                    Path(path),
                    image,
                    self.result_protected,
                )

                self.status.set(
                    f"Exported preview: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_prediction_pdf(self):

        try:

            if not self.rows:
                raise ValueError(
                    "No prediction rows to export."
                )

            image = None

            if self.cards:
                image = comparison_sheet(
                    self.card_source,
                    list(self.cards.values()),
                )

            path = self.destination(
                "Export prediction report PDF; choose a NEW filename",
                ".pdf",
            )

            if path:

                export_report_pdf(
                    Path(path),
                    title="Fruit Ripeness Prediction Report",
                    rows=self.rows,
                    fields=PREDICTION_PDF_FIELDS,
                    image=image,
                    protected=self.result_protected,
                    labels=LABELS,
                    notes=[
                        "Predictions are image-level unripe, ripe or "
                        "overripe classifications.",
                        "Class scores are uncalibrated. The visual evidence "
                        "shows the currently displayed source image.",
                    ],
                )

                self.status.set(
                    f"Exported prediction PDF: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_metrics(self):

        try:

            if not self.eval_rows:
                raise ValueError(
                    "Load validation results first."
                )

            path = self.destination(
                "Export displayed validation metrics",
                ".csv",
            )

            if path:

                export_csv(
                    Path(path),
                    self.eval_rows,
                    METRIC_FIELDS,
                    [
                        *self.protected,
                        *self.eval_protected,
                    ],
                )

                self.status.set(
                    f"Exported validation metrics: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_evaluation(self):

        try:

            if self.eval_image is None:
                raise ValueError(
                    "Load validation results first."
                )

            path = self.destination(
                "Export displayed validation comparison and confusion matrices",
                ".png",
            )

            if path:

                export_png(
                    Path(path),
                    self.eval_image,
                    [
                        *self.protected,
                        *self.eval_protected,
                    ],
                )

                self.status.set(
                    f"Exported validation comparison: {path}"
                )

        except Exception as exc:
            self.error(exc)

    def save_validation_pdf(self):

        try:

            if not self.eval_rows or self.eval_image is None:
                raise ValueError(
                    "Load validation results first."
                )

            path = self.destination(
                "Export displayed validation PDF",
                ".pdf",
            )

            if path:

                export_report_pdf(
                    Path(path),
                    title=(
                        "Validation Comparison - "
                        f"{self.scope.get()}"
                    ),
                    rows=self.eval_rows,
                    fields=METRICS_PDF_FIELDS,
                    image=self.eval_image,
                    protected=[
                        *self.protected,
                        *self.eval_protected,
                    ],
                    labels=LABELS,
                    notes=[
                        "These are development validation measurements, "
                        "not final-Test performance.",
                        "The baseline is a reference input and the hybrid "
                        "variants are separate experiments.",
                    ],
                )

                self.status.set(
                    f"Exported validation PDF: {path}"
                )

        except Exception as exc:
            self.error(exc)

    # =============================================================
    # ERROR / CLOSE
    # =============================================================

    def error(
        self,
        exc,
    ):

        self.status.set(
            str(exc)
        )

        if not self.closing:

            messagebox.showerror(
                "Fruit Ripeness",
                str(exc),
                parent=self.root,
            )

    def close(self):

        if self.busy:

            self.closing = True

            self.cancel_work()

        else:

            self.root.destroy()


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
    )

    args = parser.parse_args()

    try:

        root = tk.Tk()

    except tk.TclError as exc:

        parser.exit(
            1,
            (
                "Cannot open desktop UI: "
                f"{exc}\n"
                "Use a graphical desktop with Python Tk support. "
                "Test with python -m tkinter.\n"
            ),
        )

    App(
        root,
        args.outputs,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
