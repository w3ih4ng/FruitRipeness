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

from PIL import Image, ImageOps, ImageTk

from .processing import process_image
from .ui_core import (LABELS, METHODS, METRIC_FIELDS, RESULT_FIELDS, collect_inputs,
                      comparison_sheet, discover_runs, evaluation_rows, evaluation_sheet,
                      export_csv, export_png, prediction_card, run_batch)


class App:
    def __init__(self, root: tk.Tk, outputs: Path):
        self.root = root
        self.busy = False
        self.closing = False
        self.worker_error = ''
        self.cancel = threading.Event()
        self.events = queue.Queue(maxsize=16)
        self.inputs = []
        self.protected = []
        self.result_protected = []
        self.rows = []
        self.cards = {}
        self.card_source = ''
        self.eval_rows = []
        self.eval_image = None
        self.eval_protected = []
        self.found = {}
        self.run_vars = {}
        self.run_boxes = {}
        self.model_notes = {}
        self.buttons = []
        self.photos = []
        self.output_var = tk.StringVar(value=str(Path(outputs).resolve()))
        self.status = tk.StringVar(value='Choose images or a folder to begin.')
        self.input_note = tk.StringVar(value='No images selected')
        self.trust = tk.BooleanVar(value=False)
        self.recursive = tk.BooleanVar(value=False)
        self.method = tk.StringVar(value='hybrid')
        self.scope = tk.StringVar(value='overall')
        self.root.title('Fruit Ripeness | Comparison Studio')
        width = min(1280, max(800, root.winfo_screenwidth()-80))
        height = min(850, max(600, root.winfo_screenheight()-100))
        self.root.geometry(f'{width}x{height}')
        self.root.minsize(min(1000, width), min(700, height))
        self.root.configure(bg='#edf2ee')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10))
        style.configure('TFrame', background='#f6f8f6')
        style.configure('TLabel', background='#f6f8f6', foreground='#243b30')
        style.configure('TButton', padding=(12, 7))
        style.configure('Accent.TButton', background='#2f6547', foreground='white')
        style.configure('Treeview', rowheight=27)
        header = tk.Frame(root, bg='#214737', padx=22, pady=16)
        header.pack(fill='x')
        tk.Label(header, text='FRUIT RIPENESS', font=('Segoe UI', 21, 'bold'), bg='#214737', fg='white').pack(side='left')
        tk.Label(header, text='Comparison Studio  /  Mode A', font=('Segoe UI', 12), bg='#214737', fg='#cee3d5').pack(side='right')
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='both', expand=True, padx=12, pady=10)
        self.input_tab = ttk.Frame(self.tabs, padding=12)
        self.eval_tab = ttk.Frame(self.tabs, padding=12)
        self.models_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.input_tab, text='  Images & folders  ')
        self.tabs.add(self.eval_tab, text='  Saved validation  ')
        self.tabs.add(self.models_tab, text='  Models & runs  ')
        self.build_inputs()
        self.build_evaluation()
        self.build_models()
        footer = ttk.Frame(root, padding=(15, 6))
        footer.pack(fill='x')
        self.progress = ttk.Progressbar(footer, mode='determinate', length=190)
        self.progress.pack(side='right')
        ttk.Label(footer, textvariable=self.status, wraplength=950).pack(side='left')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh_runs()
        self.root.after(100, self.poll)

    def button(self, parent, text, command, *, accent=False):
        button = ttk.Button(parent, text=text, command=command, style='Accent.TButton' if accent else 'TButton')
        button.pack(side='left', padx=(0, 7), pady=3)
        self.buttons.append(button)
        return button

    def scroll_image(self, parent):
        frame = ttk.Frame(parent)
        canvas = tk.Canvas(frame, background='#edf2ee', highlightthickness=0)
        ybar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        xbar = ttk.Scrollbar(frame, orient='horizontal', command=canvas.xview)
        canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky='nsew')
        ybar.grid(row=0, column=1, sticky='ns')
        xbar.grid(row=1, column=0, sticky='ew')
        return frame, canvas

    def build_inputs(self):
        controls = ttk.Frame(self.input_tab)
        controls.pack(fill='x')
        self.button(controls, 'Choose image(s)', self.choose_files)
        self.button(controls, 'Choose folder', self.choose_folder)
        self.recurse_check = ttk.Checkbutton(controls, text='Include subfolders when choosing a folder', variable=self.recursive)
        self.recurse_check.pack(side='left', padx=10)
        ttk.Label(self.input_tab, textvariable=self.input_note, wraplength=900).pack(anchor='w', pady=(3, 8))
        actions = ttk.Frame(self.input_tab)
        actions.pack(fill='x')
        self.method_box = ttk.Combobox(actions, textvariable=self.method, values=METHODS, state='readonly', width=14)
        self.method_box.pack(side='left', padx=(0, 8))
        self.button(actions, 'Run selected method', lambda: self.start([self.method.get()]), accent=True)
        self.button(actions, 'Compare all six', lambda: self.start(list(METHODS)), accent=True)
        self.cancel_button = ttk.Button(actions, text='Cancel', command=self.cancel_work, state='disabled')
        self.cancel_button.pack(side='left')
        ttk.Label(self.input_tab, text='No ground truth is assumed for these inputs. Scores are uncalibrated; masks are heuristics. Original-Test content is blocked.',
                  wraplength=900).pack(anchor='w', pady=(8, 7))
        panes = ttk.Panedwindow(self.input_tab, orient='vertical')
        panes.pack(fill='both', expand=True)
        preview_frame, self.preview_canvas = self.scroll_image(panes)
        panes.add(preview_frame, weight=3)
        table_frame = ttk.Frame(panes)
        panes.add(table_frame, weight=2)
        bar = ttk.Frame(table_frame)
        bar.pack(fill='x')
        self.button(bar, 'Preview selected image', self.preview_selected)
        self.button(bar, 'Export predictions CSV', self.save_results)
        self.button(bar, 'Export preview PNG', self.save_preview)
        ttk.Label(bar, text='Select a result row to inspect its source or error.').pack(side='left', padx=6)
        table_area = ttk.Frame(table_frame)
        table_area.pack(fill='both', expand=True)
        columns = ['source', 'method', 'status', 'predicted_stage', 'processing_ms', 'prediction_ms', 'error']
        self.table = ttk.Treeview(table_area, columns=columns, show='headings', height=5, selectmode='browse')
        headings = ['Source', 'Method', 'Status', 'Prediction', 'Process ms', 'Predict ms', 'Error']
        for name, heading in zip(columns, headings):
            self.table.heading(name, text=heading)
            self.table.column(name, width=330 if name in ['source', 'error'] else 100, stretch=False)
        ybar = ttk.Scrollbar(table_area, orient='vertical', command=self.table.yview)
        xbar = ttk.Scrollbar(table_area, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        table_area.columnconfigure(0, weight=1)
        table_area.rowconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky='nsew')
        ybar.grid(row=0, column=1, sticky='ns')
        xbar.grid(row=1, column=0, sticky='ew')
        self.table.bind('<<TreeviewSelect>>', self.show_row_detail)
        self.table.bind('<Double-1>', lambda event: self.preview_selected())

    def build_evaluation(self):
        controls = ttk.Frame(self.eval_tab)
        controls.pack(fill='x')
        ttk.Label(controls, text='Fruit scope:').pack(side='left', padx=(0, 8))
        self.scope_box = ttk.Combobox(controls, textvariable=self.scope, values=['overall'], state='readonly', width=16)
        self.scope_box.pack(side='left', padx=(0, 10))
        self.button(controls, 'Load selected runs', self.load_evaluation, accent=True)
        self.button(controls, 'Export metrics CSV', self.save_metrics)
        self.button(controls, 'Export comparison PNG', self.save_evaluation)
        ttk.Label(self.eval_tab, text='Reads saved validation metrics only. Baseline is a reference, not a seventh processing method. No retraining or Test evaluation.',
                  wraplength=900).pack(anchor='w', pady=10)
        frame, self.eval_canvas = self.scroll_image(self.eval_tab)
        frame.pack(fill='both', expand=True)

    def build_models(self):
        controls = ttk.Frame(self.models_tab)
        controls.pack(fill='x')
        ttk.Label(controls, text='Outputs folder:').pack(side='left', padx=(0, 10))
        self.output_entry = ttk.Entry(controls, textvariable=self.output_var)
        self.output_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.button(controls, 'Browse', self.choose_outputs)
        self.button(controls, 'Refresh runs', self.refresh_runs)
        ttk.Label(self.models_tab, text='Newest completed run is selected initially. Choose older runs here if needed; incompatible splits/settings are rejected.',
                  wraplength=900).pack(anchor='w', pady=12)
        grid = ttk.Frame(self.models_tab)
        grid.pack(fill='x')
        for i, (method, label) in enumerate(LABELS.items()):
            ttk.Label(grid, text=label, width=32).grid(row=i, column=0, sticky='w', pady=7)
            var = tk.StringVar()
            box = ttk.Combobox(grid, textvariable=var, state='readonly', width=36)
            box.grid(row=i, column=1, sticky='ew', padx=10, pady=7)
            note = ttk.Label(grid, text='Not trained')
            note.grid(row=i, column=2, sticky='w', padx=10)
            box.bind('<<ComboboxSelected>>', lambda event: self.update_model_notes())
            self.run_vars[method], self.run_boxes[method], self.model_notes[method] = var, box, note
        grid.columnconfigure(1, weight=1)
        self.trust_check = ttk.Checkbutton(self.models_tab, variable=self.trust,
            text="I confirm these model.joblib files were generated by my team and are trusted.")
        self.trust_check.pack(anchor='w', pady=(22, 6))
        ttk.Label(self.models_tab, text='Model files can execute code when loaded. Metadata checks do not make untrusted model files safe.\n'
                  'Discovery and the validation view read JSON only; prediction requires your confirmation.\n'
                  'Processing runs locally. Source images and saved runs are never modified.', wraplength=900).pack(anchor='w')
        self.log = tk.Text(self.models_tab, height=7, wrap='word', background='white', relief='flat', padx=8, pady=8)
        self.log.pack(fill='both', expand=True, pady=12)
        self.log.configure(state='disabled')

    def log_messages(self, messages):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert('end', '\n'.join(messages) if messages else 'All discovered runs passed metadata checks.')
        self.log.configure(state='disabled')

    def selected_runs(self, methods, required=True):
        runs = []
        for method in methods:
            run = next((run for run in self.found.get(method, []) if run.path.name == self.run_vars[method].get()), None)
            if run is None and required:
                raise ValueError(f'{method}: not trained / no valid run selected. Check Models & runs.')
            if run:
                runs.append(run)
        return runs

    def update_model_notes(self):
        for method, note in self.model_notes.items():
            runs = self.selected_runs([method], required=False)
            if not runs:
                note.configure(text='Not trained / no valid run')
            else:
                run = runs[0]
                available = (run.path/'model.joblib').is_file()
                note.configure(text=f"{'Model ready' if available else 'Model missing'} | validation n={run.metrics['validation']['n_images']}")
        scopes = sorted({fruit for run in self.selected_runs(LABELS, required=False)
                         for fruit in run.metrics['validation'].get('per_fruit', {})})
        self.scope_box.configure(values=['overall', *scopes])
        if self.scope.get() not in ['overall', *scopes]:
            self.scope.set('overall')

    def refresh_runs(self):
        if self.busy:
            return
        self.trust.set(False)
        self.found, warnings = discover_runs(Path(self.output_var.get()).expanduser())
        for method, runs in self.found.items():
            names = [run.path.name for run in runs]
            old = self.run_vars[method].get()
            self.run_boxes[method].configure(values=names)
            self.run_vars[method].set(old if old in names else names[0] if names else '')
        self.update_model_notes()
        self.log_messages(warnings)
        count = sum(bool(runs) for runs in self.found.values())
        self.status.set(f'{count}/7 method/reference run groups found. Confirm trusted models on Models & runs before prediction.')

    def choose_outputs(self):
        folder = filedialog.askdirectory(parent=self.root, title='Choose the outputs folder', mustexist=True)
        if folder:
            self.output_var.set(folder)
            self.refresh_runs()

    def choose_files(self):
        files = filedialog.askopenfilenames(parent=self.root, title='Choose one or more images',
            filetypes=[('Supported images', '*.png *.jpg *.jpeg *.webp *.bmp'), ('All files', '*.*')])
        if files:
            self.set_inputs([Path(p) for p in files], recursive=False)

    def choose_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title='Choose a folder of images', mustexist=True)
        if folder:
            self.set_inputs([Path(folder)], recursive=self.recursive.get())

    def set_inputs(self, paths, recursive=False):
        try:
            self.inputs, warnings = collect_inputs(paths, recursive)
            self.protected = [p.resolve() if p.is_dir() else p.resolve().parent for p in paths]
            self.input_note.set(f'{len(self.inputs)} image(s) selected | {len(warnings)} skipped entries | ' + str(paths[0]))
            self.log_messages(warnings)
            self.status.set('Inputs selected. Previous results remain until the next run; skipped entries are listed on Models & runs.')
        except Exception as exc:
            self.error(exc)

    def set_busy(self, busy):
        self.busy = busy
        for button in self.buttons:
            button.configure(state='disabled' if busy else 'normal')
        for box in [self.method_box, self.scope_box, *self.run_boxes.values()]:
            box.configure(state='disabled' if busy else 'readonly')
        for control in [self.output_entry, self.trust_check, self.recurse_check]:
            control.configure(state='disabled' if busy else 'normal')
        self.cancel_button.configure(state='normal' if busy else 'disabled')

    def launch(self, task):
        self.cancel.clear()
        self.worker_error = ''
        self.set_busy(True)
        def worker():
            try:
                task(self.events.put)
            except Exception as exc:
                self.events.put(('error', f'{type(exc).__name__}: {exc}'))
            finally:
                self.events.put(('done', None))
        threading.Thread(target=worker, daemon=True).start()

    def start(self, methods):
        if self.busy:
            return
        try:
            if not self.inputs:
                raise ValueError('Choose at least one supported image first.')
            if not self.trust.get():
                self.tabs.select(self.models_tab)
                raise ValueError("Confirm the model files are your team's trusted files on Models & runs.")
            runs = self.selected_runs(methods)
            from .ui_core import ensure_compatible
            ensure_compatible(runs)
            paths = list(self.inputs)
            self.rows = []
            self.cards = {}
            self.card_source = ''
            self.preview_canvas.delete('all')
            self.table.delete(*self.table.get_children())
            self.result_protected = [*self.protected, *[r.path for r in runs]]
            self.progress.configure(value=0, maximum=len(paths))
            self.status.set('Starting local prediction...')
            self.launch(lambda emit: run_batch(paths, runs, trusted=True, cancel=self.cancel,
                                              emit=lambda kind, value: emit((kind, value))))
        except Exception as exc:
            self.error(exc)

    def cancel_work(self):
        self.cancel.set()
        self.status.set('Cancelling after the current model/image operation; completed rows will remain exportable.')

    def poll(self):
        try:
            # Bounded queue/batch keeps memory and the Tk event loop responsive.
            for _ in range(24):
                kind, value = self.events.get_nowait()
                if kind == 'result':
                    row, card = value
                    self.rows.append(row)
                    values = [row.get(key, '') for key in ['source', 'method', 'status', 'predicted_stage', 'processing_ms', 'prediction_ms', 'error']]
                    values[4:6] = [f'{v:.1f}' if isinstance(v, float) else v for v in values[4:6]]
                    self.table.insert('', 'end', iid=str(len(self.rows)-1), values=values)
                    if row['source'] != self.card_source:
                        self.cards = {}
                        self.card_source = row['source']
                    if card:
                        self.cards[row['method']] = card
                    self.draw_preview()
                elif kind == 'preview':
                    self.card_source, self.cards = value
                    self.draw_preview()
                elif kind == 'status':
                    self.status.set(value)
                elif kind == 'progress':
                    completed, total = value
                    self.progress.configure(value=completed, maximum=total)
                    failed = sum(r['status'] != 'ok' for r in self.rows)
                    self.status.set(f'{completed}/{total} images processed | {len(self.rows)-failed} successful method results | {failed} errors/blocked')
                elif kind == 'error':
                    self.worker_error = value
                    self.error(value)
                elif kind == 'done':
                    self.set_busy(False)
                    failed = sum(r['status'] != 'ok' for r in self.rows)
                    if self.worker_error:
                        self.status.set(f'Failed: {self.worker_error} | Completed rows remain exportable.')
                    else:
                        self.status.set(f"{'Cancelled; partial results' if self.cancel.is_set() else 'Finished'} | {len(self.rows)-failed} successful method results | {failed} errors/blocked. Select a row to inspect.")
                    if self.closing:
                        self.root.destroy()
                        return
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def display(self, canvas, image, width):
        canvas.delete('all')
        display = image.resize((width, round(image.height*width/image.width)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display, master=self.root)
        canvas.photo = photo
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.configure(scrollregion=(0, 0, display.width, display.height))

    def draw_preview(self):
        if self.cards:
            sheet = comparison_sheet(self.card_source, list(self.cards.values()))
            self.display(self.preview_canvas, sheet, min(1180, max(900, self.preview_canvas.winfo_width()-20)))
        else:
            self.preview_canvas.delete('all')
            self.preview_canvas.create_text(25, 30, anchor='nw', text='No successful preview for this image. See the result/error row.', fill='#555555')

    def show_row_detail(self, event=None):
        selection = self.table.selection()
        if selection and not self.busy:
            row = self.rows[int(selection[0])]
            self.status.set(f"{row['source']} | {row['method']} | {row.get('error') or row.get('predicted_stage', '')}")

    def preview_selected(self):
        if self.busy:
            return
        selection = self.table.selection()
        if not selection:
            self.error('Select a result row first.')
            return
        source = self.rows[int(selection[0])]['source']
        rows = [r.copy() for r in self.rows if r['source'] == source and r['status'] == 'ok']
        if not rows:
            self.error('This image has no successful predictions; inspect its error row.')
            return
        def task(emit):
            blob = Path(source).read_bytes()
            if any(hashlib.sha256(blob).hexdigest() != r['sha256'] for r in rows):
                raise ValueError('Source image changed since prediction. Run it again; old scores cannot label new pixels.')
            cards = {}
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
                for row in rows:
                    if self.cancel.is_set():
                        return
                    # Reconstruct pixels only. Scores/timings remain the original recorded result.
                    cards[row['method']] = prediction_card(row, process_image(image, row['method']))
            emit(('preview', (source, cards)))
        self.status.set('Reconstructing preview from unchanged source; preserving recorded scores and timings...')
        self.launch(task)

    def load_evaluation(self):
        try:
            runs = self.selected_runs(LABELS, required=False)
            scope = self.scope.get()
            rows = evaluation_rows(runs, scope)
            image = evaluation_sheet(runs, scope)
            self.eval_rows, self.eval_image = rows, image
            self.eval_protected = [r.path for r in runs]
            self.display(self.eval_canvas, image, 1180)
            self.status.set(f'Loaded saved validation: {len(runs)}/7 selected runs | {scope}. Missing methods are omitted, not invented.')
        except Exception as exc:
            self.error(exc)

    def destination(self, title, extension):
        return filedialog.asksaveasfilename(parent=self.root, title=title, defaultextension=extension,
            filetypes=[(extension.upper()[1:] + ' file', '*' + extension)])

    def save_results(self):
        try:
            if not self.rows:
                raise ValueError('No prediction rows to export.')
            path = self.destination('Export predictions; choose a NEW filename outside input folders', '.csv')
            if path:
                export_csv(Path(path), self.rows, RESULT_FIELDS, self.result_protected)
                self.status.set(f'Exported predictions: {path}')
        except Exception as exc:
            self.error(exc)

    def save_preview(self):
        try:
            image = comparison_sheet(self.card_source, list(self.cards.values()))
            path = self.destination('Export displayed image comparison; choose a NEW filename', '.png')
            if path:
                export_png(Path(path), image, self.result_protected)
                self.status.set(f'Exported preview: {path}')
        except Exception as exc:
            self.error(exc)

    def save_metrics(self):
        try:
            if not self.eval_rows:
                raise ValueError('Load validation results first.')
            path = self.destination('Export displayed validation metrics', '.csv')
            if path:
                export_csv(Path(path), self.eval_rows, METRIC_FIELDS, [*self.protected, *self.eval_protected])
                self.status.set(f'Exported validation metrics: {path}')
        except Exception as exc:
            self.error(exc)

    def save_evaluation(self):
        try:
            if self.eval_image is None:
                raise ValueError('Load validation results first.')
            path = self.destination('Export displayed validation comparison and confusion matrices', '.png')
            if path:
                export_png(Path(path), self.eval_image, [*self.protected, *self.eval_protected])
                self.status.set(f'Exported validation comparison: {path}')
        except Exception as exc:
            self.error(exc)

    def error(self, exc):
        self.status.set(str(exc))
        if not self.closing:
            messagebox.showerror('Fruit Ripeness', str(exc), parent=self.root)

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel_work()
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f'Cannot open desktop UI: {exc}\nUse a graphical desktop with Python Tk support. Test with python -m tkinter.\n')
    App(root, args.outputs)
    root.mainloop()


if __name__ == '__main__':
    main()
