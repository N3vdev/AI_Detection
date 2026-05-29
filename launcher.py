"""
AI Product Inspector — First-run bootstrapper
==============================================
Compile to a single EXE (run build_exe.bat):

  pyinstaller --onefile --windowed --name AI_Inspector
      --add-data "python311_embed.zip;."
      --add-data "get_pip.py;."
      launcher.py

Flow
----
  First run  → shows setup window, installs env + models, writes _env/.ready
  Every run  → _env/.ready exists → launch run_conveyor_ui.py directly
"""

import os
import sys
import subprocess
import zipfile
import threading
import traceback
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

# APP_DIR = folder containing the EXE (or the .py script during development)
APP_DIR = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent

ENV_DIR    = APP_DIR / "_env"
PYTHON_DIR = ENV_DIR / "python"
VENV_DIR   = ENV_DIR / "venv"
FLAG_FILE  = ENV_DIR / ".ready"
MODELS_DIR = APP_DIR / "_models"
LOG_FILE   = ENV_DIR / "setup.log"

VENV_PY    = VENV_DIR / "Scripts" / "python.exe"
VENV_PIP   = VENV_DIR / "Scripts" / "pip.exe"

REQS_FILE  = APP_DIR / "requirements_app.txt"
MAIN_APP   = APP_DIR / "run_conveyor_ui.py"

QWEN_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODELS = ["yolo11n.pt", "yolov8m-worldv2.pt"]


def _bundled(name: str) -> Path:
    """
    Find a file that was bundled inside the EXE via --add-data, OR falls back
    to the same directory as the EXE (for development / manual distribution).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / name
        if p.exists():
            return p
    return APP_DIR / name


EMBED_ZIP  = _bundled("python311_embed.zip")
GET_PIP_PY = _bundled("get_pip.py")


# ── GPU detection ──────────────────────────────────────────────────────────────

def _detect_gpu() -> str | None:
    """Return NVIDIA GPU name if present, else None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0:
            name = r.stdout.strip().splitlines()[0].strip()
            return name or None
    except Exception:
        pass
    return None


# ── Setup UI ───────────────────────────────────────────────────────────────────

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


class SetupUI:
    """Dark-themed progress window shown during first-run setup."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Product Inspector — First-time Setup")
        self.root.geometry("620x440")
        self.root.resizable(False, False)
        self.root.configure(bg="#0d0d0d")
        self._allow_close = False
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # block close during setup

        # Header
        tk.Label(self.root, text="AI PRODUCT INSPECTOR",
                 bg="#0d0d0d", fg="#22c55e",
                 font=("Segoe UI", 15, "bold")).pack(pady=(30, 4))
        tk.Label(self.root,
                 text="First-run setup — please do not close this window",
                 bg="#0d0d0d", fg="#444",
                 font=("Segoe UI", 9)).pack()

        # Progress bar
        self._pv = tk.DoubleVar()
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("G.Horizontal.TProgressbar",
                    background="#22c55e", troughcolor="#131313",
                    bordercolor="#1c1c1c", lightcolor="#22c55e", darkcolor="#16a34a")
        ttk.Progressbar(self.root, variable=self._pv,
                        style="G.Horizontal.TProgressbar",
                        maximum=100, length=540).pack(pady=(22, 6))

        self._step_var = tk.StringVar(value="Initialising...")
        tk.Label(self.root, textvariable=self._step_var,
                 bg="#0d0d0d", fg="#555",
                 font=("Segoe UI", 9)).pack()

        # Scrollable log
        self._log_w = scrolledtext.ScrolledText(
            self.root, height=11, width=74,
            bg="#080808", fg="#404040", insertbackground="#404040",
            font=("Consolas", 8), borderwidth=0, relief="flat",
        )
        self._log_w.pack(padx=20, pady=(14, 20))
        self._log_w.configure(state="disabled")

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(self, text: str, pct: float):
        self._step_var.set(text)
        self._pv.set(pct)
        self.root.update_idletasks()

    def log(self, line: str):
        self._log_w.configure(state="normal")
        self._log_w.insert("end", line.rstrip() + "\n")
        self._log_w.see("end")
        self._log_w.configure(state="disabled")
        self.root.update_idletasks()

    def finish_ok(self):
        self._allow_close = True
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.step("✓  Setup complete — launching app...", 100)
        self.root.after(1500, self.root.destroy)

    def finish_error(self, msg: str):
        self._allow_close = True
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.step(f"✗  Setup failed", 0)
        self._log_w.configure(fg="#f87171")
        self.log(f"\nERROR: {msg}")
        self.log(f"\nFull log saved to: {LOG_FILE}")
        self.log("Fix the issue and try again, or contact support.")
        tk.Label(self.root, text="Close this window when ready.",
                 bg="#0d0d0d", fg="#f87171", font=("Segoe UI", 9)).pack()


# ── Setup logic ────────────────────────────────────────────────────────────────

def _run(cmd, cwd=None, env=None, log=None) -> subprocess.CompletedProcess:
    """Run a command, optionally streaming output to a log callback."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if log:
        for line in (r.stdout + r.stderr).splitlines():
            if line.strip():
                log(line)
    return r


def do_setup(ui: "SetupUI | None", log_fn):
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Direct all log output to both the UI and the log file
    with open(LOG_FILE, "w", encoding="utf-8") as logf:

        def log(line: str):
            logf.write(line.rstrip() + "\n")
            logf.flush()
            log_fn(line)

        # ── 1. Extract embedded Python ────────────────────────────────────────
        if ui: ui.step("1 / 5  —  Extracting Python 3.11 runtime...", 4)
        log("=== Step 1: Python runtime")
        if not PYTHON_DIR.exists():
            if not EMBED_ZIP.exists():
                raise FileNotFoundError(
                    f"python311_embed.zip not found.\n"
                    f"Expected at: {EMBED_ZIP}\n"
                    "Ensure all distribution files are in the same folder as AI_Inspector.exe"
                )
            log(f"Extracting {EMBED_ZIP.name} ...")
            PYTHON_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(EMBED_ZIP) as z:
                z.extractall(PYTHON_DIR)
            # Enable site-packages so pip-installed packages are importable
            for pth in PYTHON_DIR.glob("python*._pth"):
                txt = pth.read_text()
                pth.write_text(txt.replace("#import site", "import site"))
            log("Runtime extracted.")
        else:
            log("Runtime already present — skipping extraction.")

        py_exe = PYTHON_DIR / "python.exe"

        # ── 2. Bootstrap pip ──────────────────────────────────────────────────
        if ui: ui.step("2 / 5  —  Bootstrapping pip...", 14)
        log("\n=== Step 2: pip")
        pip_marker = PYTHON_DIR / "_pip_ready"
        if not pip_marker.exists():
            if not GET_PIP_PY.exists():
                raise FileNotFoundError(
                    f"get_pip.py not found at {GET_PIP_PY}"
                )
            log("Bootstrapping pip ...")
            r = _run([str(py_exe), str(GET_PIP_PY), "--no-warn-script-location"],
                     cwd=str(PYTHON_DIR), log=log)
            if r.returncode != 0:
                raise RuntimeError(f"pip bootstrap failed (exit {r.returncode})")
            pip_marker.touch()
            log("pip ready.")
        else:
            log("pip already bootstrapped.")

        # ── 3. Create virtual environment ─────────────────────────────────────
        if ui: ui.step("3 / 5  —  Creating virtual environment...", 24)
        log("\n=== Step 3: virtual environment")
        if not VENV_DIR.exists():
            log("Creating venv ...")
            r = _run([str(py_exe), "-m", "venv", str(VENV_DIR)], log=log)
            if r.returncode != 0:
                raise RuntimeError(f"venv creation failed (exit {r.returncode})")
            log("venv created.")
        else:
            log("venv already exists.")

        # ── 4. Install packages ───────────────────────────────────────────────
        if ui: ui.step("4 / 5  —  Installing packages  (10–20 min)...", 33)
        log("\n=== Step 4: packages")

        gpu = _detect_gpu()
        if gpu:
            log(f"GPU detected: {gpu}")
            torch_cmd = [
                str(VENV_PIP), "install", "--upgrade",
                "torch", "torchvision", "accelerate",
                "--index-url", "https://download.pytorch.org/whl/cu121",
            ]
        else:
            log("No NVIDIA GPU — installing CPU-only PyTorch (slower inference).")
            torch_cmd = [
                str(VENV_PIP), "install", "--upgrade",
                "torch", "torchvision", "accelerate",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ]

        log("Installing PyTorch (may take several minutes) ...")
        r = _run(torch_cmd, log=log)
        if r.returncode != 0:
            raise RuntimeError(f"torch install failed (exit {r.returncode})")

        if ui: ui.step("4 / 5  —  Installing remaining packages...", 55)
        log("\nInstalling requirements_app.txt ...")
        r = _run([str(VENV_PIP), "install", "-r", str(REQS_FILE)], log=log)
        if r.returncode != 0:
            raise RuntimeError(f"requirements install failed (exit {r.returncode})")
        log("All packages installed.")

        # ── 5. Pre-download AI models ─────────────────────────────────────────
        if ui: ui.step("5 / 5  —  Downloading AI models  (~3–5 GB)...", 72)
        log(f"\n=== Step 5: AI models")

        hf_env = os.environ.copy()
        hf_env["HF_HOME"] = str(MODELS_DIR)
        hf_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        # Qwen 2.5-VL-3B
        log(f"Downloading {QWEN_MODEL} ...")
        r = _run(
            [str(VENV_PY), "-c",
             f"from huggingface_hub import snapshot_download; "
             f"snapshot_download('{QWEN_MODEL}'); print('[OK] Qwen download complete')"],
            env=hf_env, log=log,
        )
        if r.returncode != 0:
            log("[Warning] Qwen download failed — the app will re-attempt on first inspection.")
        else:
            log("Qwen model ready.")

        # YOLO models (ultralytics auto-downloads on first use, but we pre-fetch here)
        log("Pre-downloading YOLO models ...")
        for model_name in YOLO_MODELS:
            r = _run(
                [str(VENV_PY), "-c",
                 f"from ultralytics import YOLO; "
                 f"YOLO('{model_name}'); print('[OK] {model_name} ready')"],
                cwd=str(APP_DIR), env=hf_env, log=log,
            )
            if r.returncode != 0:
                log(f"[Warning] {model_name} pre-download failed — will download on first run.")

        # ── Done ──────────────────────────────────────────────────────────────
        FLAG_FILE.write_text("ready")
        log("\n✓ Setup complete. AI Inspector is ready.")


# ── App launcher ───────────────────────────────────────────────────────────────

def launch_app():
    env = os.environ.copy()
    env["HF_HOME"] = str(MODELS_DIR)
    # CREATE_NO_WINDOW so no console flickers when launched from EXE
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        [str(VENV_PY), str(MAIN_APP)],
        cwd=str(APP_DIR),
        env=env,
        creationflags=flags,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Already installed — go straight to launch
    if FLAG_FILE.exists():
        launch_app()
        return

    # ── First run: show setup UI ───────────────────────────────────────────────
    if not _HAS_TK:
        # Headless fallback (very unlikely on Windows)
        print("[AI Inspector] First-run setup — please wait...")
        try:
            do_setup(None, print)
            launch_app()
        except Exception as e:
            print(f"Setup failed: {e}")
            input("Press Enter to close.")
        return

    ui = SetupUI()
    error: list[Exception | None] = [None]

    def worker():
        try:
            do_setup(ui, ui.log)
        except Exception as e:
            error[0] = e
            traceback.print_exc()
        finally:
            ui.root.after(0, lambda: (
                ui.finish_ok() if error[0] is None else ui.finish_error(str(error[0]))
            ))

    threading.Thread(target=worker, daemon=True).start()
    ui.root.mainloop()

    if error[0] is None:
        launch_app()


if __name__ == "__main__":
    main()
