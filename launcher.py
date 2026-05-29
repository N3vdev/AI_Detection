"""
AI Product Inspector — First-run bootstrapper
==============================================
Compile to a single EXE (run build_exe.bat):

  pyinstaller --onefile --windowed --name AI_Inspector
      --add-data "python313_embed.zip;."
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

APP_DIR    = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent

ENV_DIR    = APP_DIR / "_env"
PYTHON_DIR = ENV_DIR / "python"
FLAG_FILE  = ENV_DIR / ".ready"
MODELS_DIR = APP_DIR / "_models"
LOG_FILE   = ENV_DIR / "setup.log"

REQS_FILE  = APP_DIR / "requirements_app.txt"
MAIN_APP   = APP_DIR / "run_conveyor_ui.py"

QWEN_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"


def _bundled(name: str) -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / name
        if p.exists():
            return p
    return APP_DIR / name


EMBED_ZIP  = _bundled("python313_embed.zip")
GET_PIP_PY = _bundled("get_pip.py")

# Derived after extraction
def _py_exe()  -> Path: return PYTHON_DIR / "python.exe"
def _pip_exe() -> Path: return PYTHON_DIR / "Scripts" / "pip.exe"


# ── GPU detection ──────────────────────────────────────────────────────────────

def _detect_gpu() -> str | None:
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
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

        tk.Label(self.root, text="AI PRODUCT INSPECTOR",
                 bg="#0d0d0d", fg="#22c55e",
                 font=("Segoe UI", 15, "bold")).pack(pady=(30, 4))
        tk.Label(self.root,
                 text="First-run setup — please do not close this window",
                 bg="#0d0d0d", fg="#444",
                 font=("Segoe UI", 9)).pack()

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

        self._log_w = scrolledtext.ScrolledText(
            self.root, height=11, width=74,
            bg="#080808", fg="#404040", insertbackground="#404040",
            font=("Consolas", 8), borderwidth=0, relief="flat",
        )
        self._log_w.pack(padx=20, pady=(14, 20))
        self._log_w.configure(state="disabled")

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
        self.step("✗  Setup failed", 0)
        self._log_w.configure(fg="#f87171")
        self.log(f"\nERROR: {msg}")
        self.log(f"\nFull log saved to: {LOG_FILE}")
        self.log("Fix the issue and try again, or contact support.")
        tk.Label(self.root, text="Close this window when ready.",
                 bg="#0d0d0d", fg="#f87171", font=("Segoe UI", 9)).pack()


# ── Setup logic ────────────────────────────────────────────────────────────────

def _run(cmd, cwd=None, env=None, log=None) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if log:
        for line in (r.stdout + r.stderr).splitlines():
            if line.strip():
                log(line)
    return r


def do_setup(ui: "SetupUI | None", log_fn):
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as logf:

        def log(line: str):
            logf.write(line.rstrip() + "\n")
            logf.flush()
            log_fn(line)

        # ── 1. Extract embedded Python ────────────────────────────────────────
        if ui: ui.step("1 / 4  —  Extracting Python 3.13 runtime...", 5)
        log("=== Step 1: Python runtime")
        if not PYTHON_DIR.exists():
            if not EMBED_ZIP.exists():
                raise FileNotFoundError(
                    f"python313_embed.zip not found.\n"
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
                # Try uncommenting the existing directive
                txt = txt.replace("#import site", "import site")
                # Also explicitly add the path — works even if the directive format differs
                if "Lib\\site-packages" not in txt and "Lib/site-packages" not in txt:
                    txt = txt.rstrip() + "\nLib\\site-packages\n"
                pth.write_text(txt)
            log("Runtime extracted.")
        else:
            log("Runtime already present — skipping extraction.")

        # ── 2. Bootstrap pip ──────────────────────────────────────────────────
        if ui: ui.step("2 / 4  —  Bootstrapping pip...", 18)
        log("\n=== Step 2: pip")
        pip_marker = PYTHON_DIR / "_pip_ready"
        if not pip_marker.exists():
            if not GET_PIP_PY.exists():
                raise FileNotFoundError(f"get_pip.py not found at {GET_PIP_PY}")
            log("Bootstrapping pip ...")
            r = _run([str(_py_exe()), str(GET_PIP_PY), "--no-warn-script-location"],
                     cwd=str(PYTHON_DIR), log=log)
            if r.returncode != 0:
                raise RuntimeError(f"pip bootstrap failed (exit {r.returncode})")
            pip_marker.touch()
            log("pip ready.")
        else:
            log("pip already bootstrapped.")

        # ── 3. Install packages ───────────────────────────────────────────────
        if ui: ui.step("3 / 4  —  Installing packages  (10–20 min)...", 30)
        log("\n=== Step 3: packages")

        gpu = _detect_gpu()
        if gpu:
            log(f"GPU detected: {gpu}")
            torch_index = "https://download.pytorch.org/whl/cu124"
        else:
            log("No NVIDIA GPU — installing CPU-only PyTorch (slower inference).")
            torch_index = "https://download.pytorch.org/whl/cpu"

        # --index-url (not --extra-index-url) guarantees pip picks the CUDA wheel
        # from PyTorch's server rather than the CPU build from PyPI.
        log("Installing PyTorch (may take several minutes) ...")
        r = _run([
            str(_pip_exe()), "install", "--upgrade",
            "torch", "torchvision",
            "--index-url", torch_index,
        ], log=log)
        if r.returncode != 0:
            raise RuntimeError(f"torch install failed (exit {r.returncode})")

        # accelerate lives on PyPI, not the PyTorch wheel server — install separately.
        log("Installing accelerate ...")
        r = _run([str(_pip_exe()), "install", "--upgrade", "accelerate"], log=log)
        if r.returncode != 0:
            raise RuntimeError(f"accelerate install failed (exit {r.returncode})")

        # Confirm CUDA is visible
        r = _run([str(_py_exe()), "-c",
                  "import torch; print('[CUDA]', torch.cuda.is_available(), "
                  "torch.version.cuda if torch.cuda.is_available() else 'CPU only')"],
                 log=log)

        if ui: ui.step("3 / 4  —  Installing remaining packages...", 58)
        log("\nInstalling requirements_app.txt ...")
        r = _run([str(_pip_exe()), "install", "-r", str(REQS_FILE)], log=log)
        if r.returncode != 0:
            raise RuntimeError(f"requirements install failed (exit {r.returncode})")
        log("All packages installed.")

        # ── 4. Pre-download AI models ─────────────────────────────────────────
        if ui: ui.step("4 / 4  —  Checking AI models...", 74)
        log("\n=== Step 4: AI models")

        # Check if model already exists in the system HF cache
        default_hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_cache_dir = default_hf / "hub" / ("models--" + QWEN_MODEL.replace("/", "--"))

        if model_cache_dir.exists():
            log(f"Qwen model found in existing cache: {default_hf}")
            log("Skipping download — using existing installation.")
            effective_hf_home = str(default_hf)
        else:
            if ui: ui.step("4 / 4  —  Downloading AI models  (~3–5 GB)...", 74)
            effective_hf_home = str(MODELS_DIR)
            hf_env = os.environ.copy()
            hf_env["HF_HOME"] = effective_hf_home
            hf_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            log(f"Downloading {QWEN_MODEL} ...")
            r = _run(
                [str(_py_exe()), "-c",
                 f"from huggingface_hub import snapshot_download; "
                 f"snapshot_download('{QWEN_MODEL}'); print('[OK] Qwen download complete')"],
                env=hf_env, log=log,
            )
            if r.returncode != 0:
                log("[Warning] Qwen download failed — the app will re-attempt on first inspection.")
            else:
                log("Qwen model ready.")

        log("YOLO models will be downloaded automatically on first inspection run.")

        # Save effective HF_HOME so launch_app() uses the same location
        (ENV_DIR / "hf_home.txt").write_text(effective_hf_home)

        FLAG_FILE.write_text("ready")
        log("\n✓ Setup complete. AI Inspector is ready.")


# ── App launcher ───────────────────────────────────────────────────────────────

def launch_app():
    env = os.environ.copy()
    hf_home_file = ENV_DIR / "hf_home.txt"
    env["HF_HOME"] = hf_home_file.read_text().strip() if hf_home_file.exists() else str(MODELS_DIR)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        [str(_py_exe()), str(MAIN_APP)],
        cwd=str(APP_DIR),
        env=env,
        creationflags=flags,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if FLAG_FILE.exists():
        launch_app()
        return

    if not _HAS_TK:
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
