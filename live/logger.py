"""
Session logger — redirects stdout + stderr to both console and a timestamped log file.

Usage (call once at process startup):
    from live.logger import setup_logging
    setup_logging()          # writes to logs/session_YYYY-MM-DD_HH-MM-SS.log
    setup_logging("my.log")  # explicit path
"""

import sys
import os
import threading
import datetime


class TeeLogger:
    """Writes every line to the original stream AND a log file, prepending a timestamp."""

    def __init__(self, original_stream, log_file):
        self._out  = original_stream
        self._file = log_file
        self._lock = threading.Lock()
        self._buf  = ""

    def write(self, text):
        with self._lock:
            # Always pass through to the original stream immediately
            self._out.write(text)
            self._out.flush()
            # Buffer until we have complete lines, then write with timestamps
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                entry = f"{ts}  {line}\n" if line else "\n"
                self._file.write(entry)
            self._file.flush()

    def flush(self):
        self._out.flush()
        self._file.flush()

    def fileno(self):
        # Some libraries call fileno() to check if the stream is a real file.
        # Delegate to original so they don't break.
        try:
            return self._out.fileno()
        except Exception:
            raise io.UnsupportedOperation("fileno")

    def isatty(self):
        return False


def setup_logging(log_path: str | None = None, log_dir: str = "logs") -> str:
    """
    Redirect sys.stdout and sys.stderr to TeeLogger instances.
    Returns the path of the log file being written.

    Call once at the very top of the entry point before any other imports print.
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"session_{ts}.log")

    # Open in append mode so reruns to the same path accumulate
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    # Write a header so it's easy to spot where a run starts in a long file
    header = (
        f"\n{'='*72}\n"
        f"  SESSION START  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*72}\n"
    )
    log_file.write(header)
    log_file.flush()

    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)

    print(f"[Logger] Logging to: {os.path.abspath(log_path)}")
    return log_path
