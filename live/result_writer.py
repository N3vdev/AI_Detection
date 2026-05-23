import json
import os
import sqlite3
import threading


_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS truck_sessions (
    session_id       TEXT PRIMARY KEY,
    start_time       TEXT NOT NULL,
    end_time         TEXT,
    total_products   INTEGER DEFAULT 0,
    barcode_count    INTEGER DEFAULT 0,
    vlm_count        INTEGER DEFAULT 0,
    incomplete_count INTEGER DEFAULT 0
);
"""

_CREATE_BARCODE = """
CREATE TABLE IF NOT EXISTS barcode_products (
    product_id     TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    seq_number     INTEGER NOT NULL,
    barcode_value  TEXT NOT NULL,
    trigger_time   TEXT NOT NULL,
    snapshot_cam0  TEXT,
    snapshot_cam1  TEXT,
    snapshot_cam2  TEXT,
    processing_ms  INTEGER,
    FOREIGN KEY (session_id) REFERENCES truck_sessions(session_id)
);
"""

_CREATE_VLM = """
CREATE TABLE IF NOT EXISTS vlm_products (
    product_id        TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    seq_number        INTEGER NOT NULL,
    brand             TEXT,
    product_name      TEXT,
    product_category  TEXT,
    expiry_date       TEXT,
    manufacture_date  TEXT,
    batch_number      TEXT,
    dotted_label_text TEXT,
    vlm_response_raw  TEXT,
    status            TEXT NOT NULL,
    trigger_time      TEXT NOT NULL,
    snapshot_cam0     TEXT,
    snapshot_cam1     TEXT,
    snapshot_cam2     TEXT,
    processing_ms     INTEGER,
    FOREIGN KEY (session_id) REFERENCES truck_sessions(session_id)
);
"""


class ResultWriter:
    def __init__(self, db_path, json_log_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._json_path = json_log_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.cursor()
        cur.executescript(_CREATE_SESSIONS + _CREATE_BARCODE + _CREATE_VLM)
        self._conn.commit()

    def start_session(self, session_id, start_time):
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO truck_sessions (session_id, start_time) VALUES (?, ?)",
                (session_id, start_time),
            )
            self._conn.commit()

    def end_session(self, session_id, end_time):
        with self._lock:
            self._conn.execute(
                "UPDATE truck_sessions SET end_time=? WHERE session_id=?",
                (end_time, session_id),
            )
            self._conn.commit()

    def write(self, result: dict):
        with self._lock:
            if result.get("barcode"):
                self._write_barcode(result)
            else:
                self._write_vlm(result)
            self._update_session_counts(result)
            self._append_jsonl(result)

    def _write_barcode(self, r):
        self._conn.execute(
            """INSERT OR REPLACE INTO barcode_products
               (product_id, session_id, seq_number, barcode_value,
                trigger_time, snapshot_cam0, snapshot_cam1, snapshot_cam2, processing_ms)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (r["product_id"], r["session_id"], r["seq_number"], r["barcode"],
             r["trigger_time"], r.get("snapshot_cam0"), r.get("snapshot_cam1"),
             r.get("snapshot_cam2"), r.get("processing_ms")),
        )
        self._conn.commit()

    def _write_vlm(self, r):
        self._conn.execute(
            """INSERT OR REPLACE INTO vlm_products
               (product_id, session_id, seq_number, brand, product_name, product_category,
                expiry_date, manufacture_date, batch_number, dotted_label_text,
                status, trigger_time, snapshot_cam0, snapshot_cam1, snapshot_cam2, processing_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["product_id"], r["session_id"], r["seq_number"],
             r.get("brand"), r.get("product_name"), r.get("product_category"),
             r.get("expiry_date"), r.get("manufacture_date"), r.get("batch_number"),
             r.get("dotted_label_text"), r.get("status", "Incomplete"),
             r["trigger_time"], r.get("snapshot_cam0"), r.get("snapshot_cam1"),
             r.get("snapshot_cam2"), r.get("processing_ms")),
        )
        self._conn.commit()

    def _update_session_counts(self, r):
        sid = r.get("session_id")
        if not sid:
            return
        is_barcode = bool(r.get("barcode"))
        is_incomplete = not is_barcode and r.get("status", "").startswith("Incomplete")
        self._conn.execute(
            """UPDATE truck_sessions SET
               total_products   = total_products + 1,
               barcode_count    = barcode_count    + ?,
               vlm_count        = vlm_count        + ?,
               incomplete_count = incomplete_count + ?
               WHERE session_id = ?""",
            (1 if is_barcode else 0,
             0 if is_barcode else 1,
             1 if is_incomplete else 0,
             sid),
        )
        self._conn.commit()

    def _append_jsonl(self, result):
        with open(self._json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")

    def get_session_summary(self, session_id):
        cur = self._conn.execute(
            "SELECT * FROM truck_sessions WHERE session_id=?", (session_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else {}

    def close(self):
        with self._lock:
            self._conn.close()
