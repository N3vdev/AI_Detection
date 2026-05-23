# ── Camera Sources ─────────────────────────────────────────────────────────────
# Use integer for USB/device index, or URL string for IP stream
# IP Webcam (Android): install app → Start server → use URL shown on screen
# Examples:
#   Single phone (WiFi) : ["http://192.168.1.5:8080/video"]
#   3 conveyor cameras  : ["http://192.168.1.5:8080/video",
#                          "http://192.168.1.6:8080/video",
#                          "http://192.168.1.7:8080/video"]
#   USB webcam          : [0]  or  [0, 1, 2]
CAMERA_INDICES = ["http://192.168.1.5:8080/video"]   # ← replace with your phone's IP
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FPS = 30

# ── Trigger Camera ─────────────────────────────────────────────────────────────
TRIGGER_CAMERA_INDEX = 0            # Index into CAMERA_INDICES list

# Trigger zone — normalized (0.0–1.0) region of the frame to watch
TRIGGER_ROI_Y_BAND = (0.1, 0.9)    # Tall band — catches products at any height
TRIGGER_ROI_X_CENTER_BAND = (0.1, 0.9)  # Wide band — whole frame center

# MOG2 background subtraction settings
MOG2_HISTORY = 300                  # Frames to build background model
MOG2_VAR_THRESHOLD = 40             # Sensitivity — lower = more sensitive
TRIGGER_MIN_AREA = 3000             # Min foreground px² to count as a product
                                    # Increase if getting false triggers from noise
TRIGGER_MIN_GAP_FRAMES = 20        # Consecutive clear checks before re-arming
TRIGGER_CHECK_EVERY_N_FRAMES = 2   # Run detection every Nth frame (~15fps checks)

# ── Frame Buffer & Sync ────────────────────────────────────────────────────────
FRAME_BUFFER_SIZE = 90              # Frames per camera buffer (3s at 30fps)
FRAME_SYNC_WINDOW_MS = 200          # ±ms window to pick frames at trigger time
SHARPNESS_MIN_VARIANCE = 50.0       # Laplacian variance below this = blurry

# ── Models ─────────────────────────────────────────────────────────────────────
BARCODE_DETECTOR_MODEL = "models/barcode_detector.pt"
DOTTED_OCR_MODEL = "models/dotted_ocr_retrained.pth"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── Pipeline ───────────────────────────────────────────────────────────────────
INSPECTION_QUEUE_MAX = 10
MAX_PRODUCTS = 100

# ── Storage ────────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
DB_PATH = "db/inspections.db"
JSON_LOG_PATH = "db/inspections_log.jsonl"
SNAPSHOT_JPEG_QUALITY = 95
