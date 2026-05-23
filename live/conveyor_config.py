# ── Camera Setup ───────────────────────────────────────────────────────────────
# Camera sources — use integer for USB/device, or URL string for IP stream
# IP Webcam (Android): install app → Start server → use the URL shown on screen
# Examples:
#   Single phone over WiFi : ["http://192.168.1.5:8080/video"]
#   3 cameras (conveyor)   : ["http://192.168.1.5:8080/video",
#                             "http://192.168.1.6:8080/video",
#                             "http://192.168.1.7:8080/video"]
#   USB webcam             : [0]  or  [0, 1, 2]
CAMERA_INDICES = ["http://192.168.1.5:8080/video"]   # ← replace with your phone's IP
CAMERA_RESOLUTION = (1280, 720)     # Width x Height
CAMERA_FPS = 30

# ── Trigger Camera (center camera watches for products) ────────────────────────
TRIGGER_CAMERA_INDEX = 0            # Index into CAMERA_INDICES list

# ROI: normalized (0.0–1.0) bands within the frame where products pass through
TRIGGER_ROI_Y_BAND = (0.25, 0.75)       # Vertical band
TRIGGER_ROI_X_CENTER_BAND = (0.3, 0.7) # Horizontal band — "product is centered"

TRIGGER_CONFIDENCE_THRESHOLD = 0.55     # YOLO detection confidence minimum
TRIGGER_MIN_GAP_FRAMES = 20            # Consecutive empty YOLO checks before re-arming (~2s)
TRIGGER_CHECK_EVERY_N_FRAMES = 3        # Run YOLO every Nth frame to save CPU

# ── Frame Buffer ───────────────────────────────────────────────────────────────
FRAME_BUFFER_SIZE = 90              # Frames per camera buffer (3s at 30fps)
FRAME_SYNC_WINDOW_MS = 200          # ±ms window to find matching frames at trigger
SHARPNESS_MIN_VARIANCE = 50.0       # Laplacian variance — below this = blurry

# ── Models ─────────────────────────────────────────────────────────────────────
YOLO_TRIGGER_MODEL = "yolov8n.pt"                       # Auto-downloads via ultralytics
BARCODE_DETECTOR_MODEL = "models/barcode_detector.pt"
DOTTED_OCR_MODEL = "models/dotted_ocr_retrained.pth"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"          # float16, ~6GB VRAM

# ── Pipeline ───────────────────────────────────────────────────────────────────
INSPECTION_QUEUE_MAX = 10           # Backpressure limit on the inspection queue
MAX_PRODUCTS = 100                  # Stop after this many products per session

# ── Storage ────────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
DB_PATH = "db/inspections.db"
JSON_LOG_PATH = "db/inspections_log.jsonl"
SNAPSHOT_JPEG_QUALITY = 95
