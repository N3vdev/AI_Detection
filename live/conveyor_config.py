# ── Camera Sources ─────────────────────────────────────────────────────────────
# Use integer for USB/device index, or URL string for IP stream
# IP Webcam (Android): install app → Start server → use URL shown on screen
# Examples:
#   Single phone (WiFi) : ["http://192.168.1.5:8080/video"]
#   3 conveyor cameras  : ["http://192.168.1.5:8080/video",
#                          "http://192.168.1.6:8080/video",
#                          "http://192.168.1.7:8080/video"]
#   USB webcam          : [0]  or  [0, 1, 2]
CAMERA_INDICES = ["http://192.168.0.199:8080/video"]   # ← replace with your phone's IP
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FPS = 30

# Rotate frames after capture — fixes portrait phone streams arriving as landscape
# Options: cv2.ROTATE_90_CLOCKWISE | cv2.ROTATE_90_COUNTERCLOCKWISE | cv2.ROTATE_180 | None
CAMERA_ROTATE = None   # None = no rotation (use phone's natural stream orientation)

# Preview window size (small, so it doesn't take up the whole screen)
PREVIEW_WIDTH  = 360
PREVIEW_HEIGHT = 640

# ── Trigger Camera ─────────────────────────────────────────────────────────────
TRIGGER_CAMERA_INDEX = 0            # Index into CAMERA_INDICES list
YOLO_TRIGGER_MODEL = "yolo11n.pt"   # Auto-downloads via ultralytics

# Trigger zone — normalized (0.0–1.0). Wide zone catches products at any position.
TRIGGER_ROI_Y_BAND = (0.0, 1.0)
TRIGGER_ROI_X_CENTER_BAND = (0.0, 1.0)

TRIGGER_CONFIDENCE_THRESHOLD = 0.30  # Raise to avoid empty-space false positives
TRIGGER_MIN_BOX_AREA = 0.05          # Object must cover ≥5% of frame (filters tiny noise)
TRIGGER_ENTER_FRAMES = 3             # Consecutive detections to confirm product entered
TRIGGER_LEAVE_FRAMES = 8             # Consecutive empty frames to confirm product left

# ── Frame Buffer & Sync ────────────────────────────────────────────────────────
FRAME_BUFFER_SIZE = 90              # Frames per camera buffer (3s at 30fps)
FRAME_SYNC_WINDOW_MS = 200          # ±ms window to pick frames at trigger time
SHARPNESS_MIN_VARIANCE = 50.0       # Laplacian variance below this = blurry

# ── Models ─────────────────────────────────────────────────────────────────────
BARCODE_DETECTOR_MODEL = "models/barcode_detector.pt"
DOTTED_OCR_MODEL = "models/dotted_ocr_retrained.pth"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# YOLO-World open-vocabulary region detector (auto-downloads)
# yolov8s-worldv2.pt = fast (~50ms), yolov8m-worldv2.pt = more accurate (~100ms)
YOLO_WORLD_MODEL = "yolov8m-worldv2.pt"

# ── Pipeline ───────────────────────────────────────────────────────────────────
INSPECTION_QUEUE_MAX = 10
MAX_PRODUCTS = 100

# ── Storage ────────────────────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
DB_PATH = "db/inspections.db"
JSON_LOG_PATH = "db/inspections_log.jsonl"
SNAPSHOT_JPEG_QUALITY = 95

# ── Debug ──────────────────────────────────────────────────────────────────────
# Set True while tuning — saves per-step pipeline images to debug_snapshots/.
# Set False for production: avoids filling disk with hundreds of debug folders.
SAVE_DEBUG_SNAPSHOTS = True
