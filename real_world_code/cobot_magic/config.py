"""
Global configuration.

Any field that depends on the deployment environment should be overridable via
environment variables so the repository never embeds a user-specific absolute
path or secret. The Language Action (LA) and Vision Action (VA) endpoints
follow the same naming convention as the LaViRA simulation repo:

    https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code
"""
import os


# --- Robot motion parameters ---
LINEAR_SPEED = 0.25  # Forward speed in m/s
ANGULAR_SPEED_DEG = 15  # Rotation speed in deg/s
ARM_SPEED = 10  # Arm motion speed multiplier

# --- Navigation tuning ---
DIRECTION_OFFSET = 5  # Correction for the final rotation angle, in degrees
DISTANCE_OFFSET = 1.0  # Correction for the final forward distance, in meters
WINDOW_SIZE = 31  # Pixel window size used when sampling depth

# --- Panoramic view configuration ---
NUM_DIRECTIONS = 7  # Number of panoramic directions: 6, 7, or 8 (Cobot Magic uses 7 because the dual arms cannot reach the rear-facing pose)

# --- Language Action (LA) model ---
# Used for the direction-decision call.
# Recommended setups:
#   - API mode:   LA_MODEL_NAME=gemini-3.1-pro      (set LA_BASE_URL / LA_API_KEY accordingly)
#   - Local mode: LA_MODEL_NAME=Qwen3.5-27B-Q4      (4-bit quantized, served locally via vLLM)
# Defaults below target the local 4-bit setup so the repo runs without API keys.
LA_API_KEY = os.environ.get("LA_API_KEY", "EMPTY")
LA_BASE_URL = os.environ.get("LA_BASE_URL", "http://localhost:8000/v1")
LA_MODEL_NAME = os.environ.get("LA_MODEL_NAME", "Qwen3.5-27B-Q4")

# --- Vision Action (VA) model ---
# Used for target-detection / bounding-box calls.
# Recommended setups:
#   - API mode:   VA_MODEL_NAME=Qwen3.5-27B         (full-precision, hosted)
#   - Local mode: VA_MODEL_NAME=Qwen3.5-27B-Q4      (same quantized model as LA)
VA_API_KEY = os.environ.get("VA_API_KEY", "EMPTY")
VA_BASE_URL = os.environ.get("VA_BASE_URL", "http://localhost:8000/v1")
VA_MODEL_NAME = os.environ.get("VA_MODEL_NAME", "Qwen3.5-27B-Q4")

# --- Web server configuration ---
SERVER_HOST = os.environ.get("COBOT_HTTP_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("COBOT_HTTP_PORT", "5000"))
SSL_CERT_PATH = os.environ.get("COBOT_SSL_CERT", "cert.pem")  # Relative to the working directory
SSL_KEY_PATH = os.environ.get("COBOT_SSL_KEY", "key.pem")  # Relative to the working directory

# --- Speech recognition (Whisper) ---
# faster-whisper model directory. Defaults to ./models/faster-whisper-base
# relative to this file; override with COBOT_WHISPER_MODEL to point at any
# pre-downloaded snapshot.
LOCAL_WHISPER_MODEL_PATH = os.environ.get(
    "COBOT_WHISPER_MODEL",
    os.path.join(os.path.dirname(__file__), "models", "faster-whisper-base"),
)

# --- Web demo Flask secret ---
# The default is intentionally non-sensitive. Set FLASK_SECRET_KEY in production.
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me")

# --- Default instruction for text-instruction mode ---
DEFAULT_TEXT_INSTRUCTION = "go to the chair in front of you"
