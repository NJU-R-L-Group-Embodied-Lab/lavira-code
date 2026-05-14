"""
Global configuration for the Unitree Go1 LaViRA deployment.

Every environment-specific field is overridable via environment variables so
the repository ships zero absolute paths and zero embedded secrets. The
Language Action (LA) and Vision Action (VA) endpoints follow the same naming
convention as the LaViRA simulation repo
(https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code).
"""
import argparse
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Static defaults (overridable via env vars)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- Language Action (LA) model ---
    # API profile: gemini-3.1-pro (or any hosted OpenAI-compatible endpoint)
    # Local profile: Qwen3.5-27B-Q4 served via vLLM / llama-server
    LA_API_KEY = os.environ.get("LA_API_KEY", "EMPTY")
    LA_BASE_URL = os.environ.get("LA_BASE_URL", "http://localhost:8000/v1")
    LA_MODEL_NAME = os.environ.get("LA_MODEL_NAME", "Qwen3.5-27B-Q4")

    # --- Vision Action (VA) model ---
    # API profile: Qwen3.5-27B (full precision, hosted)
    # Local profile: Qwen3.5-27B-Q4 (same vLLM instance as LA)
    VA_API_KEY = os.environ.get("VA_API_KEY", "EMPTY")
    VA_BASE_URL = os.environ.get("VA_BASE_URL", "http://localhost:8000/v1")
    VA_MODEL_NAME = os.environ.get("VA_MODEL_NAME", "Qwen3.5-27B-Q4")

    # --- Output directories ---
    OUTPUT_DIR = os.environ.get("COBOT_OUTPUT_DIR", "outputs")
    PANORAMA_DIR = "panorama_images"  # relative; rewritten in parse()
    CURRENT_VIEW_IMG = "current_view/current.png"

    # --- Session paths (populated in parse) ---
    SESSION_DIR = ""
    LOG_DIR = ""
    IMG_DIR = ""
    IPLANNER_DIR = ""
    BBOX_DIR = ""

    # --- Robot / iPlanner ---
    IPLANNER_URL = os.environ.get("IPLANNER_URL", "http://localhost:8888")

    # --- Camera extrinsics (calibration) ---
    CAMERA_HEIGHT = float(os.environ.get("CAMERA_HEIGHT", "0.3"))  # meters
    CAMERA_ROLL_CORRECTION = float(os.environ.get("CAMERA_ROLL_CORRECTION", "0.0"))

    # --- Unitree SDK ---
    # Native sdk binaries live outside the repo. Set this env var to the
    # directory containing robot_interface.so (e.g. the arm64 build of
    # unitree_legged_sdk). The default is empty, which causes the loader to
    # fall back to whatever is already on sys.path / PYTHONPATH.
    UNITREE_SDK_PATH = os.environ.get("UNITREE_SDK_PATH", "")
    UNITREE_HOST = os.environ.get("UNITREE_HOST", "192.168.123.161")
    UNITREE_LOCAL_PORT = int(os.environ.get("UNITREE_LOCAL_PORT", "8080"))
    UNITREE_REMOTE_PORT = int(os.environ.get("UNITREE_REMOTE_PORT", "8082"))

    # --- Web demo ---
    SERVER_HOST = os.environ.get("COBOT_HTTP_HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("COBOT_HTTP_PORT", "5000"))
    SSL_CERT_PATH = os.environ.get("COBOT_SSL_CERT", "cert.pem")
    SSL_KEY_PATH = os.environ.get("COBOT_SSL_KEY", "key.pem")
    FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me")

    # --- Speech recognition (Whisper) ---
    LOCAL_WHISPER_MODEL_PATH = os.environ.get(
        "COBOT_WHISPER_MODEL",
        os.path.join(_HERE, "models", "faster-whisper-base"),
    )

    # --- Default instruction (text mode) ---
    DEFAULT_INSTRUCTION = os.environ.get(
        "COBOT_DEFAULT_INSTRUCTION", "go to the chair in front of you"
    )

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Unitree Go1 LaViRA VLN entry point."
        )
        self._setup_arguments()

    def _setup_arguments(self):
        self.parser.add_argument(
            "--demo", action="store_true",
            help="Launch the voice + web SocketIO demo. Otherwise run a "
                 "single VLN task from the command line.",
        )
        self.parser.add_argument(
            "--instruction", type=str, default=None,
            help="Natural-language VLN instruction (defaults to "
                 "config.DEFAULT_INSTRUCTION).",
        )
        self.parser.add_argument(
            "--output-dir", type=str, default=None,
            help="Override the output directory.",
        )

    def parse(self):
        args = self.parser.parse_args()

        if args.output_dir:
            Config.OUTPUT_DIR = args.output_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Config.SESSION_DIR = os.path.join(Config.OUTPUT_DIR, timestamp)
        Config.LOG_DIR = os.path.join(Config.SESSION_DIR, "logs")
        Config.IMG_DIR = os.path.join(Config.SESSION_DIR, "images")
        Config.PANORAMA_DIR = os.path.join(Config.IMG_DIR, "panorama")
        Config.IPLANNER_DIR = os.path.join(Config.IMG_DIR, "iplanner")
        Config.BBOX_DIR = os.path.join(Config.IMG_DIR, "bbox")

        for d in (Config.SESSION_DIR, Config.LOG_DIR, Config.IMG_DIR,
                  Config.PANORAMA_DIR, Config.IPLANNER_DIR, Config.BBOX_DIR):
            os.makedirs(d, exist_ok=True)

        print(f"[Config] LA: {Config.LA_MODEL_NAME} @ {Config.LA_BASE_URL}")
        print(f"[Config] VA: {Config.VA_MODEL_NAME} @ {Config.VA_BASE_URL}")
        print(f"[Config] Session directory: {Config.SESSION_DIR}")
        return args


# Module-level singleton for argparse-driven entry points.
config_manager = Config()
