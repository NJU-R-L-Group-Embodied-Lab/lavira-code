"""
Logging, JSON, and image-encoding helpers used across the codebase.
"""
import os
import json
import base64
import re
from io import BytesIO
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from colorama import Fore, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = ""

        def __getattr__(self, _name):
            return ""

    Fore = _NoColour()


# ---------------------------------------------------------------------------
# Coloured stdout logging
# ---------------------------------------------------------------------------
def print_step(step_num: int, description: str):
    print(Fore.CYAN + f"\n[STEP {step_num}] {description}")


def print_action(action: str, details: str = ""):
    print(Fore.GREEN + f"[ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str):
    print(Fore.BLUE + f"[INFO] {info}")


def print_warning(warning: str):
    print(Fore.YELLOW + f"[WARNING] {warning}")


def print_error(error: str):
    print(Fore.RED + f"[ERROR] {error}")


def print_success(success: str):
    print(Fore.GREEN + f"[SUCCESS] {success}")


def print_model_interaction(model_name, prompt, response,
                            speed=None, duration=None, prompt_speed=None):
    """Pretty-print a model call. Truncates base64 image payloads in the prompt."""
    print(Fore.YELLOW + "=" * 60)
    print(Fore.YELLOW + f"MODEL: {model_name}")
    print(Fore.CYAN + "-" * 20 + " PROMPT " + "-" * 20)

    clean_prompt = str(prompt)
    if "data:image" in clean_prompt:
        clean_prompt = re.sub(
            r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+", "[IMAGE_BASE64]", clean_prompt
        )
    print(clean_prompt)

    print(Fore.GREEN + "-" * 20 + " RESPONSE " + "-" * 20)
    print(response)

    if any(v is not None for v in (speed, duration, prompt_speed)):
        print(Fore.MAGENTA + "-" * 20 + " STATS " + "-" * 20)
        if prompt_speed is not None:
            print(Fore.MAGENTA + f"Input speed:  {prompt_speed:.2f} tokens/s")
        if speed is not None:
            print(Fore.MAGENTA + f"Output speed: {speed:.2f} tokens/s")
        if duration is not None:
            print(Fore.MAGENTA + f"Duration:     {duration:.2f}s")
    print(Fore.YELLOW + "=" * 60)


# ---------------------------------------------------------------------------
# Output / image helpers
# ---------------------------------------------------------------------------
def save_output(output_dir: str, filename: str, content: Union[Dict, List, str]):
    """Persist a JSON-serialisable object or a raw string to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if isinstance(content, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
    print_action(f"Saved output file: {filename}")


def numpy_to_base64(img_np: np.ndarray, max_size: int = 1024, quality: int = 85) -> str:
    """Encode an OpenCV (BGR) image array as a base64 JPEG string."""
    if img_np is None:
        return ""

    h, w = img_np.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h))

    success, buffer = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        print_error("Failed to encode image to JPEG")
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def img_to_base64(img_path: str, max_size: int = 256, quality: int = 85) -> str:
    """Read an image file from disk and convert it to a base64 JPEG string."""
    if not os.path.exists(img_path):
        print_error(f"Image file not found: {img_path}")
        return ""
    try:
        img = PILImage.open(img_path).convert("RGB")
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print_error(f"Failed to convert image to base64: {e}")
        return ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def safe_json_loads(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from an LLM response.

    Falls back through three layers: direct parse, extract first ``{...}``
    block and fix common syntax errors, and manual key extraction.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        print_error("JSON parsing failed - no JSON structure found")
        return {}

    json_str = json_match.group()
    try:
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r"(\{|\,\s*)(\w+)\s*:", r'\1"\2":', json_str)
        json_str = re.sub(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}])", r':"\1"\2', json_str)
        json_str = re.sub(r":\s*(true|false|null)\s*([,}])", r":\1\2", json_str)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        result = {}
        for key in ("description", "reasoning", "turn_direction", "action"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
            if m:
                result[key] = m.group(1).strip()
        return result
