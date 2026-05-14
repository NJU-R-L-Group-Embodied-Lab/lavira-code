"""
Flask + SocketIO interactive demo for the Unitree Go1 VLN task.

Users either type an instruction or press-and-hold the microphone button to
dictate one. The single VLN controller runs the navigation loop on a
background thread; status / response messages are pushed back via SocketIO.
"""
import os
import time

import cv2
import rospy
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

from config import Config
from utils import print_error

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = Config.FLASK_SECRET_KEY
socketio = SocketIO(app, async_mode="gevent")

# Injected by main.py
controller = None
local_whisper_model = None


def set_controller(c):
    global controller
    controller = c


def set_whisper_model(m):
    global local_whisper_model
    local_whisper_model = m


# ---------------------------------------------------------------------------
# Pages and video feed
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream the current front camera as MJPEG."""

    def generate():
        while not rospy.is_shutdown():
            if controller is not None:
                img = controller.robot.camera_data.get("camera1", {}).get("rgb_image")
                if img is not None:
                    success, buf = cv2.imencode(
                        ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                    )
                    if success:
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
            time.sleep(0.05)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    rospy.loginfo("Client connected")
    socketio.emit("response", {"message": "Robot ready. Type or speak an instruction."})


@socketio.on("disconnect")
def handle_disconnect():
    rospy.loginfo("Client disconnected")


@socketio.on("text_command")
def handle_text_command(data):
    """Run VLN from a typed instruction. Payload: ``{"instruction": "<text>"}``."""
    instruction = (data or {}).get("instruction", "").strip()
    if not instruction:
        socketio.emit("response", {"message": "Please type an instruction first."})
        return
    if controller is None:
        socketio.emit("response", {"message": "Robot not initialised yet."})
        return
    controller.start_new_task(instruction)


@socketio.on("audio_command")
def handle_audio_command(audio_bytes):
    """Transcribe the recorded audio with whisper and run the VLN task."""
    if not local_whisper_model:
        socketio.emit("response", {"message": "Local speech recognition is not available."})
        return
    if controller is None:
        socketio.emit("response", {"message": "Robot not initialised yet."})
        return

    temp_audio_path = os.path.join(Config.SESSION_DIR or ".", "temp_audio.webm")
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)
    try:
        socketio.emit("status_update", {"message": "Transcribing audio locally..."})
        segments, _ = local_whisper_model.transcribe(temp_audio_path, beam_size=5)
        transcript = "".join(seg.text for seg in segments).strip()
        socketio.emit("status_update", {"message": f"Understood: '{transcript}'"})
        if transcript:
            controller.start_new_task(transcript)
        else:
            socketio.emit("response", {"message": "Did not recognise any speech."})
    except Exception as e:
        print_error(f"Whisper error: {e}")
        socketio.emit("response", {"message": f"Could not process audio: {e}"})
    finally:
        try:
            os.remove(temp_audio_path)
        except OSError:
            pass
