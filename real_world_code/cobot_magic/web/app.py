import os
import rospy
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

import config


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = config.FLASK_SECRET_KEY
socketio = SocketIO(app, async_mode='gevent')

# Global controller / model handles. Both are injected from main.py.
controller = None
local_whisper_model = None


def set_controller(c):
    global controller
    controller = c


def set_whisper_model(m):
    global local_whisper_model
    local_whisper_model = m


@app.route('/')
def index():
    """Serve the main interaction page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Serve the live MJPEG video stream."""

    def generate():
        while not rospy.is_shutdown():
            if controller is not None:
                jpeg_bytes = controller.get_front_image_jpeg()
                if jpeg_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            rospy.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    rospy.loginfo("Client connected to WebSocket.")
    socketio.emit('response', {'message': 'Hello! I am ready for your command.'})


@socketio.on('disconnect')
def handle_disconnect():
    rospy.loginfo("Client disconnected from WebSocket.")


@socketio.on('audio_command')
def handle_audio_command(audio_data):
    """Handle an audio command from the web client using local faster-whisper."""
    rospy.loginfo("Received audio command from client.")
    if not local_whisper_model:
        socketio.emit('response', {'message': 'Sorry, the local voice recognition service is not available.'})
        return

    temp_audio_path = "temp_audio.webm"
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_data)

    try:
        socketio.emit('status_update', {'message': 'Transcribing audio locally...'})
        segments, _ = local_whisper_model.transcribe(temp_audio_path, beam_size=5)
        transcript = "".join(segment.text for segment in segments)

        rospy.loginfo(f"Local whisper transcription: '{transcript}'")
        socketio.emit('status_update', {'message': f"Understood: '{transcript}'. Starting task."})

        if controller is not None:
            controller.start_new_task(transcript.strip())

    except Exception as e:
        rospy.logerr(f"Error during local whisper transcription: {e}")
        socketio.emit('response', {'message': f"Sorry, I couldn't process the audio. Error: {e}"})
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
