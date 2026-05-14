#!/usr/bin/env python3
"""
Cobot Magic navigation entry point.

Usage:
- Text-instruction mode (default):
    python main.py --instruction "go to the chair in front of you"
- Voice / web demo mode:
    python main.py --demo
"""
import argparse
import os
import sys
import time
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description="Cobot Magic VLM-driven navigation entry point.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Launch the voice + web SocketIO demo. Otherwise run the text-instruction loop.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Natural-language instruction for text-instruction mode "
             "(defaults to config.DEFAULT_TEXT_INSTRUCTION).",
    )
    parser.add_argument(
        "--cycle-pause",
        type=float,
        default=2.0,
        help="Seconds to pause between navigation cycles in text-instruction mode.",
    )
    return parser.parse_args()


def run_demo(timestamp):
    """Voice + web SocketIO demo mode."""
    # gevent monkey patching must happen before importing ROS / Flask / etc.
    from gevent import monkey
    monkey.patch_all()

    import rospy
    import config
    from web.app import app, socketio, set_controller, set_whisper_model
    from robot.nav_controller import IntegratedVisionNavController

    # --- Load the local whisper model if available ---
    if config.LOCAL_WHISPER_MODEL_PATH and os.path.exists(config.LOCAL_WHISPER_MODEL_PATH):
        try:
            from faster_whisper import WhisperModel
            print("Loading local whisper model on CPU...")
            local_whisper_model = WhisperModel(
                config.LOCAL_WHISPER_MODEL_PATH, device="cpu", compute_type="int8"
            )
            set_whisper_model(local_whisper_model)
            print("Local whisper model loaded successfully on CPU.")
        except ImportError:
            print("faster_whisper is not installed. Voice commands will be unavailable.")
        except Exception as e:
            print(f"Error loading local whisper model: {e}")
    else:
        print(
            f"Whisper model path not found: {config.LOCAL_WHISPER_MODEL_PATH}. "
            f"Voice commands will be unavailable."
        )

    rospy.init_node("integrated_vision_nav_controller", anonymous=True)
    controller = IntegratedVisionNavController(socketio_instance=socketio, timestamp=timestamp)
    set_controller(controller)
    rospy.on_shutdown(controller.shutdown)

    rospy.loginfo(
        f"Starting HTTPS web server on https://{config.SERVER_HOST}:{config.SERVER_PORT}"
    )
    socketio.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        keyfile=config.SSL_KEY_PATH,
        certfile=config.SSL_CERT_PATH,
        server_side=True,
    )


def run_text_instruction(timestamp, instruction, cycle_pause):
    """Text-instruction mode with no web server and no speech recognition."""
    import rospy
    from robot.nav_controller import IntegratedVisionNavController

    rospy.init_node("integrated_vision_nav_controller", anonymous=True)
    controller = IntegratedVisionNavController(
        socketio_instance=None, timestamp=timestamp, instruction=instruction
    )
    rospy.on_shutdown(controller.shutdown)

    rospy.loginfo(f"Text-instruction mode started. Instruction: {instruction}")
    while not rospy.is_shutdown():
        task_complete = controller.run_navigation_cycle(timestamp)
        if task_complete:
            rospy.loginfo("Navigation task complete. Stopping.")
            rospy.signal_shutdown("Task complete.")
            break
        rospy.sleep(cycle_pause)


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(timestamp, exist_ok=True)

    try:
        if args.demo:
            run_demo(timestamp)
        else:
            import config
            instruction = args.instruction or config.DEFAULT_TEXT_INSTRUCTION
            run_text_instruction(timestamp, instruction, args.cycle_pause)

    except Exception as e:
        # rospy.ROSInterruptException flows through here as well; treat it gently.
        try:
            import rospy
            if isinstance(e, rospy.ROSInterruptException):
                print("\nROS interrupted. Exiting.")
                return
        except Exception:
            pass
        print(f"\nAn unhandled error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
