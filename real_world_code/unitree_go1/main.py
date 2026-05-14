#!/usr/bin/env python3
"""
Unitree Go1 LaViRA VLN entry point.

Usage:
    # Headless text-instruction VLN run
    python main.py --instruction "go to the chair in front of you"

    # Voice + web demo (browser microphone, status streamed back over SocketIO)
    python main.py --demo
"""
import os
import signal
import sys
import time
import traceback

from config import config_manager
from utils import print_error, print_info, print_success, print_warning


_robot_instance = None


def _signal_handler(_sig, _frame):
    print_info("Signal received, shutting down...")
    if _robot_instance is not None:
        try:
            _robot_instance.stop_robot()
        except Exception:
            pass
    os._exit(0)


def run_demo(robot):
    """Voice + web SocketIO demo mode."""
    from gevent import monkey
    monkey.patch_all()

    import rospy
    from config import Config
    from robot import IntegratedVisionNavController
    from web.app import (app, set_controller, set_whisper_model, socketio)

    if Config.LOCAL_WHISPER_MODEL_PATH and os.path.exists(Config.LOCAL_WHISPER_MODEL_PATH):
        try:
            from faster_whisper import WhisperModel
            print_info("Loading local whisper model on CPU...")
            whisper_model = WhisperModel(
                Config.LOCAL_WHISPER_MODEL_PATH, device="cpu", compute_type="int8",
            )
            set_whisper_model(whisper_model)
            print_success("Whisper model loaded.")
        except ImportError:
            print_warning("faster_whisper not installed; voice commands disabled.")
        except Exception as e:
            print_error(f"Failed to load whisper model: {e}")
    else:
        print_warning(
            f"Whisper model path '{Config.LOCAL_WHISPER_MODEL_PATH}' missing; "
            f"voice commands disabled. Text-only commands still work."
        )

    controller = IntegratedVisionNavController(robot, socketio_instance=socketio)
    set_controller(controller)
    rospy.on_shutdown(robot.stop_robot)

    print_info(
        f"Starting HTTPS web server on https://{Config.SERVER_HOST}:{Config.SERVER_PORT}"
    )
    socketio.run(
        app,
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        keyfile=Config.SSL_KEY_PATH if os.path.exists(Config.SSL_KEY_PATH) else None,
        certfile=Config.SSL_CERT_PATH if os.path.exists(Config.SSL_CERT_PATH) else None,
        server_side=True,
    )


def run_headless(robot, args):
    """Single VLN run with no web UI."""
    from config import Config
    from robot import IntegratedVisionNavController

    instruction = args.instruction or Config.DEFAULT_INSTRUCTION
    if not robot.planner_client.initialized:
        print_warning("iPlanner client not initialised; navigation may fail.")

    controller = IntegratedVisionNavController(robot, instruction=instruction)
    controller.run()


def main():
    global _robot_instance

    signal.signal(signal.SIGINT, _signal_handler)
    args = config_manager.parse()

    # RobotController has heavy imports (rospy, Unitree SDK); import after argparse.
    from robot import RobotController

    try:
        robot = RobotController()
        _robot_instance = robot
    except Exception as e:
        print_error(f"Failed to initialise robot: {e}")
        traceback.print_exc()
        return

    try:
        if args.demo:
            run_demo(robot)
        else:
            run_headless(robot, args)
    except KeyboardInterrupt:
        print_info("Interrupted by user")
    except Exception as e:
        print_error(f"Runtime error: {e}")
        traceback.print_exc()
    finally:
        if _robot_instance is not None:
            print_info("Stopping robot movement...")
            try:
                _robot_instance.stop_robot()
            except Exception:
                pass
            time.sleep(1.0)
            print_info("Stopping image saver...")
            try:
                _robot_instance.stop_saver()
            except Exception:
                pass
            saver_thread = getattr(_robot_instance, "saver_thread", None)
            if saver_thread and saver_thread.is_alive():
                saver_thread.join(timeout=2.0)
        print_info("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
