"""
Integrated Vision Navigation Controller for the Unitree Go1 VLN task.

Drives the per-cycle pipeline:

1. Capture a 4-direction panorama by switching between the four head cameras.
2. Language Action model picks a direction and (optionally) signals STOP.
3. Base rotates 90° / 180° in the chosen direction.
4. Vision Action model returns either ``STOP`` or a bbox of the next target.
5. The bbox centre is back-projected through the front-camera depth to a goal
   in the robot frame; iPlanner generates a trajectory; a pure-pursuit loop
   drives the goal with parallel replanning, leaving a safety buffer.

Supports both modes:
- Voice / web: ``socketio_instance`` is set, ``start_new_task()`` runs the loop
  on a background thread and emits status / response updates to the browser.
- Text: ``socketio_instance`` is ``None``; the caller drives
  ``run_navigation_cycle()`` (or ``run()``) directly.
"""
import os
import threading
import time
from typing import List, Optional

import numpy as np

from ai_client import LaViRAVisionClient
from config import Config
from robot.navigation_api import LaViRANavigationAPI
from utils import (
    img_to_base64, print_action, print_error, print_info, print_success,
    print_warning,
)


def _make_client() -> LaViRAVisionClient:
    return LaViRAVisionClient(
        la_api_key=Config.LA_API_KEY,
        la_base_url=Config.LA_BASE_URL,
        la_model_name=Config.LA_MODEL_NAME,
        va_api_key=Config.VA_API_KEY,
        va_base_url=Config.VA_BASE_URL,
        va_model_name=Config.VA_MODEL_NAME,
    )


class IntegratedVisionNavController:
    """High-level VLN controller for the Go1."""

    SAFE_DISTANCE_M = 1.5  # leave this much room short of the depth-derived goal
    DEFAULT_FORWARD_GOAL = (1.5, 0.0)

    def __init__(self, robot, instruction: Optional[str] = None,
                 socketio_instance=None,
                 client: Optional[LaViRAVisionClient] = None):
        self.robot = robot
        self.socketio = socketio_instance
        self.instruction = instruction
        self.global_target = instruction

        self.client = client or _make_client()
        self.api = LaViRANavigationAPI(self.client)

        self.current_step = 1
        self.todo_list = ""
        self.visited_targets: List[str] = []
        self.is_task_running = False
        self.task_thread = None

    # ------------------------------------------------------------------ #
    # SocketIO helpers
    # ------------------------------------------------------------------ #
    def _emit_status(self, message):
        if self.socketio is not None:
            self.socketio.emit("status_update", {"message": message})
        print_info(f"[STATUS] {message}")

    def _emit_response(self, message):
        if self.socketio is not None:
            self.socketio.emit("response", {"message": message})
        print_info(f"[RESPONSE] {message}")

    # ------------------------------------------------------------------ #
    # Public lifecycle
    # ------------------------------------------------------------------ #
    def stop_task(self):
        print_info("Stop signal received")
        self.is_task_running = False

    def start_new_task(self, instruction):
        """Background-thread launcher for the web demo."""
        if self.is_task_running:
            msg = "Task already in progress."
            self._emit_response(msg)
            return False, msg

        self.instruction = instruction
        self.global_target = instruction
        self.current_step = 1
        self.todo_list = ""
        self.visited_targets = []

        self.is_task_running = True
        self.task_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.task_thread.start()
        self._emit_status(f"Task started: {instruction}")
        return True, "ok"

    def run(self):
        """Headless / synchronous entry point."""
        self.is_task_running = True
        self._run_loop()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _run_loop(self):
        print_info(f"Starting VLN: {self.instruction}")

        try:
            while self.robot.running and self.is_task_running:
                cycle_done = self._run_cycle()
                if cycle_done:
                    self._emit_response(
                        "Navigation complete. I have arrived at the destination."
                    )
                    break
                self.current_step += 1
        finally:
            self.is_task_running = False
            time.sleep(2.0)
            print_success("VLN task finished")

    def _run_cycle(self) -> bool:
        """Returns True when the task is fully complete."""
        if not self.robot.capture_all_directions(self.current_step):
            print_error("Failed to capture panorama")
            return False

        panorama = self._get_panorama_frames()
        if self.current_step == 1 and not self.todo_list:
            self.todo_list = "No TODO list yet. Please generate one."

        decision = self.api.language_action(
            instruction=self.instruction,
            global_target=self.global_target,
            todo_list=self.todo_list,
            iplanner_history=self.robot.iplanner_history,
            panorama_frames=panorama,
            current_step=self.current_step,
        )
        if decision.get("updated_todo_list"):
            self.todo_list = decision["updated_todo_list"]
            print_info(f"Updated TODO List:\n{self.todo_list}")
            self._emit_status(f"Step {self.current_step}: TODO list updated")

        turn_direction = decision.get("turn_direction", "front").lower()
        raw_stop = decision.get("stop", False)
        strategic_stop = (raw_stop if isinstance(raw_stop, bool)
                          else str(raw_stop).lower() == "true")

        if turn_direction not in ("front", "right", "left", "behind"):
            print_warning(f"Unknown direction '{turn_direction}'; defaulting to front")
            turn_direction = "front"

        strategic_reasoning = decision.get("reasoning", "")
        strategic_goal = f"Go {turn_direction}. {strategic_reasoning}"

        if turn_direction == "left":
            self.robot.rotate_left(90)
        elif turn_direction == "right":
            self.robot.rotate_right(90)
        elif turn_direction == "behind":
            self.robot.rotate_right(180)

        time.sleep(0.5)
        if not self.robot.capture_current_image():
            print_warning("Failed to capture current view after rotation")
            return False

        img_np = self.robot.direction_images[self.robot.current_direction]["rgb"]
        bbox_decision = self.api.vision_action(
            img_np=img_np,
            instruction=self.instruction,
            global_target=self.global_target,
            strategic_goal=strategic_goal,
            strategic_stop=False,
            current_step=self.current_step,
        )
        action = bbox_decision.get("action", "NAVIGATE")
        bbox = bbox_decision.get("bbox_2d")

        should_stop_now = False
        if strategic_stop:
            if bbox and len(bbox) == 4:
                print_info("Strategic stop: executing FINAL approach to target.")
                action = "NAVIGATE"
            else:
                print_success("Strategic stop without bbox: stopping now.")
                should_stop_now = True
        elif action == "STOP":
            if bbox and len(bbox) == 4:
                print_info("Tactical STOP with bbox: executing FINAL approach.")
                action = "NAVIGATE"
                strategic_stop = True
            else:
                print_success("Tactical STOP without bbox: stopping now.")
                should_stop_now = True

        if should_stop_now:
            return True

        target_pixel = None
        goal_2d = self.DEFAULT_FORWARD_GOAL
        if bbox and len(bbox) == 4:
            target_pixel, goal_2d = self._bbox_to_goal(bbox)

        rgb = self.robot.direction_images[self.robot.current_direction]["rgb"]
        depth = self.robot.direction_images[self.robot.current_direction]["depth"]

        print_action(f"Requesting plan from iPlanner to {goal_2d}...")
        traj, _ = self.robot.planner_client.get_plan(rgb, depth, goal_2d)
        if traj is not None:
            ts = time.strftime("%H%M%S")
            self.robot.project_trajectory_to_image(
                traj, rgb,
                save_name=f"step{self.current_step}_plan_{ts}.jpg",
                target_pixel=target_pixel,
                target_3d=(goal_2d[0], goal_2d[1], 0.0),
            )

        if traj is not None and len(traj) > 0:
            self._execute_with_safety_buffer(traj, goal_2d)
        else:
            print_warning("Planner failed; holding position.")

        if bbox and bbox_decision.get("target"):
            self.visited_targets.append(bbox_decision["target"])

        return bool(strategic_stop)

    # ------------------------------------------------------------------ #
    # Panorama / geometry
    # ------------------------------------------------------------------ #
    def _get_panorama_frames(self):
        view_defs = [
            {"angle": 0,   "label": f"Image 1: FORWARD view (Step {self.current_step}).",  "file": "view_0.png"},
            {"angle": 90,  "label": f"Image 2: 90° RIGHT view (Step {self.current_step}).", "file": "view_90.png"},
            {"angle": 180, "label": f"Image 3: 180° BEHIND view (Step {self.current_step}).", "file": "view_180.png"},
            {"angle": 270, "label": f"Image 4: 90° LEFT view (Step {self.current_step}).",  "file": "view_270.png"},
        ]
        step_dir = os.path.join(Config.PANORAMA_DIR, f"step{self.current_step}")
        return [{
            "angle": v["angle"], "label": v["label"],
            "img_base64": img_to_base64(os.path.join(step_dir, v["file"])),
        } for v in view_defs]

    def _bbox_to_goal(self, bbox):
        x1, y1, x2, y2 = bbox
        box_h = y2 - y1
        cx = int((x1 + x2) / 2)
        cy = int(y2 - 0.25 * box_h)
        target_pixel = (cx, cy)

        depth_val = self.robot.get_depth_value(cx, cy, self.robot.current_direction)
        if depth_val is None:
            print_warning("No depth at bbox; using default forward goal")
            return target_pixel, self.DEFAULT_FORWARD_GOAL

        cam = self.robot.camera_data[self.robot.current_camera]
        fx, cx_intr = cam.get("fx"), cam.get("cx")
        if fx is None or cx_intr is None:
            print_warning("Missing camera intrinsics; using default forward goal")
            return target_pixel, self.DEFAULT_FORWARD_GOAL

        z_3d = depth_val
        x_3d = (cx - cx_intr) * z_3d / fx
        goal_x = z_3d      # robot frame: forward
        goal_y = -x_3d     # robot frame: left
        print_info(
            f"Calculated goal from bbox: ({goal_x:.2f}, {goal_y:.2f}) "
            f"(depth {depth_val:.2f}m)"
        )
        return target_pixel, (goal_x, goal_y)

    # ------------------------------------------------------------------ #
    # Trajectory execution
    # ------------------------------------------------------------------ #
    def _execute_with_safety_buffer(self, traj, goal_2d):
        try:
            path_arr = np.array(traj)
            diffs = path_arr[1:] - path_arr[:-1]
            dists = np.linalg.norm(diffs[:, :2], axis=1)
            total_dist = float(np.sum(dists))
            target_dist = total_dist - self.SAFE_DISTANCE_M

            if target_dist <= 0:
                print_warning(
                    f"Target too close ({total_dist:.2f}m < {self.SAFE_DISTANCE_M}m); holding."
                )
                return

            cutoff_idx = len(path_arr) - 1
            current_dist = 0.0
            for i, d in enumerate(dists):
                current_dist += d
                if current_dist >= target_dist:
                    cutoff_idx = i + 1
                    break

            truncated = traj[:cutoff_idx + 1]
            if len(truncated) == 0:
                self.robot.execute_trajectory(traj, goal_local=goal_2d)
                return

            final_pt = truncated[-1]
            new_goal = (final_pt[0], final_pt[1])
            print_warning(
                f"Truncating trajectory to leave {self.SAFE_DISTANCE_M}m buffer "
                f"({target_dist:.2f}/{total_dist:.2f}m). New goal: {new_goal}"
            )
            self.robot.execute_trajectory(truncated, goal_local=new_goal)
        except Exception as e:
            print_error(f"Trajectory truncation failed: {e}")
            self.robot.execute_trajectory(traj, goal_local=goal_2d)
