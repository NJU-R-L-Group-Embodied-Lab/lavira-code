"""
Unitree Go1 robot controller.

Wraps the Unitree High-level SDK plus four ROS RGB-D cameras, exposes panorama
capture, base rotation, depth lookup, and iPlanner-driven trajectory
execution.

The Unitree SDK shared object (``robot_interface.so``) lives outside the repo.
Set ``UNITREE_SDK_PATH`` to its directory if it is not already on
``PYTHONPATH``.
"""
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, Image

import config as _config_mod
from config import Config
from utils import print_action, print_error, print_info, print_success, print_warning
from robot.iplanner_client import IPlannerRemoteClient


def _import_unitree_sdk():
    """Import the Unitree High-level SDK, optionally extending ``sys.path``."""
    if Config.UNITREE_SDK_PATH:
        sys.path.append(Config.UNITREE_SDK_PATH)
    try:
        import robot_interface as sdk  # noqa: WPS433 (runtime import is intentional)
        return sdk
    except ImportError:
        print_warning(
            "robot_interface SDK not found. Set UNITREE_SDK_PATH to the directory "
            "containing robot_interface.so to enable real-hardware motion."
        )
        return None


class RobotController:
    """Real-hardware controller for the Unitree Go1 quadruped."""

    NUM_CAMERAS = 4
    BASIC_TURN_ANGLE = 30
    ROTATION_CORRECTION_FACTOR = 0.75  # Empirical compensation for rotation overshoot

    def __init__(self):
        if not rospy.get_node_uri():
            rospy.init_node("lavira_go1", anonymous=True)

        self._sdk = _import_unitree_sdk()
        self._init_unitree_sdk()

        # Per-camera buffers
        self.camera_data = {}
        for i in range(1, self.NUM_CAMERAS + 1):
            self.camera_data[f"camera{i}"] = {
                "rgb_image": None, "depth_image": None,
                "camera_info": None, "camera_info_received": False,
                "fx": None, "fy": None, "cx": None, "cy": None,
            }

        # Per-direction snapshot buffers used by tasks
        self.direction_images = {
            d: {"rgb": None, "depth": None}
            for d in ("front", "right", "behind", "left")
        }

        self.current_direction = "front"
        self.current_camera = "camera1"
        self.running = True
        self.planning_lock = threading.Lock()
        self.iplanner_history = []

        # iPlanner client
        self.planner_client = IPlannerRemoteClient(Config.IPLANNER_URL)
        try:
            if not self.planner_client.reset():
                print_warning("Failed to reset planner client (server might be down).")
        except Exception as e:
            print_warning(f"iPlanner reset failed: {e}")

        self._setup_ros()

        # Background image saver
        self.saver_running = True
        self.saver_thread = threading.Thread(target=self._image_saver_loop, daemon=True)
        self.saver_thread.start()

        print_info(f"RobotController initialised with {self.NUM_CAMERAS} cameras")

    # ------------------------------------------------------------------ #
    # SDK / ROS setup
    # ------------------------------------------------------------------ #
    def _init_unitree_sdk(self):
        if self._sdk is None:
            self.udp = None
            self.cmd = None
            self.state = None
            return
        try:
            HIGHLEVEL = 0xEE
            self.udp = self._sdk.UDP(
                HIGHLEVEL, Config.UNITREE_LOCAL_PORT,
                Config.UNITREE_HOST, Config.UNITREE_REMOTE_PORT,
            )
            self.cmd = self._sdk.HighCmd()
            self.state = self._sdk.HighState()
            self.udp.InitCmdData(self.cmd)
        except Exception as e:
            print_error(f"Failed to initialise Unitree SDK: {e}")
            self.udp = None

    def _setup_ros(self):
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.subscribers = []
        for i in range(1, self.NUM_CAMERAS + 1):
            cam = f"camera{i}"
            self.subscribers.extend([
                rospy.Subscriber(f"/{cam}/color/image_raw", Image,
                                 self._rgb_callback, callback_args=cam),
                rospy.Subscriber(f"/{cam}/depth/image_raw", Image,
                                 self._depth_callback, callback_args=cam),
                rospy.Subscriber(f"/{cam}/color/camera_info", CameraInfo,
                                 self._camera_info_callback, callback_args=cam),
            ])

    # ------------------------------------------------------------------ #
    # Background saver
    # ------------------------------------------------------------------ #
    def _image_saver_loop(self):
        """Save the front camera frame at ~10 FPS for debugging / playback."""
        attempts = 0
        base_dir = None
        while attempts < 10:
            if Config.LOG_DIR:
                base_dir = Config.LOG_DIR
                break
            time.sleep(0.5)
            attempts += 1

        if not base_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "outputs", timestamp)

        save_dir = os.path.join(base_dir, "..", "images", "front_view")
        save_dir = os.path.normpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print_info(f"Front-camera saver writing to: {save_dir}")

        interval = 1.0 / 10.0
        while self.saver_running:
            start = time.time()
            img = self.camera_data.get("camera1", {}).get("rgb_image")
            if img is not None:
                try:
                    ts = datetime.now().strftime("%H%M%S_%f")
                    cv2.imwrite(os.path.join(save_dir, f"front_{ts}.jpg"), img)
                except Exception:
                    pass
            time.sleep(max(0.0, interval - (time.time() - start)))

    # ------------------------------------------------------------------ #
    # Camera callbacks
    # ------------------------------------------------------------------ #
    def _rgb_callback(self, msg, camera_name):
        try:
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if "rgb" in msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.camera_data[camera_name]["rgb_image"] = cv_image
        except Exception:
            pass

    def _depth_callback(self, msg, camera_name):
        try:
            cv_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self.camera_data[camera_name]["depth_image"] = cv_image
        except Exception:
            pass

    def _camera_info_callback(self, msg, camera_name):
        data = self.camera_data[camera_name]
        if not data["camera_info_received"] and msg.K and len(msg.K) >= 9:
            data["camera_info"] = msg
            data["fx"], data["fy"] = msg.K[0], msg.K[4]
            data["cx"], data["cy"] = msg.K[2], msg.K[5]
            data["camera_info_received"] = True
            print_success(f"{camera_name} intrinsics received")

    # ------------------------------------------------------------------ #
    # Image capture
    # ------------------------------------------------------------------ #
    def get_depth_value(self, u: int, v: int, direction: str) -> Optional[float]:
        depth = self.direction_images[direction]["depth"]
        if depth is None:
            print_warning(f"Depth image for {direction} is empty")
            return None

        try:
            k = 2
            h, w = depth.shape
            depth_patch = depth[max(0, v - k):min(h, v + k + 1),
                                max(0, u - k):min(w, u + k + 1)] / 1000.0
            valid = depth_patch[depth_patch > 0]
            if valid.size == 0:
                print_warning(f"Invalid depth at ({u}, {v})")
                return None
            return float(np.percentile(valid, 30))
        except Exception as e:
            print_error(f"Error getting depth value: {e}")
            return None

    def capture_current_image(self) -> bool:
        curr_rgb = self.camera_data[self.current_camera]["rgb_image"]
        curr_depth = self.camera_data[self.current_camera]["depth_image"]
        if curr_rgb is None or curr_depth is None:
            print_warning(f"Failed to capture image for {self.current_direction}")
            return False
        self.direction_images[self.current_direction]["rgb"] = curr_rgb.copy()
        self.direction_images[self.current_direction]["depth"] = curr_depth.copy()
        print_action(f"Captured {self.current_direction} view from {self.current_camera}")
        return True

    def capture_all_directions(self, current_step: int) -> bool:
        print_action(f"Step {current_step} - capturing all 4 directions")
        directions = ["front", "right", "behind", "left"]
        direction_files = {
            "front": "view_0.png", "right": "view_90.png",
            "behind": "view_180.png", "left": "view_270.png",
        }
        camera_mapping = {
            "front": "camera1", "right": "camera2",
            "behind": "camera3", "left": "camera4",
        }

        step_dir = os.path.join(Config.PANORAMA_DIR, f"step{current_step}")
        os.makedirs(step_dir, exist_ok=True)

        for direction in directions:
            target_camera = camera_mapping[direction]
            if self.current_camera != target_camera:
                self.current_camera = target_camera
                rospy.sleep(0.5)
            rospy.sleep(0.2)

            self.current_direction = direction
            if not self.capture_current_image():
                rospy.sleep(1.0)
                if not self.capture_current_image():
                    return False

            rgb_img = self.direction_images[direction]["rgb"]
            if rgb_img is not None:
                h, w = rgb_img.shape[:2]
                max_size = 1024
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    rgb_img = cv2.resize(rgb_img, (int(w * scale), int(h * scale)))
                cv2.imwrite(os.path.join(step_dir, direction_files[direction]), rgb_img)

        self.current_camera = "camera1"
        self.current_direction = "front"
        rospy.sleep(0.5)
        return True

    # ------------------------------------------------------------------ #
    # Base motion
    # ------------------------------------------------------------------ #
    def rotate_left(self, angle):
        if not self.udp:
            return
        print_action(f"Rotating LEFT: {angle} degrees")
        self._perform_rotation(angle, direction=1)

    def rotate_right(self, angle):
        if not self.udp:
            return
        print_action(f"Rotating RIGHT: {angle} degrees")
        self._perform_rotation(angle, direction=-1)

    def _perform_rotation(self, angle, direction):
        corrected_angle = angle * self.ROTATION_CORRECTION_FACTOR
        duration = int(1000 * (corrected_angle / 90))
        yaw_speed = 0.87 * direction
        print_action(f"Rotation: {angle} deg -> {corrected_angle:.2f} deg ({duration} ms)")

        motiontime = 0
        while motiontime < duration:
            time.sleep(0.002)
            motiontime += 1
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity = [0, 0]
            self.cmd.yawSpeed = yaw_speed
            self.cmd.footRaiseHeight = 0.1
            self.udp.SetSend(self.cmd)
            self.udp.Send()

        self._send_stop_cmd()
        time.sleep(1.0)
        print_success("Rotation complete")

    def stop_robot(self):
        print_action("Stopping robot...")
        if not self.udp:
            return
        for _ in range(10):
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            self._send_stop_cmd()
            self.udp.Send()
            time.sleep(0.002)

    def stop_saver(self):
        print_action("Stopping image saver thread...")
        self.saver_running = False

    def _send_stop_cmd(self):
        self.cmd.mode = 0
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.velocity = [0, 0]
        self.cmd.yawSpeed = 0
        self.udp.SetSend(self.cmd)

    # ------------------------------------------------------------------ #
    # Trajectory execution + visualisation (uses iPlanner)
    # ------------------------------------------------------------------ #
    def project_trajectory_to_image(self, trajectory_points, img_bgr,
                                    save_name="planned_path.png",
                                    target_pixel=None, target_3d=None):
        """Back-project a trajectory onto the camera image and save it."""
        if trajectory_points is None or len(trajectory_points) == 0:
            return
        cam = self.camera_data["camera1"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        if None in (fx, fy, cx, cy):
            print_warning("Camera intrinsics missing; cannot draw trajectory")
            return

        vis = img_bgr.copy()
        h, w = vis.shape[:2]
        cam_height = Config.CAMERA_HEIGHT
        roll_corr = Config.CAMERA_ROLL_CORRECTION

        points_2d = []
        for pt in trajectory_points:
            sim_z = pt[0]
            sim_x = -pt[1]
            sim_y = cam_height + sim_x * roll_corr
            if sim_z <= 0.01:
                continue
            u = int((sim_x * fx) / sim_z + cx)
            v = int((sim_y * fy) / sim_z + cy)
            if 0 <= u < w and 0 <= v < h:
                points_2d.append((u, v))

        for i in range(len(points_2d) - 1):
            cv2.line(vis, points_2d[i], points_2d[i + 1], (0, 255, 0), 2)
        if points_2d:
            cv2.circle(vis, points_2d[-1], 4, (0, 0, 255), -1)

        if target_pixel is not None:
            tx, ty = target_pixel
            cv2.drawMarker(vis, (tx, ty), (0, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            lines = [f"Target Pixel: ({tx}, {ty})"]
            if target_3d:
                lines.append(f"Target 3D: ({target_3d[0]:.2f}, {target_3d[1]:.2f}, {target_3d[2]:.2f})m")
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_y = ty + 25
            for line in lines:
                cv2.putText(vis, line, (tx + 10, base_y), font, 0.5, (0, 0, 0), 3)
                cv2.putText(vis, line, (tx + 10, base_y), font, 0.5, (0, 0, 255), 1)
                base_y += 20

        save_path = os.path.join(Config.IPLANNER_DIR, save_name)
        cv2.imwrite(save_path, vis)
        self.iplanner_history.append(save_path)
        print_action(f"Saved iPlanner visualisation: {save_path}")

    def execute_trajectory(self, trajectory_points, goal_local=None,
                           target_speed=0.3, lookahead_dist=0.4,
                           max_yaw_speed=0.8, goal_tolerance=0.5,
                           replan_interval=0.5, max_execution_time=60.0):
        """Pure-pursuit execution with parallel iPlanner replanning."""
        if trajectory_points is None or len(trajectory_points) < 2:
            print_warning("Trajectory too short to execute")
            return

        print_action(f"Executing trajectory with {len(trajectory_points)} points")
        path = np.array(trajectory_points)
        current_goal = (path[-1].copy() if goal_local is None
                        else np.array([goal_local[0], goal_local[1], 0.0]))

        start_time = time.time()
        last_replan_time = start_time
        current_idx = 0
        is_planning = False
        next_path = None

        def replan_worker(rgb, depth, goal):
            nonlocal next_path, is_planning
            try:
                traj, fear = self.planner_client.get_plan(rgb, depth, (goal[0], goal[1]))
                try:
                    ts = datetime.now().strftime("%H%M%S_%f")
                    self.project_trajectory_to_image(
                        traj, rgb, save_name=f"replan_{ts}.jpg",
                        target_3d=(goal[0], goal[1], 0.0),
                    )
                except Exception as vis_e:
                    print_warning(f"Visualisation failed: {vis_e}")
                if fear is not None and fear <= 0.6 and traj is not None and len(traj) > 0:
                    with self.planning_lock:
                        next_path = traj
            except Exception as e:
                print_error(f"Replan failed: {e}")
            finally:
                is_planning = False

        while self.running:
            now = time.time()
            if now - start_time > max_execution_time:
                print_warning("Execution timeout")
                break

            with self.planning_lock:
                if next_path is not None:
                    path = np.array(next_path)
                    current_idx = 0
                    next_path = None

            if not is_planning and (now - last_replan_time > replan_interval):
                if self.capture_current_image():
                    rgb = self.direction_images[self.current_direction]["rgb"]
                    depth = self.direction_images[self.current_direction]["depth"]
                    if rgb is not None:
                        is_planning = True
                        last_replan_time = now
                        threading.Thread(
                            target=replan_worker,
                            args=(rgb.copy(), depth.copy(), current_goal.copy()),
                            daemon=True,
                        ).start()

            found_target = False
            for i in range(current_idx, len(path)):
                if np.linalg.norm(path[i][:2]) > lookahead_dist:
                    current_idx = i
                    found_target = True
                    break
            if not found_target:
                current_idx = len(path) - 1

            target_point = path[current_idx]
            dx, dy = target_point[0], target_point[1]
            dist_to_target = math.sqrt(dx ** 2 + dy ** 2)
            dist_to_global_goal = np.linalg.norm(current_goal[:2])

            if dist_to_global_goal < goal_tolerance:
                print_success(f"Reached goal area (dist: {dist_to_global_goal:.2f}m)")
                break

            alpha = math.atan2(dy, dx)
            cmd_yaw_speed = float(np.clip(
                (2.0 * target_speed * math.sin(alpha)) / max(dist_to_target, 1e-3),
                -max_yaw_speed, max_yaw_speed,
            ))
            current_speed = max(0.1, target_speed * (1.0 - abs(alpha) / math.pi))

            dt = 0.05
            loop_cycles = int(dt / 0.002)
            if self.udp:
                for _ in range(loop_cycles):
                    self.udp.Recv()
                    self.udp.GetRecv(self.state)
                    self.cmd.mode = 2
                    self.cmd.gaitType = 1
                    self.cmd.velocity = [current_speed, 0]
                    self.cmd.yawSpeed = cmd_yaw_speed
                    self.udp.SetSend(self.cmd)
                    self.udp.Send()
                    time.sleep(0.002)
            else:
                time.sleep(dt)

            d_dist = current_speed * dt
            d_yaw = cmd_yaw_speed * dt
            cos_t, sin_t = math.cos(-d_yaw), math.sin(-d_yaw)
            path = np.array([[p[0] * cos_t - p[1] * sin_t - d_dist,
                              p[0] * sin_t + p[1] * cos_t,
                              p[2]] for p in path])
            current_goal = np.array([
                current_goal[0] * cos_t - current_goal[1] * sin_t - d_dist,
                current_goal[0] * sin_t + current_goal[1] * cos_t,
                current_goal[2],
            ])

        self.stop_robot()
