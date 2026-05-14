import math
import os
import threading
import time
import numpy as np
import rospy
from PIL import Image, ImageDraw, ImageFont

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as ROSImage, CameraInfo

import config
from ai_client.vision_client import LaViRAVisionClient
from robot.arm_controller import AutomatedController
from robot.navigation_api import LaViRANavigationAPI


class IntegratedVisionNavController:
    """
    Integrated vision-driven navigation controller.

    Supports two execution modes:
    - Voice / Web mode: pass a SocketIO instance and call ``start_new_task()`` to
      run the navigation loop in a background thread.
    - Text-instruction mode: leave ``socketio_instance`` as ``None`` and have the
      caller drive ``run_navigation_cycle()`` in its own loop.
    """

    def __init__(self, socketio_instance=None, timestamp=None, instruction=None):
        self.socketio = socketio_instance
        self.instruction = instruction
        self.is_running_task = False
        self.bridge = CvBridge()
        self.task_thread = None
        self.timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")

        # --- Robot state and data ---
        self.current_front_rgb = None
        self.current_front_depth = None
        self.current_left_rgb = None
        self.current_right_rgb = None
        self.depth_fx = self.depth_fy = self.depth_cx = self.depth_cy = None
        self.image_lock = threading.Lock()

        # --- Frame saving setup ---
        self.frame_save_path = os.path.join(self.timestamp, "front_camera_frames")
        os.makedirs(self.frame_save_path, exist_ok=True)
        rospy.loginfo(f"Saving front camera frames to: {self.frame_save_path}")

        # --- Camera resolutions ---
        self.rgb_width, self.rgb_height = 1280, 720
        self.depth_width, self.depth_height = 1280, 720

        # --- ROS publishers / subscribers ---
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/camera/color/image_raw", ROSImage, self.front_rgb_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", ROSImage, self.front_depth_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera_l/color/image_raw", ROSImage, self.left_rgb_callback)
        rospy.Subscriber("/camera_r/color/image_raw", ROSImage, self.right_rgb_callback)
        rospy.loginfo("Subscribed to all camera and robot topics.")
        rospy.sleep(2)

        # --- Navigation state ---
        self.visited_targets = []

        # --- Arm control ---
        self.arm_controller = AutomatedController()
        # Three symmetric dual-arm poses used to sweep the left/right cameras
        # across the panoramic field of view.
        self.TARGET_SEQUENCE = [
            {'left': [0.8, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1], 'right': [-0.8, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
            {'left': [1.6, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1], 'right': [-1.6, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
            {'left': [2.4, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1], 'right': [-2.4, 0.7, 0.0, 0.0, -0.85, 0.0, 0.1]},
        ]

        # --- AI / VLM initialization ---
        self.initialize_models()
        self.api = LaViRANavigationAPI(self.model, image_width=self.rgb_width, image_height=self.rgb_height)

    # ------------------------------------------------------------------ #
    # Video stream / SocketIO helpers
    # ------------------------------------------------------------------ #
    def get_front_image_jpeg(self):
        """Return the latest front-camera frame encoded as JPEG bytes."""
        with self.image_lock:
            if self.current_front_rgb is None:
                return None
            try:
                import simplejpeg
                return simplejpeg.encode_jpeg(
                    self.current_front_rgb, quality=75, colorspace='BGR', fastdct=True
                )
            except Exception as e:
                rospy.logwarn_throttle(10, f"Failed to encode JPEG: {e}")
                return None

    def _emit_status(self, message):
        if self.socketio is not None:
            self.socketio.emit('status_update', {'message': message})
        rospy.loginfo(f"[STATUS_UPDATE] {message}")

    def _emit_response(self, message):
        if self.socketio is not None:
            self.socketio.emit('response', {'message': message})
        rospy.loginfo(f"[RESPONSE] {message}")

    # ------------------------------------------------------------------ #
    # Task loop (used by the web demo)
    # ------------------------------------------------------------------ #
    def start_new_task(self, instruction):
        if self.is_running_task:
            msg = "Task already in progress."
            self._emit_response(msg)
            rospy.logwarn("Cannot start a new task, another one is already running.")
            return False, msg

        self.instruction = instruction
        self.visited_targets = []
        self.is_running_task = True
        self.task_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self.task_thread.start()
        msg = f"Task started: {self.instruction}"
        self._emit_status(msg)
        return True, msg

    def _navigation_loop(self):
        cycle_count = 1
        while not rospy.is_shutdown() and self.is_running_task:
            self._emit_status(f"Starting navigation cycle {cycle_count}...")
            task_complete = self.run_navigation_cycle(self.timestamp)
            if task_complete:
                self._emit_response("Navigation task complete. I have arrived at the final destination.")
                self.is_running_task = False
                break

            self._emit_status(f"Cycle {cycle_count} finished. Pausing before next cycle.")
            cycle_count += 1
            rospy.sleep(2)

        if self.is_running_task:
            self._emit_status("Navigation stopped due to system shutdown.")
        rospy.loginfo("Navigation loop finished.")
        self.is_running_task = False

    # ------------------------------------------------------------------ #
    # VLM client
    # ------------------------------------------------------------------ #
    def initialize_models(self):
        rospy.loginfo(
            f"Initializing LaViRA client: LA model '{config.LA_MODEL_NAME}' @ {config.LA_BASE_URL}, "
            f"VA model '{config.VA_MODEL_NAME}' @ {config.VA_BASE_URL}"
        )
        self.model = LaViRAVisionClient(
            la_api_key=config.LA_API_KEY,
            la_base_url=config.LA_BASE_URL,
            la_model_name=config.LA_MODEL_NAME,
            va_api_key=config.VA_API_KEY,
            va_base_url=config.VA_BASE_URL,
            va_model_name=config.VA_MODEL_NAME,
        )
        rospy.loginfo("LaViRA client ready.")

    # ------------------------------------------------------------------ #
    # Camera callbacks
    # ------------------------------------------------------------------ #
    def front_rgb_callback(self, msg):
        with self.image_lock:
            self.current_front_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def front_depth_callback(self, msg):
        with self.image_lock:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32) / 1000.0
            self.current_front_depth = depth_img

    def left_rgb_callback(self, msg):
        with self.image_lock:
            self.current_left_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def right_rgb_callback(self, msg):
        with self.image_lock:
            self.current_right_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def camera_info_callback(self, msg):
        with self.image_lock:
            self.depth_fx, self.depth_fy = msg.K[0], msg.K[4]
            self.depth_cx, self.depth_cy = msg.K[2], msg.K[5]

    # ------------------------------------------------------------------ #
    # Base motion
    # ------------------------------------------------------------------ #
    def move_distance(self, distance):
        if distance <= 0:
            return
        twist = Twist()
        twist.linear.x = config.LINEAR_SPEED
        duration = abs(distance / config.LINEAR_SPEED)
        rate = rospy.Rate(10)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        self.cmd_vel_pub.publish(Twist())

    def rotate_angle(self, angle_deg):
        if abs(angle_deg) < 1.0:
            return
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360

        twist = Twist()
        twist.angular.z = math.radians(config.ANGULAR_SPEED_DEG) if angle_deg > 0 else -math.radians(
            config.ANGULAR_SPEED_DEG
        )
        duration = abs(angle_deg / config.ANGULAR_SPEED_DEG)
        rate = rospy.Rate(10)
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        self.cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------ #
    # Pixel -> world motion
    # ------------------------------------------------------------------ #
    def pixel_to_robot_motion(self, target_pixel, timestamp, name, window_size=7, fallback_distance=1.0):
        with self.image_lock:
            if self.current_front_depth is None:
                raise ValueError("Front depth image not available")
            if self.depth_fx is None:
                raise ValueError("Depth camera intrinsics not available")
            depth_image = self.current_front_depth.copy()
            fx_d, fy_d = self.depth_fx, self.depth_fy
            cx_d, cy_d = self.depth_cx, self.depth_cy

        u_depth, v_depth = int(target_pixel[0]), int(target_pixel[1])
        half_w = window_size // 2
        u_min, u_max = max(0, u_depth - half_w), min(self.depth_width, u_depth + half_w + 1)
        v_min, v_max = max(0, v_depth - half_w), min(self.depth_height, v_depth + half_w + 1)

        depth_patch = depth_image[v_min:v_max, u_min:u_max].flatten()
        valid_depths = depth_patch[depth_patch > 0]

        if len(valid_depths) == 0:
            Z = float(fallback_distance)
            rospy.logwarn(f"No valid depth in window. Using fallback distance: {Z}m")
        else:
            Z = float(np.median(valid_depths))
            rospy.loginfo(
                f"Depth patch {window_size}x{window_size}, {len(valid_depths)} valid points, median depth: {Z:.2f}m"
            )

        # The depth value is a Euclidean distance, so back out the z component.
        x_normalized = (u_depth - cx_d) / fx_d
        y_normalized = (v_depth - cy_d) / fy_d
        norm = math.sqrt(x_normalized ** 2 + y_normalized ** 2 + 1)
        z_camera = Z / norm
        x_camera = x_normalized * z_camera

        horizontal_angle_deg = -math.degrees(math.atan2(x_camera, z_camera))
        horizontal_distance = math.sqrt(x_camera ** 2 + z_camera ** 2)

        self._save_depth_debug_image(depth_image, u_depth, v_depth, timestamp, name)
        return horizontal_angle_deg, horizontal_distance

    def _save_depth_debug_image(self, depth_image, u, v, timestamp, name):
        try:
            depth_valid = depth_image[depth_image > 0]
            if depth_valid.size == 0:
                return
            depth_min, depth_max = np.min(depth_valid), np.max(depth_image)
            depth_norm = (255 * (depth_image - depth_min) / (depth_max - depth_min + 1e-6)).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(depth_norm, mode='L').convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 0, 0))
            save_dir = os.path.join(timestamp, "depth_dir")
            os.makedirs(save_dir, exist_ok=True)
            img.save(os.path.join(save_dir, f"{name}.png"))
        except Exception as e:
            rospy.logwarn(f"Failed to save depth debug image: {e}")

    # ------------------------------------------------------------------ #
    # Panoramic capture
    # ------------------------------------------------------------------ #
    def collect_panoramic_images(self):
        """
        Sweep the dual-arm-mounted cameras to capture 7 panoramic views.

        The dual arms reach every yaw around the robot **except** the rear
        (180°), so the panorama has 7 slots, not 8:

            idx 0: front camera                          (   0°)
            idx 1: left  arm, pose 1                     (  45°)
            idx 2: left  arm, pose 2                     (  90°)
            idx 3: left  arm, pose 3                     ( 135°)
            idx 4: right arm, pose 3                     ( 225°)
            idx 5: right arm, pose 2                     ( 270°)
            idx 6: right arm, pose 1                     ( 315°)

        Returns ``(panorama_images, arm_reset_thread)``. The caller must
        ``start()`` the reset thread to run the arm return motion in parallel
        with the downstream VLM call.
        """
        rospy.loginfo("Collecting 7-direction panorama using arms.")
        panorama_images = [None] * 7

        with self.image_lock:
            if self.current_front_rgb is not None:
                panorama_images[0] = self.current_front_rgb.copy()
                rospy.loginfo("Captured image 1/7 (Forward view, 0 deg)")
            else:
                rospy.logwarn("Front RGB image is not available for 0-degree view.")

        # left-arm poses fill indices 1..3 (front-left / left / back-left);
        # right-arm poses fill indices 4..6 (back-right / right / front-right).
        sequence_to_pano_idx = [
            {'left': 1, 'right': 6},
            {'left': 2, 'right': 5},
            {'left': 3, 'right': 4},
        ]
        for i, target_pos in enumerate(self.TARGET_SEQUENCE):
            rospy.loginfo(f"Moving arms to sequence position {i + 1}...")
            self.arm_controller.move_to_goal(target_pos['left'], target_pos['right'])
            rospy.sleep(1.0)
            with self.image_lock:
                left_idx = sequence_to_pano_idx[i]['left']
                if self.current_left_rgb is not None:
                    panorama_images[left_idx] = self.current_left_rgb.copy()
                    rospy.loginfo(f"Captured image {left_idx + 1}/7 (Left arm view)")
                else:
                    rospy.logwarn(f"Left arm RGB image not available for sequence {i + 1}")
                right_idx = sequence_to_pano_idx[i]['right']
                if self.current_right_rgb is not None:
                    panorama_images[right_idx] = self.current_right_rgb.copy()
                    rospy.loginfo(f"Captured image {right_idx + 1}/7 (Right arm view)")
                else:
                    rospy.logwarn(f"Right arm RGB image not available for sequence {i + 1}")

        rospy.loginfo("Starting to return arms to initial position in parallel.")
        home_pos = self.TARGET_SEQUENCE[0]
        arm_reset_thread = threading.Thread(
            target=self.arm_controller.move_to_goal,
            args=(home_pos['left'], home_pos['right']),
        )
        return panorama_images, arm_reset_thread

    # ------------------------------------------------------------------ #
    # Main navigation cycle (single iteration)
    # ------------------------------------------------------------------ #
    def run_navigation_cycle(self, timestamp):
        """Run one full navigation cycle. Return True when the task is done."""
        rospy.loginfo("=" * 20 + " NEW NAVIGATION CYCLE " + "=" * 20)
        panorama_images, arm_reset_thread = self.collect_panoramic_images()
        arm_reset_thread.start()

        if sum(1 for img in panorama_images if img is not None) < 4:
            rospy.logerr("Failed to collect enough images. Aborting cycle.")
            arm_reset_thread.join()
            return False

        rospy.loginfo("Calling Language Action model for direction decision...")
        nav_result = self.api.language_action(
            self.instruction, self.visited_targets, panorama_images, num_directions=config.NUM_DIRECTIONS
        )

        while arm_reset_thread.is_alive():
            if rospy.is_shutdown():
                rospy.logwarn("Shutdown requested. Aborting cycle.")
                arm_reset_thread.join()
                return False
            arm_reset_thread.join(timeout=0.1)
        rospy.loginfo("Arm reset complete.")

        if not nav_result['success']:
            rospy.logwarn(f"Language Action decision failed: {nav_result.get('error', 'Unknown error')}")
            return False

        target_angle = nav_result['target_angle']
        progress_analysis = nav_result['progress_analysis']
        rospy.loginfo(
            f"LA decision: navigate towards angle {target_angle} deg. Reason: {nav_result['reasoning']}"
        )
        self.rotate_angle(target_angle)
        rospy.sleep(2.0)

        with self.image_lock:
            if self.current_front_rgb is None:
                rospy.logerr("Cannot get front RGB image for detection.")
                return False
            front_image_for_detection = self.current_front_rgb.copy()

        rospy.loginfo("Calling Vision Action model for target detection...")
        target_result = self.api.vision_action(
            self.instruction, front_image_for_detection, progress_analysis, self.visited_targets
        )

        if not target_result['success']:
            rospy.logwarn(f"Vision Action detection failed: {target_result.get('error', 'Unknown error')}")
            return False

        bbox = target_result['bbox']
        rospy.loginfo(f"VA target: '{bbox['target']}' with action '{bbox['action']}'.")
        self.save_bbox_image(front_image_for_detection, bbox, timestamp, f"bbox_{len(self.visited_targets)}")

        if bbox['action'] == 'STOP':
            rospy.loginfo("VA action is 'STOP'. Task complete.")
            return True

        u_center = bbox['x1'] + bbox['width'] // 2
        v_center = bbox['y1'] + bbox['height'] // 2
        try:
            final_rotate_deg, final_distance = self.pixel_to_robot_motion(
                (u_center, v_center), timestamp,
                f"depth_{len(self.visited_targets)}", config.WINDOW_SIZE,
            )

            abs_rotate_deg = abs(final_rotate_deg)
            new_abs_rotate_deg = max(0, abs_rotate_deg - config.DIRECTION_OFFSET)
            final_rotate_deg = math.copysign(new_abs_rotate_deg, final_rotate_deg)
            final_distance -= config.DISTANCE_OFFSET

            rospy.loginfo(
                f"Calculated motion: rotate {final_rotate_deg:.2f} deg, move {final_distance:.2f}m."
            )
            self.rotate_angle(final_rotate_deg)
            self.move_distance(final_distance)
            self.visited_targets.append(bbox['target'])
            rospy.loginfo(f"Waypoint added. Visited targets: {self.visited_targets}")

        except ValueError as e:
            rospy.logerr(f"Could not calculate motion: {e}")

        return False

    # ------------------------------------------------------------------ #
    # Debug output and shutdown
    # ------------------------------------------------------------------ #
    def save_bbox_image(self, image, bbox, timestamp, name):
        try:
            pil_img = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
            draw = ImageDraw.Draw(pil_img)
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except OSError:
                font = ImageFont.load_default(size=20)
            draw.text((x1, max(0, y1 - 20)), bbox.get('target', 'target'), fill="red", font=font)

            save_dir = os.path.join(timestamp, "bbox_images")
            os.makedirs(save_dir, exist_ok=True)
            pil_img.save(os.path.join(save_dir, f"{name}.jpg"))
        except Exception as e:
            rospy.logerr(f"Failed to save bbox image: {e}")

    def shutdown(self):
        rospy.loginfo("Shutting down the integrated controller.")
        self.cmd_vel_pub.publish(Twist())
        self.arm_controller.stop_control_thread()
        self.model.print_usage_stats()
