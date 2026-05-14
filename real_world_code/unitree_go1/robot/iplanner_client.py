"""HTTP client for the iPlanner remote service."""
import json

import cv2
import numpy as np
import requests


class IPlannerRemoteClient:
    """Thin requests-based wrapper around the iPlanner Flask server."""

    def __init__(self, server_url="http://localhost:8888"):
        self.server_url = server_url
        self.session = requests.Session()
        self.initialized = False

    def reset(self, intrinsic=None, stop_threshold=0.1, batch_size=1):
        """Initialise / reset the planner on the server."""
        url = f"{self.server_url}/navigator_reset"
        payload = {
            "intrinsic": intrinsic or [[384.0, 0.0, 320.0],
                                       [0.0, 384.0, 240.0],
                                       [0.0, 0.0, 1.0]],
            "stop_threshold": stop_threshold,
            "batch_size": batch_size,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=2)
            if resp.status_code == 200:
                print(f"iPlanner server connected at {self.server_url}")
                self.initialized = True
                return True
            print(f"iPlanner init failed: {resp.text}")
            return False
        except Exception as e:
            if "Connection refused" in str(e):
                print(f"Cannot connect to iPlanner server at {self.server_url}. Is it running?")
            else:
                print(f"iPlanner connection error: {e}")
            return False

    def get_plan(self, rgb_img, depth_img_mm, goal_local):
        """Request a path plan.

        Args:
            rgb_img: ``(H, W, 3)`` BGR or RGB array.
            depth_img_mm: ``(H, W)`` uint16 depth image in millimetres.
            goal_local: ``(x, y)`` goal in the robot frame.

        Returns:
            ``(trajectory_points, fear_value)`` on success, otherwise ``(None, None)``.
        """
        if not self.initialized and not self.reset():
            return None, None

        url = f"{self.server_url}/pointgoal_step"

        goal_payload = {
            "goal_x": [float(goal_local[0])],
            "goal_y": [float(goal_local[1])],
        }

        _, rgb_encoded = cv2.imencode(".png", rgb_img)

        # Scale depth from millimetres to 0.1-millimetre units as expected by the server.
        depth_scaled = (depth_img_mm.astype(np.uint32) * 10).astype(np.uint16)
        _, depth_encoded = cv2.imencode(".png", depth_scaled)

        files = {
            "image": ("rgb.png", rgb_encoded.tobytes(), "image/png"),
            "depth": ("depth.png", depth_encoded.tobytes(), "image/png"),
        }
        data = {"goal_data": json.dumps(goal_payload)}

        try:
            resp = self.session.post(url, files=files, data=data, timeout=2)
            if resp.status_code == 200:
                result = resp.json()
                traj = result["trajectory"][0]
                fear = result["all_values"][0][0]
                return np.array(traj), fear
            print(f"iPlanner request failed: {resp.status_code} - {resp.text}")
            return None, None
        except Exception as e:
            print(f"iPlanner network error: {e}")
            return None, None
