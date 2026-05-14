"""
LaViRA navigation API wrappers for the Go1 VLN task.

Two methods mirror the simulation repo's contract:
- ``language_action`` builds the panorama / history / TODO message and asks the
  LA model for the next direction.
- ``vision_action`` shows the post-rotation front view to the VA model and
  returns either ``STOP`` or a bounding box of the next intermediate target.

Both methods consume ``LaViRAVisionClient`` and produce the same JSON shapes
that the simulation pipeline uses, so prompts can be A/B compared.
"""
import os
import time

import cv2

import prompts
from ai_client import LaViRAVisionClient
from config import Config
from utils import (
    img_to_base64, numpy_to_base64, print_model_interaction, print_step,
    safe_json_loads, save_output,
)


class LaViRANavigationAPI:
    """LA + VA prompt orchestration for the Go1 4-direction VLN loop."""

    def __init__(self, client: LaViRAVisionClient):
        self.client = client

    # ------------------------------------------------------------------ #
    # Language Action
    # ------------------------------------------------------------------ #
    def language_action(self, instruction, global_target, todo_list,
                        iplanner_history, panorama_frames, current_step):
        """Strategic-direction call. Returns the parsed JSON dict."""
        print_step(current_step, "Language Action: deciding direction")

        history_content = []
        recent_history = iplanner_history[-5:]
        if recent_history:
            history_content.append({
                "type": "text",
                "text": "Recent visual history (trajectory plans):",
            })
            for i, path in enumerate(recent_history):
                b64 = img_to_base64(path)
                history_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                history_content.append({
                    "type": "text",
                    "text": f"History Plan -{len(recent_history) - i}",
                })
        else:
            history_content.append({"type": "text", "text": "No visual history yet."})

        content = [{
            "type": "text",
            "text": f'Navigation Task: "{instruction}"\n\n- Current Step: {current_step}',
        }]
        content.extend(history_content)
        content.append({"type": "text", "text": "Current panorama views:"})

        for frame in panorama_frames:
            if not frame.get("img_base64"):
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['img_base64']}"},
            })
            content.append({"type": "text", "text": frame["label"]})

        prompt = prompts.get_la_navigation_prompt(
            instruction, global_target, todo_list,
            "Visual history provided above.", current_step,
        )
        content.append({"type": "text", "text": prompt})

        result_text, stats = self.client.generate_with_la(
            [{"role": "user", "content": content}], max_new_tokens=1024,
        )
        print_model_interaction(
            "LA Model (Strategic Brain)", content, result_text,
            speed=stats["output_speed"], duration=stats["duration"],
            prompt_speed=stats["prompt_speed"],
        )
        save_output(Config.LOG_DIR, f"step{current_step}_navigation_raw.txt", result_text)
        return safe_json_loads(result_text)

    # ------------------------------------------------------------------ #
    # Vision Action
    # ------------------------------------------------------------------ #
    def vision_action(self, img_np, instruction, global_target,
                      strategic_goal, strategic_stop, current_step):
        """Tactical bbox / NAVIGATE / STOP call.

        Returns the parsed JSON dict with the bbox already de-normalised from
        Qwen's ``[0, 1000]`` range into pixel coordinates.
        """
        print_step(current_step, "Vision Action: verifying view")
        if img_np is None:
            return {}

        img_base64 = numpy_to_base64(img_np)
        if not img_base64:
            return {}

        prompt = prompts.get_va_prompt(
            instruction, global_target, strategic_goal, strategic_stop,
        )
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            {"type": "text", "text": prompt},
        ]

        result_text, stats = self.client.generate_with_va(
            [{"role": "user", "content": content}], max_new_tokens=1024,
        )
        print_model_interaction(
            "VA Model (Tactical Eyes)", content, result_text,
            speed=stats["output_speed"], duration=stats["duration"],
            prompt_speed=stats["prompt_speed"],
        )
        save_output(Config.LOG_DIR, f"step{current_step}_bbox_raw.txt", result_text)

        parsed = safe_json_loads(result_text)
        bbox = parsed.get("bbox_2d", [])
        if bbox and len(bbox) == 4:
            h, w = img_np.shape[:2]
            x1, y1, x2, y2 = bbox
            parsed["bbox_2d"] = [
                x1 / 1000.0 * w, y1 / 1000.0 * h,
                x2 / 1000.0 * w, y2 / 1000.0 * h,
            ]

        save_path = os.path.join(Config.BBOX_DIR, f"step{current_step}_bbox_vis.png")
        self._draw_bbox(img_np.copy(), parsed, save_path)
        return parsed

    # ------------------------------------------------------------------ #
    # Visualisation helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _draw_bbox(img, bbox_data, save_path):
        bbox = bbox_data.get("bbox_2d", [])
        if not bbox or len(bbox) != 4:
            cv2.imwrite(save_path, img)
            return
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y1, y2 = max(0, int(y1)), min(h, int(y2))
        action = bbox_data.get("action", "NAVIGATE")
        color = (0, 255, 0) if action == "NAVIGATE" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, action, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite(save_path, img)
