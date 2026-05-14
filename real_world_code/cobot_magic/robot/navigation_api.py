import base64
import io
import json
import re
import numpy as np
from PIL import Image

import prompts


# Angle / label definitions for the 6-, 7-, and 8-direction panoramic layouts.
# The 7-direction layout matches the Cobot Magic hardware: the dual arms sweep
# the wrist-mounted cameras through six angles around the robot, but the rear
# pose (180°) is mechanically unreachable, so it is omitted.
VIEW_DEFINITIONS = {
    6: [
        {'angle': 0, 'name': 'image1', 'label': 'Image 1: Current FORWARD view (0°)'},
        {'angle': 60, 'name': 'image2', 'label': 'Image 2: Front-RIGHT view (60°)'},
        {'angle': 120, 'name': 'image3', 'label': 'Image 3: Back-RIGHT view (120°)'},
        {'angle': 180, 'name': 'image4', 'label': 'Image 4: BEHIND view (180°)'},
        {'angle': 240, 'name': 'image5', 'label': 'Image 5: Back-LEFT view (240°)'},
        {'angle': 300, 'name': 'image6', 'label': 'Image 6: Front-LEFT view (300°)'},
    ],
    7: [
        {'angle': 0,   'name': 'image1', 'label': 'Image 1: Current FORWARD view (0°)'},
        {'angle': 45,  'name': 'image2', 'label': 'Image 2: Front-LEFT view (45°)'},
        {'angle': 90,  'name': 'image3', 'label': 'Image 3: LEFT view (90°)'},
        {'angle': 135, 'name': 'image4', 'label': 'Image 4: Back-LEFT view (135°)'},
        # Back view (180°) intentionally skipped: the arms cannot reach it.
        {'angle': 225, 'name': 'image5', 'label': 'Image 5: Back-RIGHT view (225°)'},
        {'angle': 270, 'name': 'image6', 'label': 'Image 6: RIGHT view (270°)'},
        {'angle': 315, 'name': 'image7', 'label': 'Image 7: Front-RIGHT view (315°)'},
    ],
    8: [
        {'angle': 0, 'name': 'image1', 'label': 'Image 1: Current FORWARD view (0°)'},
        {'angle': 45, 'name': 'image2', 'label': 'Image 2: Front-LEFT view (45°)'},
        {'angle': 90, 'name': 'image3', 'label': 'Image 3: LEFT view (90°)'},
        {'angle': 135, 'name': 'image4', 'label': 'Image 4: Back-LEFT view (135°)'},
        {'angle': 180, 'name': 'image5', 'label': 'Image 5: BEHIND view (180°)'},
        {'angle': 225, 'name': 'image6', 'label': 'Image 6: Back-RIGHT view (225°)'},
        {'angle': 270, 'name': 'image7', 'label': 'Image 7: RIGHT view (270°)'},
        {'angle': 315, 'name': 'image8', 'label': 'Image 8: Front-RIGHT view (315°)'},
    ],
}


class LaViRANavigationAPI:
    """Wraps the two LaViRA calls: Language Action (direction) and Vision Action (target)."""

    def __init__(self, model_client, image_width, image_height):
        self.model = model_client
        self.width = image_width
        self.height = image_height

    def img_to_base64(self, img):
        """Encode an image as a base64 PNG string for API transmission."""
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def language_action(self, instruction, visited_targets, panorama_images, num_directions=7):
        """Language Action call: choose a panoramic direction to navigate toward."""
        if num_directions not in VIEW_DEFINITIONS:
            raise ValueError(f"num_directions must be one of {sorted(VIEW_DEFINITIONS)}")
        if len(panorama_images) < num_directions:
            raise ValueError(f"Need at least {num_directions} panoramic images")

        panorama_images = panorama_images[:num_directions]
        view_definitions = VIEW_DEFINITIONS[num_directions]

        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]

        if visited_targets:
            history_text = "\nPreviously visited targets:\n"
            for i, target in enumerate(visited_targets):
                history_text += f"  - Target {i + 1}: {target}\n"
        else:
            history_text = "\nNo targets visited yet."
        content.append({"type": "text", "text": history_text})

        content.append({"type": "text", "text": f"\nCurrent {num_directions}-directional views:"})

        for i, view in enumerate(view_definitions):
            if i < len(panorama_images):
                if panorama_images[i] is None:
                    content.append({"type": "text", "text": f"{view['label']} - IMAGE NOT AVAILABLE"})
                    continue
                img_base64 = self.img_to_base64(panorama_images[i])
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
                content.append({"type": "text", "text": view['label']})

        action_choices = ""
        for i, view in enumerate(view_definitions):
            action_choices += (
                f"{i + 1}. Select {view['name']} ({view['angle']}°) - "
                f"navigate toward the {view['label'].split(':')[1].strip()}\n"
            )

        prompt = prompts.get_la_prompt(num_directions, action_choices)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=1024,
            temperature=0,
            use_la=True,
        )

        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                action = response_data.get('action', 'image1').lower()
                progress_analysis = response_data.get('progress_analysis', '')
                reasoning = response_data.get('reasoning', '')

                image_to_angle = {view['name']: view['angle'] for view in view_definitions}

                selected_image = 'image1'
                for img_name in image_to_angle.keys():
                    if img_name in action:
                        selected_image = img_name
                        break

                target_angle = image_to_angle[selected_image]
                selected_image_idx = int(selected_image.replace('image', '')) - 1

                return {
                    'success': True,
                    'selected_image': selected_image,
                    'selected_image_idx': selected_image_idx,
                    'target_angle': target_angle,
                    'progress_analysis': progress_analysis,
                    'reasoning': reasoning,
                    'raw_response': output_text,
                }

            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f"JSON parsing error: {e}",
                    'selected_image': 'image1',
                    'selected_image_idx': 0,
                    'target_angle': 0,
                    'progress_analysis': 'Unable to analyze due to parsing error',
                    'reasoning': 'Fallback to forward navigation',
                    'raw_response': output_text,
                }

        return {
            'success': False,
            'error': 'No JSON found in response',
            'selected_image': 'image1',
            'selected_image_idx': 0,
            'target_angle': 0,
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation',
            'raw_response': output_text,
        }

    def vision_action(self, instruction, selected_image, progress_analysis, visited_targets):
        """Vision Action call: locate the next target's bounding box, or decide STOP."""
        if visited_targets and len(visited_targets) > 0:
            visited_targets_str = "\n\nPreviously visited targets:\n"
            for i, target in enumerate(visited_targets):
                visited_targets_str += f"  - Target {i + 1}: {target}\n"
        else:
            visited_targets_str = "\n\nNo targets visited yet."

        content = [{
            'type': 'text',
            'text': 'You are a robot performing navigation task. Look at this image and help the robot navigate.'
        }]

        img_base64 = self.img_to_base64(selected_image)
        content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{img_base64}"}})

        progress_info = ""
        if progress_analysis:
            progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"

        prompt = prompts.get_va_prompt(
            instruction, self.width, self.height, visited_targets_str, progress_info
        )
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=1024,
            temperature=0,
            use_la=False,
        )

        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                action_decision = response_data.get('action', 'NAVIGATE').upper()
                bbox_2d = response_data.get(
                    'bbox_2d',
                    [self.width // 4, self.height // 4, 3 * self.width // 4, 3 * self.height // 4],
                )

                if len(bbox_2d) >= 4:
                    x1, y1, x2, y2 = bbox_2d[:4]
                else:
                    x1, y1, x2, y2 = self.width // 4, self.height // 4, 3 * self.width // 4, 3 * self.height // 4

                bbox = {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress'),
                }

                bbox['x1'] = max(0, min(bbox['x1'], self.width - 1))
                bbox['y1'] = max(0, min(bbox['y1'], self.height - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], self.width))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], self.height))

                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']

                return {
                    'success': True,
                    'bbox': bbox,
                    'raw_response': output_text,
                }

            except json.JSONDecodeError as e:
                return {
                    'success': False,
                    'error': f"JSON parsing error: {e}",
                    'bbox': self._get_fallback_bbox(),
                    'raw_response': output_text,
                }

        return {
            'success': False,
            'error': 'No JSON found in response',
            'bbox': self._get_fallback_bbox(),
            'raw_response': output_text,
        }

    def _get_fallback_bbox(self):
        x1, y1 = self.width // 4, self.height // 4
        x2, y2 = 3 * self.width // 4, 3 * self.height // 4
        return {
            'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'target': 'fallback target', 'action': 'NAVIGATE',
            'reasoning': 'Fallback due to error', 'progress': 'Unknown',
        }
