import cv2
import numpy as np
import os
from habitat import logger
from collections import Sequence
from vlnce_baselines.utils.constant import legend_color_palette
from habitat.core.simulator import Observations
import time
from PIL import Image
import torch

def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image():
    vis_image = np.ones((755, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Goal: "
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    vis_image = add_class(vis_image, 0, "out of map", legend_color_palette)
    vis_image = add_class(vis_image, 1, "obstacle", legend_color_palette)
    vis_image = add_class(vis_image, 2, "free space", legend_color_palette)
    vis_image = add_class(vis_image, 3, "agent trajecy", legend_color_palette)
    vis_image = add_class(vis_image, 4, "waypoint", legend_color_palette)

    return vis_image


def add_class(vis_image, id, name, color_palette):
    text = f"{id}:{name}"
    font_color = (0, 0, 0)
    class_color = list(color_palette[id])
    class_color.reverse() # BGR -> RGB
    fontScale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = 20, 50
    start_x, start_y = 25, 540 # x is horizontal and point to right, y is vertical and point to down
    gap_x, gap_y = 280, 30
    
    x1 = start_x + (id % 4) * gap_x
    y1 = start_y + (id // 4) * gap_y
    x2 = x1 + w
    y2 = y1 + h
    text_x = x2 + 10
    text_y = (y1 + y2) // 2 + 5
    
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), class_color, thickness=cv2.FILLED)
    vis_image = cv2.putText(vis_image, text, (text_x, text_y),
                            font, fontScale, font_color, thickness, cv2.LINE_AA)
    
    return vis_image


def add_text(image: np.ndarray, text: str, position: Sequence):
    textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
    x, y = position
    x += (480 - textsize[0]) // 2
    image = cv2.putText(image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=0, thickness=1, lineType=cv2.LINE_AA)
    
    return image

ACTION_TO_TEXT = ['stop', 'forward', 'turn left', 'turn right']
ACTION_TO_DISPLAY = {
    0: 'STOP',
    1: 'FORWARD',
    2: 'TURN LEFT',
    3: 'TURN RIGHT',
    'stop': 'STOP',
    'forward': 'FORWARD',
    'turn left': 'TURN LEFT',
    'turn right': 'TURN RIGHT',
}

class LaViRAVisualizer:
    def __init__(self, mapping_module, visualize=False, save_dir="./visualizations", width=640, height=480):
        self.visualize = visualize
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.current_episode_id = None
        self.current_step = 0
        if self.visualize:
            os.makedirs(self.save_dir, exist_ok=True)
        self.rgb_history = []
        self.mapping_module = mapping_module

    def sync(self, step, ep_id):
        self.current_step = step
        self.current_episode_id = ep_id

    def update_map(self, mapping_module):
        self.mapping_module = mapping_module

    def _save_rgb_with_bbox(self, rgb_image, bbox):
        """Save RGB image with bounding box annotation as a separate file"""
        if not self.visualize:
            return

        try:
            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                # Make a copy to avoid modifying the original
                annotated_image = rgb_image.copy()
            else:
                annotated_image = np.array(rgb_image)

            # Draw bounding box
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            target_description = bbox.get('target', 'target')
            action_decision = bbox.get('action', 'NAVIGATE')

            # Choose color based on action decision
            box_color = (0, 255, 0) if action_decision == 'NAVIGATE' else (0, 0,
                                                                           255)  # Green for navigate, Red for stop

            # Draw rectangle with thicker border for better visibility
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), box_color, 4)

            # Create enhanced label with action info
            label = f"{action_decision}: {target_description}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9  # Slightly larger font
            thickness = 2

            # Get text size to create background with more padding
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Calculate text position with more spacing
            padding = 15  # Increased padding
            text_y = y - padding if y - padding > text_height + padding else y + height + text_height + padding

            # Draw text background with more generous padding using same color as box
            bg_x1 = max(0, x - 5)
            bg_y1 = max(0, text_y - text_height - padding)
            bg_x2 = min(self.width, x + text_width + padding * 2)
            bg_y2 = min(self.height, text_y + baseline + 5)

            cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)

            # Draw text with better positioning
            text_x = max(padding, x + 5)
            cv2.putText(annotated_image, label, (text_x, text_y - 5),
                        font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

            # Save to episode-specific folder with chronological naming
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            img_folder = os.path.join(self.save_dir, episode_id)

            # Create episode folder if it doesn't exist
            os.makedirs(img_folder, exist_ok=True)

            # Use current step for chronological ordering
            filename = f"bbox_step{self.current_step:04d}.png"
            filepath = os.path.join(img_folder, filename)

            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, annotated_image_bgr)

            # logger.info(f"Saved RGB with bbox annotation: {filepath}")

        except Exception as e:
            logger.info(f"Error saving RGB with bbox: {e}")

    def _save_depth(
            self,
            depth_image: np.ndarray,
            step: int,
            invert: bool = False,
            save_16bit: bool = False,
            use_colormap: bool = True,
            colormap: int = cv2.COLORMAP_JET
    ):
        """
        Save depth:
          \- If save_16bit=True: 16-bit single channel PNG (no colormap).
          \- Else if use_colormap=True: colored PNG using OpenCV colormap (default Jet).
          \- Else: 8-bit grayscale PNG.
        Args:
            depth_image: float32 depth normalized or raw in [0,1] (will be clipped).
            invert: if True, near becomes bright.
            save_16bit: save a 16-bit raw depth (no colormap).
            use_colormap: apply colormap (ignored if save_16bit=True).
            colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
        Returns:
            np.ndarray saved array (8-bit color/grayscale or 16-bit).
        """
        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 1.0)
            if invert:
                depth = 1.0 - depth

            episode_id = str(self.current_episode_id) if getattr(self, "current_episode_id", None) else "unknown"
            out_dir = os.path.join(self.save_dir, episode_id)
            os.makedirs(out_dir, exist_ok=True)

            if save_16bit:
                depth_u16 = (depth * 65535.0 + 0.5).astype(np.uint16)
                fname = f"depth16_step{step:04d}.png"
                path = os.path.join(out_dir, fname)
                cv2.imwrite(path, depth_u16)
                return depth_u16

            depth_u8 = (depth * 255.0 + 0.5).astype(np.uint8)

            if use_colormap:
                colored = cv2.applyColorMap(depth_u8, colormap)
                fname = f"depth_color_step{step:04d}.png"
                path = os.path.join(out_dir, fname)
                cv2.imwrite(path, colored)
                return colored
            else:
                fname = f"depth_gray_step{step:04d}.png"
                path = os.path.join(out_dir, fname)
                cv2.imwrite(path, depth_u8)
                return depth_u8
        except Exception as e:
            logger.info(f"Error saving depth: {e}")
            return None

    def _create_combined_image(self, rgb_image, metadata, step, visited_targets, target_coords=None, reasoning_info=None):
        """Create a combined image with RGB on left, VLM map on right, instruction text, and reasoning info

        Args:
            rgb_image: RGB observation image
            metadata: Metadata dict containing episode info
            step: Current step number
            target_coords: Tuple of (target_map_x, target_map_y) or None
            reasoning_info: Dict containing reasoning and progress_analysis from LLM
        """
        # Create goal tensor for visualization if target coordinates are available
        goal_tensor = None
        if target_coords is not None and target_coords[0] is not None and target_coords[1] is not None:
            goal_tensor = torch.tensor([target_coords[0], target_coords[1]], dtype=torch.float32)

        # Generate VLM map
        vlm_map = self.mapping_module.create_vlm_map_from_state(
            self.current_episode_id, 0, goal_tensor,
            output_size=(1024, 1024), visited_targets=visited_targets, display_last=True
        ).copy()

        cv2.imwrite(f'saved_rgb_images/{self.current_episode_id}/debug_vlm_map_step{step:04d}.png', vlm_map)

        # Convert RGB image to proper format if needed
        if isinstance(rgb_image, np.ndarray):
            rgb_display = rgb_image.copy()
        else:
            rgb_display = np.array(rgb_image)

        # Ensure RGB is in correct format (H, W, 3)
        if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
            pass  # Already correct format
        else:
            logger.info(f"Warning: Unexpected RGB shape: {rgb_display.shape}")

        # Convert VLM map from BGR to RGB for consistent display
        if len(vlm_map.shape) == 3:
            vlm_map_display = cv2.cvtColor(vlm_map, cv2.COLOR_BGR2RGB)
        else:
            vlm_map_display = vlm_map.copy()

        # Get target height from RGB image
        target_height = rgb_display.shape[0]

        # Resize VLM map to match RGB height while preserving aspect ratio
        vlm_old_h, vlm_old_w = vlm_map_display.shape[:2]
        if vlm_old_h == 0 or vlm_old_w == 0:
            vlm_target_width = target_height
        else:
            vlm_target_width = int(float(target_height) / vlm_old_h * vlm_old_w)

        # Resize VLM map
        vlm_map_resized = cv2.resize(
            vlm_map_display,
            (vlm_target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Pad images to same height with white background
        def pad_to_height(img, target_height):
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            if img.shape[0] < target_height:
                pad_height = target_height - img.shape[0]
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
            return img

        rgb_padded = pad_to_height(rgb_display, target_height)
        vlm_map_padded = pad_to_height(vlm_map_resized, target_height)

        # Combine horizontally: RGB on left, VLM map on right
        frame = np.concatenate((rgb_padded, vlm_map_padded), axis=1)

        # Prepare metadata for text display
        instruction = metadata.get('instruction', 'No instruction')
        destination = metadata.get('destination', 'unknown')
        action = metadata.get('action', 'unknown')

        action_display = 'UNKNOWN'
        try:
            if isinstance(action, dict):
                action = action.get('action', '')

            if isinstance(action, str) and action.isdigit():
                action = int(action)
            elif isinstance(action, str):
                pass

            # Try to get display action text
            if isinstance(action, int) and 0 <= action < len(ACTION_TO_TEXT):
                action_display = ACTION_TO_DISPLAY.get(action, ACTION_TO_TEXT[action].upper())
            elif isinstance(action, str):
                action_display = ACTION_TO_DISPLAY.get(action, action.upper())
            else:
                action_display = 'UNKNOWN'
        except Exception as e:
            logger.info(f"Error parsing action: {e}")
            action_display = 'UNKNOWN'

        # Create instruction area - increased height to accommodate reasoning info
        instruction_height = 360  # Further increased from 280 to accommodate longer reasoning text
        white_bg = np.ones((instruction_height, frame.shape[1], 3), dtype=np.uint8) * 255

        # Text rendering setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25

        # Function to wrap long text
        def wrap_text(text, max_width, font, font_scale, thickness):
            if not text:
                return [text]

            words = text.split(' ')
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

                if text_size[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        lines.append(word)

            if current_line:
                lines.append(current_line)

            return lines if lines else [text]

        # Calculate max width for text
        max_text_width = frame.shape[1] - 30

        y_pos = 20
        base_text_lines = [
            f"Step: {step} | Episode: {metadata.get('episode_id', 'N/A')} | ACTION: {action_display}",
            f"Destination: {destination}",
            "",
            f"Instruction: {instruction}"
        ]

        # Draw text lines
        for line in base_text_lines:
            if line.strip():
                color = (0, 0, 0)
                current_font_scale = font_scale
                current_thickness = thickness

                if "Step:" in line and "ACTION:" in line:
                    color = (0, 0, 255)
                    current_font_scale = 0.7
                    current_thickness = 2
                elif "Destination:" in line:
                    color = (0, 165, 255)
                elif "Instruction:" in line:
                    color = (255, 0, 0)
                    # Wrap instruction text
                    if len(line) > 20:
                        instruction_text = line.replace("Instruction: ", "")
                        wrapped_instruction = wrap_text(instruction_text, max_text_width - 120, font,
                                                        current_font_scale, current_thickness)

                        # Draw "Instruction: " first
                        cv2.putText(
                            white_bg,
                            "Instruction: ",
                            (15, y_pos),
                            font,
                            current_font_scale,
                            color,
                            current_thickness,
                            lineType=cv2.LINE_AA
                        )
                        y_pos += line_height

                        # Draw wrapped lines with indentation
                        for wrapped_line in wrapped_instruction:
                            cv2.putText(
                                white_bg,
                                wrapped_line,
                                (30, y_pos),
                                font,
                                current_font_scale,
                                color,
                                current_thickness,
                                lineType=cv2.LINE_AA
                            )
                            y_pos += line_height
                        continue

                # Regular text drawing
                cv2.putText(
                    white_bg,
                    line,
                    (15, y_pos),
                    font,
                    current_font_scale,
                    color,
                    current_thickness,
                    lineType=cv2.LINE_AA
                )
            y_pos += line_height

        # Add reasoning and progress analysis information if available
        if reasoning_info:
            y_pos += 5  # Add some spacing

            # Draw progress analysis
            progress = reasoning_info.get('progress_analysis', '')
            if progress:
                cv2.putText(
                    white_bg,
                    "Progress Analysis: ",
                    (15, y_pos),
                    font,
                    0.5,  # Slightly smaller font
                    (0, 128, 0),  # Dark green
                    1,
                    lineType=cv2.LINE_AA
                )
                y_pos += line_height

                # Wrap progress analysis text
                wrapped_progress = wrap_text(progress, max_text_width - 60, font, 0.5, 1)
                for wrapped_line in wrapped_progress:
                    if y_pos < instruction_height - 50:  # Give more space for reasoning below
                        cv2.putText(
                            white_bg,
                            wrapped_line,
                            (30, y_pos),
                            font,
                            0.5,
                            (0, 128, 0),
                            1,
                            lineType=cv2.LINE_AA
                        )
                        y_pos += 18  # Slightly tighter line spacing for better space utilization

            # Draw reasoning
            reasoning = reasoning_info.get('reasoning', '')
            if reasoning and y_pos < instruction_height - 70:  # More generous space check
                y_pos += 5  # Add some spacing
                cv2.putText(
                    white_bg,
                    "Reasoning: ",
                    (15, y_pos),
                    font,
                    0.5,  # Slightly smaller font
                    (128, 0, 128),  # Purple
                    1,
                    lineType=cv2.LINE_AA
                )
                y_pos += line_height

                # Wrap reasoning text
                wrapped_reasoning = wrap_text(reasoning, max_text_width - 60, font, 0.5, 1)
                for wrapped_line in wrapped_reasoning:
                    if y_pos < instruction_height - 15:  # Use almost all available space
                        cv2.putText(
                            white_bg,
                            wrapped_line,
                            (30, y_pos),
                            font,
                            0.5,
                            (128, 0, 128),
                            1,
                            lineType=cv2.LINE_AA
                        )
                        y_pos += 18  # Same tighter spacing as progress analysis

        # Create header for the two panels
        header_height = 25
        header_bg = np.ones((header_height, frame.shape[1], 3), dtype=np.uint8) * 240

        rgb_width = rgb_padded.shape[1]
        vlm_width = vlm_map_padded.shape[1]

        headers = ["RGB View", "VLM Navigation Map"]
        header_colors = [(0, 100, 0), (0, 0, 139)]  # Dark green, Dark red

        # Position headers in center of each panel
        x_positions = [
            rgb_width // 2,  # RGB center
            rgb_width + vlm_width // 2  # VLM map center
        ]

        for i, (header, color, x_pos) in enumerate(zip(headers, header_colors, x_positions)):
            text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_centered = max(0, x_pos - text_size[0] // 2)

            cv2.putText(
                header_bg,
                header,
                (x_centered, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        # Combine all components: header + main frame + instruction area
        final_frame = np.concatenate([header_bg, frame, white_bg], axis=0)

        return Image.fromarray(final_frame.astype(np.uint8))

    def _save_rgb_frame(self, obs: Observations, step: int, visited_targets, episode_id: str = None, target_coords: tuple = None):
        """Save RGB frame with metadata, instruction text and top-down map"""
        if not self.visualize:
            return

        rgb_image = obs['rgb']
        episode_id = episode_id or self.current_episode_id
        episode_id = str(episode_id) if not isinstance(episode_id, str) else episode_id

        metadata = {
            'episode_id': episode_id,
            'step': step,
            'timestamp': time.time(),
            'instruction': getattr(self, 'instruction', ''),
            'pose': obs.get('sensor_pose', None),
            'destination': getattr(self, 'destination', 'unknown'),
            'action': getattr(self, '_action', 'unknown')
        }

        # 保存单独的RGB文件
        self._save_individual_rgb(rgb_image, episode_id, step)

        # logger.info(target_coords)
        # Create combined image with RGB, top-down map, and value map heatmap
        combined_image = self._create_combined_image(rgb_image, metadata, step, visited_targets, target_coords)

        img_folder = os.path.join(self.save_dir, episode_id)
        img_path = os.path.join(img_folder, f"combined_step{step:04d}.png")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        combined_image.save(img_path)

    def _save_individual_rgb(self, rgb_image, episode_id, step):
        """Save individual RGB image as a separate file"""
        try:
            episode_id = str(episode_id) if episode_id else "unknown"

            # Create episode-specific RGB folder
            rgb_folder = os.path.join(self.save_dir, episode_id, "rgb_only")
            os.makedirs(rgb_folder, exist_ok=True)

            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                rgb_display = rgb_image.copy()
            else:
                rgb_display = np.array(rgb_image)

            # Save RGB image
            filename = f"rgb_step{step:04d}.png"
            filepath = os.path.join(rgb_folder, filename)

            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
                rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, rgb_bgr)
            else:
                # Fallback: save as PIL Image
                if isinstance(rgb_image, np.ndarray):
                    Image.fromarray(rgb_image).save(filepath)
                else:
                    rgb_image.save(filepath)

        except Exception as e:
            logger.info(f"Error saving individual RGB: {e}")

    def _save_waypoint_panorama_rgb(self, panorama_frames, waypoint_id, step):
        """Save the 4 directional RGB images for a waypoint"""
        try:
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"

            # Create waypoint-specific folder
            waypoint_folder = os.path.join(self.save_dir, episode_id, "waypoints",
                                           f"waypoint_{waypoint_id:02d}_step{step:04d}")
            os.makedirs(waypoint_folder, exist_ok=True)

            # Direction names for the 4 cardinal directions
            direction_names = ['forward', 'left', 'behind', 'right']

            for i, frame in enumerate(panorama_frames):
                if i >= 4:  # Only save first 4 frames (forward, left, behind, right)
                    break

                rgb_image = frame['rgb']
                direction_name = direction_names[i]
                angle = frame.get('angle', i * 90)

                # Convert RGB image to proper format
                if isinstance(rgb_image, np.ndarray):
                    rgb_display = rgb_image.copy()
                else:
                    rgb_display = np.array(rgb_image)

                # Save RGB image with direction name
                filename = f"{direction_name}_{angle:03d}deg.png"
                filepath = os.path.join(waypoint_folder, filename)

                # Convert RGB to BGR for cv2 saving
                if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
                    rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, rgb_bgr)
                else:
                    # Fallback: save as PIL Image
                    if isinstance(rgb_image, np.ndarray):
                        Image.fromarray(rgb_image).save(filepath)
                    else:
                        rgb_image.save(filepath)

            logger.info(f"Saved waypoint {waypoint_id} panorama RGB images to: {waypoint_folder}")

        except Exception as e:
            logger.info(f"Error saving waypoint panorama RGB: {e}")
