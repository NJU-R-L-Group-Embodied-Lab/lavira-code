import base64
import copy
import gzip
import io
import json
import os
import re
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image
from fastdtw import fastdtw
from skimage.morphology import binary_closing
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
import supervision as sv

from habitat import logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

# Import Habitat visualization utilities
from habitat.utils.visualizations import maps

from .utils.prompts import *
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.utils.constant import base_classes, map_channels
from .utils.api import LaViRA_API
from .utils.depth_utils import get_world_xz_from_pixel
from .utils.visualization import LaViRAVisualizer

import warnings

warnings.filterwarnings('ignore')
 
class VLMReasoningAgent:
    def __init__(self, visualizer: LaViRAVisualizer):
        va_api_key = os.getenv('VA_API_KEY', None)
        va_base_url = os.getenv("VA_BASE_URL", None)
        va_model_name = os.getenv('VA_MODEL_NAME', 'Qwen/Qwen2.5-VL-32B-Instruct')

        la_api_key = os.getenv('LA_API_KEY', None)
        la_base_url = os.getenv('LA_BASE_URL', None)
        la_model_name = os.getenv("LA_MODEL_NAME", 'gpt-4o-2024-11-20')
        self.model = LaViRA_API(
            la_api_key=la_api_key,
            la_base_url=la_base_url,
            la_model_name=la_model_name,
            va_model_name=va_model_name,
            va_api_key=va_api_key,
            va_base_url=va_base_url
        )
        self.model.eval()
        self.visualizer = visualizer


    def img_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64

    def navigate_or_backtrack(self, instruction, visited_targets):
        """
        Use Language Action Model to analyze instruction + history + 4-dir images，decide where to nav or backtrack
        return LA：navigate to [left, right, forward, behind] / backtrack to <waypoint_id>
        """

        panorama_images = visited_targets[-1]['panorama_frames'] if visited_targets else []

        # History：[init image] -> "turn xxx" -> [dir image] -> "go to xxx" -> [arrival image] -> ...
        history_content = []

        for i, target in enumerate(visited_targets[:-1]):
            if 'init_image' in target:
                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(target['init_image'])}"}})
                history_content.append({"type": "text", "text": f"Waypoint {i}: Initial view"})

            if 'turn_action' in target:
                history_content.append({"type": "text", "text": f"Action: {target['turn_action']}"})

            if 'dir_image' in target:
                history_content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self.img_to_base64(target['dir_image'])}"}})
                history_content.append({"type": "text", "text": f"After turn view"})

            if 'description' in target:
                history_content.append({"type": "text", "text": f"Navigate to: {target['description']}"})

        # Current 4 views:
        current_views = []
        view_definitions = [
            {'angle': 0, 'name': 'forward', 'label': 'Current FORWARD view'},
            {'angle': 90, 'name': 'left', 'label': 'View after turning LEFT'},
            {'angle': 180, 'name': 'behind', 'label': 'View after turning BEHIND'},
            {'angle': 270, 'name': 'right', 'label': 'View after turning RIGHT'}
        ]

        for view in view_definitions:
            angle = view['angle']
            frame_idx = angle // 90
            if frame_idx < len(panorama_images):
                rgb_image = panorama_images[frame_idx]['rgb']
                if isinstance(rgb_image, np.ndarray):
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    img = Image.fromarray(rgb_image)
                else:
                    img = rgb_image
                current_views.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.img_to_base64(img)}"}})
                current_views.append({"type": "text", "text": view['label']})

        # Backtrack check
        num_waypoints = len([t for t in visited_targets[:-1] if 'description' in t])
        should_consider_backtrack = 1

        # Prompt construction based on whether backtrack or not
        content = [{"type": "text", "text": f"Navigation Task: \"{instruction}\"\n\nNavigation History:"}]
        content.extend(history_content)
        content.append({"type": "text", "text": "\nCurrent 4-directional views:"})
        content.extend(current_views)

        if should_consider_backtrack and num_waypoints > 0:
            waypoint_list = ""
            for i, target in enumerate(visited_targets[:-1]):
                if 'description' in target:
                    waypoint_list += f"  - Waypoint {i}: {target['description']}\n"

            prompt = LA_PROMPT_BACKTRACK.format(waypoint_list=waypoint_list)
        else:
            prompt = LA_PROMPT_NO_BACKTRACK

        logger.info(prompt)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        output_text = self.model.generate(
            messages=messages,
            max_new_tokens=8192,
            temperature=0.7,
            use_la=True
        )

        logger.info('LA-response:')
        logger.info(f"{output_text}")
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

        # Json parse
        while not json_match:
            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=8192,
                temperature=0.7,
                use_la=True
            )
            logger.info('Retried.')
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
            except:
                response_data = {}
            action = response_data.get('action', 'navigate to forward') or ''
            progress_analysis = response_data.get('progress_analysis', '')
            reasoning = response_data.get('reasoning', '')
            action = action.lower()
            if action.startswith('backtrack to'):
                waypoint_id = action.split('backtrack to ')[-1].strip()
                if waypoint_id.startswith('waypoint'):
                    waypoint_id = waypoint_id.split('waypoint')[-1].strip()
                # logger.info('Waypoint:%s', waypoint_id)
                try:
                    waypoint_id = int(waypoint_id)
                    return {
                        'action': 'BACKTRACK',
                        'waypoint': waypoint_id,
                        'progress_analysis': progress_analysis,
                        'reasoning': reasoning
                    }
                except:
                    pass

            if 'forward' in action:
                direction = 'forward'
            elif 'left' in action:
                direction = 'left'
            elif 'right' in action:
                direction = 'right'
            elif 'behind' in action:
                direction = 'behind'
            else:
                direction = 'forward'

            return {
                'action': 'NAVIGATE',
                'direction': direction,
                'progress_analysis': progress_analysis,
                'reasoning': reasoning
            }

        return {
            'action': 'NAVIGATE',
            'direction': 'forward',
            'progress_analysis': 'Unable to analyze due to parsing error',
            'reasoning': 'Fallback to forward navigation'
        }

    def query_llm(self,
                  instruction: str,
                  visited_targets: List[Dict[str, str]],
                  rgb_image: np.ndarray,
                  width: int,
                  height: int,
                  current_step: int,
                  progress_analysis: str = None):
        try:
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                img = Image.fromarray(rgb_image)
            else:
                img = rgb_image

            # Build visited targets history string
            visited_targets_str = ""
            # visited_targets
            if len(visited_targets) > 1:
                visited_targets_str = f"\n\nPreviously visited targets:\n"
                for i, target in enumerate(visited_targets[:-1], 0):
                    visited_targets_str += f"{i}. {target['description']} (Step {target['step']})\n"
            else:
                visited_targets_str = "\n\nNo targets visited yet."

            content = [{
                'type': 'text',
                'text': f'You are a robot performing navigation task. Look at this image and help the robot navigate.'
            }]

            img_base64 = self.img_to_base64(img)

            content.append({"type": "image_url", 'image_url': {'url': f"data:image/png;base64,{img_base64}"}})

            progress_info = ""
            if progress_analysis:
                progress_info = f"\nProgress Analysis from Navigation Decision: {progress_analysis}\n"

            prompt = VA_PROMPT.format(instruction=instruction, current_step=current_step,
                                      width=width, height=height,
                                      visited_targets_str=visited_targets_str,
                                      progress_info=progress_info)
            logger.info(prompt)
            content.append({
                "type": "text",
                "text": prompt
            })

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            output_text = self.model.generate(
                messages=messages,
                max_new_tokens=4096,
                temperature=0
            )
            logger.info('LLM Output:')
            logger.info("%s", output_text)

            # Try to parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            while not json_match:
                output_text = self.model.generate(
                    messages=messages,
                    max_new_tokens=4096,
                    temperature=0
                )
                logger.info('Retried LLM Output:')
                logger.info("%s", output_text)
                json_match = re.search(r'\{.*\}', output_text, re.DOTALL)

            if json_match:
                try:
                    response_data = json.loads(json_match.group())
                except:
                    response_data = {}

                # Extract action decision
                action_decision = response_data.get('action', 'NAVIGATE').upper()

                # Extract bbox_2d in [x1, y1, x2, y2] format
                bbox_2d = response_data.get('bbox_2d', [width // 4, height // 4, 3 * width // 4,
                                                        3 * height // 4])

                # Ensure we have 4 coordinates
                if len(bbox_2d) >= 4:
                    x1, y1, x2, y2 = bbox_2d[:4]
                else:
                    # Fallback if bbox_2d is malformed
                    x1, y1, x2, y2 = width // 4, height // 4, 3 * width // 4, 3 * height // 4

                # Convert to x, y, width, height format for internal use
                x = int(x1)
                y = int(y1)
                img_w, img_h = width, height
                width = int(x2 - x1)
                height = int(y2 - y1)

                bbox = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'target': response_data.get('target', 'unknown target'),
                    'action': action_decision,
                    'reasoning': response_data.get('reasoning', 'No reasoning provided'),
                    'progress': response_data.get('progress', 'Unknown progress')
                }

                # Ensure bbox is within image bounds
                bbox['x1'] = max(0, min(bbox['x1'], img_w - 1))
                bbox['y1'] = max(0, min(bbox['y1'], img_h - 1))
                bbox['x2'] = max(bbox['x1'] + 1, min(bbox['x2'], img_w))
                bbox['y2'] = max(bbox['y1'] + 1, min(bbox['y2'], img_h))

                # Update x, y, width, height based on bounded coordinates
                bbox['x'] = bbox['x1']
                bbox['y'] = bbox['y1']
                bbox['width'] = bbox['x2'] - bbox['x1']
                bbox['height'] = bbox['y2'] - bbox['y1']

                # logger.info("Parsed response:", bbox)
                # logger.info(f"BBox 2D format - x1:{bbox['x1']}, y1:{bbox['y1']}, x2:{bbox['x2']}, y2:{bbox['y2']}")
                # logger.info(f"Action: {action_decision}, Target: {bbox['target']}")
                # logger.info(f"Reasoning: {bbox['reasoning']}")
                # logger.info(f"Progress: {bbox['progress']}")

                # Record this target if it's a new navigation target
                if action_decision == 'NAVIGATE' and bbox['target'] != 'unknown target':
                    target_record = {
                        'step': current_step,
                        'description': bbox['target'],
                        'bbox': {
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'x1': bbox['x1'],
                            'y1': bbox['y1'],
                            'x2': bbox['x2'],
                            'y2': bbox['y2']
                        },
                        'reasoning': bbox['reasoning']
                    }

                    if visited_targets is not None:
                        visited_targets[-1].update(target_record)
                        # logger.info(f"Added target to history: {bbox['target']}")
                        # logger.info(visited_targets[-1])

                # Save RGB image with bounding box annotation
                self.visualizer._save_rgb_with_bbox(rgb_image, bbox)

                return bbox
            else:
                logger.info("Failed to parse JSON response, using default")

        except Exception as e:
            logger.info(f"Error in query_llm: {e}")

        # Fallback bbox (center of image)
        x1_fallback = width // 4
        y1_fallback = height // 4
        x2_fallback = 3 * width // 4
        y2_fallback = 3 * height // 4

        fallback_bbox = {
            'x': x1_fallback,
            'y': y1_fallback,
            'width': x2_fallback - x1_fallback,
            'height': y2_fallback - y1_fallback,
            'x1': x1_fallback,
            'y1': y1_fallback,
            'x2': x2_fallback,
            'y2': y2_fallback,
            'target': 'fallback target',
            'action': 'NAVIGATE',
            'reasoning': 'Fallback due to parsing error',
            'progress': 'Unknown due to error'
        }

        self.visualizer._save_rgb_with_bbox(rgb_image, fallback_bbox)

        return fallback_bbox

    def reset(self):
        # self.model.reset_stats()
        pass



@baseline_registry.register_trainer(name="lavira")
class LaViRA(BaseTrainer):
    def __init__(self, config, r2r) -> None:
        super().__init__()
        self.backtrack_steps = 0
        self.r2r = r2r
        self.device = get_device(config.TORCH_GPU_ID)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.config = config
        self.map_args = config.MAP
        self.resolution = config.MAP.MAP_RESOLUTION
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)

        self.trans = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH),
                                             interpolation=Image.NEAREST)
                                         ])

        self.classes = []
        self.current_episode_id = None
        self.current_detections = None
        self.map_channels = map_channels
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)


        self.visualize = True
        self.save_dir = getattr(config, 'RGB_SAVE_DIR', './saved_rgb_images')
        self.visualizer = LaViRAVisualizer(None, self.visualize, self.save_dir, self.width, self.height)

        self.visited_targets = []  # List of targets the agent has identified/visited
        self.current_step = 0  # Track current step for navigation decisions

        # Distance thresholds for target management (in map units)
        self.target_reached_threshold = getattr(config, 'TARGET_REACHED_THRESHOLD', 15.0)
        self.agent = VLMReasoningAgent(self.visualizer)

    def _set_eval_config(self) -> None:
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.config.freeze()

    def _init_envs(self) -> None:
        # logger.info("start to initialize environments")

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        logger.info(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()
        # logger.info("initializing environments finished!")

    def _collect_val_traj(self) -> None:
        if not self.r2r:
            role = self.config.TASK_CONFIG.DATASET.ROLES
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        if self.r2r:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
                gt_data = json.load(f)
        else:
            with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role=role[0])) as f:
                gt_data = json.load(f)

        self.gt_data = gt_data

    def _calculate_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
        pred_path = np.array(info['position']['position'])
        distances = np.array(info['position']['distance'])
        gt_length = distances[0]
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]
        metric['success'] = 1. if distances[-1] <= 3. else 0.
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        logger.info(f'ep{ep_id}:{self.state_eps[ep_id]}')

    def _initialize_policy(self) -> None:
        # logger.info("start to initialize policy")
        self.segment_module = GroundedSAM(self.config, self.device)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        self.visualizer.update_map(self.mapping_module)

        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()

    def _concat_obs(self, obs: Observations) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (h, w, c)->(c, h, w)

        return state

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)  # [3, h, w]
        rgb = rgb[:, :, ::-1]  # RGB to BGR
        depth = state[:, :, 3:4]  # [1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH

        sem_seg_pred = self._get_sem_pred(rgb)  # [num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth)  # [1, h, w]

        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH  # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8)))  # resize
            depth = depth[ds // 2::ds, ds // 2::ds]  # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)  # (4+num_detected_classes, h, w)

        return state

    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        cls2 = self.classes.copy()
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=cls2)
        if self.visualize:
            cv2.imwrite(f'saved_rgb_images/{self.current_episode_id}/step{self.current_step}_mask.png',
                        annotated_images)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        # logger.info("current step detected classes (before filtering): ", labels)

        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)

        return masks.transpose(1, 2, 0)

    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)

        return class_names

    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        if masks.shape[0] > 0:  # Check if there are any masks
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx)  # dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)

            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]

            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))

        return final_masks

    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        # logger.info('max:',depth.max())
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99  # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 1.0  # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0

        return depth

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)

        return state  # state.shape=(c,h,w)

    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                                 [(0, max_channels - state.shape[0]),
                                  (0, 0),
                                  (0, 0)],
                                 mode='constant')
                          for state in n_states], axis=0)

        return torch.from_numpy(batch).to(self.device)

    def _process_classes(self, base_class: List, target_class: List) -> List:
        for item in target_class:
            if item in base_class:
                base_class.remove(item)
        base_class.extend(target_class)

        return base_class


    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        # logger.info(f'{navigable_index}, {not_navigable_index}')
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)

        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)

        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=disk(kernel_size))

        return floor

    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int = 3) -> tuple:
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)

        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)

        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)

        selem = disk(3)
        obstacles_closed = binary_closing(obstacles, footprint=selem)
        objects_closed = binary_closing(objects, footprint=selem)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        # stairs should remain navigable even if overlapped with objects
        # navigable = np.logical_or(navigable, stairs_mask)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=selem)

        untraversable = np.logical_or(objects_closed, obstacles_closed)
        # ensure stairs override untraversable
        untraversable[navigable_closed == 1] = 0
        # untraversable[stairs_mask == 1] = 0
        untraversable = remove_small_objects(untraversable, min_size=64)
        untraversable = binary_closing(untraversable, footprint=disk(3))
        traversable = np.logical_not(untraversable)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        # free_mask = np.logical_or(free_mask, stairs_mask)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=selem)
        traversable = np.logical_or(floor, traversable)

        explored_area = binary_closing(explored_area, footprint=selem)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversable, floor, frontiers.astype(np.uint8)

    def _maps_initialization(self):
        obs = self.envs.reset()  # type(obs): list
        self.instruction = obs[0]['instruction']['text']
        self.destination = "goal"
        self.classes = self.base_classes.copy()
        self.current_episode_id = self.envs.current_episodes()[0].episode_id

        self.visualizer._save_rgb_frame(obs[0], 0, None, self.current_episode_id)

        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses, self.current_step)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)

    def _look_around(self):
        full_pose, obs, dones, infos = None, None, None, None
        for step in range(0, 12):
            self._action = HabitatSimActions.TURN_LEFT
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            self.current_step = step
            if dones[0]:
                return full_pose, obs, dones, infos

            # Save RGB frame during look around phase
            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id)

            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

        return full_pose, obs, dones, infos


    def reset(self) -> None:
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversable = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)

        # Reset target tracking
        self.visited_targets = []
        self.current_step = 0
        self.backtrack_steps = 0

        self.policy.reset()
        self.mapping_module.reset()
        self.agent.reset()

    def _get_camera_intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix for depth projection"""
        hfov = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        width = self.width
        height = self.height
        vfov = 2 * np.arctan(height / width * np.tan(hfov / 2))

        fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
        fy = height / (2 * np.tan(np.deg2rad(vfov / 2)))
        cx = width / 2
        cy = height / 2

        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        return intrinsics

    def get_panorama(self, obs: Observations, step: int):
        """
        Turn around(12 turns) to get panorama
        """

        panorama_frames = []

        for turn_step in range(1, 12 + 1):
            turn_action = [{"action": HabitatSimActions.TURN_LEFT}]  # 30 deg
            turn_outputs = self.envs.step(turn_action)
            turn_obs, _, turn_dones, turn_infos = [list(x) for x in zip(*turn_outputs)]

            if turn_dones[0]:
                # logger.info("Episode ended during panorama collection")
                # Return signal that episode is done - caller should handle this
                return {'turn_direction': 'episode_done', 'episode_finished': True}

            panorama_frames.append({
                'rgb': turn_obs[0]['rgb'].copy(),
                'angle': turn_step * 30 % 360,
                'step': turn_step
            })

            # Update map
            batch_obs = self._batch_obs(turn_obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in turn_obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses, self.current_step)
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step + turn_step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
        panorama_frames = [panorama_frames[-1]] + panorama_frames[:-1]
        # logger.info([x['angle'] for x in panorama_frames])
        # logger.info(f"Collected {len(panorama_frames)} panorama frames")

        return panorama_frames[::3]

    def rollout(self):
        """
        Execute a whole episode using bounding box target navigation
        """
        self._maps_initialization()
        look_around_results = self._look_around()
        if look_around_results[1] is None:
            logger.info("Episode finished during look_around. Exiting rollout.")
            if look_around_results[3]:  # infos
                self._calculate_metric(look_around_results[3])
            return

        full_pose, obs, dones, infos = look_around_results

        # logger.info('Sensor pose', obs[0]['sensor_pose'])

        # --- Initialize ---
        action_list = []
        going_to_stop = False
        panorama_got = False
        navigate_or_not = False
        collided = 0
        search_destination = False
        current_pose = full_pose[0] if full_pose is not None else None

        target_map_x, target_map_y = None, None

        max_steps_to_target = 30  # Renavigate after 30 steps
        target_set_step = None  # Record the steps after target set

        # Initial map status
        full_map = self.mapping_module.get_full_map()

        for step in range(12, self.max_step):
            # import sys
            # sys.stdout.flush()
            # logger.info(action_list, panorama_got)
            # =================================================================
            # 1. (ANALYZE STATE for step N)
            #
            # =================================================================
            if dones[0]:
                self._calculate_metric(infos)
                return
            self.visualizer.instruction = self.instruction
            self.visualizer.destination = self.destination
            self.visualizer._action = self._action

            logger.info(f"\nepisode:{self.current_episode_id}, step:{step}")

            # logger.info(f"instr: {self.instruction}")
            # logger.info(f"Targets visited: {len(self.visited_targets)}")

            last_pose = current_pose
            current_pose = full_pose[0]
            self.current_step = step
            self.visualizer.sync(step, self.current_episode_id)

            position = current_pose[:2] * 100 / self.resolution
            agent_map_x, agent_map_y = int(position[0]), int(position[1])
            # logger.info("full pose: ", current_pose)

            self.visualizer._save_rgb_frame(obs[0], step, self.visited_targets, self.current_episode_id, (target_map_x, target_map_y))

            # =================================================================
            # 2. PLAN/DECIDE for step N
            #    Four steps：
            #    Step 1: When there's no target, turn around to get panorama
            #    Step 2: After getting the panorama, navigate_or_backtrack & query_llm to get target
            #    Step 3: Navigate to the target
            #    Step 4: If timeout or arrival, go back to step1
            # =================================================================

            if not action_list:
                # Timeout
                if target_map_x is not None and target_map_y is not None and target_set_step is not None:
                    steps_since_target_set = step - target_set_step
                    if steps_since_target_set >= max_steps_to_target:
                        # logger.info(f"Target timeout: {steps_since_target_set} steps since target set, exceeding {max_steps_to_target} limit.")

                        if len(self.visited_targets) > 0:
                            self.visited_targets.pop()

                        # Step 4: Reset
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        # logger.info("Reset navigation state due to timeout.")

                # Check if arrival
                if target_map_x is not None and target_map_y is not None:
                    distance_to_target = np.sqrt((target_map_x - agent_map_x) ** 2 + (target_map_y - agent_map_y) ** 2)
                    # logger.info(f"Agent: ({agent_map_x}, {agent_map_y}), Target: ({target_map_x}, {target_map_y})")
                    # logger.info(f"Distance to target: {distance_to_target:.2f} (threshold: {self.target_reached_threshold})")
                    if distance_to_target < self.target_reached_threshold:
                        # logger.info(f"Target reached! Distance: {distance_to_target:.2f}")

                        # Check arrival image
                        if len(self.visited_targets) > 0:
                            dist_calc = lambda target: np.sqrt(
                                (target['world_coords'][0] - self.visited_targets[-1]['world_coords'][0]) ** 2 + (
                                            target['world_coords'][1] - self.visited_targets[-1]['world_coords'][
                                        1]) ** 2) if 'world_coords' in target else float('inf')
                            for target in self.visited_targets[:-1]:
                                if dist_calc(target) < self.target_reached_threshold:
                                    # logger.info('Removed duplicate waypoint due to proximity. Distance: %f', dist_calc(target))
                                    self.visited_targets.pop()
                                    break

                        # Step 4: Reset status
                        panorama_got = False
                        navigate_or_not = False
                        target_map_x, target_map_y = None, None
                        target_set_step = None
                        # logger.info("Reset navigation state - target reached.")

                # Step 1: No target and haven't got the panorama -> turn around first
                if target_map_x is None and not panorama_got and going_to_stop:
                    # logger.info('Final stop.')
                    action_list.append(0)  # STOP action
                elif target_map_x is None and not navigate_or_not:
                    # logger.info("Step 1: Getting panorama and deciding navigation direction...")
                    current_rgb = obs[0]['rgb'].copy()

                    panorama_frames = self.get_panorama(obs[0], step)
                    if 'episode_finished' in panorama_frames:
                        break

                    # New waypoint record
                    waypoint_id = len(self.visited_targets)
                    self.visited_targets.append({
                        'step': step,
                        'init_image': Image.fromarray(current_rgb) if isinstance(current_rgb,
                                                                                 np.ndarray) else current_rgb,
                        'panorama_frames': panorama_frames,
                        'world_coords': (agent_map_x, agent_map_y)
                    })


                    self.visualizer._save_waypoint_panorama_rgb(panorama_frames, waypoint_id, step)

                    decision = self.agent.navigate_or_backtrack(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets
                    )
                    # logger.info(f"Navigation decision: {decision}")

                    if decision.get('action', 'NAVIGATE') == 'BACKTRACK':
                        target_waypoint_id = decision.get('waypoint', 0)
                        if isinstance(target_waypoint_id, int) and target_waypoint_id < len(self.visited_targets) - 1:
                            target_map_x, target_map_y = self.visited_targets[target_waypoint_id]['world_coords']
                            self.visited_targets.pop()  # remove unfinished waypoint
                            # (f"Backtracking to waypoint {target_waypoint_id} at ({target_map_x}, {target_map_y})")
                            self.visited_targets = self.visited_targets[:target_waypoint_id + 1]
                        else:
                            # logger.info("Invalid waypoint ID for backtrack, continuing with navigation")
                            decision['action'] = 'NAVIGATE'
                        panorama_got = True
                    if decision.get('action', 'NAVIGATE') == 'NAVIGATE':
                        navigate_or_not = True
                        direction = decision.get('direction', 'forward')
                        progress_analysis = decision.get('progress_analysis', '')
                        reasoning = decision.get('reasoning', '')

                        # Save decision
                        self.visited_targets[-1].update({
                            'progress_analysis': progress_analysis,
                            'reasoning': reasoning,
                            'direction_decision': direction
                        })

                        # Get corresponding image
                        direction_map = {'forward': 0, 'left': 90, 'behind': 180, 'right': 270}
                        target_angle = direction_map.get(direction, 0)
                        frame_idx = target_angle // 90

                        if frame_idx < len(panorama_frames):
                            dir_rgb = panorama_frames[frame_idx]['rgb']
                            self.visited_targets[-1]['dir_image'] = Image.fromarray(dir_rgb) if isinstance(dir_rgb,
                                                                                                           np.ndarray) else dir_rgb
                            self.visited_targets[-1]['turn_action'] = f"turn {direction}"

                        # Add action according to LA
                        if direction == 'left':
                            action_list.extend([2] * 3)  # left 3*30
                        elif direction == 'right':
                            action_list.extend([3] * 3)  # right 3*30
                        elif direction == 'behind':
                            action_list.extend([2] * 6)  # left 6*30

                        panorama_got = True
                        # logger.info(f"Step 1 completed: Direction decision = {direction}, added turn actions")

                # Step 2: After getting to the right direction, query_llm to get target position
                elif target_map_x is None and panorama_got and not action_list:
                    # logger.info("Step 2: Querying LLM for specific target...")

                    progress_analysis = self.visited_targets[-1].get('progress_analysis', '')

                    bbox = self.agent.query_llm(
                        instruction=self.instruction,
                        visited_targets=self.visited_targets,
                        rgb_image=obs[0]['rgb'],
                        width=self.width,
                        height=self.height,
                        current_step=self.current_step,
                        progress_analysis=progress_analysis,
                    )
                    # logger.info(f"LLM response: {bbox}")

                    self.visited_targets[-1].update({
                        'description': bbox.get('target', 'unknown target'),
                        'bbox': bbox,
                        'llm_reasoning': bbox.get('reasoning', ''),
                        'llm_progress': bbox.get('progress', '')
                    })

                    if bbox.get('action', 'NAVIGATE') == 'STOP':
                        # logger.info("LLM decided STOP - going last")
                        going_to_stop = True

                    depth_image = self._preprocess_depth(obs[0]['depth'], 0.1, 5.0) / 100.0
                    coords = (int((bbox.get('x1', 0) + bbox.get('x2', 0)) / 2.0), int(bbox.get('y2', 0)))

                    # Convert pixel to map position
                    # Note that we don't want the target position untraversible, thus we reduce depth and make the target closer to the agent if so
                    while True:
                        target = get_world_xz_from_pixel(
                            pixel_coords=coords,
                            depth_image=depth_image,
                            full_pose=current_pose,
                            camera_intrinsics=self._get_camera_intrinsics(),
                        )
                        new_target_x = int(target[0] * 100.0 / self.resolution)
                        new_target_y = int(target[1] * 100.0 / self.resolution)
                        new_target_x = max(0, min(new_target_x, self.map_shape[0] - 1))
                        new_target_y = max(0, min(new_target_y, self.map_shape[1] - 1))

                        if self.traversable[new_target_y, new_target_x] == 1 or depth_image.max() < 0.1:
                            target_map_x, target_map_y = new_target_x, new_target_y
                            target_set_step = step
                            # logger.info(f"Target set at map coordinates: ({target_map_x}, {target_map_y}) at step {step}")

                            waypoint = np.array([target_map_y, target_map_x])
                            navigation_action = self.policy._get_action(
                                current_pose, waypoint, full_map[0], self.traversable,
                                self.collision_map, step, self.current_episode_id,
                                self.detected_classes, search_destination
                            )
                            action_list.append(navigation_action)
                            # logger.info(f"Added initial navigation action: {navigation_action}")
                            break
                        depth_image = depth_image - 0.1

                    panorama_got = False  # Reset
                    # logger.info("Step 2 completed: Target acquired from LLM")

                # Step 3: Have target -> navigate
                elif target_map_x is not None and target_map_y is not None and not action_list:
                    # logger.info(f"Step 3: Continuing navigation to target ({target_map_x}, {target_map_y})")
                    waypoint = np.array([target_map_y, target_map_x])
                    navigation_action = self.policy._get_action(
                        current_pose, waypoint, full_map[0], self.traversable,
                        self.collision_map, step, self.current_episode_id,
                        self.detected_classes, search_destination
                    )
                    action_list.append(navigation_action)
                    # logger.info(f"Added navigation action: {navigation_action}")


            if action_list:
                # =================================================================
                # 3. ACT for step N
                #
                # =================================================================
                self._action = action_list[0]
                action_list.pop(0)
                actions = [{"action": self._action}]

                # logger.info(f'Action actually performed: {self._action}')

                outputs = self.envs.step(actions)

                # =================================================================
                # 4. UPDATE for step N+1
                #
                # =================================================================
                obs, _, dones, infos = [list(x) for x in zip(*outputs)]
                # logger.info('Sensor pose', obs[0]['sensor_pose'])

                if not dones[0]:
                    batch_obs = self._batch_obs(obs)
                    poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
                    self.mapping_module(batch_obs, poses, self.current_step)
                    full_map, full_pose, one_step_full_map = \
                        self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
                    self.mapping_module.one_step_full_map.fill_(0.)
                    self.mapping_module.one_step_local_map.fill_(0.)

                    self.traversable, self.floor, self.frontiers = self._process_map(step, full_map[0])
                    self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])

                    last_pose = current_pose
                    current_pose = full_pose[0]
                    if last_pose is not None and current_pose is not None:
                        displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                        if displacement < 0.2 * 100 / self.resolution:
                            collided += 1
                        else:
                            collided = 0
                            replan = False
                        if collided >= 30:
                            fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                                 f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt")
                            with open(fname, "a") as f:
                                f.writelines(
                                    f"id: {str(self.current_episode_id)}; step: {str(step)}; collided: {str(collided)}\n")

                    current_action = self._action
                    if last_pose is not None and current_action is not None and current_action == 1:
                        collision_map = collision_check_fmm(last_pose, current_pose, self.resolution,
                                                            self.mapping_module.map_shape)
                        self.collision_map = np.logical_or(self.collision_map, collision_map)
                    self.traversable[self.collision_map == 1] = 0
                else:
                    self._calculate_metric(infos)
                    return
            else:
                pass
        self._calculate_metric(infos)

    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        self.agent.reset()


        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))

        self.state_eps = {}
        t1 = time.time()
        for i in tqdm(range(eps_to_eval)):
            self.rollout()
            self.reset()

        self.envs.close()

        logger.info("=== FINAL MODEL USAGE STATISTICS ===")
        final_stats = self.agent.model.print_usage_stats()

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        stats_fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR,
                                   f"model_usage_stats_{split}_r{self.local_rank}_w{self.world_size}.json")
        with open(stats_fname, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Model usage statistics saved to: {stats_fname}")

        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        logger.info("test time: %d", t2 - t1)


def merge_model_usage_stats(stats_dir, split="val_unseen"):
    import glob
    import json

    pattern = os.path.join(stats_dir, f"model_usage_stats_{split}_r*_w*.json")
    stat_files = glob.glob(pattern)

    if not stat_files:
        print(f"No model usage stat files found in {stats_dir} with pattern {pattern}")
        return

    merged_stats = {
        'la': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'va': {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'total_calls': 0,
        'total_tokens': 0,
        'num_processes': 0,
        'process_stats': []
    }

    for stat_file in stat_files:
        try:
            with open(stat_file, 'r') as f:
                stats = json.load(f)

            merged_stats['la']['calls'] += stats['la']['calls']
            merged_stats['la']['input_tokens'] += stats['la']['input_tokens']
            merged_stats['la']['output_tokens'] += stats['la']['output_tokens']
            merged_stats['la']['total_tokens'] += stats['la']['total_tokens']

            merged_stats['va']['calls'] += stats['va']['calls']
            merged_stats['va']['input_tokens'] += stats['va']['input_tokens']
            merged_stats['va']['output_tokens'] += stats['va']['output_tokens']
            merged_stats['va']['total_tokens'] += stats['va']['total_tokens']

            merged_stats['total_calls'] += stats['total_calls']
            merged_stats['total_tokens'] += stats['total_tokens']
            merged_stats['num_processes'] += 1

            process_info = {
                'file': os.path.basename(stat_file),
                'stats': stats
            }
            merged_stats['process_stats'].append(process_info)

            print(f"Loaded stats from: {stat_file}")

        except Exception as e:
            print(f"Error loading {stat_file}: {e}")

    merged_file = os.path.join(stats_dir, f"merged_model_usage_stats_{split}.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_stats, f, indent=2)

    print("=== MERGED MODEL USAGE STATISTICS ===")
    print(f"Number of processes: {merged_stats['num_processes']}")
    print(f"Language Action Model:")
    print(f"  - Total calls: {merged_stats['la']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['la']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['la']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['la']['total_tokens']:,}")
    print(f"Vision Action Model:")
    print(f"  - Total calls: {merged_stats['va']['calls']:,}")
    print(f"  - Total input tokens: {merged_stats['va']['input_tokens']:,}")
    print(f"  - Total output tokens: {merged_stats['va']['output_tokens']:,}")
    print(f"  - Total tokens: {merged_stats['va']['total_tokens']:,}")
    print(f"OVERALL TOTAL:")
    print(f"  - Total calls: {merged_stats['total_calls']:,}")
    print(f"  - Total tokens: {merged_stats['total_tokens']:,}")
    print(f"Merged statistics saved to: {merged_file}")
    print("=====================================")

    return merged_stats
