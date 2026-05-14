"""
LaViRA prompt templates for the real-world (Cobot Magic / Aloha) stack.

Adapted from the simulation prompts in the LaViRA repo
(https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code/blob/main/vlnce_baselines/utils/prompts.py).
The real-world deployment uses 6- or 8-direction panoramic views instead of
the 4-direction simulator layout, so the LA prompt is parameterised.
"""


def get_la_prompt(num_directions, action_choices):
    """Language Action prompt: choose a panoramic direction to head toward."""
    image_options = "image1|image2|image3|image4|image5|image6"
    if num_directions == 7:
        image_options += "|image7"
    elif num_directions == 8:
        image_options += "|image7|image8"

    return f"""Based on the navigation history and current {num_directions}-directional views, decide which image direction to navigate toward:

Choose one of the {num_directions} images:
{action_choices}

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen image>",
    "action": "{image_options}"
}}

Guidelines:
- Consider instruction completion progress
- Choose the image that best advances toward goal
- Look for relevant objects and clear paths in the images
- Focus on the visual content to make the decision"""


def get_va_prompt(instruction, width, height, visited_targets_str, progress_info):
    """Vision Action prompt: detect the next target's bounding box or decide STOP."""
    return f"""Navigation Task: "{instruction}"

Current situation:
- Image size: {width}x{height} pixels{visited_targets_str}{progress_info}

Your task:
1. Analyze at what stage the current instruction has been completed and what should be done next
2. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
3. Decide if the robot should STOP (if task is completed or very close to final goal)

Response format (JSON):
{{
    "progress": "<assessment of how close to completing the instruction>",
    "reasoning": "<brief explanation of decision>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- If you see the final destination mentioned in instruction, consider STOP action
- If already very close to the goal object, choose STOP
- If still need to navigate, choose NAVIGATE and provide bounding box of next target
- Target description should be specific and clear
- Consider the instruction completion progress based on visited targets
- Use the progress analysis to inform your decision
- bbox_2d should be in format [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right"""
