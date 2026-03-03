
LA_PROMPT_BACKTRACK = """Based on the navigation history and current 4-directional views, decide the next action:

Available waypoints for backtracking:
{waypoint_list}

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward  
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen action>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>",
}}

Guidelines:
- Consider instruction completion progress
- Backtrack only if current path seems unproductive or dead-end
- Choose direction that best advances toward goal
"""

LA_PROMPT_NO_BACKTRACK = f"""Based on the navigation history and current 4-directional views, decide the next navigation direction:

Choose the best direction:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward
3. navigate to right - turn right and go forward  
4. navigate to behind - turn around and go forward

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen direction>",
    "action": "navigate to forward|left|right|behind",
}}

Guidelines:
- Consider instruction completion progress
- Choose direction that best advances toward goal
- Look for relevant objects and clear paths
"""


VA_PROMPT = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
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
- Use the progress analysis to inform your decision"""