"""
LaViRA prompt templates for the Unitree Go1 VLN task.

The contract names (JSON keys, ``NAVIGATE`` / ``STOP``, direction verbs) match
the LaViRA simulation repo so prompts can be A/B compared.
"""


def get_la_navigation_prompt(instruction, global_target, current_todo_list,
                             history_info, current_step=1):
    """LA prompt for the strategic-direction call."""
    allowed_dirs = '"front", "left", "right"'
    if current_step == 1:
        allowed_dirs = '"front", "left", "right", "behind"'

    return f"""
**ROLE**: You are an intelligent robot navigator using a dynamic checklist to guide your actions.
**MISSION**: "{instruction}"
**GLOBAL TARGET**: "{global_target}"

**Current TODO List**:
{current_todo_list}

**Task**:
1. **Update the TODO list**:
   - If the "Current TODO List" is empty or None, GENERATE a new checklist based on the MISSION and visual views.
   - If a list exists, check if the current step is completed.
   - If completed, mark it as [x] and append "Result: ...".
   - If the plan is stuck, add new items or modify steps.
2. **Decide the next action**:
   - Choose strictly from: {allowed_dirs}.

**HISTORY ANALYSIS**:
{history_info}
(Note: also refer to the 'Recent Visual History' images provided above.)

3. **Stop decision**:
   - Set "stop": true if you are sure you have reached the final goal.

**OUTPUT REQUIREMENT**:
- **DIRECTLY** output the JSON object.
- **DO NOT** output any analysis, reasoning, or markdown text before or after the JSON.
- **KEEP CONCISE**: limit `progress_analysis` and `reasoning` to one or two short sentences.

**JSON RESPONSE FORMAT**:
{{
    "progress_analysis": "Short assessment of current progress...",
    "reasoning": "Short explanation for chosen action...",
    "updated_todo_list": "The full updated Markdown checklist string (with [x] and [ ])",
    "turn_direction": "front" or "right" or "left" or "behind",
    "stop": true or false,
    "expected_landmark": "What to look for next"
}}
"""


def get_va_prompt(instruction, global_target, strategic_goal, strategic_stop):
    """VA prompt for the tactical bbox / NAVIGATE / STOP call."""
    return f"""
**ROLE**: You are a robot navigator's TACTICAL EYES.
**MISSION**: "{instruction}"
**GLOBAL TARGET**: "{global_target}"
**CURRENT STRATEGY**: "{strategic_goal}"
**STRATEGIC STOP SIGNAL**: {strategic_stop}

**INPUT**: You are looking at the CURRENT VIEW after turning.

**TASK**:
1. **Verification**: do you see the object/area mentioned in CURRENT STRATEGY?
2. **Targeting**: draw a bounding box (bbox_2d) around the best navigation target.
   - If the GLOBAL TARGET is visible, box it.
   - Otherwise, box the landmark mentioned in CURRENT STRATEGY.
3. **Action decision (NAVIGATE vs STOP)**:
   - **NAVIGATE**: if the target is far away or not centered.
   - **STOP**: only if the GLOBAL TARGET is clearly visible, centered, and occupies more than 20% of the image height.
   - **SPECIAL CASE**: if STRATEGIC STOP SIGNAL is true, verify we are at the goal. If yes, output STOP.

**OUTPUT REQUIREMENT**:
- **DIRECTLY** output the JSON object.
- **DO NOT** output any analysis or text before or after the JSON.
- Start immediately with `{{`.

**JSON FORMAT**:
{{
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "Name of the object in the bbox",
    "stop_reasoning": "Only fill this if stopping."
}}
"""
