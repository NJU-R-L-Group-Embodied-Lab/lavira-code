# Agilex Cobot Magic — Real-World Deployment for LaViRA

> **LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision-and-Language Navigation in Continuous Environments.** Ding et al., arXiv [2510.19655](https://arxiv.org/abs/2510.19655).

## Pipeline

Per navigation cycle:
1. Sweep the two arm-mounted cameras to capture a 7-direction panorama (the rear pose is mechanically unreachable, so the back view is skipped).
2. **Language Action (LA)** call picks the panoramic direction toward the goal.
3. Base rotates; **Vision Action (VA)** call returns a bbox of the next target or STOP.
4. Bbox centre is back-projected through depth intrinsics into a rotate-then-move command.

Arm reset and VLM inference run in parallel to shorten each cycle.

## Layout

```
cobot_magic/
├── main.py                # Entry point; --demo toggles voice/web vs. text-instruction
├── config.py              # Env-var overridable config
├── prompts.py             # LA / VA prompt templates
├── ai_client/
│   └── vision_client.py   # Two-endpoint OpenAI-compatible client
├── robot/
│   ├── arm_controller.py  # Dual-arm joint controller
│   ├── navigation_api.py  # LA + VA wrappers
│   └── nav_controller.py  # Per-cycle pipeline (single class, both modes)
└── web/
    ├── app.py             # Flask + SocketIO backend
    └── templates/index.html
```

## Install

```bash
conda create -n cobot_magic python=3.10 && conda activate cobot_magic
pip install -r requirements.txt
```

Requires Ubuntu + ROS Noetic with a working Cobot Magic bringup (dual arms, base, RealSense front camera, two arm-mounted cameras).

## Deployment profiles

| Profile | LA model | VA model |
|---------|----------|----------|
| Local (default) | `Qwen3.5-27B-Q4` | `Qwen3.5-27B-Q4` (same vLLM instance) |
| API | `gemini-3.1-pro` | `Qwen3.5-27B` |

**Local profile** — launch one vLLM server and the defaults work:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B-Q4 \
    --served-model-name Qwen3.5-27B-Q4 \
    --quantization awq --max-model-len 8192
```

**API profile** — set env vars:

```bash
export LA_BASE_URL=...   LA_API_KEY=...   LA_MODEL_NAME=gemini-3.1-pro
export VA_BASE_URL=...   VA_API_KEY=...   VA_MODEL_NAME=Qwen3.5-27B
```

Both slots accept any OpenAI-compatible endpoint, so mix-and-match is fine.

## Run

```bash
roscore
roslaunch your_cobot_magic_bringup.launch     # in another terminal
```

Text-instruction mode (headless):

```bash
python main.py --instruction "go to the chair in front of you"
```

Voice + web demo (requires `cert.pem`/`key.pem` and a whisper snapshot):

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
huggingface-cli download Systran/faster-whisper-base --local-dir ./models/faster-whisper-base
python main.py --demo
# Browse to https://<robot-ip>:5000
```

## Env vars

All knobs are env-var overridable; see `config.py` for defaults.

| Variable | Default |
|----------|---------|
| `LA_BASE_URL` / `LA_API_KEY` / `LA_MODEL_NAME` | `http://localhost:8000/v1` / `EMPTY` / `Qwen3.5-27B-Q4` |
| `VA_BASE_URL` / `VA_API_KEY` / `VA_MODEL_NAME` | same as LA |
| `COBOT_HTTP_HOST` / `COBOT_HTTP_PORT` | `0.0.0.0` / `5000` |
| `COBOT_SSL_CERT` / `COBOT_SSL_KEY` | `cert.pem` / `key.pem` |
| `COBOT_WHISPER_MODEL` | `./models/faster-whisper-base` |
| `FLASK_SECRET_KEY` | `change-me` (set this if exposing the demo) |

Numeric tuning (speeds, offsets, depth window, `NUM_DIRECTIONS`, default instruction) lives in `config.py`.

## Citation

```bibtex
@article{ding2025lavira,
  title   = {LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision-and-Language Navigation in Continuous Environments},
  author  = {Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal = {arXiv preprint arXiv:2510.19655},
  year    = {2025}
}
```
