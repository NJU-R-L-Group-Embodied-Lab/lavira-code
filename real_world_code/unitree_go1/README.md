# Unitree Go1 — Real-World VLN Deployment for LaViRA

> **LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision-and-Language Navigation in Continuous Environments.** Ding et al., arXiv [2510.19655](https://arxiv.org/abs/2510.19655).

## Pipeline

Per navigation cycle:

1. Switch through the four head cameras to capture a 4-direction panorama.
2. **Language Action (LA)** call updates a markdown TODO list, picks `front` / `right` / `left` / `behind`, and may flag `stop`.
3. The base rotates 90° (or 180° for `behind`).
4. **Vision Action (VA)** call returns either `STOP` or a bbox of the next intermediate target.
5. The bbox centre is back-projected through the front-camera depth to a robot-frame goal; iPlanner produces a trajectory; a pure-pursuit loop drives it with parallel replanning, leaving a safety buffer.

## Layout

```
unitree_go1/
├── main.py                # Entry; --demo for web, otherwise headless VLN
├── config.py              # Env-var overridable configuration
├── prompts.py             # LA + VA prompt templates
├── utils.py               # Logging, JSON, base64 helpers
├── ai_client/
│   └── vision_client.py   # LaViRAVisionClient (two-endpoint LA + VA)
├── robot/
│   ├── robot_controller.py  # Unitree SDK + ROS + 4 cameras + iPlanner
│   ├── iplanner_client.py   # HTTP wrapper for the iPlanner server
│   ├── navigation_api.py    # LA + VA prompt orchestration
│   └── nav_controller.py    # IntegratedVisionNavController (per-cycle pipeline)
├── iplanner/                # iPlanner server + model code (kept as-is)
└── web/
    ├── app.py               # Flask + SocketIO interactive demo
    └── templates/index.html
```

## Install

```bash
conda create -n unitree_go1 python=3.10 && conda activate unitree_go1
pip install -r requirements.txt
```

You also need:
- ROS Noetic with the camera drivers publishing under `/cameraN/color|depth/image_raw` for `N = 1..4`.
- The Unitree High-level SDK (`robot_interface.so`). Set `UNITREE_SDK_PATH` to its directory, or put it on `PYTHONPATH`.
- An iPlanner server. Launch the one bundled under `iplanner/` and point `IPLANNER_URL` at it.

## Deployment profiles

| Profile | LA model | VA model |
|---------|----------|----------|
| Local (default) | `Qwen3.5-27B-Q4` | `Qwen3.5-27B-Q4` (same vLLM instance) |
| API | `gemini-3.1-pro` | `Qwen3.5-27B` |

Local profile — one vLLM server, defaults already point at it:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B-Q4 \
    --served-model-name Qwen3.5-27B-Q4 \
    --quantization awq --max-model-len 8192
```

API profile — set env vars per slot:

```bash
export LA_BASE_URL=...   LA_API_KEY=...   LA_MODEL_NAME=gemini-3.1-pro
export VA_BASE_URL=...   VA_API_KEY=...   VA_MODEL_NAME=Qwen3.5-27B
```

## Run

```bash
roscore
roslaunch your_go1_bringup.launch        # cameras + UDP bridge
python iplanner/iplanner_server.py       # iPlanner Flask server (port 8888)
```

Headless VLN:

```bash
python main.py --instruction "go to the chair in front of you"
# Omit --instruction to use config.DEFAULT_INSTRUCTION
```

Interactive web demo (typed or voice instruction):

```bash
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
huggingface-cli download Systran/faster-whisper-base --local-dir ./models/faster-whisper-base
python main.py --demo
# Open https://<robot-ip>:5000
```

## Environment variables

All knobs are env-var overridable; see `config.py` for defaults.

| Variable | Default |
|----------|---------|
| `LA_BASE_URL` / `LA_API_KEY` / `LA_MODEL_NAME` | `http://localhost:8000/v1` / `EMPTY` / `Qwen3.5-27B-Q4` |
| `VA_BASE_URL` / `VA_API_KEY` / `VA_MODEL_NAME` | same as LA |
| `IPLANNER_URL` | `http://localhost:8888` |
| `UNITREE_SDK_PATH` | `""` (empty: relies on `PYTHONPATH`) |
| `UNITREE_HOST` / `UNITREE_LOCAL_PORT` / `UNITREE_REMOTE_PORT` | `192.168.123.161` / `8080` / `8082` |
| `CAMERA_HEIGHT` / `CAMERA_ROLL_CORRECTION` | `0.3` / `0.0` |
| `COBOT_HTTP_HOST` / `COBOT_HTTP_PORT` | `0.0.0.0` / `5000` |
| `COBOT_SSL_CERT` / `COBOT_SSL_KEY` | `cert.pem` / `key.pem` |
| `COBOT_WHISPER_MODEL` | `./models/faster-whisper-base` |
| `COBOT_OUTPUT_DIR` | `outputs` |
| `FLASK_SECRET_KEY` | `change-me` (set this if exposing the demo) |

## Citation

```bibtex
@article{ding2025lavira,
  title   = {LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision-and-Language Navigation in Continuous Environments},
  author  = {Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal = {arXiv preprint arXiv:2510.19655},
  year    = {2025}
}
```
