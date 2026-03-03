

# [ICRA 2026] LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision-and-Language Navigation in Continuous Environments
## TODO List
- [x] Conda Environment Setup
- [x] Evaluation Code
    - [x] R2R-CE
    - [x] RxR-CE
- [ ] Docker Environment Setup
- [ ] Real-world Deployment Code（Aloha & Go1）

## Environment Setup

### 1. Create Conda Environment

The codebase has been tested with Python 3.9. Python 3.8 should also be compatible.

```shell
conda create -n lavira python=3.9
conda activate lavira
```

### 2. Install Habitat-Sim & Habitat-Lab

Habitat-Sim is compiled locally in this setup. Alternatively, a precompiled conda version is available.

**Habitat-Sim**
```shell
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout tags/v0.1.7
pip install -r requirements.txt
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" python setup.py install --headless
```

**Habitat-Lab**

> **Note:** The `tensorflow==1.13.1` dependency must be removed from `habitat_baselines/rl/requirements.txt` prior to installation to avoid dependency conflicts.

```shell
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/v0.1.7
cd habitat_baselines/rl
vi requirements.txt  # Remove the line: tensorflow==1.13.1
cd ../../            # Return to habitat-lab root directory

pip install torch==2.1.0+cu121 torchvision==0.16.0 torchaudio==2.1.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
# Installs both habitat and habitat_baselines.
# If installation fails due to network issues, please retry.
python setup.py develop --all
```

### 3. GroundingDINO

GroundingDINO is used to construct semantic maps for robot action planning.

```shell
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e . --no-build-isolation
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# Note: Upgrading setuptools and wheel may be required.
```

**Phrase-to-Class Mapping Optimization**

Following [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), we recommend replacing the default `phrases2classes` function in Grounded-SAM with a minimum edit distance-based implementation for more robust and stable prediction outputs.

```shell
pip install nltk
```

Locate the `phrases2classes` method at Line 235 of `<YOUR_PATH>/GroundingDINO/groundingdino/util/inference.py`. Comment out the original implementation and replace it with the following:

```python
# @staticmethod
# def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
#     class_ids = []
#     for phrase in phrases:
#         try:
#             class_ids.append(classes.index(phrase))
#         except ValueError:
#             class_ids.append(None)
#     return np.array(class_ids)

from nltk.metrics import edit_distance

@staticmethod
def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    class_ids = []
    for phrase in phrases:
        if phrase in classes:
            class_ids.append(classes.index(phrase))
        else:
            distances = np.array([edit_distance(phrase, c) for c in classes])
            class_ids.append(np.argmin(distances))
    return np.array(class_ids)
```

### 4. Other dependencies for LaViRA
Then install the remaining dependencies:

```shell
pip install setuptools==58.5.3 meson-python ninja
pip install -r requirements.txt --use-pep517 --no-build-isolation
```

---

## Dataset Preparation

Download the [Matterport3D](https://niessner.github.io/Matterport/) scene dataset and the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) episode dataset, and place them under the `data/` directory as described below.

**Matterport3D Scenes**

After obtaining the official download script (`download_mp.py`) by submitting the Terms of Use agreement at the [Matterport3D project page](https://niessner.github.io/Matterport/), run:

```shell
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

**VLN-CE Episodes**

Download `R2R_VLNCE_v1-3_preprocessed.zip` (~250 MB) using the following command:

```shell
gdown https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr
```

---

## Pretrained Weights

Download the required GroundedSAM model weights [here](https://drive.google.com/drive/folders/1RvB3z8wi19saplpFYw07NwTdgVkBbH2G) and place them under `data/grounded_sam/`. The expected directory structure for the `data/` folder is as follows:

```shell
data
├── grounded_sam
│   ├── groundingdino_swint_ogc.pth
│   ├── GroundingDINO_SwinT_OGC.py
│   ├── repvit_sam.pt
│   └── sam_vit_h_4b8939.pth
├── datasets
│   └── R2R_VLNCE_v1-3_preprocessed
│       ├── embeddings.json.gz
│       ├── envdrop
│       │   ├── envdrop_gt.json.gz
│       │   └── envdrop.json.gz
│       ├── joint_train_envdrop
│       │   ├── joint_train_envdrop_gt.json.gz
│       │   └── joint_train_envdrop.gz
│       ├── test
│       │   ├── test.json
│       │   └── test.json.gz
│       ├── train
│       │   ├── train_gt.json.gz
│       │   └── train.json.gz
│       ├── val_seen
│       │   ├── val_seen_gt.json.gz
│       │   └── val_seen.json.gz
│       └── val_unseen
│           ├── val_unseen_gt.json
│           ├── val_unseen_gt.json.gz
│           ├── val_unseen.json
│           └── val_unseen.json.gz
└── scene_datasets
    └── mp3d
        ├── 17DRP5sb8fy
        │   ├── 17DRP5sb8fy.glb
        │   ├── 17DRP5sb8fy.house
        │   ├── 17DRP5sb8fy.navmesh
        │   └── 17DRP5sb8fy_semantic.ply
        ├── 1LXtFkjw3qL
        │   ├── 1LXtFkjw3qL.glb
        │   ├── 1LXtFkjw3qL.house
        │   ├── 1LXtFkjw3qL.navmesh
        │   └── 1LXtFkjw3qL_semantic.ply
        └── ...
```

---

## Evaluation

### OpenNav_R2R-CE_100 download
Download from [here](https://drive.google.com/drive/folders/1RvB3z8wi19saplpFYw07NwTdgVkBbH2G?usp=sharing) to get `val_unseen.json.gz` and `val_unseen_gt.json.gz` and place them under `data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen`.


Set the following environment variables before running evaluation:

| Variable | Description |
|---|---|
| `LA_API_KEY` | API key for the Language Action Model |
| `LA_BASE_URL` | Base URL for the Language Action Model endpoint |
| `LA_MODEL_NAME` | Model name for the Language Action Model |
| `VA_API_KEY` | API key for the Vision Action Model |
| `VA_BASE_URL` | Base URL for the Vision Action Model endpoint |
| `VA_MODEL_NAME` | Model name for the Vision Action Model |

**Recommended Models:** We recommend using `gemini-2.5-pro` or `gpt-4o` as the Language Action Model, and `Qwen2.5-VL-32B-Instruct` as the Vision Action Model.

> **Note:** Some models (e.g., Qwen3-VL, Gemini) return bounding box coordinates in a relative format. If you wish to use a different Vision Action Model, the coordinate parsing logic may need to be adjusted accordingly.

Configure `CUDA_VISIBLE_DEVICES` and the number of parallel processes (`nprocess`) in `eval_scripts/r2r.sh`, then run:

```shell
bash eval_scripts/r2r.sh
```

On a server equipped with an NVIDIA RTX 4090 GPU using 20 parallel processes, evaluation on the OpenNav-100 episodes completes in approximately 30 minutes.

## Simple Visualization
Set in[`lavira_main.py` Line 463](https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code/blob/main/vlnce_baselines/lavira_main.py#L463)：

```python
self.visualize = True
```
After finishing evaluation, run 
```python
python server.py
```
The website will run on http://0.0.0.0:9999 by default.

## Credits
If you find this work useful, please cite our paper:

```bibtex
@article{ding2025lavira,
  title={LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments},
  author={Ding, Hongyu and Xu, Ziming and Fang, Yudong and Wu, You and Chen, Zixuan and Shi, Jieqi and Huo, Jing and Zhang, Yifan and Gao, Yang},
  journal={arXiv preprint arXiv:2510.19655},
  year={2025}
}
```

Our code is adapted from [CA-Nav](https://github.com/Chenkehan21/CA-Nav-code), thanks for their work!
