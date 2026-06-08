# LAVIRA Docker image
# Reverse-engineered from the lavira:v4 container that the authors run in production.
#
# Build:  docker build -t lavira:dev .
# Run:    docker run --gpus all -it --rm \
#             -v $(pwd):/workspace/lavira-code-oss \
#             -v /path/to/datasets:/workspace/lavira-code-oss/data \
#             --env-file .env \
#             lavira:dev
#
# Stack reproduced inside the container:
#   - Ubuntu 22.04, CUDA 11.8 + cuDNN 8.9 (devel base, needed for habitat-sim build)
#   - Conda env `lavira` on Python 3.8
#   - habitat-sim 0.1.7 built from source @ commit 856d4b08 (with --headless --with-cuda --bullet)
#   - habitat-lab @ commit d6ed1c0a (facebookresearch/habitat-lab)
#   - PyTorch 2.4.1 + cu118
#   - GroundingDINO @ commit 57535c5a (IDEA-Research)
#   - segment-anything @ commit dca509fe (facebookresearch)
#   - Remaining Python deps pinned in requirements.txt

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV NVIDIA_VISIBLE_DEVICES=all

# --- 1. System packages ---
# Build deps for habitat-sim 0.1.7 (cmake, assimp, boost, mesa EGL).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    pkg-config \
    unzip \
    ffmpeg \
    libboost-all-dev \
    libassimp-dev \
    libarmadillo-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglvnd-dev \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libopenexr-dev \
    libglm-dev \
    libbullet-dev \
    libfreetype6-dev \
    libfreeimage-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Miniconda + lavira env (Python 3.8) ---
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.10.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
# Newer conda (25+) requires `conda tos accept` for Anaconda channels; the
# pinned Miniconda3-py38_23.10.0-1 installer ships conda 23.10 which neither
# requires nor supports that subcommand, so we skip it.

RUN conda create -n lavira python=3.8 cmake=3.14.0 -y && \
    conda clean -a -y
ENV PATH=/opt/conda/envs/lavira/bin:$PATH
SHELL ["conda", "run", "--no-capture-output", "-n", "lavira", "/bin/bash", "-lc"]

# --- 3. PyTorch (cu118 wheels) ---
RUN pip install --no-cache-dir \
        torch==2.4.1+cu118 \
        torchvision==0.19.1+cu118 \
        torchaudio==2.4.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# GitHub clones over GnuTLS occasionally die with "TLS connection was non-properly
# terminated (-110)" on large repos. Workarounds: raise the post buffer + force
# HTTP/1.1 so we avoid HTTP/2 multiplexing that triggers the GnuTLS bug.
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.version HTTP/1.1

# --- 4. habitat-sim 0.1.7 (built from source, headless + bullet + cuda) ---
# Shallow-clone via the v0.1.7 tag to avoid pulling the full history (~500MB).
# Inline 5x retry handles intermittent github TLS drops.
WORKDIR /root
RUN for i in 1 2 3 4 5; do \
        git clone --depth 1 --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git && break; \
        echo "[retry $i] habitat-sim clone failed, retrying" >&2; \
        rm -rf habitat-sim; sleep 5; \
    done && \
    cd habitat-sim && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install --headless --with-cuda --bullet

# --- 5. habitat-lab @ pinned commit ---
# Pinned commit is on master (no tag) — use partial clone (no blobs upfront)
# so git only fetches metadata, then checkout lazy-fetches the needed blobs.
# Patch InstructionSensor: habitat-lab @ d6ed1c0a initialises observation_space
# with spaces.Discrete(0), which fails the assert n>0 in gym >=0.22. The actual
# instruction tensor is filled at sim step by the task, so a Dict() placeholder
# is the standard workaround used in the production lavira container.
WORKDIR /root
RUN for i in 1 2 3 4 5; do \
        git clone --filter=blob:none https://github.com/facebookresearch/habitat-lab.git && break; \
        echo "[retry $i] habitat-lab clone failed, retrying" >&2; \
        rm -rf habitat-lab; sleep 5; \
    done && \
    cd habitat-lab && \
    git checkout d6ed1c0a0e786f16f261de2beafe347f4186d0d8 && \
    sed -i 's|self.observation_space = spaces.Discrete(0)|self.observation_space = spaces.Dict()|' habitat/tasks/vln/vln.py && \
    pip install --no-cache-dir -e .

# --- 6. GroundingDINO @ pinned commit ---
# FORCE_CUDA=1 + CUDA_HOME + TORCH_CUDA_ARCH_LIST: build the MultiScaleDeformableAttention
# CUDA extension (_C) even though no GPU is visible at docker build time. Without
# these vars, setup.py falls back to a CPU-only build and runtime crashes with
# `NameError: name '_C' is not defined`. Arch list covers Ampere (sm_80, A100/A800)
# through Hopper (sm_90, H100); add older archs (7.5 for T4 / 8.6 for A6000) if
# you target those GPUs.
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
WORKDIR /workspace
RUN for i in 1 2 3 4 5; do \
        git clone --filter=blob:none https://github.com/IDEA-Research/GroundingDINO.git && break; \
        echo "[retry $i] GroundingDINO clone failed, retrying" >&2; \
        rm -rf GroundingDINO; sleep 5; \
    done && \
    cd GroundingDINO && \
    git checkout 57535c5a79791cb76e36fdb64975271354f10251 && \
    pip install --no-cache-dir -e .

# --- 7. Segment Anything @ pinned commit ---
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf

# --- 7b. Patch GroundingDINO to honor a BERT_LOCAL_PATH env var ---
# GroundingDINO's text encoder defaults to "bert-base-uncased", which transformers
# tries to fetch from HF Hub. Users who already have bert checkpoints locally
# (e.g. via data/grounded_sam/bert-base-uncased/) can point at them by setting
# BERT_LOCAL_PATH in the environment, avoiding the need to pre-warm HF cache.
RUN sed -i '1i import os' /workspace/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py && \
    sed -i 's|self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)|text_encoder_type = os.environ.get("BERT_LOCAL_PATH", text_encoder_type)\n        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)|' /workspace/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py && \
    sed -i 's|if text_encoder_type == "bert-base-uncased":|if text_encoder_type == "bert-base-uncased" or "bert" in os.path.basename(text_encoder_type).lower():|' /workspace/GroundingDINO/groundingdino/util/get_tokenlizer.py && \
    sed -i '1i import os' /workspace/GroundingDINO/groundingdino/util/get_tokenlizer.py

# --- 8. Remaining python deps ---
COPY requirements.txt /tmp/lavira-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lavira-requirements.txt

# --- 9. Register the NVIDIA EGL vendor with GLVND ---
# libEGL_nvidia.so.0 is injected at runtime by the NVIDIA Container Toolkit,
# but the ICD JSON that points GLVND at it is missing from the base CUDA image.
# Without it, Magnum's WindowlessEglApplication can't find an EGL device for
# the CUDA GPU and habitat-sim crashes at sim init with
#   "unable to find EGL device for CUDA device 0".
USER root
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Mount this repo at /workspace/lavira-code-oss when running.
WORKDIR /workspace/lavira-code-oss
CMD ["/bin/bash"]
