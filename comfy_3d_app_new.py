import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict
import shutil
import os

import modal
from fastapi import File, UploadFile, Form
import base64

import logging
import json
from pathlib import Path
import time
import requests
import urllib.request
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "build-essential",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libegl1-mesa-dev",
        "libgl1",
        "libglib2.0-0",
        "libgl1-mesa-dev",
        "libgl1-mesa-glx",
        "libgles2",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "libglvnd0",
        "libglx0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
        "python3.11",
        "python3.11-dev",
        "python3.11-venv",
        "wget"
    )
    .pip_install(
        "comfy-cli==1.0.33",
        "kiui")
    .env({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LD_LIBRARY_PATH": "/usr/lib64:$LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "PYOPENGL_PLATFORM": "egl"
    })
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    # Clone and setup ComfyUI
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /app",
        "cd /app && git reset --hard 29c2e26724d4982a3e33114eb9064f1a11f4f4ed",
        "cd /app && pip install -r requirements.txt"
    )
    # Setup ComfyUI-3D-Pack
    .run_commands(
        "mkdir -p /app/custom_nodes/ComfyUI-3D-Pack",
        "cd /app/custom_nodes/ComfyUI-3D-Pack",
        "pip install ninja rembg[gpu] open_clip_torch"
    )
    # # Install Essential nodes
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack && "
    #     "cd ComfyUI-Impact-Pack && "
    #     "git reset --hard ab17f8886945b0d36478950fe532164a8b569cc7 && "
    #     "pip install -r requirements.txt"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/kijai/ComfyUI-KJNodes && "
    #     "cd ComfyUI-KJNodes && "
    #     "git reset --hard ffafc9c2c675ce4e89386c725def331a88274004 && "
    #     "pip install -r requirements.txt"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && "
    #     "cd ComfyUI-VideoHelperSuite && "
    #     "git reset --hard 70faa9bcef65932ab72e7404d6373fb300013a2e && "
    #     "pip install -r requirements.txt"
    # )
    # # Install Extra nodes
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus && "
    #     "cd ComfyUI_IPAdapter_plus && "
    #     "git reset --hard 13fedc634d1abf19d289fc7a7b5a74589465206c"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive && "
    #     "cd ComfyUI_UltimateSDUpscale && "
    #     "git reset --hard 70083f5d449c498ee0fb35f5293c91cebac4b758"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack && "
    #     "cd ComfyUI-Inspire-Pack && "
    #     "git reset --hard cadf604de528be62e4fbb1e3d12c51c98f20f50b && "
    #     "pip install -r requirements.txt"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/edenartlab/eden_comfy_pipelines && "
    #     "cd eden_comfy_pipelines && "
    #     "git reset --hard 1b64dd507e8560466a8a50b8b8a704547890f525 && "
    #     "pip install -r requirements.txt"
    # )
    # .run_commands(
    #     "cd /app/custom_nodes && "
    #     "git clone https://github.com/WASasquatch/was-node-suite-comfyui && "
    #     "cd was-node-suite-comfyui && "
    #     "git reset --hard e036c1aa1b228c31473f78e020f47f0ce94d4c80 && "
    #     "pip install -r requirements.txt"
    # )
    # Install ComfyUI-Manager
    .run_commands(
        "cd /app/custom_nodes && "
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git && "
        "cd ComfyUI-Manager && "
        "git reset --hard 2b8e76197ae970dbd7854a09a5ef57731dc1c82f"
    )
    # .run_commands(
    #     "git clone --recursive https://github.com/NVlabs/nvdiffrast",
    #     "pip install nvdiffrast/."
    # )
    .run_commands(
        gpu="A100",
        "git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git",
        "cd ComfyUI-3D-Pack && pip install -r requirements.txt && python install.py",
    )
    # .run_commands(
    #     "comfy node install ComfyUI-3D-Pack"
    # )
)

app = modal.App(name="image-to-3d-comfyui-new", image=image)


@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A100",
)
@modal.web_server(8006, startup_timeout=180)
def ui():
    output = subprocess.check_output(["nvidia-smi"], text=True)
    print("nvidia-smi output: ", output)
    subprocess.Popen("comfy tracking disable", shell=True)
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8006", shell=True)



