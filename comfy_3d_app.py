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
import websocket
import json
from pathlib import Path
import time
import requests
import urllib.request
import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "requests-toolbelt"])
from requests_toolbelt import MultipartEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .apt_install("ffmpeg")
    .apt_install("libsm6")
    .apt_install("libxext6")
    .pip_install("comfy-cli==1.0.33")
    .pip_install("requests-toolbelt")
    .pip_install("websocket-client")
    .pip_install("kiui")
    .pip_install("torch")
    .pip_install("torchvision")
    .pip_install("torch_scatter")
    .run_commands(
        "git clone https://github.com/facebookresearch/pytorch3d.git",
        "pip install pytorch3d/.")
    # .run_commands(
    #     "git clone https://github.com/ashawkey/diff-gaussian-rasterization.git"
    #     "pip install diff-gaussian-rasterization/.")
    .run_commands(
        "git clone --recursive https://github.com/NVlabs/nvdiffrast",
        "pip install nvdiffrast/."
    )
    .run_commands(
        "comfy --skip-prompt install --nvidia",
    )
    # .run_commands(
    #     "git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git"
    # )
    .run_commands(
        "comfy node install ComfyUI-3D-Pack",
        # "python ComfyUI-3D-Pack/_Pre_Builds/_Build_Scripts/auto_build_all.py"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/tencent/Hunyuan3D-1/blob/main/mvd_std/uc_text_emb.pt --relative-path custom_nodes/ComfyUI-3D-Pack/Checkpoints/Diffusers/tencent/Hunyuan3D-1"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/tencent/Hunyuan3D-1/blob/main/mvd_std/uc_text_emb_2.pt --relative-path custom_nodes/ComfyUI-3D-Pack/Checkpoints/Diffusers/tencent/Hunyuan3D-1"
    )
    .run_commands(
        "ls -R /root/comfy/ComfyUI/models"
    )
)

app = modal.App(name="image-to-3d-comfyui", image=image)



@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A100",
)
@modal.web_server(8005, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8005", shell=True)

