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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .run_commands("export DEBIAN_FRONTEND=noninteractive")
    .apt_install("git", "gcc", "g++")  # install git to clone ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.3.1")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands("nvcc --version")
    .run_commands("pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124")
    .run_commands(
        # Set debconf to non-interactive mode
        "export DEBIAN_FRONTEND=noninteractive",
        # Preconfigure keyboard settings to avoid prompts
        "echo 'keyboard-configuration keyboard-configuration/layout select English (US)' | debconf-set-selections",
        "echo 'keyboard-configuration keyboard-configuration/layoutcode string us' | debconf-set-selections",
        "echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections"
    )
    .run_commands(        
        "apt-get install -y clang",
        "apt-get install -y libomp-dev",
        "apt-get install --no-install-recommends -y libegl1",
        "apt-get install --no-install-recommends -y libegl1-mesa-dev",
        "apt-get install --no-install-recommends -y libgl1",
        "apt-get install --no-install-recommends -y libglib2.0-0",
        "apt-get install --no-install-recommends -y libgl1-mesa-dev",
        "apt-get install --no-install-recommends -y libgl1-mesa-glx",
        "apt-get install --no-install-recommends -y libgles2",
        "apt-get install --no-install-recommends -y libgles2-mesa-dev",
        "apt-get install --no-install-recommends -y libglib2.0-0",
        "apt-get install --no-install-recommends -y libglvnd-dev",
        "apt-get install --no-install-recommends -y libglvnd0",
        "apt-get install --no-install-recommends -y libglx0",
        "apt-get install --no-install-recommends -y libsm6",
        "apt-get install --no-install-recommends -y libxext6",
        "apt-get install --no-install-recommends -y libxrender1",
        "pip install onnxruntime-gpu==1.20.0"
    )
    # .run_commands("apt-get -y install nvidia-driver-525")
    # .env({
    #     "CUDA_HOME": "/usr/local/cuda",
    #     "PATH": "/usr/local/cuda/bin:$PATH"
    # })
    # .run_commands("nvidia-smi")
    
)

image = (

    image
    .pip_install("wheel")
    .run_commands(
        # Clone the ComfyUI-3D-Pack repository
        "git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git",
        "cd ComfyUI-3D-Pack && python install.py",
        gpu="A100"
    )
)

image = (
    image.run_commands(  # download a custom node
        "comfy node install ComfyUI-3D-Pack",
        gpu="A100"
    )
)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(  # needs to be empty for Volume mount to work
        "rm -rf /root/comfy/ComfyUI/models"
    )
)

app = modal.App(name="comfy-3d-app", image=image)

vol = modal.Volume.from_name("comfyui-3d-models", create_if_missing=True)

@app.function(gpu="any")
def check_nvidia_smi():
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version: 550.90.07" in output
    assert "CUDA Version: 12.4" in output
    return output

@app.function(
    volumes={"/root/3d_models": vol},
)
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=f"/root/3d_models/{model_type}",
    )

@app.local_entrypoint()
def download_models():
    models_to_download = [
        # format is (huggingface repo_id, the model filename, comfyui models subdirectory we want to save the model in)
        (
            "tencent/Hunyuan3D-1",
            "mvd_lite/vae/diffusion_pytorch_model.safetensors",
            "Hunyuan3D",
        ),
        (
            "tencent/Hunyuan3D-1",
            "mvd_lite/vision_encoder/model.safetensors",
            "Hunyuan3D",
        ),
        (
            "tencent/Hunyuan3D-1",
            "mvd_lite/unet/diffusion_pytorch_model.safetensors",
            "Hunyuan3D",
        ),
        (
            "tencent/Hunyuan3D-1",
            "mvd_lite/text_encoder/model.safetensors",
            "Hunyuan3D",
        ),
        
    ]
    list(hf_download.starmap(models_to_download))


@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A100",
    volumes={"/root/comfy/ComfyUI/models": vol},
)
@modal.web_server(8005, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8005", shell=True)


@app.cls(
    allow_concurrent_inputs=10,
    container_idle_timeout=300,
    gpu="A100",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "workflow_hunyuan_3d_api.json",
            "/root/workflow_hunyuan_3d_api.json",
        ),
    ],
    volumes={"/root/comfy/ComfyUI/models": vol},
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_hunyuan_3d_api.json"):
        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200"
        result = subprocess.run(cmd, shell=True, check=True)
        # Check if the command was successful
        if result.returncode == 0:
            # Command was successful
            return {"status": "success", "output": result.stdout}
        else:
            # Command failed
            return {"status": "error", "output": result.stderr}

        # # completed workflows write output images to this directory
        # output_dir = "/root/comfy/ComfyUI/output"
        # # looks up the name of the output image file based on the workflow
        # workflow = json.loads(Path(workflow_path).read_text())
        # file_prefix = [
        #     node.get("inputs")
        #     for node in workflow.values()
        #     if node.get("class_type") == "SaveImage"
        # ][0]["filename_prefix"]

        # # returns the image as bytes
        # for f in Path(output_dir).iterdir():
        #     if f.name.startswith(file_prefix):
        #         return f.read_bytes()

    @modal.web_endpoint(method="POST")
    async def api(self, image: UploadFile = File(...)):
        from fastapi import Response
        

        local_img_path = f"/tmp/{image.filename}"
        with open(local_img_path, "wb") as f:
            f.write(await image.read())
        

        workflow_data = json.loads(
            (Path(__file__).parent / "workflow_hunyuan_3d_api.json").read_text()
        )
        for node in workflow_data["nodes"]:
            if node["id"] == 9:
                node["widgets_values"][0] = local_img_path

        client_id = uuid.uuid4().hex
        new_workflow_file = f"/tmp/{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # Run inference
        result = self.infer.local(new_workflow_file)

        return Response(
            content=f"""
            Executed inference for {new_workflow_file}; status: {result.get("status")}; output: {result.get("output")}""",
            media_type="text/plain"
        )

       