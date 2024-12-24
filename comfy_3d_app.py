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
import glob
import shutil
from requests_toolbelt import MultipartEncoder
from comfy_utils import upload_image, queue_prompt, track_progress, interupt_prompt, get_history, get_node_info_by_class, clear_comfy_cache

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
        "apt-get install -y libglfw3-dev",
        "apt-get install -y libgles2-mesa",
        "apt-get install -y mesa-utils",
        "apt-get install -y xvfb",
        "apt-get install -y libgl1-mesa-dri",
        "apt-get install -y x11-xserver-utils",
        # "apt-get install -y pkg-config",
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
    .env({
        "DISPLAY": ":99",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "MESA_GL_VERSION_OVERRIDE": "4.5",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        "PYTHONPATH": "/usr/lib/python3/dist-packages"
    })
    .run_commands(
        "Xvfb :99 -screen 0 1024x768x24 &"
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
        "cd /root/comfy/ComfyUI/custom_nodes && "
        "rm -rf ComfyUI-3D-Pack && "
        "git clone https://github.com/bansaltushar92/ComfyUI-3D-Pack.git && "
        # "git clone https://github.com/1halfplusminus/ComfyUI-3D-Pack.git &&"
        "cd ComfyUI-3D-Pack && "
        "git checkout 37059a42994ee1e5636d6962823098969c2b47f4"
    )
    .run_commands(
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-3D-Pack && "
        "python install.py && "
        "pip install -r requirements.txt && "
        "pip install objprint varname",
        gpu="A100"
    )
)

# image = image.run_commands(
#     "pip install slangtorch==1.3.0",
#     "python -c 'import slangtorch; print(\"slangtorch version:\", slangtorch.__version__)'"
# )

# image = (
#     image.run_commands(  # download a custom node
#         "comfy node install ComfyUI-3D-Pack",
#         gpu="A100"
#     )
# )

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "rm -rf /root/comfy/ComfyUI/models"
    )
)


image = (
    image.pip_install("aiohttp==3.11.11")
    .pip_install("yarl==1.18.3")
    .pip_install("requests-toolbelt==1.0.0")
    .pip_install("websocket_client==1.8.0")
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
@modal.web_server(8005, startup_timeout=300)
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
        modal.Mount.from_local_file(
            Path(__file__).parent / "squirrel_girl_1.png",
            "/root/squirrel_girl_1.png",
        ),
    ],
    volumes={"/root/comfy/ComfyUI/models": vol},
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        print("Inside launch_comfy_background")
        # first check if the server is already running
        if not self.wait_for_comfyui_server(timeout=10):
            cmd = "comfy launch --background"
            subprocess.run(cmd, shell=True, check=True)
        time.sleep(10)
    
    def open_websocket_connection(self,server_address):
        client_id=str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
        return ws, client_id

    
    def wait_for_comfyui_server(self, timeout=600):
        print("Inside wait_for_comfyui_server")
        server_address = "127.0.0.1:8188"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{server_address}/")
                print(response.status_code, f"http://{server_address}/")
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException as e:
                print("RequestException: ", e)
                pass
            time.sleep(1)
        return False

    @modal.method()
    def infer(self, workflow_path: str, image_path: str, client_id: str, output_dir_path: str):
        print("Inside infer")
        # runs the comfy run --workflow command as a subprocess
        # if not self.wait_for_comfyui_server(timeout=10):
        #     raise Exception("ComfyUI server is not ready")
        # logger.info("ComfyUI server is ready")
        server_address = "127.0.0.1:8188"
        # ws, server_address, client_id = self.open_websocket_connection(server_address)
        print("Uploading image")
        upload_image(image_path, "image.png", server_address, client_id)
        print("Uploaded image")
        with open(workflow_path, 'r') as f:
            prompt = json.load(f)

        # Update image nodes in the workflow
        prompt["9"]["inputs"]["image"] = "image.png"
        print("Queuing prompt")
        prompt_id = queue_prompt(prompt, client_id, server_address)['prompt_id']

        while True:
            print("Getting history")
            history = get_history(prompt_id, server_address)
            if history:
                break
            time.sleep(1)
        

        # try:
        #     # Upload images
        #     logger.info("Uploading images...")
        #     self.upload_image(image_path, "image.png", server_address)

        #     # Load and prepare the workflow
        #     logger.info("Preparing workflow...")
        #     with open(workflow_path, 'r') as f:
        #         workflow = json.load(f)

        #     # Update image nodes in the workflow
        #     workflow["9"]["inputs"]["image"] = "image.png"

        #     # Queue prompt
        #     logger.info("Queueing prompt...")
        #     prompt_id = self.queue_prompt(workflow, client_id, server_address)
        #     logger.info(f"Prompt queued with ID: {prompt_id}")

        #     # Wait for execution to complete
        #     max_retries = 180  # 15 minutes total
        #     retry_delay = 5
        #     for attempt in range(max_retries):
        #         try:
        #             history = self.get_history(prompt_id, server_address)
        #             print(f"Attempt {attempt + 1}: History status - {json.dumps(history.get(prompt_id, {}), indent=2)}")
        #             # check if there are any obj and mtl files in the output_dir_path
        #             if os.path.exists(output_dir_path):
        #                 obj_files = glob.glob(f"{output_dir_path}/*.obj")
        #                 mtl_files = glob.glob(f"{output_dir_path}/*.mtl")
        #                 print(f"Found {len(obj_files)} obj files and {len(mtl_files)} mtl files in {output_dir_path}")

        #             logger.info(f"Attempt {attempt + 1}: History status - {json.dumps(history.get(prompt_id, {}), indent=2)}")
                    
        #             if prompt_id in history:
        #                 status = history[prompt_id]
        #                 if status.get('status', {}).get('completed', False):
        #                     logger.info("Execution completed")
        #                     break
        #                 elif 'error' in status:
        #                     raise Exception(f"Execution failed: {status['error']}")
        #             else:
        #                 logger.warning(f"Prompt ID {prompt_id} not found in history")
        #         except RequestException as e:
        #             logger.warning(f"Error getting history: {e}. Retrying...")
                
        #         time.sleep(retry_delay)
        #     else:
        #         raise TimeoutError(f"Execution did not complete within the expected time ({max_retries * retry_delay} seconds)")

        #     # Wait a bit more to ensure file system sync
        #     time.sleep(10)

        # except Exception as e:
        #     logger.error(f"An error occurred during inference: {e}")
        #     raise

    def upload_image(self, input_path, name, server_address):
        with open(input_path, 'rb') as file:
            multipart_data = MultipartEncoder(
                fields={
                    'image': (name, file, 'image/png'),
                    'type': 'input',
                    'overwrite': 'true'
                }
            )
            headers = {'Content-Type': multipart_data.content_type}
            response = requests.post(f"http://{server_address}/upload/image", data=multipart_data, headers=headers)
            response.raise_for_status()
            print(f"successfully uploaded image for {input_path}/{name}/{server_address}/{response.status_code}")
        return response.json()

    def queue_prompt(self, prompt, client_id, server_address):
        logger.info(f"Queueing prompt for client ID: {client_id}")
        p = {"prompt": prompt, "client_id": client_id}
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(f"http://{server_address}/prompt", json=p, headers=headers)
            response.raise_for_status()
            result = response.json()
            print(f"successfully queued prompt for {client_id}/{server_address}/{response.status_code}/{result}")
            logger.info(f"Prompt queued successfully. Response: {json.dumps(result, indent=2)}")
            return result['prompt_id']
        except Exception as e:
            logger.error(f"Error queueing prompt: {e}")
            raise

    def get_history(self, prompt_id, server_address):
        response = requests.get(f"http://{server_address}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()



    @modal.web_endpoint(method="POST")
    async def api(self, image: UploadFile = File(...)):
        from fastapi import Response
        
        local_img_path = f"/tmp/{image.filename}"
        with open(local_img_path, "wb") as f:
            f.write(await image.read())
        
        workflow_data = json.loads(
            (Path(__file__).parent / "workflow_hunyuan_3d_api.json").read_text()
        )
        
        workflow_data["9"]["inputs"]["image"] = local_img_path
        output_folder_uuid = uuid.uuid4().hex
        output_folder_path = f"/root/comfy/ComfyUI/models/{output_folder_uuid}"
        # create output folder if it doesn't exist
        os.makedirs(output_folder_path, exist_ok=True)
        workflow_data["15"]["inputs"]["save_path"] = f"{output_folder_path}/mesh_1.obj"
        workflow_data["17"]["inputs"]["save_path"] = f"{output_folder_path}/mesh_t_1.obj"
        
        client_id = uuid.uuid4().hex
        new_workflow_file = f"/tmp/{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # Run inference
        self.infer.local(new_workflow_file, local_img_path, client_id, output_folder_path)
        response_data = {
            "files": {}
        }
            
        # Clean up: delete the output folder
        if os.path.exists(output_folder_path):
            # Gather all .obj and .mtl files from the output folder
            obj_files = glob.glob(f"{output_folder_path}/*.obj")
            mtl_files = glob.glob(f"{output_folder_path}/*.mtl")

            # Read the contents of the .obj and .mtl files
            for obj_file in obj_files:
                with open(obj_file, 'r') as f:
                    response_data["files"][os.path.basename(obj_file)] = f.read()

            for mtl_file in mtl_files:
                with open(mtl_file, 'r') as f:
                    response_data["files"][os.path.basename(mtl_file)] = f.read()

            shutil.rmtree(output_folder_path)
            print(f"Deleted output folder: {output_folder_path}")
        else:
            print(f"Output folder does not exist: {output_folder_path}")

        return Response(content=json.dumps(response_data), media_type="application/json")
       