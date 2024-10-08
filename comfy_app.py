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
    .run_commands(
        "comfy --skip-prompt install --nvidia",
    )
    .run_commands(
        "comfy node install ComfyUI-ToonCrafter"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/Doubiiu/ToonCrafter/resolve/main/model.ckpt --relative-path custom_nodes/ComfyUI-ToonCrafter/ToonCrafter/checkpoints/tooncrafter_512_interp_v1"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/Doubiiu/ToonCrafter/resolve/main/sketch_encoder.ckpt --relative-path custom_nodes/ComfyUI-ToonCrafter/ToonCrafter/checkpoints"
    )
    .run_commands(
        "comfy node install ComfyUI-VideoHelperSuite"
    )
    .run_commands(
        "ls -R /root/comfy/ComfyUI/models"
    )
)

app = modal.App(name="tooncrafter-comfyui", image=image)



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



@app.cls(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=300,
    gpu="A100",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "workflow_api.json",
            "/root/workflow_api.json",
        ),
    ]
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)
        time.sleep(60)

    def wait_for_comfyui_server(self, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://127.0.0.1:8188/")
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False


    @modal.method()
    def infer(self, workflow_path: str, start_img_path: str, end_img_path: str):
        logger.info("Starting inference...")
        if not self.wait_for_comfyui_server():
            raise Exception("ComfyUI server is not ready")

        server_address = "127.0.0.1:8188"

        try:
            # Upload images
            logger.info("Uploading images...")
            self.upload_image(start_img_path, "start_image.png", server_address)
            self.upload_image(end_img_path, "end_image.png", server_address)

            # Load and prepare the workflow
            logger.info("Preparing workflow...")
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Update image nodes in the workflow
            workflow["12"]["inputs"]["image"] = "start_image.png"
            workflow["14"]["inputs"]["image"] = "end_image.png"

            # Queue prompt
            logger.info("Queueing prompt...")
            client_id = workflow["13"]["inputs"]["filename_prefix"]
            prompt_id = self.queue_prompt(workflow, client_id, server_address)
            logger.info(f"Prompt queued with ID: {prompt_id}")

            # Wait for execution to complete
            max_retries = 180  # 15 minutes total
            retry_delay = 5
            for attempt in range(max_retries):
                try:
                    history = self.get_history(prompt_id, server_address)
                    logger.info(f"Attempt {attempt + 1}: History status - {json.dumps(history.get(prompt_id, {}), indent=2)}")
                    
                    if prompt_id in history:
                        status = history[prompt_id]
                        if status.get('status', {}).get('completed', False):
                            logger.info("Execution completed")
                            break
                        elif 'error' in status:
                            raise Exception(f"Execution failed: {status['error']}")
                    else:
                        logger.warning(f"Prompt ID {prompt_id} not found in history")
                except RequestException as e:
                    logger.warning(f"Error getting history: {e}. Retrying...")
                
                time.sleep(retry_delay)
            else:
                raise TimeoutError(f"Execution did not complete within the expected time ({max_retries * retry_delay} seconds)")

            # Wait a bit more to ensure file system sync
            time.sleep(10)

            # Get output images
            logger.info("Searching for output file...")
            output_dirs = [
                "/root/comfy/ComfyUI/output",
                "/root/comfy/output",
                "/tmp/comfy/output"
            ]

            file_prefix = workflow["13"]["inputs"]["filename_prefix"]

            for output_dir in output_dirs:
                logger.info(f"Checking directory: {output_dir}")
                logger.info(f"Directory contents: {list(Path(output_dir).glob('*'))}")

                for f in Path(output_dir).glob('*'):
                    logger.info(f"Found file: {f}")
                    if f.name.startswith(file_prefix):
                        logger.info(f"Found matching output file: {f}")
                        try:
                            return f.read_bytes()
                        except PermissionError:
                            logger.error(f"Permission denied when trying to read {f}")
                        except Exception as e:
                            logger.error(f"Error reading file {f}: {e}")

            logger.error(f"No output file found with prefix: {file_prefix}")
            raise FileNotFoundError(f"No output file found with prefix: {file_prefix}")

        except Exception as e:
            logger.error(f"An error occurred during inference: {e}")
            raise

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
        return response.json()

    def queue_prompt(self, prompt, client_id, server_address):
        logger.info(f"Queueing prompt for client ID: {client_id}")
        p = {"prompt": prompt, "client_id": client_id}
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(f"http://{server_address}/prompt", json=p, headers=headers)
            response.raise_for_status()
            result = response.json()
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
    async def api(self, 
                  start_image: UploadFile = File(...),
                  end_image: UploadFile = File(...),
                  prompt: str = Form(...),
                  steps: int = Form(50),
                  frame_count: int = Form(5),
                  seed: int = Form(123),
                  eta: float = Form(1.0),
                  cfg_scale: float = Form(7.5),
                  fps: int = Form(1)):
        from fastapi import Response
        import json

        # Save uploaded images temporarily
        start_img_path = f"/tmp/{start_image.filename}"
        end_img_path = f"/tmp/{end_image.filename}"
        with open(start_img_path, "wb") as f:
            f.write(await start_image.read())
        with open(end_img_path, "wb") as f:
            f.write(await end_image.read())

        workflow_data = json.loads(
            (Path(__file__).parent / "workflow_api.json").read_text()
        )

        # Update ToonCrafter node
        tooncrafter_node = workflow_data["15"]
        tooncrafter_node["inputs"].update({
            "prompt": prompt,
            "steps": steps,
            "frame_count": frame_count,
            "seed": seed,
            "eta": eta,
            "cfg_scale": cfg_scale,
            "fps": fps,
            "ckpt_name": "tooncrafter_512_interp_v1/model.ckpt"  # Make sure this matches the actual filename
        })

        # Update image loading nodes
        workflow_data["12"]["inputs"]["image"] = f"/tmp/{start_image.filename}"
        workflow_data["14"]["inputs"]["image"] = f"/tmp/{end_image.filename}"

        # Save updated workflow
        client_id = uuid.uuid4().hex
        workflow_data["13"]["inputs"]["filename_prefix"] = client_id
        new_workflow_file = f"/tmp/{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # Run inference
        self.infer.local(new_workflow_file, start_img_path, end_img_path)

        # Get all output images
        output_dir = "/root/comfy/ComfyUI/output"
        output_images = []
        for f in Path(output_dir).iterdir():
            if f.name.startswith(client_id):
                with open(f, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    output_images.append({
                        "filename": f.name,
                        "data": encoded_image
                    })

        # Clean up temporary files
        os.remove(start_img_path)
        os.remove(end_img_path)
        os.remove(new_workflow_file)

        return Response(content=json.dumps(output_images), media_type="application/json")
