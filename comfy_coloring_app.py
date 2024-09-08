import json
import subprocess
import uuid
import tempfile
from pathlib import Path
from typing import Dict
import shutil
import os
import base64

import modal
from fastapi import File, UploadFile, Form
import requests
from requests_toolbelt import MultipartEncoder
import logging
import time
import zipfile
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .apt_install("ffmpeg")
    .apt_install("libsm6")
    .apt_install("libxext6")
    .pip_install("comfy-cli==1.0.33")
    .pip_install("requests-toolbelt")
    .pip_install("Pillow")
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
)

app = modal.App(name="tooncrafter-comfyui-coloring", image=image)

@app.cls(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=300,
    gpu="A100",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "workflow_sketch_color_api.json",
            "/root/workflow_sketch_color_api.json",
        ),
    ]
)
class ComfyColoringUI:
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
    def infer(self, workflow_path: str, start_img_path: str, end_img_path: str, video_path: str):
        logger.info("Starting inference...")
        if not self.wait_for_comfyui_server():
            raise Exception("ComfyUI server is not ready")

        server_address = "127.0.0.1:8188"

        try:
            # Upload images and video
            logger.info("Uploading images and video...")
            self.upload_image(start_img_path, "start_image.png", server_address)
            self.upload_image(end_img_path, "end_image.png", server_address)
            self.upload_image(video_path, "input_video.mp4", server_address)

            # Load and prepare the workflow
            logger.info("Preparing workflow...")
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Queue prompt
            logger.info("Queueing prompt...")
            client_id = workflow["6"]["inputs"]["filename_prefix"]
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

            file_prefix = workflow["6"]["inputs"]["filename_prefix"]

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

    def background_conversion(self, input_path: str, output_path: str, to_white: bool):
        from PIL import Image
        img = Image.open(input_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        if to_white:
            new_img = Image.new('RGB', img.size, (255, 255, 255))
            new_img.paste(img, (0, 0), img)
            new_img.save(output_path)
        else:
            datas = img.getdata()
            new_data = []
            for item in datas:
                if item[0] > 200 and item[1] > 200 and item[2] > 200:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)
            img.save(output_path, "PNG")
    

    def transform_filename(self, original_path: str, suffix: str):
        # Split the path into directory, filename, and extension
        directory, filename = os.path.split(original_path)
        name, ext = os.path.splitext(filename)
        
        # Create the new filename with "_bg" added before the extension
        new_filename = f"{name}{suffix}{ext}"
        
        # Join the directory with the new filename
        return os.path.join(directory, new_filename)


    def get_history(self, prompt_id, server_address):
        response = requests.get(f"http://{server_address}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    @modal.web_endpoint(method="POST")
    async def api(self, 
                  start_image: UploadFile = File(...),
                  end_image: UploadFile = File(...),
                  video: UploadFile = File(...),
                  prompt: str = Form(...),
                  steps: int = Form(50),
                  frame_count: int = Form(10),
                  seed: int = Form(123),
                  eta: float = Form(1.0),
                  cfg_scale: float = Form(7.5),
                  fps: int = Form(8)):
        from fastapi import Response
        import json

        # Save uploaded files temporarily
        start_img_path = f"/tmp/{start_image.filename}"
        end_img_path = f"/tmp/{end_image.filename}"

        video_path = f"/tmp/{video.filename}"
        with open(start_img_path, "wb") as f:
            f.write(await start_image.read())
        with open(end_img_path, "wb") as f:
            f.write(await end_image.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())
        # chose background conversion filename as _bg_converted.png
        start_white_bg_path = self.transform_filename(start_img_path, "_white_bg")
        end_white_bg_path = self.transform_filename(end_img_path, "_white_bg")
        self.background_conversion(start_img_path, start_white_bg_path, to_white=True)
        self.background_conversion(end_img_path, end_white_bg_path, to_white=True)
        

        workflow_data = json.loads(
            (Path(__file__).parent / "workflow_sketch_color_api.json").read_text()
        )

        # Update ToonCrafterWithSketch node
        tooncrafter_node = workflow_data["1"]
        tooncrafter_node["inputs"].update({
            "prompt": prompt,
            "steps": steps,
            "frame_count": frame_count,
            "seed": seed,
            "eta": eta,
            "cfg_scale": cfg_scale,
            "fps": fps,
            "ckpt_name": "tooncrafter_512_interp_v1/model.ckpt"
        })

        # Update image and video loading nodes
        workflow_data["3"]["inputs"]["image"] = start_white_bg_path
        workflow_data["4"]["inputs"]["image"] = end_white_bg_path
        workflow_data["7"]["inputs"]["video"] = video_path

        # Save updated workflow
        client_id = uuid.uuid4().hex
        workflow_data["6"]["inputs"]["filename_prefix"] = client_id
        new_workflow_file = f"/tmp/{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # Run inference
        result = self.infer.local(new_workflow_file, start_img_path, end_img_path, video_path)

        # Get all output files
        output_dir = "/root/comfy/ComfyUI/output"
        output_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            from PIL import Image
            for f in Path(output_dir).iterdir():
                if f.name.startswith(client_id):
                    with open(f, "rb") as file:
                        # Create a temporary file path
                        temp_file_path = os.path.join(temp_dir, f.name)
                        trans_bg_file_path = self.transform_filename(temp_file_path, "_trans_bg")
                        # Read the file content and save it to the temporary file
                        with open(f, "rb") as source_file, open(temp_file_path, "wb") as temp_file:
                            temp_file.write(source_file.read())
                    
                        # Make background transparent
                        self.background_conversion(start_img_path, trans_bg_file_path, to_white=False)
                        
                        # Open the transparent image
                        with Image.open(trans_bg_file_path) as transparent_img:
                            # Save to bytes
                            img_byte_arr = io.BytesIO()
                            transparent_img.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            output_files.append({
                                "filename": f.name,
                                "data": base64.b64encode(img_byte_arr).decode('utf-8')
                            })

        # Clean up temporary files
        os.remove(start_img_path)
        os.remove(start_white_bg_path)
        os.remove(end_img_path)
        os.remove(end_white_bg_path)
        os.remove(video_path)
        os.remove(new_workflow_file)

        return Response(content=json.dumps(output_files), media_type="application/json")

