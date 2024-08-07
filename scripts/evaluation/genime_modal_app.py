import modal
import os
import uuid
from pydantic import BaseModel
import shutil
import sys
sys.path.append("/root/project")

stub = modal.Stub("tooncrafter-model")

class InterpolaterRequest(BaseModel):
    video_url_a: str
    video_url_b: str
    prompt: str
    eta: float 
    cfg_scale: float 
    steps: int 
    fps: int
    frame_stride: int
    width: int
    height: int
    video_len: int


def download_model():
    import os
    import subprocess

    os.makedirs("/root/project/checkpoints/tooncrafter_512_interp_v1", exist_ok=True)

    subprocess.run([
        "wget",
        "https://huggingface.co/Doubiiu/ToonCrafter/resolve/main/model.ckpt",
        "-O",
        "/root/project/checkpoints/tooncrafter_512_interp_v1/model.ckpt"
    ], check=True)

    print("Model downloaded successfully.")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel")
    .apt_install(
        "wget", "tzdata", "vim", "git",
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev"
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "America/New_York"})
    .run_commands(
        "ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime",
        "dpkg-reconfigure --frontend noninteractive tzdata"
    )
    .run_commands(
        "rm -rf /root/project",  # Remove existing project directory
        "ls",
        "git clone https://github.com/345ishaan/ToonCrafter.git /root/project",
        "cd /root/project && pip install -r requirements.txt"
    )
    .run_function(download_model)
)

# We'll use a persistent directory instead of a volume for now
persistent_dir = modal.NetworkFileSystem.persisted("tooncrafter-model-weights")

@stub.cls(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_name("supabase-credentials")],
    network_file_systems={"/model": persistent_dir}
)
class ToonCrafter:
    def __init__(self):
        self.model = None
        self.supa_client = None

    @modal.enter()
    def load_model(self):
        if self.model is None:
            from scripts.evaluation.genime_inference import InterPolater
            self.model = InterPolater()

    def init_supa_client(self):
        if self.supa_client is None:
            SUPABASE_URL = os.getenv("SUPABASE_URL")                                                                                                                                               
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            from supabase import create_client, Client
            self.supa_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    @modal.method()
    def create_interpolation(self, req: InterpolaterRequest):
        self.init_supa_client()
        save_dir = f"/tmp/ToonCrafter/server_response/{str(uuid.uuid4())}"
        os.makedirs(save_dir, exist_ok=True)
        img_urls = [(req.video_url_a, req.video_url_b)]
        prompts = [req.prompt]
        self.model.infer(
            img_urls=img_urls,
            prompts=prompts,
            save_dir=save_dir,
            eta=req.eta, 
            cfg_scale=req.cfg_scale, 
            steps=req.steps,
            fps=req.fps,
            width=req.width,
            height=req.height,
            frame_stride=req.frame_stride,
            video_len=req.video_len
        )
        
        
        save_fname = os.path.join(save_dir, 'final_concat.mp4')
        zip_fname = os.path.join(save_dir, 'final_concat.zip')
        # return (save_fname, zip_fname)
        
        if not os.path.exists(save_fname):
            raise Exception("Could not retrieve final lip sync")
        
        with open(save_fname, 'rb') as f:
            self.supa_client.storage.from_("genime-bucket").upload(
                file=f,
                path=save_fname, 
                file_options={"content-type": "video/mp4"}
            )
        
        with open(zip_fname, "rb") as f:
            self.supa_client.storage.from_("genime-bucket").upload(
                path=zip_fname,
                file=f,
                file_options={"content-type": "application/zip"}
            )
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        return (save_fname, zip_fname)

@stub.function()
@modal.web_endpoint(method="POST")
def endpoint(request: InterpolaterRequest):
    print("Received data:", request)
    tooncrafter = ToonCrafter()
    return tooncrafter.create_interpolation.remote(request)


if __name__ == "__main__":
    stub.serve()