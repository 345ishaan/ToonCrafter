import os
import logging
import uvicorn
import uuid
from pydantic import BaseModel

from fastapi import FastAPI

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
#from genime_creator import GenimeCreator
from typing import List
from supabase import create_client, Client
import sys
sys.path.append("/home/ToonCrafter")
from scripts.evaluation.genime_inference import InterPolater
import shutil

app = FastAPI()

SUPABASE_URL = "https://ttvaarlnqssopdguetwq.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR0dmFhcmxucXNzb3BkZ3VldHdxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMDcxODMwNSwiZXhwIjoyMDM2Mjk0MzA1fQ.TFoNnJhvghFsinuAjbNILN1ohJkz41vbv9y-4Eva12g"

supa_client = create_client(SUPABASE_URL, SUPABASE_KEY)

interpolater = InterPolater()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

#genime_creator = GenimeCreator()


class InterpolaterRequest(BaseModel):
    video_url_a: str
    video_url_b: str
    prompt: str


@app.get("/test")
async def root():
    return "Hello World"


@app.post("/create-interpolation")
async def create_interpolation(req: InterpolaterRequest):
    """Endpoint for motion interpolation
    """

    save_dir = f"/home/ToonCrafter/server_response/{str(uuid.uuid4())}"
    interpolater.infer(image_urls=[(req.video_url_a, req.video_url_b)],
        prompt=[req.prompt],
        save_dir=save_dir)
    save_fname = os.path.join(save_dir, 'final_concat.mp4')
    zip_fname = os.path.join(save_dir, 'final_concat.zip')
    if not os.path.exists(save_fname):
        raise Exception("Could not retrieve final lip sync")
    with open(save_fname, 'rb') as f:
        supa_client.storage.from_("genime-bucket").upload(file=f,
                path=save_fname, 
                file_options={"content-type": "video/mp4"})
    with open(zip_fname, "rb") as f:
        supa_client.storage.from_("genime-bucket").upload(
                path=zip_fname,
                file=f,
                file_options={"content-type": "application/zip"}
        )
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    return save_fname


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
