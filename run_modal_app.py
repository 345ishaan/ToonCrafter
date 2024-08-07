import requests
import json

url = "https://genime--tooncrafter-model-endpoint.modal.run"

data = {
    "video_url_a": "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/hanumanji.jpg?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2hhbnVtYW5qaS5qcGciLCJpYXQiOjE3MjMwMTU0NjYsImV4cCI6MTcyMzYyMDI2Nn0.yZ9sTJthRJ2u39ajts4q0AJVHE9DiZ2r9nt6iqwNA7U&t=2024-08-07T07%3A24%3A26.428Z",
    "video_url_b": "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/hanumanji_2.jpg?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2hhbnVtYW5qaV8yLmpwZyIsImlhdCI6MTcyMzAxNTQ3NCwiZXhwIjoxNzIzNjIwMjc0fQ.5jwZsrrRygg5wovMkyO0jNcNv0N8nMxNuV5UWFyo_jM&t=2024-08-07T07%3A24%3A34.839Z",
    "prompt": "man tearing his chest to reveal a divine image inside",
    "eta": 1.0,
    "cfg_scale": 7.5,
    "steps": 50,
    "fps": 10,
    "frame_stride": 10,
    "width": 512,
    "height": 320,
    "video_len": 16
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=data, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response Headers: {response.headers}")
print(f"Response Content: {response.text}")

if response.status_code == 200:
    result = response.json()
    print("Interpolation successful!")
    print("Video file:", result[0])
    print("ZIP file:", result[1])
else:
    print("Error:", response.status_code)
    print(response.text)