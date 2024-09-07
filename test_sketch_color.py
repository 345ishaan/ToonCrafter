import requests
import json
import base64

url = "https://genime--tooncrafter-comfyui-coloring-comfycoloringui-api-dev.modal.run"

files = {
    'start_image': ('start_image.jpg', open('/Users/tusharbansal/Desktop/kid_with_white_bg.png', 'rb'), 'image/jpeg'),
    'end_image': ('end_image.jpg', open('/Users/tusharbansal/Desktop/kid_end_with_white_bg.png', 'rb'), 'image/jpeg'),
    'video': ('input_video.mp4', open('/Users/tusharbansal/Desktop/kidfinal3.gif', 'rb'), 'video/mp4')
}

data = {
    'prompt': "kid dancing",
    'steps': 50,
    'frame_count': 10,
    'seed': 123,
    'eta': 1.0,
    'cfg_scale': 7.5,
    'fps': 8
}

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    print("Sketch coloring successful!")
    result = response.json()
    print("Number of output files:", len(result))
    
    # Save each file
    for i, file_data in enumerate(result):
        filename = file_data['filename']
        file_content = base64.b64decode(file_data['data'])
        
        with open(f"output_{i}_{filename}", "wb") as f:
            f.write(file_content)
        
        print(f"Saved: output_{i}_{filename}")

else:
    print("Error:", response.status_code)
    print(response.text)