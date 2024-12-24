import requests

url = "https://genime--comfy-3d-app-comfyui-api.modal.run"


files = {
    'image': ('squirrel_girl.png', open('squirrel_girl_1.png', 'rb'), 'image/png'),
}

# Send a POST request to the API endpoint
response = requests.post(url, files=files, timeout=600)

result = response.json()
print(result)