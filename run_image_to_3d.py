import requests
import json

url = "https://genime--comfy-3d-app-comfyui-api.modal.run"

# data = {
#     "image": "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/hanumanji.jpg?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2hhbnVtYW5qaS5qcGciLCJpYXQiOjE3MjMwMTU0NjYsImV4cCI6MTcyMzYyMDI2Nn0.yZ9sTJthRJ2u39ajts4q0AJVHE9DiZ2r9nt6iqwNA7U&t=2024-08-07T07%3A24%3A26.428Z",
# }

# headers = {
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=data, headers=headers)



with open("squirrel_girl_1.png", "rb") as image_file:
    # Prepare the files dictionary for the request
    files = {
        "image": image_file
    }

    # Send a POST request to the API endpoint
    response = requests.post(url, files=files, timeout=3600)

# Check if the request was successful
if response.status_code == 200:
    # Print the response content
    print("Response Content:")
    print(response.json())  # This will print the content returned by the API
else:
    print(f"Failed to retrieve response. Status Code: {response.status_code}")
    print(f"Response Content: {response.json()}")


# if response.status_code == 200:
#     result = response.json()
# else:
#     print("Error:", response.status_code)
#     print(response.text)