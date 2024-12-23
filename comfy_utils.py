import json
import urllib.request
import urllib.parse
import requests
from requests_toolbelt import MultipartEncoder


def upload_image(input_path, name, server_address, image_type="input", overwrite=False):
	with open(input_path, 'rb') as file:
		multipart_data = MultipartEncoder(
            fields= {
                'image': (name, file, 'image/png'),
                'type': image_type,
                'overwrite': str(overwrite).lower()
            }
		)

		headers = { 'Content-Type': multipart_data.content_type }
		response = requests.post(f"http://{server_address}/upload/image", data=multipart_data, headers=headers)
		response.raise_for_status()
	return response.json()
		
def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
    return json.loads(urllib.request.urlopen(req).read())

def interupt_prompt(server_address):
    req =  urllib.request.Request("http://{}/interrupt".format(server_address), data={})
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id, server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_node_info_by_class(node_class, server_address):
    with urllib.request.urlopen("http://{}/object_info/{}".format(server_address, node_class)) as response:
        return json.loads(response.read())

def clear_comfy_cache(server_address, unload_models=False, free_memory=False):
    clear_data = {
    "unload_models": unload_models,
    "free_memory": free_memory
    }
    data = json.dumps(clear_data).encode('utf-8')

    with urllib.request.urlopen("http://{}/free".format(server_address), data=data) as response:
        return response.read()
    

def track_progress(prompt, ws, prompt_id):
    node_ids = list(prompt.keys())
    finished_nodes = []

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'progress':
                data = message['data']
                current_step = data['value']
                print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
            if message['type'] == 'execution_cached':
                data = message['data']
                for itm in data['nodes']:
                    if itm not in finished_nodes:
                        finished_nodes.append(itm)
                        print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] not in finished_nodes:
                    finished_nodes.append(data['node'])
                    print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')


                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data
    return