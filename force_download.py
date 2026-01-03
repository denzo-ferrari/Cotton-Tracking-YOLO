import torch
import requests

# Use the reliable Hugging Face backup mirror
backup_url = 'https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt'
file_name = 'osnet_x0_25_msmt17.pt'

print(f"Attempting download from backup: {backup_url}")

try:
    # We use requests with a fake user-agent to bypass firewalls
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(backup_url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"\nSUCCESS! {file_name} saved. You are ready to track.")
    else:
        print(f"Failed with status code: {response.status_code}")

except Exception as e:
    print(f"An error occurred: {e}")