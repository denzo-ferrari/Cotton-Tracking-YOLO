import torch

# The official URL for the specific Re-ID model you need
url = 'https://github.com/ultralytics/assets/releases/download/v8.2.0/osnet_x0_25_msmt17.pt'
file_name = 'osnet_x0_25_msmt17.pt'

print(f"Downloading {file_name}...")
torch.hub.download_url_to_file(url, file_name)
print("Download complete! You can now run your tracker.")