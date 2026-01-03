import os
import requests

# The exact filename required
filename = 'osnet_x0_25_msmt17.pt'

# 1. Delete the "bad" file if it exists
if os.path.exists(filename):
    print(f"Deleting incompatible file: {filename}")
    os.remove(filename)

# 2. Download the Correct 'Packaged' Version
# We use v8.1.0 because it is known to be stable and packaged correctly
url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/osnet_x0_25_msmt17.pt"

print(f"Downloading correct version from: {url}")

try:
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("\nSUCCESS! The correct file has been installed.")
        print("You can now run 'python track_webcam_v2.py'")
    else:
        print(f"Failed to download. Status Code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")