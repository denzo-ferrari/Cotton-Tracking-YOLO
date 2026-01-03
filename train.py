

from roboflow import Roboflow
from ultralytics import YOLO


rf = Roboflow(api_key="Ckiw5pJlNNOE3yWBojfW")
project = rf.workspace("deniz-drin5").project("cotton-boll-and-cluster")
version = project.version(2)
dataset = version.download("yolov11")
                

# REPLACE THE BLOCK ABOVE with your actual Roboflow code snippet!
# For this example, I will assume your dataset downloaded to a folder named "dataset"
# If Roboflow downloads it to a folder with a different name, update the 'data=' line below.

# --- Step 2: Initialize Model ---
# We load a pre-trained model to start (Transfer Learning). 
# 'yolo11n.pt' is the latest Nano model (fastest). You can also use 'yolov8n.pt'.
model = YOLO('yolo11n.pt') 

# --- Step 3: Train ---
if __name__ == '__main__':
    # Train the model
    results = model.train(
        data= f'{dataset.location}/data.yaml',  # Points to the downloaded data config
        epochs=50,                             # How many times to go through the data
        imgsz=640,                             # Image size
        plots=True                             # Save training graphs
    )
    
    # The best model will be saved to: runs/detect/train/weights/best.pt
    print("Training Complete. Model saved to runs/detect/train/weights/best.pt")