# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path = "Src/demo/image/zrn.jpg"
output = model(Image.open(image_path), save=True, conf=0.5)
results = Detections.from_ultralytics(output[0])
