from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")
# Train the model
model.train(data="C:/Users/VJ_Mahesh/OneDrive/Desktop/fog vehicle detection/dataset.yaml", epochs=50, imgsz=640, batch=16)
