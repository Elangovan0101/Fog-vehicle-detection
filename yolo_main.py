from ultralytics import YOLO

# Load the model
model = YOLO("runs/detect/train2/weights/best.pt")

# Run validation
metrics = model.val(data="C:/Users/VJ_Mahesh/OneDrive/Desktop/fog vehicle detection/dataset.yaml")

# Accessing the mAP50 score (usually stored as the first value in the maps array)
mAP50_score = metrics.maps[0]  # Assuming index 0 corresponds to mAP@0.5
print("Accuracy Score (mAP50):", mAP50_score)
