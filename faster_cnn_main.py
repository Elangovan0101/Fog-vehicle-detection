import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define class names for COCO dataset (index 2 for "car" and 7 for "truck")
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    # Other classes omitted for brevity
]

# Define a transformation to apply to the input images
transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to PyTorch tensor
])

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate the intersection area
    x_inter1 = max(x1, x1g)
    y_inter1 = max(y1, y1g)
    x_inter2 = min(x2, x2g)
    y_inter2 = min(y2, y2g)
    
    if x_inter1 < x_inter2 and y_inter1 < y_inter2:  # There's an intersection
        intersection_area = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
    else:
        intersection_area = 0

    # Calculate the areas of the predicted and ground truth boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Function to perform vehicle detection on a single image
def detect_vehicles(image_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    
    with torch.no_grad():
        predictions = model([img_tensor])

    boxes = []
    labels = []
    scores = predictions[0]['scores'].numpy()
    pred_boxes = predictions[0]['boxes'].numpy()
    pred_labels = predictions[0]['labels'].numpy()

    # Ensure that pred_labels values are within the range of COCO_INSTANCE_CATEGORY_NAMES
    for i, score in enumerate(scores):
        if score > threshold:
            label_idx = pred_labels[i]
            if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):  # Check if the index is within range
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                if label in ['car', 'truck']:
                    boxes.append(pred_boxes[i])
                    labels.append(label)
            else:
                print(f"Warning: Predicted label index {label_idx} is out of range.")  # Debugging out-of-range labels

    img_cv2 = cv2.imread(image_path)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_cv2, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return boxes, labels

# Evaluation function to calculate precision, recall, and F1 score
def evaluate_predictions(ground_truths, predictions, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for gt_boxes, gt_labels in ground_truths:
        matched = [False] * len(gt_boxes)
        
        for pred_box, pred_label in predictions:
            match_found = False
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label == pred_label:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= iou_threshold and not matched[i]:
                        tp += 1
                        matched[i] = True
                        match_found = True
                        break

            if not match_found:
                fp += 1
        
        fn += len(gt_boxes) - sum(matched)  # Ground truth boxes not matched by predictions

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    return precision, recall, f1_score

# Test the function on an image and evaluate accuracy
image_path = "C:/Users/VJ_Mahesh/OneDrive/Desktop/fog vehicle detection/datasets/images/test/mist-147.jpg"
detected_boxes, detected_labels = detect_vehicles(image_path, threshold=0.5)

# Replace this with the actual ground truth for the image
ground_truth_boxes = [[100, 150, 400, 450]]  # Example coordinates
ground_truth_labels = ['car']                # Example label

# Evaluate the model's predictions
evaluate_predictions([(ground_truth_boxes, ground_truth_labels)], list(zip(detected_boxes, detected_labels))) 
