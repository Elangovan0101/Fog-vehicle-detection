import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict

# Example: Change this to your model loading code (RetinaNet, YOLO, etc.)
# Assuming you are using a model like RetinaNet here:
# model = torch.load('path_to_your_model.pth')

# Replace with the actual model loading code for YOLO, RetinaNet, etc.
from torchvision.models.detection import retinanet_resnet50_fpn
model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Example COCO Category Names (you can replace this with your own classes)
COCO_INSTANCE_CATEGORY_NAMES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

# Example image transform (resize and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function to calculate IoU (Intersection over Union)
def calculate_iou(pred_box, gt_box):
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box

    # Calculate intersection area
    x1_int = max(x1_pred, x1_gt)
    y1_int = max(y1_pred, y1_gt)
    x2_int = min(x2_pred, x2_gt)
    y2_int = min(y2_pred, y2_gt)

    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)

    # Calculate areas of the bounding boxes
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # Calculate union area
    union_area = area_pred + area_gt - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Function to evaluate Precision, Recall, and F1 score
def evaluate_predictions(ground_truths, predictions, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    for gt_boxes, gt_labels in ground_truths:
        matched = [False] * len(gt_boxes)
        
        for pred_box, pred_label in predictions:
            match_found = False
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label == pred_label:
                    iou = calculate_iou(pred_box, gt_box)
                    print(f"IOU between predicted: {pred_box} and ground truth: {gt_box} is {iou:.2f}")
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
    f1_score_value = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score_value:.2f}")
    return precision, recall, f1_score_value

# Function to perform vehicle detection on a single image and debug predictions
def detect_vehicles(image_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes = []
    labels = []
    scores = predictions[0]['scores'].numpy()
    pred_boxes = predictions[0]['boxes'].numpy()
    pred_labels = predictions[0]['labels'].numpy()

    # Filter predictions by threshold score
    for i, score in enumerate(scores):
        if score > threshold:
            label_idx = pred_labels[i]
            if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):  # Check if index is valid
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                if label in ['car', 'truck']:  # Modify based on your specific classes
                    boxes.append(pred_boxes[i])
                    labels.append(label)
                    print(f"Predicted: {label} | Box: {pred_boxes[i]} | Score: {score}")  # Debugging line

    img_cv2 = cv2.imread(image_path)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_cv2, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return boxes, labels

# Example ground truth format (adjust this with your actual ground truth data)
ground_truth_boxes = [
    [[100, 150, 400, 450]],  # Coordinates of the ground truth bounding boxes
]
ground_truth_labels = [
    ['car'],  # Labels corresponding to the ground truth boxes
]

# Test the detection and evaluation with an image
image_path = r"C:\Users\VJ_Mahesh\OneDrive\Desktop\fog vehicle detection\datasets\images\test\mist-966.jpg"  # Replace with your test image path
pred_boxes, pred_labels = detect_vehicles(image_path, threshold=0.5)

# Evaluate the model performance using Precision, Recall, and F1 Score
precision, recall, f1_score_value = evaluate_predictions(
    ground_truths=[(ground_truth_boxes[0], ground_truth_labels[0])],  # Format your ground truth data
    predictions=list(zip(pred_boxes, pred_labels)),  # Format your predicted data
    iou_threshold=0.5
)
