import torch
from torchvision.models.detection import retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define transformation for the input image
def transform(image):
    return F.to_tensor(image).unsqueeze(0)

# Load both RetinaNet and Faster R-CNN pre-trained models
retina_model = retinanet_resnet50_fpn(pretrained=True)
fasterrcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)

# Set the models to evaluation mode
retina_model.eval()
fasterrcnn_model.eval()

def compute_iou(pred_box, gt_box):
    # Calculate intersection over union (IoU)
    x1, y1, x2, y2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    # Calculate intersection
    inter_x1 = max(x1, gx1)
    inter_y1 = max(y1, gy1)
    inter_x2 = min(x2, gx2)
    inter_y2 = min(y2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    pred_area = (x2 - x1) * (y2 - y1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)

    union_area = pred_area + gt_area - inter_area

    # Return IoU
    return inter_area / union_area if union_area != 0 else 0

def hybrid_detect(image_path, threshold=0.3):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)

    # RetinaNet prediction
    with torch.no_grad():
        retina_preds = retina_model(img_tensor)
    retina_boxes = retina_preds[0]['boxes'].numpy()
    retina_scores = retina_preds[0]['scores'].numpy()

    # FasterRCNN prediction
    with torch.no_grad():
        faster_preds = fasterrcnn_model(img_tensor)
    faster_boxes = faster_preds[0]['boxes'].numpy()
    faster_scores = faster_preds[0]['scores'].numpy()

    # Combine predictions from both models
    combined_boxes = list(retina_boxes) + list(faster_boxes)
    combined_scores = list(retina_scores) + list(faster_scores)

    # Filter combined predictions by threshold
    combined_boxes = [box for box, score in zip(combined_boxes, combined_scores) if score > threshold]

    return combined_boxes, img

def visualize_predictions(image_path, ground_truth_boxes, threshold=0.3):
    # Get predictions
    pred_boxes, img = hybrid_detect(image_path, threshold)

    img_cv2 = cv2.imread(image_path)

    # Draw ground truth boxes (if available)
    for gt_box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, gt_box)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for GT

    # Draw predicted boxes
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red for Prediction
        cv2.putText(img_cv2, f"Pred", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the image with predictions and ground truth
    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def evaluate_predictions(pred_boxes, ground_truth_boxes):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Loop over predictions and ground truth to calculate true positives, false positives, and false negatives
    for pred_box in pred_boxes:
        best_iou = 0
        for gt_box in ground_truth_boxes:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)

        if best_iou >= 0.5:  # If IoU is greater than or equal to 0.5, consider it a true positive
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth_boxes) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print evaluation results
    print(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

# Example ground truth boxes (you need to replace these with actual ground truth data)
ground_truth_boxes = [
    [100, 150, 400, 450],  # Example ground truth box
]

# Example usage
image_path = r'C:\Users\VJ_Mahesh\OneDrive\Desktop\fog vehicle detection\datasets\images\train\foggy-014.jpg'
pred_boxes, img = hybrid_detect(image_path, threshold=0.3)
evaluate_predictions(pred_boxes, ground_truth_boxes)
visualize_predictions(image_path, ground_truth_boxes, threshold=0.3)
