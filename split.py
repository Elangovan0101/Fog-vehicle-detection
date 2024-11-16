import os
import shutil
import random

# Define paths
base_path = r"C:\Users\VJ_Mahesh\OneDrive\Desktop\fog vehicle detection\datasets"
images_path = os.path.join(base_path, r"C:\Users\VJ_Mahesh\Downloads\Fog\Fog")  # Folder with all images
labels_path = r"C:\Users\VJ_Mahesh\OneDrive\Desktop\fog vehicle detection\Fog_YOLO_darknet"  # Folder with YOLO labels
output_images_path = os.path.join(base_path, "images")
output_labels_path = os.path.join(base_path, "labels")

# Create train, val, and test directories for images and labels
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_images_path, split), exist_ok=True)
    os.makedirs(os.path.join(output_labels_path, split), exist_ok=True)

# List all images
all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Set split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate split counts
total_images = len(all_images)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# Split images into train, val, and test
train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

def move_files(image_list, split):
    for image_file in image_list:
        # Move image
        src_image = os.path.join(images_path, image_file)
        dst_image = os.path.join(output_images_path, split, image_file)
        shutil.copy(src_image, dst_image)

        # Move corresponding label file
        label_file = image_file.rsplit(".", 1)[0] + ".txt"
        src_label = os.path.join(labels_path, label_file)
        dst_label = os.path.join(output_labels_path, split, label_file)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

# Move files to their respective directories
move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("Dataset split and organized into train, val, and test sets.")
