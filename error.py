import os

# Define paths
labels_path = r"C:\Users\VJ_Mahesh\OneDrive\Desktop\fog vehicle detection\datasets\labels"  # Path to labels directory
max_classes = 2  # Number of classes specified in dataset.yaml (0, 1, 2)

def clean_labels():
    for split in ["train", "val", "test"]:
        split_path = os.path.join(labels_path, split)
        for file in os.listdir(split_path):
            if file.endswith(".txt"):
                file_path = os.path.join(split_path, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id <= max_classes:
                        new_lines.append(line)
                    else:
                        print(f"Warning: Removing class {class_id} in {file_path} as it exceeds max class {max_classes}")

                # Rewrite the cleaned label file
                with open(file_path, "w") as f:
                    f.writelines(new_lines)

clean_labels()
