import matplotlib.pyplot as plt
import numpy as np

# Model names and their respective accuracies (replace with actual accuracy values)
models = [
    "RetinaNet", "YOLO", "YOLO + SSD Hybrid", "Faster CNN", "RetinaNet + SSD Hybrid"
]
accuracies = [0.85, 0.78, 0.82, 0.88, 0.84]  # Example accuracy values, replace with your actual values

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar plot
bars = ax.bar(models, accuracies, color=['#FF6F61', '#6B8E23', '#FFD700', '#20B2AA', '#8A2BE2'])

# Add labels and title
ax.set_xlabel("Models", fontsize=14, fontweight='bold')
ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight='bold')
ax.set_ylim(0, 1)  # Set y-axis limits to 0-1 for accuracy

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2),
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the x-ticks and y-ticks for better readability
ax.tick_params(axis='both', labelsize=12)

# Add a grid for better visual appeal
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
