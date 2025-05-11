import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
from PIL import Image

from swinloopmodel import SwinSiamese  # Your Swin-based Siamese model
from swinloop_dataset import OverlapDataset  # Dataset class

# ====== CONFIGURATION ======
model_path = "swin_siamese_best.pth"  # Path to your trained model weights
csv_path = "SAMPLE_RANDOM/test.csv"  # Path to test CSV
img_dir = "SAMPLE_RANDOM/IMAGES"  # Directory with the images
model_name = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== TRANSFORMATIONS ======
processor = AutoImageProcessor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ====== LOAD DATASET ======
test_dataset = OverlapDataset(csv_file=csv_path, root_dir=img_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ====== LOAD MODEL ======
model = SwinSiamese().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ====== PERFORM PREDICTION ON FIRST BATCH ======
img1_batch, img2_batch, labels = next(iter(test_loader))
img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)
with torch.no_grad():
    outputs = model(img1_batch, img2_batch).squeeze()
predictions = (outputs > 0.5).float().cpu().numpy()

# ====== VISUALIZE MONTAGE ======

def create_montage(images, predictions, title, filename):
    """Creates a montage of images with prediction overlay."""
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i, ax in enumerate(axes):
        img = transforms.ToPILImage()(images[i].cpu())
        ax.imshow(img)
        ax.set_title(f"Pred: {int(predictions[i])}")
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[INFO] Saved montage: {filename}")
    plt.close()

# Save montages
create_montage(img1_batch, predictions, "Image 1 Predictions", "montage_input1.png")
create_montage(img2_batch, predictions, "Image 2 Predictions", "montage_input2.png")

# ====== PRINT SAMPLE OUTPUTS ======
print("First 10 predictions:", predictions[:10])
print("First 10 labels:     ", labels.numpy()[:10])
print("[INFO] Prediction visualization complete.")
